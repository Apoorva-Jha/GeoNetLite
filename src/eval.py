import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

from models.unet import UnetZ500


# -------------------------
# Dataset
# -------------------------
class Z500NpyDataset(Dataset):
    """
    Simple dataset that reads preprocessed Z500 samples from .npy files.

    Expects:
        xTest.npy, yTest.npy in dataDir with shapes (N, C, H, W)
    """

    def __init__(self, dataDir: Path, split: str = "test"):
        super().__init__()
        self.dataDir = Path(dataDir)

        # Normalize split to lowercase for consistent file naming
        split = split.lower()
        xPath = self.dataDir / f"x{split.capitalize()}.npy"
        yPath = self.dataDir / f"y{split.capitalize()}.npy"

        if not xPath.exists():
            raise FileNotFoundError(f"Missing file: {xPath}")
        if not yPath.exists():
            raise FileNotFoundError(f"Missing file: {yPath}")

        # Use memory mapping to avoid loading everything at once into RAM
        self.xArray = np.load(xPath, mmap_mode="r")
        self.yArray = np.load(yPath, mmap_mode="r")

        if self.xArray.shape != self.yArray.shape:
            raise ValueError(
                f"X and Y shapes do not match: {self.xArray.shape} vs {self.yArray.shape}"
            )

        self.numSamples = self.xArray.shape[0]

    def __len__(self):
        return self.numSamples

    def __getitem__(self, idx: int):
        x = self.xArray[idx]  # (C, H, W)
        y = self.yArray[idx]  # (C, H, W)

        # Convert to torch tensors (float32)
        x = torch.from_numpy(x.astype(np.float32))
        y = torch.from_numpy(y.astype(np.float32))

        return {"x": x, "y": y}


# -------------------------
# Metrics
# -------------------------
def computeCosLatWeights(latArray: np.ndarray) -> torch.Tensor:
    """
    Compute cosine-latitude area weights from a 1D latitude array (degrees).

    Returns:
        weights2d: (H, 1) tensor that can be broadcast over lon.
    """
    latRad = np.deg2rad(latArray)
    weights = np.cos(latRad)  # shape (H,)
    weights = np.clip(weights, 0.0, None)  # just in case of numerical weirdness

    # Shape (H, 1) so it broadcasts over lon
    weights2d = torch.from_numpy(weights.astype(np.float32))[:, None]
    return weights2d


def rmseWeighted(
    yTrue: torch.Tensor,
    yPred: torch.Tensor,
    latWeights2d: torch.Tensor
) -> float:
    """
    Area-weighted RMSE over (N, C, H, W) using cos(lat) weights.

    latWeights2d: (H, 1) tensor (cos(lat)), broadcast over lon.
    """
    # yTrue/yPred: (N, C, H, W)
    diff = yPred - yTrue
    sqErr = diff ** 2  # (N, C, H, W)

    H = yTrue.shape[-2]
    W = yTrue.shape[-1]

    # Build full 2D weights (H, W)
    w2d = latWeights2d.to(yTrue.device).repeat(1, W)  # (H, W)

    # Broadcast to (N, C, H, W)
    w4d = w2d[None, None, :, :]

    weightedSqErrSum = (sqErr * w4d).sum()
    weightsSum = w4d.sum() * yTrue.shape[0] * yTrue.shape[1]  # N * C copies

    mse = (weightedSqErrSum / weightsSum).item()
    return float(np.sqrt(mse))


# -------------------------
# Evaluation
# -------------------------
def evaluateModel(
    model: torch.nn.Module,
    dataLoader: DataLoader,
    latWeights2d: torch.Tensor,
    device: torch.device,
):
    """
    Evaluate UNet vs persistence baseline.

    Returns:
        dict with RMSEs.
    """
    model.eval()

    allYTrue = []
    allYPred = []
    allYPers = []

    with torch.no_grad():
        for batch in tqdm(dataLoader, desc="Evaluating", unit="batch"):
            x = batch["x"].to(device)  # (B, C, H, W)
            y = batch["y"].to(device)  # (B, C, H, W)

            yPred = model(x)  # (B, C, H, W)

            # Persistence baseline: yPers(t+lead) = x(t)
            yPers = x

            allYTrue.append(y.cpu())
            allYPred.append(yPred.cpu())
            allYPers.append(yPers.cpu())

    # Stack into (N, C, H, W)
    yTrueAll = torch.cat(allYTrue, dim=0)
    yPredAll = torch.cat(allYPred, dim=0)
    yPersAll = torch.cat(allYPers, dim=0)

    # Move weights to same device as tensors for metric calculation
    latWeights2d = latWeights2d.to(yTrueAll.device)

    rmseModel = rmseWeighted(yTrueAll, yPredAll, latWeights2d)
    rmsePers = rmseWeighted(yTrueAll, yPersAll, latWeights2d)

    metrics = {
        "rmse_model": rmseModel,
        "rmse_persistence": rmsePers,
    }

    return {
    "metrics": metrics,
    "yTrue": yTrueAll,
    "yPred": yPredAll,
    "yPers": yPersAll
}


def saveEvalResults(
    outDir: Path,
    metrics: dict,
    yTrue: torch.Tensor,
    yPred: torch.Tensor,
    yPers: torch.Tensor
):
    """
    Save evaluation outputs for downstream visualization.
    
    Saves:
      - metrics.json (RMSEs, experiment metadata)
      - yTrue.npy    (N, 1, H, W)
      - yPred.npy    (N, 1, H, W)
      - yPers.npy    (N, 1, H, W)
    """
    outDir.mkdir(parents=True, exist_ok=True)

    # Save metrics JSON
    metricsPath = outDir / "metrics.json"
    with metricsPath.open("w") as f:
        json.dump(metrics, f, indent=4)
    
    # Save arrays (convert torch → numpy)
    np.save(outDir / "yTrue.npy", yTrue.cpu().numpy())
    np.save(outDir / "yPred.npy", yPred.cpu().numpy())
    np.save(outDir / "yPers.npy", yPers.cpu().numpy())

    print(f"Saved evaluation results to: {outDir}")


# -------------------------
# Main
# -------------------------
def parseArgs():
    parser = argparse.ArgumentParser(
        description="Evaluate UNet Z500 emulator against persistence baseline."
    )

    parser.add_argument(
        "--data-dir",
        type=str,
        default="data/processed",
        help="Directory with xTest.npy, yTest.npy, lat.npy",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="/home/apoorva-jha/Documents/GeoNetLite/runs/z500_unet/checkpoint_epoch009_valloss0.0019.pt",
        help="Path to trained UNet checkpoint (.pt)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        help="Batch size for evaluation.",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=4,
        help="Number of DataLoader workers.",
    )
    parser.add_argument(
        "--no-cuda",
        action="store_true",
        help="Force evaluation on CPU even if CUDA is available.",
    )

    return parser.parse_args()


def main():
    args = parseArgs()

    dataDir = Path(args.data_dir)
    checkpointPath = Path(args.checkpoint)

    if not checkpointPath.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpointPath}")

    # Device
    useCuda = (not args.no_cuda) and torch.cuda.is_available()
    device = torch.device("cuda" if useCuda else "cpu")
    print(f"Using device: {device}")

    # Load latitude array for area-weighted metrics
    latPath = dataDir / "lat.npy"
    if not latPath.exists():
        raise FileNotFoundError(
            f"Latitude file not found: {latPath}. "
            "Please save lat.npy during preprocessing."
        )

    latArray = np.load(latPath)  # shape (H,)
    latWeights2d = computeCosLatWeights(latArray)  # (H, 1)

    # Dataset + DataLoader
    print("Loading test dataset...")
    testDataset = Z500NpyDataset(dataDir=dataDir, split="test")

    testLoader = DataLoader(
        testDataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False,
    )

    # Model
    print("Loading model...")
    # Adjust hyperparameters to match training
    model = UnetZ500(inChannels=1, outChannels=1, baseChannels=32, useResidual=False)
    
    # Load checkpoint (handle both full checkpoint dict and state dict)
    checkpoint = torch.load(checkpointPath, map_location=device)
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    model.to(device)

    # Evaluate model
    print("Evaluating UNet vs persistence baseline...")
    results = evaluateModel(model, testLoader, latWeights2d, device)

    metrics = results["metrics"]
    yTrue   = results["yTrue"]
    yPred   = results["yPred"]
    yPers   = results["yPers"]

    # Determine output directory name from checkpoint
    leadStr = checkpointPath.stem  # e.g., unetLead6h
    outDir = Path("results") / leadStr

    saveEvalResults(outDir, metrics, yTrue, yPred, yPers)

    print("\n=== Final Metrics ===")
    print(json.dumps(metrics, indent=4))


if __name__ == "__main__":
    main()

    