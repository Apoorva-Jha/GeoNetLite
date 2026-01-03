import argparse
import json
import shutil
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

try:
    import yaml
except ImportError:
    yaml = None

from models.unet import UnetZ500


def loadConfigYaml(configPath: str) -> dict:
    """Load YAML config file and return as dict."""
    if yaml is None:
        raise ImportError("PyYAML not installed. Install with: pip install pyyaml")
    
    with open(configPath, 'r') as f:
        config = yaml.safe_load(f)
    return config


def mergeConfigWithArgs(config: dict, args) -> dict:
    """Merge YAML config with command-line args (args override config)."""
    # Type mappings for conversion
    typeMap = {
        'batchSize': int, 'numWorkers': int,
        'baseChannels': int
    }
    
    # Start with empty dict (no defaults)
    merged = {}
    
    # Flatten nested config dict
    for section, values in config.items():
        if isinstance(values, dict):
            merged.update(values)
        else:
            merged[section] = values
    
    # Override with command-line args (only if not None)
    for key, value in vars(args).items():
        if value is not None and key != 'config':
            # Convert to correct type if in typeMap
            if key in typeMap:
                value = typeMap[key](value)
            merged[key] = value
    
    # Ensure critical fields have correct types
    for key, expected_type in typeMap.items():
        if key in merged:
            merged[key] = expected_type(merged[key])
    
    return merged


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


def maeWeighted(
    yTrue: torch.Tensor,
    yPred: torch.Tensor,
    latWeights2d: torch.Tensor
) -> float:
    """Area-weighted Mean Absolute Error."""
    diff = torch.abs(yPred - yTrue)
    H = yTrue.shape[-2]
    W = yTrue.shape[-1]
    w2d = latWeights2d.to(yTrue.device).repeat(1, W)
    w4d = w2d[None, None, :, :]
    weightedSum = (diff * w4d).sum()
    weightsSum = w4d.sum() * yTrue.shape[0] * yTrue.shape[1]
    return float((weightedSum / weightsSum).item())


def anomalyCorrelation(
    yTrue: torch.Tensor,
    yPred: torch.Tensor,
    latWeights2d: torch.Tensor
) -> float:
    """
    Anomaly Correlation Coefficient (ACC).
    Measures spatial pattern correlation of anomalies.
    """
    # Compute anomalies (deviations from temporal mean)
    yTrueMean = yTrue.mean(dim=0, keepdim=True)
    yPredMean = yPred.mean(dim=0, keepdim=True)
    
    truthAnom = yTrue - yTrueMean
    predAnom = yPred - yPredMean
    
    # Area-weighted correlation
    H = yTrue.shape[-2]
    W = yTrue.shape[-1]
    w2d = latWeights2d.to(yTrue.device).repeat(1, W)
    w4d = w2d[None, None, :, :]
    
    numerator = (truthAnom * predAnom * w4d).sum()
    denominator = torch.sqrt(
        (truthAnom ** 2 * w4d).sum() * (predAnom ** 2 * w4d).sum()
    )
    
    if denominator < 1e-10:
        return 0.0
    return float((numerator / denominator).item())


def skillScore(rmseModel: float, rmsePers: float) -> float:
    """
    Skill score: percentage improvement of model over persistence.
    Positive = model better, Negative = persistence better
    """
    if rmsePers == 0:
        return 0.0
    return float(100 * (1 - rmseModel / rmsePers))


def loadModelFromCheckpoint(checkpointPath: Path, device: torch.device, 
                            baseChannels: int = 32, useResidual: bool = False):
    """
    Load model from checkpoint with config validation.
    """
    if not checkpointPath.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpointPath}")
    
    checkpoint = torch.load(checkpointPath, map_location=device)
    
    # Try to load config from checkpoint
    if isinstance(checkpoint, dict) and 'config' in checkpoint:
        config = checkpoint['config']
        baseChannels = config.get('baseChannels', baseChannels)
        useResidual = config.get('useResidual', useResidual)
        print(f"[Checkpoint] Loaded config from checkpoint: baseChannels={baseChannels}, useResidual={useResidual}")
    
    # Create model with config
    model = UnetZ500(
        inChannels=1,
        outChannels=1,
        baseChannels=baseChannels,
        useResidual=useResidual,
    )
    
    # Load weights
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    
    return model.to(device)


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
        dict with metrics and predictions.
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

    # Compute all metrics
    rmseModel = rmseWeighted(yTrueAll, yPredAll, latWeights2d)
    rmsePers = rmseWeighted(yTrueAll, yPersAll, latWeights2d)
    
    maeModel = maeWeighted(yTrueAll, yPredAll, latWeights2d)
    maePers = maeWeighted(yTrueAll, yPersAll, latWeights2d)
    
    accModel = anomalyCorrelation(yTrueAll, yPredAll, latWeights2d)
    accPers = anomalyCorrelation(yTrueAll, yPersAll, latWeights2d)
    
    skillRMSE = skillScore(rmseModel, rmsePers)
    skillMAE = skillScore(maeModel, maePers)

    metrics = {
        "rmse": {
            "model": round(rmseModel, 6),
            "persistence": round(rmsePers, 6),
            "skill": round(skillRMSE, 2),
        },
        "mae": {
            "model": round(maeModel, 6),
            "persistence": round(maePers, 6),
            "skill": round(skillMAE, 2),
        },
        "anomaly_correlation": {
            "model": round(accModel, 4),
            "persistence": round(accPers, 4),
        },
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
    yPers: torch.Tensor,
    dataDir: Path = None,
):
    """
    Save evaluation outputs for downstream visualization.
    
    Saves:
      - metrics.json (all error metrics)
      - yTrue.npy, yPred.npy, yPers.npy (N, C, H, W)
      - lat.npy, lon.npy (if available)
      - climatology.npz (if available)
    """
    outDir.mkdir(parents=True, exist_ok=True)

    # Save metrics JSON
    metricsPath = outDir / "metrics.json"
    with metricsPath.open("w") as f:
        json.dump(metrics, f, indent=2)
    print(f"✅ Metrics saved to {metricsPath}")
    
    # Save arrays (convert torch → numpy)
    np.save(outDir / "yTrue.npy", yTrue.cpu().numpy())
    np.save(outDir / "yPred.npy", yPred.cpu().numpy())
    np.save(outDir / "yPers.npy", yPers.cpu().numpy())
    print(f"✅ Predictions saved to {outDir}")

    # Copy supporting files for visualization if dataDir provided
    if dataDir is not None:
        dataDir = Path(dataDir)
        
        # Copy lat/lon arrays
        for fname in ["lat.npy", "lon.npy"]:
            src = dataDir / fname
            dst = outDir / fname
            if src.exists() and not dst.exists():
                shutil.copy(src, dst)
                print(f"✅ Copied {fname} to results")
            elif not src.exists():
                print(f"⚠️  {fname} not found in {dataDir}")
        
        # Copy climatology from parent directory
        climSrc = dataDir.parent / "climatology.npz"
        climDst = outDir / "climatology.npz"
        if climSrc.exists() and not climDst.exists():
            shutil.copy(climSrc, climDst)
            print(f"✅ Copied climatology.npz to results")
        elif not climSrc.exists():
            print(f"⚠️  climatology.npz not found in {climSrc.parent}")

    print(f"\n✅ All results saved to: {outDir}")


# -------------------------
# Main
# -------------------------
def parseArgs():
    parser = argparse.ArgumentParser(
        description="Evaluate UNet Z500 emulator against persistence baseline."
    )

    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="YAML config file (e.g., config/evalConfig.yaml)",
    )
    # Core overrides (commonly changed)
    parser.add_argument("--checkpoint", type=str, default=None, help="Checkpoint path")
    parser.add_argument("--batchSize", type=int, default=None, help="Batch size")
    parser.add_argument("--device", type=str, default=None, help="Device (cuda/cpu)")
    parser.add_argument("--baseChannels", type=int, default=None, help="Base channels")
    parser.add_argument("--outDir", type=str, default=None, help="Output directory")

    return parser.parse_args()


def main():
    args = parseArgs()

    # Load YAML config if provided
    configPath = args.config
    if configPath:
        config = loadConfigYaml(configPath)
        args = argparse.Namespace(**mergeConfigWithArgs(config, args))
        print(f"✓ Loaded config from {configPath}")

    dataDir = Path(args.dataDir)
    checkpointPath = Path(args.checkpoint)

    if not checkpointPath.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpointPath}")

    # Device setup
    if args.device == "cuda" and not torch.cuda.is_available():
        print("⚠️  CUDA requested but not available, falling back to CPU")
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load latitude array for area-weighted metrics
    latPath = dataDir / "lat.npy"
    if not latPath.exists():
        raise FileNotFoundError(
            f"Latitude file not found: {latPath}\n"
            f"Expected in: {dataDir.absolute()}"
        )

    latArray = np.load(latPath)  # shape (H,)
    latWeights2d = computeCosLatWeights(latArray)  # (H, 1)

    # Dataset + DataLoader
    print("Loading test dataset...")
    testDataset = Z500NpyDataset(dataDir=dataDir, split="test")
    print(f"  Dataset size: {len(testDataset)} samples")

    testLoader = DataLoader(
        testDataset,
        batch_size=args.batchSize,
        shuffle=False,
        num_workers=args.numWorkers,
        pin_memory=True,
        drop_last=False,
    )

    # Load model from checkpoint (with config handling)
    print("Loading model...")
    startLoadTime = time.time()
    model = loadModelFromCheckpoint(
        checkpointPath, 
        device, 
        baseChannels=args.baseChannels,
        useResidual=args.useResidual
    )
    loadTime = time.time() - startLoadTime
    print(f"  Model loaded in {loadTime:.2f}s")

    # Evaluate model
    print("\nEvaluating UNet vs persistence baseline...")
    startEvalTime = time.time()
    results = evaluateModel(model, testLoader, latWeights2d, device)
    evalTime = time.time() - startEvalTime

    metrics = results["metrics"]
    yTrue = results["yTrue"]
    yPred = results["yPred"]
    yPers = results["yPers"]

    # Add timing and dataset info to metrics
    metrics["evaluation_time_seconds"] = round(evalTime, 2)
    metrics["num_samples"] = len(testDataset)
    metrics["batch_size"] = args.batchSize
    metrics["checkpoint"] = str(checkpointPath)
    metrics["device"] = str(device)

    # Determine output directory
    if hasattr(args, 'saveDir') and args.saveDir:
        outDir = Path(args.saveDir)
    else:
        # Use checkpoint stem as directory name
        checkpointName = checkpointPath.stem
        outDir = Path("results") / checkpointName

    # Save results (including supporting files)
    saveEvalResults(outDir, metrics, yTrue, yPred, yPers, dataDir=dataDir)

    # Print summary
    print("\n" + "="*70)
    print("EVALUATION SUMMARY")
    print("="*70)
    print(json.dumps(metrics, indent=2))
    print("="*70)
    print(f"Evaluation completed in {evalTime:.2f} seconds")


if __name__ == "__main__":
    main()