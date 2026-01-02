import argparse
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

# Adjust this import to match your actual dataset file / class name
from data.dataset import Z500ForecastDataset  
from models.unet import UnetZ500


def parseArgs():
    parser = argparse.ArgumentParser(description="Train UNet Z500 Emulator")

    parser.add_argument(
        "--dataDir",
        type=str,
        default="data/processed",
        help="Directory with preprocessed Z500 arrays.",
    )
    parser.add_argument(
        "--saveDir",
        type=str,
        default="runs/z500_unet",
        help="Directory to save checkpoints and logs.",
    )
    parser.add_argument(
        "--batchSize",
        type=int,
        default=32,
        help="Batch size.",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=20,
        help="Number of training epochs.",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=1e-3,
        help="Learning rate.",
    )
    parser.add_argument(
        "--numWorkers",
        type=int,
        default=4,
        help="Number of DataLoader workers.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device: 'cuda' or 'cpu'.",
    )
    parser.add_argument(
        "--baseChannels",
        type=int,
        default=32,
        help="Number of base channels in UNet.",
    )
    parser.add_argument(
        "--useResidual",
        action="store_true",
        help="If set, UNet uses residual output (x + f(x)).",
    )

    return parser.parse_args()


def getDataLoaders(
    dataDir: Path,
    batchSize: int,
    numWorkers: int,
):
    """
    Build train and validation DataLoaders.

    Assumes Z500ForecastDataset(split='train'/'val') reads from preprocessed .npy files.
    """

    trainDataset = Z500ForecastDataset(split="train", processedDir=dataDir)
    valDataset = Z500ForecastDataset(split="val", processedDir=dataDir)

    trainLoader = DataLoader(
        trainDataset,
        batch_size=batchSize,
        shuffle=True,
        num_workers=numWorkers,
        pin_memory=True,  # prefetch to pinned CPU memory for faster GPU transfer
        drop_last=True,   # drop last incomplete batch for consistent batch shapes
    )

    valLoader = DataLoader(
        valDataset,
        batch_size=batchSize,
        shuffle=False,
        num_workers=numWorkers,
        pin_memory=True,
        drop_last=False,
    )

    return trainLoader, valLoader


def saveCheckpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    valLoss: float,
    saveDir: Path,
):
    """
    Save model + optimizer state.
    """
    saveDir.mkdir(parents=True, exist_ok=True)
    ckptPath = saveDir / f"checkpoint_epoch{epoch:03d}_valloss{valLoss:.4f}.pt"
    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "val_loss": valLoss,
        },
        ckptPath,
    )
    print(f"Saved checkpoint to {ckptPath}")


def trainOneEpoch(
    model: torch.nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    lossFn: nn.Module,
    device: torch.device,
) -> float:
    """
    Single training epoch.
    """
    model.train()
    runningLoss = 0.0

    for batch in tqdm(loader, desc="Train", leave=False):
        x = batch["x"].to(device)  # (B, 1, H, W)
        y = batch["y"].to(device)  # (B, 1, H, W)

        optimizer.zero_grad()
        yPred = model(x)
        loss = lossFn(yPred, y)
        loss.backward()
        optimizer.step()

        runningLoss += loss.item() * x.size(0)

    epochLoss = runningLoss / len(loader.dataset)
    return epochLoss


def evaluate(
    model: torch.nn.Module,
    loader: DataLoader,
    lossFn: nn.Module,
    device: torch.device,
) -> float:
    """
    Validation loop.
    """
    model.eval()
    runningLoss = 0.0

    with torch.no_grad():
        for batch in tqdm(loader, desc="Val", leave=False):
            x = batch["x"].to(device)
            y = batch["y"].to(device)

            yPred = model(x)
            loss = lossFn(yPred, y)
            runningLoss += loss.item() * x.size(0)

    epochLoss = runningLoss / len(loader.dataset)
    return epochLoss


def main():
    args = parseArgs()

    dataDir = Path(args.dataDir)
    saveDir = Path(args.saveDir)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    trainLoader, valLoader = getDataLoaders(
        dataDir=dataDir,
        batchSize=args.batchSize,
        numWorkers=args.numWorkers,
    )

    # Model / loss / optimizer
    model = UnetZ500(
        inChannels=1,
        outChannels=1,
        baseChannels=args.baseChannels,
        useResidual=args.useResidual,
    ).to(device)

    lossFn = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    bestValLoss = float("inf")

    for epoch in range(1, args.epochs + 1):
        print(f"\nEpoch {epoch}/{args.epochs}")

        trainLoss = trainOneEpoch(
            model=model,
            loader=trainLoader,
            optimizer=optimizer,
            lossFn=lossFn,
            device=device,
        )

        valLoss = evaluate(
            model=model,
            loader=valLoader,
            lossFn=lossFn,
            device=device,
        )

        print(f"Train loss: {trainLoss:.6f} | Val loss: {valLoss:.6f}")

        # simple "best model" checkpointing
        if valLoss < bestValLoss:
            bestValLoss = valLoss
            saveCheckpoint(
                model=model,
                optimizer=optimizer,
                epoch=epoch,
                valLoss=valLoss,
                saveDir=saveDir,
            )


if __name__ == "__main__":
    main()