import argparse
import json
import random
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

try:
    import yaml
except ImportError:
    yaml = None

# Adjust this import to match your actual dataset file / class name
from data.dataset import Z500ForecastDataset  
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
    merged = {}
    
    # Type mappings for conversion
    typeMap = {
        'epochs': int, 'batchSize': int, 'numWorkers': int,
        'lr': float, 'seed': int, 'patience': int, 'gradClip': float,
        'baseChannels': int
    }
    
    # Flatten nested config dict
    for section, values in config.items():
        if isinstance(values, dict):
            merged.update(values)
        else:
            merged[section] = values
    
    # Override with command-line args (only if not None)
    for key, value in vars(args).items():
        if value is not None:
            # Convert to correct type if in typeMap
            if key in typeMap:
                value = typeMap[key](value)
            merged[key] = value
    
    # Ensure critical fields have correct types
    for key, expected_type in typeMap.items():
        if key in merged:
            merged[key] = expected_type(merged[key])
    
    return merged


def setSeed(seed: int = 42):
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    print(f"[Reproducibility] Random seed set to {seed}")


def parseArgs():
    parser = argparse.ArgumentParser(description="Train UNet Z500 Emulator")

    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="YAML config file (e.g., config/trainConfig.yaml)",
    )
    # Core overrides (commonly changed)
    parser.add_argument("--epochs", type=int, default=None, help="Number of epochs")
    parser.add_argument("--batchSize", type=int, default=None, help="Batch size")
    parser.add_argument("--lr", type=float, default=None, help="Learning rate")
    parser.add_argument("--device", type=str, default=None, help="Device (cuda/cpu)")
    parser.add_argument("--saveDir", type=str, default=None, help="Checkpoint directory")

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
    gradClip: float = 1.0,
) -> float:
    """
    Single training epoch with optional gradient clipping.
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
        
        # Gradient clipping
        if gradClip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=gradClip)
        
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

    # Load YAML config if provided
    if args.config:
        config = loadConfigYaml(args.config)
        args = argparse.Namespace(**mergeConfigWithArgs(config, args))
        print(f"✓ Loaded config from {args.config}")

    # Set seeds for reproducibility
    setSeed(args.seed)

    dataDir = Path(args.dataDir)
    saveDir = Path(args.saveDir)
    saveDir.mkdir(parents=True, exist_ok=True)

    # Device setup
    if args.device == "cuda" and not torch.cuda.is_available():
        print("⚠️  CUDA requested but not available, falling back to CPU")
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Save config for reproducibility
    configPath = saveDir / "config.json"
    with open(configPath, "w") as f:
        json.dump(vars(args), f, indent=2)
    print(f"Config saved to {configPath}")

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

    # Loss function (configurable)
    if args.lossFn == "mse":
        lossFn = nn.MSELoss()
    elif args.lossFn == "mae":
        lossFn = nn.L1Loss()
    elif args.lossFn == "huber":
        lossFn = nn.HuberLoss()
    else:
        raise ValueError(f"Unknown loss function: {args.lossFn}")
    print(f"Loss function: {args.lossFn}")

    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    
    # Learning rate scheduler
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs
    )
    print(f"LR scheduler: CosineAnnealingLR (T_max={args.epochs})")

    bestValLoss = float("inf")
    epocsNoImprove = 0
    
    # Training history
    trainingHistory = {
        "epochs": [],
        "trainLoss": [],
        "valLoss": [],
        "lr": [],
    }

    for epoch in range(1, args.epochs + 1):
        print(f"\n{'='*60}")
        print(f"Epoch {epoch}/{args.epochs}")
        print(f"{'='*60}")

        trainLoss = trainOneEpoch(
            model=model,
            loader=trainLoader,
            optimizer=optimizer,
            lossFn=lossFn,
            device=device,
            gradClip=args.gradClip,
        )

        valLoss = evaluate(
            model=model,
            loader=valLoader,
            lossFn=lossFn,
            device=device,
        )

        currentLr = optimizer.param_groups[0]["lr"]
        print(f"Train loss: {trainLoss:.6f} | Val loss: {valLoss:.6f} | LR: {currentLr:.6e}")

        # Record history
        trainingHistory["epochs"].append(epoch)
        trainingHistory["trainLoss"].append(float(trainLoss))
        trainingHistory["valLoss"].append(float(valLoss))
        trainingHistory["lr"].append(float(currentLr))

        # Best model checkpointing
        if valLoss < bestValLoss:
            bestValLoss = valLoss
            epocsNoImprove = 0
            saveCheckpoint(
                model=model,
                optimizer=optimizer,
                epoch=epoch,
                valLoss=valLoss,
                saveDir=saveDir,
            )
        else:
            epocsNoImprove += 1
            print(f"⚠️  No improvement ({epocsNoImprove}/{args.patience})")
            
            # Early stopping
            if epocsNoImprove >= args.patience:
                print(f"\n🛑 Early stopping triggered at epoch {epoch}")
                break

        # Learning rate step
        scheduler.step()

    # Save training history
    historyPath = saveDir / "training_history.json"
    with open(historyPath, "w") as f:
        json.dump(trainingHistory, f, indent=2)
    print(f"\n✅ Training history saved to {historyPath}")
    print(f"✅ Best validation loss: {bestValLoss:.6f}")


if __name__ == "__main__":
    main()