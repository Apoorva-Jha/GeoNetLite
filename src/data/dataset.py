import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from pathlib import Path


class Z500ForecastDataset(Dataset):
    """
    Loads preprocessed ERA5 Z500 forecasting data.
    Each item is a dict containing:
        - x: input field   (1, lat, lon)
        - y: target field  (1, lat, lon)
    """

    def __init__(self, processedDir, split="train", subsetSize=None):
        """
        processedDir: Path to data/processed/
        split: "train", "val", or "test"
        subsetSize: optionally use only N samples (debug mode)
        """

        self.processedDir = Path(processedDir)

        if split == "train":
            xPath = self.processedDir / "xTrain.npy"
            yPath = self.processedDir / "yTrain.npy"
        elif split == "val":
            xPath = self.processedDir / "xVal.npy"
            yPath = self.processedDir / "yVal.npy"
        elif split == "test":
            xPath = self.processedDir / "xTest.npy"
            yPath = self.processedDir / "yTest.npy"
        else:
            raise ValueError(f"Unknown split: {split}")

        if not xPath.exists():
            raise FileNotFoundError(f"Cannot find file {xPath}. Run preprocessing first.")

        self.x = np.load(xPath)
        self.y = np.load(yPath)

        if subsetSize is not None:
            self.x = self.x[:subsetSize]
            self.y = self.y[:subsetSize]

        # Convert to float32 if needed
        self.x = self.x.astype(np.float32)
        self.y = self.y.astype(np.float32)

    def __len__(self):
        return self.x.shape[0]

    def __getitem__(self, idx):
        # Convert numpy → torch
        xTensor = torch.from_numpy(self.x[idx])
        yTensor = torch.from_numpy(self.y[idx])
        return {"x": xTensor, "y": yTensor}


def getDataLoaders(
    processedDir,
    batchSize=32,
    numWorkers=4,
    subsetSize=None,
    shuffleTrain=True
):
    """
    Returns PyTorch dataloaders for train/val/test.

    subsetSize: if set, dataloaders return only N samples (for debugging)
    shuffleTrain: True for random shuffle
    """

    processedDir = Path(processedDir)

    trainDs = Z500ForecastDataset(processedDir, split="train", subsetSize=subsetSize)
    valDs   = Z500ForecastDataset(processedDir, split="val",   subsetSize=subsetSize)
    testDs  = Z500ForecastDataset(processedDir, split="test",  subsetSize=subsetSize)

    trainLoader = DataLoader(
        trainDs,
        batch_size=batchSize,
        shuffle=shuffleTrain,
        num_workers=numWorkers,
        pin_memory=True,
        drop_last=True,
    )

    valLoader = DataLoader(
        valDs,
        batch_size=batchSize,
        shuffle=False,
        num_workers=numWorkers,
        pin_memory=True,
        drop_last=False,
    )

    testLoader = DataLoader(
        testDs,
        batch_size=batchSize,
        shuffle=False,
        num_workers=numWorkers,
        pin_memory=True,
        drop_last=False,
    )

    return trainLoader, valLoader, testLoader


# num_workers > 0 enables multiple CPU processes to load data in parallel.
# This keeps the GPU fully utilized instead of waiting for batches.

# pin_memory=True moves tensors to pinned (page-locked) RAM.
# CUDA can transfer pinned memory to GPU much faster via DMA.

# drop_last=True ensures all batches have equal size.
# This stabilizes training (especially BatchNorm) and avoids tiny final batches.