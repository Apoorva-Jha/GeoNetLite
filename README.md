# GeoNetLite

A lightweight UNet-based emulator for short-range Z500 atmospheric flow prediction, benchmarked against persistence and climatology using ERA5 reanalysis.

## Overview

**GeoNetLite** trains a U-Net neural network to predict Z500 geopotential height from ERA5 reanalysis data (1979-2019). The model achieves competitive skill against persistence baseline and climatology through area-weighted RMSE metrics.

- **Architecture**: UNet with configurable depth and residual connections
- **Data**: ERA5 reanalysis, 5.625° resolution, 32 latitude × 64 longitude
- **Training**: 20 epochs with learning rate scheduling and early stopping
- **Evaluation**: Area-weighted RMSE, MAE, skill score, and anomaly correlation

## Quick Start

### 1. Environment Setup

```bash
# Create conda environment
conda create -n geonet python=3.10
conda activate geonet

# Install dependencies
pip install -r requirements.txt
```

### 2. Data Preparation

```bash
# Preprocess ERA5 data (generates 2.1GB)
python src/data/preprocess.py
```

Outputs:
- `data/processed/xTrain.npy`, `yTrain.npy` (training)
- `data/processed/xVal.npy`, `yVal.npy` (validation)
- `data/processed/xTest.npy`, `yTest.npy` (testing)
- `data/processed/climatology.npz` (12-month climatology)
- `data/processed/lat.npy`, `lon.npy` (coordinates)

### 3. Train Model

```bash
# Default training (20 epochs, MSE loss, batch size 32)
python src/train.py

# Custom configuration
python src/train.py \
  --epochs 30 \
  --batchSize 64 \
  --lr 1e-3 \
  --lossFn mse \
  --baseChannels 32 \
  --useResidual \
  --patience 5 \
  --gradClip 1.0
```

**Output**: Checkpoint saved to `runs/z500_unet/checkpoint_epoch*.pt`

### 4. Evaluate Model

```bash
# Quick evaluation
python src/eval.py \
  --checkpoint runs/z500_unet/checkpoint_epoch009_valloss0.0019.pt \
  --baseChannels 32

# Full evaluation with config file
python src/eval.py --config evalConfig.yaml
```

**Output**: Metrics and predictions saved to `results/checkpoint_epoch*`

### 5. Visualize Results

```bash
# Generate all 5 visualizations (static, anomaly, spectral, RMSE, GIF)
python src/visualize.py \
  --resultDir results/checkpoint_epoch009_valloss0.0019 \
  --latPath data/processed/lat.npy \
  --lonPath data/processed/lon.npy
```

**Output**:
- `fig_staticPanels.png` - Truth vs UNet vs Persistence vs Climatology
- `fig_anomalyPanels.png` - Anomaly deviations from climatology
- `fig_spectralDiagnosis.png` - Power spectra by zonal wavenumber
- `fig_rmseTimeSeries.png` - Temporal RMSE comparison
- `anim_forecasts.gif` - Animated forecast sequence

## Training Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--dataDir` | `data/processed` | Preprocessed data directory |
| `--saveDir` | `runs/z500_unet` | Checkpoint save directory |
| `--batchSize` | 32 | Training batch size |
| `--epochs` | 20 | Number of epochs |
| `--lr` | 1e-3 | Learning rate |
| `--numWorkers` | 4 | DataLoader workers |
| `--device` | cuda | Device (cuda or cpu) |
| `--baseChannels` | 32 | UNet base channels |
| `--useResidual` | False | Use residual connections |
| `--lossFn` | mse | Loss function (mse, mae, huber) |
| `--seed` | 42 | Random seed |
| `--patience` | 5 | Early stopping patience |
| `--gradClip` | 1.0 | Gradient clipping (0 = disabled) |

## Evaluation Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--dataDir` | `data/processed` | Data directory with test arrays |
| `--checkpoint` | (required) | Path to trained checkpoint |
| `--baseChannels` | 32 | UNet base channels (must match training) |
| `--useResidual` | False | Residual flag (must match training) |
| `--batchSize` | 32 | Evaluation batch size |
| `--numWorkers` | 4 | DataLoader workers |
| `--device` | cuda | Device (cuda or cpu) |
| `--config` | None | YAML config file (overrides defaults) |

## Visualization Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--resultDir` | (required) | Directory with yTrue.npy, yPred.npy, yPers.npy |
| `--sampleIdx` | 0 | Sample index for static/anomaly/spectral |
| `--gifStartIdx` | 0 | GIF start frame |
| `--gifEndIdx` | 96 | GIF end frame |
| `--gifStep` | 1 | GIF frame stride |
| `--latPath` | None | Path to lat.npy |
| `--lonPath` | None | Path to lon.npy |

## Configuration Files

- **`evalConfig.yaml`** - Evaluation settings with checkpoint path
- See `EVALCONFIG_GUIDE.md` for detailed config examples

## Project Structure

```
GeoNetLite/
├── src/
│   ├── train.py           # Training script
│   ├── eval.py            # Evaluation script
│   ├── visualize.py       # Visualization script
│   ├── models/
│   │   └── unet.py        # UNet architecture
│   └── data/
│       ├── dataset.py     # Dataset classes
│       └── preprocess.py  # Preprocessing pipeline
├── data/
│   ├── raw/               # ERA5 NetCDF files
│   └── processed/         # Preprocessed arrays
├── results/               # Evaluation outputs
├── runs/                  # Training checkpoints
├── notebooks/             # Jupyter notebooks
│   └── 01_results_overview.ipynb
├── evalConfig.yaml        # Evaluation config template
└── README.md
```

## Notebooks

**`notebooks/01_results_overview.ipynb`** - Load predictions and reproduce:
- RMSE time series with skill improvement
- Spectral diagnosis (zonal wavenumber)
- Anomaly panels (deviations from climatology)
- Performance metrics table

Run with: `jupyter notebook notebooks/01_results_overview.ipynb`

## Key Metrics

- **RMSE**: Root mean square error (area-weighted by latitude)
- **MAE**: Mean absolute error
- **Skill Score**: Percentage improvement over persistence baseline
- **Anomaly Correlation**: Pattern correlation with observations

## Requirements

- Python 3.8+
- PyTorch 2.0+
- NumPy, Pandas, xarray
- Cartopy (visualization)
- Imageio (GIF generation)

Install with: `pip install -r requirements.txt`

## References

- ERA5 Reanalysis: Hersbach et al. (2020)
- U-Net Architecture: Ronneberger et al. (2015)

## License

MIT License - See LICENSE file for details.
