import argparse
import json
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import imageio.v2 as imageio

import cartopy.crs as ccrs
import cartopy.feature as cfeature

# python src/visualize.py --resultDir results/checkpoint_epoch005_valloss0.0019/ --latPath data/processed/lat.npy --lonPath data/processed/lon.npy

# -----------------------------
# IO helpers
# -----------------------------
def loadNpy(path: Path) -> np.ndarray:
    if not path.exists():
        raise FileNotFoundError(f"Missing file: {path}")
    return np.load(path)

def ensure3d(arr: np.ndarray) -> np.ndarray:
    """
    Expected shapes:
      - (N, H, W)
      - (N, 1, H, W) -> will squeeze channel dim
      - (H, W) -> will add N=1
    """
    if arr.ndim == 4:
        # (N, C, H, W)
        if arr.shape[1] != 1:
            raise ValueError(f"Expected channel=1, got shape {arr.shape}")
        arr = arr[:, 0, :, :]
    elif arr.ndim == 2:
        arr = arr[None, :, :]
    elif arr.ndim != 3:
        raise ValueError(f"Unexpected array shape: {arr.shape}")
    return arr

def loadLatLon(resultDir: Path):
    # Prefer lat/lon saved alongside results if you have them
    latPath = resultDir / "lat.npy"
    lonPath = resultDir / "lon.npy"
    if latPath.exists() and lonPath.exists():
        lat = np.load(latPath)
        lon = np.load(lonPath)
        return lat, lon
    return None, None


# -----------------------------
# Plotting core
# -----------------------------
def getLonLat2d(lon: np.ndarray, lat: np.ndarray):
    # lon: (W,), lat: (H,) -> meshgrid gives (H, W)
    lon2d, lat2d = np.meshgrid(lon, lat)
    return lon2d, lat2d

def normalizeLon(lon: np.ndarray) -> np.ndarray:
    """
    If lon is 0..360, keep it.
    If lon is -180..180, keep it.
    Just return as-is, but ensure increasing order.
    """
    lon = np.asarray(lon)
    if np.any(np.diff(lon) < 0):
        lon = np.sort(lon)
    return lon

def maybeFlipLat(field2d: np.ndarray, lat: np.ndarray):
    """
    Ensure latitude is increasing (south->north) for consistent meshgrid.
    If lat is decreasing, flip both lat and field.
    """
    lat = np.asarray(lat)
    if lat.size == 0:
        raise ValueError("lat is empty — your latitude subset produced 0 rows.")
    if lat[0] > lat[-1]:
        lat = lat[::-1]
        field2d = field2d[::-1, :]
    return field2d, lat

def plotFieldOnAxis(ax, field2d, lat, lon, title, vmin=None, vmax=None):
    # fix lat orientation if needed
    field2d, lat = maybeFlipLat(field2d, lat)

    lon = normalizeLon(lon)

    lon2d, lat2d = getLonLat2d(lon, lat)

    img = ax.pcolormesh(
        lon2d,
        lat2d,
        field2d,
        transform=ccrs.PlateCarree(),
        shading="auto",   # IMPORTANT: avoids many shape mismatch issues
        vmin=vmin,
        vmax=vmax,
    )
    ax.set_title(title)
    ax.coastlines(linewidth=0.7)
    ax.add_feature(cfeature.BORDERS, linewidth=0.4, alpha=0.5)
    ax.gridlines(draw_labels=False, linewidth=0.3, alpha=0.4)
    return img

def computeCommonVminVmax(fieldsList):
    # robust scaling for consistent colorbars across panels
    # filter out None values
    fields = [f for f in fieldsList if f is not None]
    if not fields:
        return 0.0, 1.0
    stack = np.stack(fields, axis=0)
    vmin = np.nanpercentile(stack, 2)
    vmax = np.nanpercentile(stack, 98)
    return float(vmin), float(vmax)


# -----------------------------
# Diagnostics
# -----------------------------
def computeAnomaly(y, yClim):
    """
    y, yClim shape: (N, H, W)
    If yClim is (H, W), broadcast.
    """
    if yClim.ndim == 2:
        return y - yClim[None, :, :]
    return y - yClim

def zonalWavenumberSpectrum(field2d):
    """
    Simple spectral diagnostic:
      - remove zonal mean
      - FFT along lon axis
      - return power vs zonal wavenumber k
    """
    f = field2d - np.nanmean(field2d, axis=1, keepdims=True)  # remove zonal mean per-lat
    f = np.nan_to_num(f, nan=0.0)

    fft = np.fft.rfft(f, axis=1)
    power = np.mean(np.abs(fft) ** 2, axis=0)  # average over lat

    k = np.arange(power.shape[0])
    return k, power

def computeLatWeights(lat: np.ndarray) -> np.ndarray:
    """Compute cosine-latitude area weights from latitude array."""
    latRad = np.deg2rad(lat)
    weights = np.cos(latRad)
    weights = weights / weights.mean()  # Normalize
    return weights

def computeRMSETimeSeries(yTrue: np.ndarray, yPred: np.ndarray, lat: np.ndarray) -> np.ndarray:
    """
    Compute area-weighted RMSE for each sample.
    
    Args:
        yTrue: (N, H, W) - true values (already 3D)
        yPred: (N, H, W) - predictions (already 3D)
        lat:   (H,) - latitude array
    
    Returns:
        rmse: (N,) - RMSE for each sample
    """
    latWeights = computeLatWeights(lat)  # (H,)
    
    # Flatten spatial dimensions: (N, H, W) -> (N, H*W)
    yTrue_flat = yTrue.reshape(yTrue.shape[0], -1)
    yPred_flat = yPred.reshape(yPred.shape[0], -1)
    
    # Compute squared errors per sample
    diff = yPred_flat - yTrue_flat  # (N, H*W)
    sqErr = diff ** 2  # (N, H*W)
    
    # Apply area weights: replicate lat weights across W
    weights_flat = np.tile(latWeights, yTrue.shape[2])  # (H*W,)
    
    # Compute weighted MSE and then RMSE
    wmse = (sqErr * weights_flat).mean(axis=1)  # (N,)
    rmse = np.sqrt(wmse)  # (N,)
    
    return rmse

def makeRMSETimeSeries(outDir: Path, yTrue, yPred, yPers, lat):
    """
    Create RMSE time series plot comparing UNet vs Persistence.
    
    Args:
        outDir: Output directory
        yTrue: (N, H, W) - true values
        yPred: (N, H, W) - UNet predictions
        yPers: (N, H, W) - Persistence baseline
        lat: (H,) - latitude array
    """
    # Compute RMSE time series
    rmseModel = computeRMSETimeSeries(yTrue, yPred, lat)
    rmsePers = computeRMSETimeSeries(yTrue, yPers, lat)
    
    # Create plot
    fig, ax = plt.subplots(figsize=(14, 6))
    
    ax.plot(rmseModel, label="UNet", color="steelblue", linewidth=1.5, alpha=0.8)
    ax.plot(rmsePers, label="Persistence", color="orange", linewidth=1.5, alpha=0.8)
    
    ax.set_xlabel("Sample Index", fontsize=12)
    ax.set_ylabel("Area-weighted RMSE", fontsize=12)
    ax.set_title("RMSE Time Series (Test Set)", fontsize=14, fontweight="bold")
    ax.legend(fontsize=11, loc="upper right")
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    figPath = outDir / "fig_rmseTimeSeries.png"
    fig.savefig(figPath, dpi=150, bbox_inches='tight')
    plt.close(fig)
    return figPath

def makeSpectralDiagnosis(outDir: Path, yTrue, yPred, yPers, yClim, sampleIdx=0):
    truth = yTrue[sampleIdx]
    pred  = yPred[sampleIdx]
    pers  = yPers[sampleIdx]
    
    # Handle climatology indexing: (12, H, W) -> use average or modulo
    if yClim is not None:
        if yClim.ndim == 3 and yClim.shape[0] == 12:
            clim = np.mean(yClim, axis=0)  # average across months
        elif yClim.ndim == 3:
            clim = yClim[sampleIdx % yClim.shape[0]]
        else:
            clim = yClim
    else:
        clim = None

    kT, pT = zonalWavenumberSpectrum(truth)
    kP, pP = zonalWavenumberSpectrum(pred)
    kR, pR = zonalWavenumberSpectrum(pers)

    plt.figure(figsize=(8, 4))
    plt.semilogy(kT[1:], pT[1:], label="Truth")
    plt.semilogy(kP[1:], pP[1:], label="UNet")
    plt.semilogy(kR[1:], pR[1:], label="Persistence")
    if clim is not None:
        kC, pC = zonalWavenumberSpectrum(clim)
        plt.semilogy(kC[1:], pC[1:], label="Climatology")
    plt.xlabel("Zonal wavenumber k")
    plt.ylabel("Power (log scale)")
    plt.title("Spectral diagnosis (zonal wavenumber power)")
    plt.legend()
    plt.tight_layout()
    figPath = outDir / "fig_spectralDiagnosis.png"
    plt.savefig(figPath, dpi=200)
    plt.close()
    return figPath


# -----------------------------
# Figures & GIF
# -----------------------------
def makeStaticPanels(outDir: Path, yTrue, yPred, yPers, yClim, lat, lon, sampleIdx=0):
    truth = yTrue[sampleIdx]
    pred  = yPred[sampleIdx]
    pers  = yPers[sampleIdx]
    
    # Handle climatology indexing: (12, H, W) -> use average or modulo
    if yClim is not None:
        if yClim.ndim == 3 and yClim.shape[0] == 12:
            clim = np.mean(yClim, axis=0)  # average across months
        elif yClim.ndim == 3:
            clim = yClim[sampleIdx % yClim.shape[0]]
        else:
            clim = yClim
    else:
        clim = None

    vmin, vmax = computeCommonVminVmax([truth, pred, pers, clim])

    # Use GridSpec for better layout control with Cartopy
    fig = plt.figure(figsize=(14, 9))
    gs = fig.add_gridspec(3, 2, height_ratios=[1, 1, 0.08], hspace=0.3, wspace=0.2)
    
    axs = [
        fig.add_subplot(gs[0, 0], projection=ccrs.PlateCarree()),
        fig.add_subplot(gs[0, 1], projection=ccrs.PlateCarree()),
        fig.add_subplot(gs[1, 0], projection=ccrs.PlateCarree()),
        fig.add_subplot(gs[1, 1], projection=ccrs.PlateCarree()),
    ]

    img00 = plotFieldOnAxis(axs[0], truth, lat, lon, "Z500 Truth", vmin, vmax)
    img01 = plotFieldOnAxis(axs[1], pred,  lat, lon, "Z500 UNet",  vmin, vmax)
    img10 = plotFieldOnAxis(axs[2], pers,  lat, lon, "Z500 Persistence", vmin, vmax)
    
    if clim is not None:
        img11 = plotFieldOnAxis(axs[3], clim,  lat, lon, "Z500 Climatology", vmin, vmax)
    else:
        axs[3].text(0.5, 0.5, "Climatology data\nnot available", 
                    ha='center', va='center', transform=axs[3].transAxes, fontsize=12)
        axs[3].set_title("Z500 Climatology", fontsize=11)
        img11 = None

    # Add colorbar below the plots
    cax = fig.add_subplot(gs[2, :])
    cbar = fig.colorbar(img00, cax=cax, orientation="horizontal", pad=0.1)
    cbar.set_label("Z500 (units of your saved arrays)", fontsize=10)

    fig.suptitle(f"Static panels (sampleIdx={sampleIdx})", fontsize=14, y=0.98)

    figPath = outDir / "fig_staticPanels.png"
    fig.savefig(figPath, dpi=150, bbox_inches='tight')
    plt.close(fig)
    return figPath

def makeAnomalyPanels(outDir: Path, yTrue, yPred, yPers, yClim, lat, lon, sampleIdx=0):
    # anomaly relative to climatology
    if yClim is None:
        print("[visualize] yClim is None, skipping anomaly panels")
        return None
    
    # yClim shape: (12, H, W) - climatology by month
    # Use average climatology across all months for stability
    if yClim.ndim == 3 and yClim.shape[0] == 12:
        clim2d = np.mean(yClim, axis=0)  # average across months -> (H, W)
    elif yClim.ndim == 3:
        # If not 12 months, try to use sample index modulo number of months
        clim2d = yClim[sampleIdx % yClim.shape[0]]
    else:
        clim2d = yClim

    truthA = computeAnomaly(yTrue, clim2d)[sampleIdx]
    predA  = computeAnomaly(yPred, clim2d)[sampleIdx]
    persA  = computeAnomaly(yPers, clim2d)[sampleIdx]
    climA  = clim2d - clim2d  # zeros

    vmin, vmax = computeCommonVminVmax([truthA, predA, persA])

    # Use GridSpec for better layout (no tight_layout needed)
    fig = plt.figure(figsize=(14, 9))
    gs = fig.add_gridspec(3, 2, height_ratios=[1, 1, 0.08], hspace=0.3, wspace=0.2)
    
    axs = [
        fig.add_subplot(gs[0, 0], projection=ccrs.PlateCarree()),
        fig.add_subplot(gs[0, 1], projection=ccrs.PlateCarree()),
        fig.add_subplot(gs[1, 0], projection=ccrs.PlateCarree()),
        fig.add_subplot(gs[1, 1], projection=ccrs.PlateCarree()),
    ]

    img00 = plotFieldOnAxis(axs[0], truthA, lat, lon, "Anomaly Truth (Truth - Clim)", vmin, vmax)
    img01 = plotFieldOnAxis(axs[1], predA,  lat, lon, "Anomaly UNet (Pred - Clim)",  vmin, vmax)
    img10 = plotFieldOnAxis(axs[2], persA,  lat, lon, "Anomaly Persistence (Pers - Clim)", vmin, vmax)
    img11 = plotFieldOnAxis(axs[3], climA,  lat, lon, "Anomaly Climatology (0)", vmin, vmax)

    # Add colorbar below
    cax = fig.add_subplot(gs[2, :])
    cbar = fig.colorbar(img00, cax=cax, orientation="horizontal", pad=0.1)
    cbar.set_label("Z500 anomaly (units)", fontsize=10)

    fig.suptitle(f"Anomaly panels (sampleIdx={sampleIdx})", fontsize=14, y=0.98)

    figPath = outDir / "fig_anomalyPanels.png"
    fig.savefig(figPath, dpi=200)
    plt.close(fig)
    return figPath

def makeForecastGif(outDir: Path, yTrue, yPred, yPers, yClim, lat, lon, startIdx=0, endIdx=48, step=1):
    """
    Makes an animated GIF over indices [startIdx, endIdx).
    Uses matplotlib frames + imageio (lazy import).
    """

    frames = []
    endIdx = min(endIdx, yTrue.shape[0])

    for i in range(startIdx, endIdx, step):
        truth = yTrue[i]
        pred  = yPred[i]
        pers  = yPers[i]
        
        # Handle climatology indexing: (12, H, W) -> use average or modulo
        if yClim is not None:
            if yClim.ndim == 3 and yClim.shape[0] == 12:
                clim = np.mean(yClim, axis=0)  # average across months
            elif yClim.ndim == 3:
                clim = yClim[i % yClim.shape[0]]
            else:
                clim = yClim
        else:
            clim = None

        vmin, vmax = computeCommonVminVmax([truth, pred, pers, clim])

        # Use GridSpec for better layout
        fig = plt.figure(figsize=(14, 9))
        gs = fig.add_gridspec(3, 2, height_ratios=[1, 1, 0.08], hspace=0.3, wspace=0.2)
        
        axs = [
            fig.add_subplot(gs[0, 0], projection=ccrs.PlateCarree()),
            fig.add_subplot(gs[0, 1], projection=ccrs.PlateCarree()),
            fig.add_subplot(gs[1, 0], projection=ccrs.PlateCarree()),
            fig.add_subplot(gs[1, 1], projection=ccrs.PlateCarree()),
        ]

        img00 = plotFieldOnAxis(axs[0], truth, lat, lon, "Truth", vmin, vmax)
        plotFieldOnAxis(axs[1], pred,  lat, lon, "UNet", vmin, vmax)
        plotFieldOnAxis(axs[2], pers,  lat, lon, "Persistence", vmin, vmax)
        
        if clim is not None:
            plotFieldOnAxis(axs[3], clim,  lat, lon, "Climatology", vmin, vmax)
        else:
            axs[3].text(0.5, 0.5, "Climatology data\nnot available", 
                        ha='center', va='center', transform=axs[3].transAxes, fontsize=12)
            axs[3].set_title("Climatology", fontsize=11)

        # Add colorbar below
        cax = fig.add_subplot(gs[2, :])
        fig.colorbar(img00, cax=cax, orientation="horizontal", pad=0.1)
        fig.suptitle(f"Forecast comparison (idx={i})", y=0.98, fontsize=14)

        # render to array (use buffer instead of deprecated tostring_rgb)
        fig.canvas.draw()
        buf = fig.canvas.buffer_rgba()
        frame = np.frombuffer(buf, dtype=np.uint8)
        frame = frame.reshape(fig.canvas.get_width_height()[::-1] + (4,))[:, :, :3]  # drop alpha channel
        frames.append(frame)

        plt.close(fig)

    gifPath = outDir / "anim_forecasts.gif"
    imageio.mimsave(gifPath, frames, duration=0.25)
    return gifPath


# -----------------------------
# Main
# -----------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--resultDir", type=str, required=True,
                        help="Path like results/checkpoint_epoch003_valloss0.0032")
    parser.add_argument("--sampleIdx", type=int, default=0)
    parser.add_argument("--gifStartIdx", type=int, default=0)
    parser.add_argument("--gifEndIdx", type=int, default=96)
    parser.add_argument("--gifStep", type=int, default=1)

    # If you did NOT save lat/lon into resultDir, you can pass them here
    parser.add_argument("--latPath", type=str, default=None)
    parser.add_argument("--lonPath", type=str, default=None)

    args = parser.parse_args()

    resultDir = Path(args.resultDir)
    resultDir.mkdir(parents=True, exist_ok=True)

    yTrue = ensure3d(loadNpy(resultDir / "yTrue.npy"))
    yPred = ensure3d(loadNpy(resultDir / "yPred.npy"))
    yPers = ensure3d(loadNpy(resultDir / "yPers.npy"))
    
    # yClim is optional - try multiple locations
    yClim = None
    climPath = resultDir / "yClim.npy"
    if climPath.exists():
        yClim = ensure3d(loadNpy(climPath))
        print(f"[visualize] Loaded yClim from {climPath}")
    else:
        # Try loading from preprocessed data directory
        projectRoot = Path(__file__).resolve().parents[1]  # GeoNetLite root
        climNpzPath = projectRoot / "data" / "processed" / "climatology.npz"
        if climNpzPath.exists():
            try:
                climData = np.load(climNpzPath)
                yClim = ensure3d(climData["yClim"])
                print(f"[visualize] Loaded yClim from {climNpzPath}")
            except Exception as e:
                print(f"[visualize] Failed to load climatology from {climNpzPath}: {e}")
                yClim = None
        
        if yClim is None:
            print("[visualize] yClim.npy not found in resultDir or data/processed/, climatology comparisons will be skipped.")

    lat, lon = loadLatLon(resultDir)

    if lat is None or lon is None:
        if args.latPath is None or args.lonPath is None:
            raise ValueError(
                "lat/lon not found in resultDir and --latPath/--lonPath not provided.\n"
                "Fix: save lat.npy and lon.npy into the result folder, OR call visualize.py with --latPath and --lonPath."
            )
        lat = np.load(args.latPath)
        lon = np.load(args.lonPath)

    # Basic sanity: match shapes
    h, w = yTrue.shape[1], yTrue.shape[2]
    if lat.shape[0] != h or lon.shape[0] != w:
        raise ValueError(
            f"Shape mismatch:\n"
            f"  yTrue: (N,H,W)=({yTrue.shape[0]},{h},{w})\n"
            f"  lat: {lat.shape}\n"
            f"  lon: {lon.shape}\n"
            f"Expected lat=(H,) and lon=(W,)."
        )

    fig1 = makeStaticPanels(resultDir, yTrue, yPred, yPers, yClim, lat, lon, args.sampleIdx)
    fig2 = makeAnomalyPanels(resultDir, yTrue, yPred, yPers, yClim, lat, lon, args.sampleIdx)
    fig3 = makeSpectralDiagnosis(resultDir, yTrue, yPred, yPers, yClim, args.sampleIdx)
    fig4 = makeRMSETimeSeries(resultDir, yTrue, yPred, yPers, lat)
    gif1 = makeForecastGif(resultDir, yTrue, yPred, yPers, yClim, lat, lon,
                           startIdx=args.gifStartIdx, endIdx=args.gifEndIdx, step=args.gifStep)

    summary = {
        "resultDir": str(resultDir),
        "figStaticPanels": str(fig1),
        "figAnomalyPanels": str(fig2),
        "figSpectralDiagnosis": str(fig3),
        "figRMSETimeSeries": str(fig4),
        "gifForecasts": str(gif1),
        "sampleIdx": args.sampleIdx,
        "gifRange": [args.gifStartIdx, args.gifEndIdx, args.gifStep],
    }
    with open(resultDir / "vizSummary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()