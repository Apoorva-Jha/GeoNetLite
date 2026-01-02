import os
import glob
from pathlib import Path
import numpy as np
import xarray as xr

def loadRawZ500(rawDir):
    """
    Load raw ERA5 geopotential files and combine them into a single dataset.
    Ensures consistent dimension ordering (time, lat, lon).
    """
    fileList = sorted(glob.glob(str(rawDir / "*.nc")))
    if len(fileList) == 0:
        raise FileNotFoundError(f"No .nc files found in {rawDir}")

    def loadOne(path):
        ds = xr.open_dataset(path)
        # Force dimension order
        ds = ds.transpose("time", "lat", "lon")
        return ds

    datasets = [loadOne(f) for f in fileList]
    z500 = xr.concat(datasets, dim="time")
    return z500


def convertToHeight(zDa):
    """
    Convert geopotential (m^2/s^2) to geopotential height (m).
    """
    g = 9.80665
    return zDa / g


def subsetLatBand(zDa, latMin=20, latMax=90):
    """
    Subset latitude to [latMin, latMax], handling ascending/descending lat.
    """
    latName = "lat"

    lats = zDa[latName]
    if lats.values[0] < lats.values[-1]:
        # ascending lat → slice(latMin, latMax)
        return zDa.sel(lat=slice(latMin, latMax))
    else:
        # descending lat → slice(latMax, latMin)
        return zDa.sel(lat=slice(latMax, latMin))


def coarsenGrid(zDa, latFactor=1, lonFactor=1):
    """
    Optional grid coarsening. Use factors >= 1.
    If both factors == 1 → no coarsening applied.
    """
    if latFactor == 1 and lonFactor == 1:
        return zDa

    return zDa.coarsen(
        lat=latFactor,
        lon=lonFactor,
        boundary="trim"
    ).mean()


def assignYears(zDa):
    """
    Add a 'year' coordinate for easy splitting.
    """
    years = zDa["time.year"].values
    zDa = zDa.assign_coords(year=("time", years))
    return zDa


def normalizeTrain(zDa, trainYears):
    """
    Compute mean/std from training years and return normalized zDa + stats.
    """
    trainDa = zDa.sel(time=zDa["time.year"].isin(trainYears))

    meanVal = float(trainDa.mean().values)
    stdVal = float(trainDa.std().values)

    zNorm = (zDa - meanVal) / (stdVal + 1e-8)
    return zNorm, meanVal, stdVal


def createLeadPairs(zDa, leadHours):
    """
    Build input-target pairs for a given lead time.
    Assumes hourly data: leadHours = number of hours forward.
    Returns (X, Y) arrays of shape (N, 1, lat, lon).
    """

    # Ensure sorted time
    da = zDa.sortby("time")

    total = da.shape[0]
    lead = leadHours  # because hourly

    X = da.isel(time=slice(0, total - lead)).values
    Y = da.isel(time=slice(lead, total)).values

    # Reshape to (N, 1, lat, lon)
    X = X[:, np.newaxis, :, :]
    Y = Y[:, np.newaxis, :, :]

    return X.astype(np.float32), Y.astype(np.float32)


def splitByYears(X, Y, years, trainYears, valYears, testYears, leadHours):
    """
    Split X and Y into train/val/test by year.
    years: full-length year array (same length as original time dimension)
    We trim 'years' to match X/Y length by dropping the last 'leadHours' entries.
    """

    # Align years with X/Y (drop last leadHours timestamps)
    yearsXY = years[:-leadHours]

    def mask(yearList):
        return np.isin(yearsXY, yearList)

    trainMask = mask(trainYears)
    valMask   = mask(valYears)
    testMask  = mask(testYears)

    return (
        X[trainMask], Y[trainMask],
        X[valMask],  Y[valMask],
        X[testMask], Y[testMask]
    )

def computeMonthlyClimatology(yTrain, timeTrainY):
    """
    yTrain: (N, 1, H, W) normalized targets
    timeTrainY: (N,) datetime64 array aligned with yTrain samples (target times)

    Returns:
        clim: (12, H, W) monthly mean climatology in normalized space
    """
    # months 1..12
    months = timeTrainY.astype("datetime64[M]").astype(int) % 12  # 0..11

    yTrain2d = yTrain[:, 0, :, :]  # (N, H, W)
    clim = np.zeros((12, yTrain2d.shape[1], yTrain2d.shape[2]), dtype=np.float32)

    for m in range(12):
        mask = months == m
        if mask.sum() == 0:
            raise ValueError(f"No samples found for month index {m}. Check time alignment.")
        clim[m] = yTrain2d[mask].mean(axis=0)

    return clim

def saveNumpyArrays(outDir, meanVal, stdVal, splits, zH, clim):
    """
    Save preprocessed arrays and metadata (mean, std, latitude).
    
    Args:
        outDir: Output directory path
        meanVal: Mean value for normalization
        stdVal: Std value for normalization
        splits: Tuple of (xTrain, yTrain, xVal, yVal, xTest, yTest)
        zH: Original height DataArray (to extract latitude)
    """
     
    outDir.mkdir(parents=True, exist_ok=True)

    (
        xTrain, yTrain,
        xVal, yVal,
        xTest, yTest
    ) = splits

    np.save(outDir / "xTrain.npy", xTrain)
    np.save(outDir / "yTrain.npy", yTrain)
    np.save(outDir / "xVal.npy",  xVal)
    np.save(outDir / "yVal.npy",  yVal)
    np.save(outDir / "xTest.npy", xTest)
    np.save(outDir / "yTest.npy", yTest)

    np.savez(outDir / "z500Stats.npz", mean=meanVal, std=stdVal)

    lat = zH["lat"].values
    lon = zH["lon"].values
    np.save(outDir / "lat.npy", lat)
    np.save(outDir / "lon.npy", lon)

    np.savez(outDir / "climatology.npz", yClim=clim)

    print(f"Saved yClim (shape: {clim.shape}) to {outDir / 'climatology.npz'}")



def main():
    projectRoot = Path(__file__).resolve().parents[2]
    rawDir = projectRoot / "data" / "raw" / "geopotential_500"
    outDir = projectRoot / "data" / "processed"

    # User choices
    leadHours = 6  # change to 12 if you want 12-hour forecasts
    latMin, latMax = 20, 90

    trainYears = list(range(1979, 2010))
    valYears =   list(range(2010, 2014))
    testYears =  list(range(2014, 2019))

    print("Loading raw Z500…")
    z500 = loadRawZ500(rawDir)
    z500Da = z500["z"]

    print("Converting to height…")
    zH = convertToHeight(z500Da)

    print("Subsetting latitude band…")
    zH = subsetLatBand(zH, latMin, latMax)

    print("Assigning years…")
    zH = assignYears(zH)

    print("Normalizing (train-year mean/std)…")
    zNorm, meanVal, stdVal = normalizeTrain(zH, trainYears)

    print(f"Creating lead pairs: lead={leadHours} hours")
    X, Y = createLeadPairs(zNorm, leadHours)

    # Full time array from the raw (after subsetting); aligned with zH/zNorm
    fullTime = zNorm["time"].values  # length = total
    
    # Extract years from zNorm (same as zH since we just normalized it)
    years = zNorm["year"].values  # aligned with X before lead-pairing, length = total

    # Target times for Y (since Y starts at time index leadHours)
    timeY = fullTime[leadHours:]     # length matches Y along axis 0

    # Years aligned to X/Y already in splitByYears (yearsXY = years[:-leadHours])
    yearsXY = years[:-leadHours]     # aligns with X (and also with Y index 0..)

    # Build train mask (same as used for X/Y train split)
    trainMask = np.isin(yearsXY, trainYears)

    # Extract train target times matching yTrain samples
    timeTrainY = timeY[trainMask]

    # Split
    splits = splitByYears(X, Y, years, trainYears, valYears, testYears, leadHours)

    # Unpack so we can compute climatology from yTrain
    xTrain, yTrain, xVal, yVal, xTest, yTest = splits

    # Compute monthly climatology baseline in normalized space
    yClim = computeMonthlyClimatology(yTrain, timeTrainY)

    print("Saving arrays to processed directory…")
    saveNumpyArrays(outDir, meanVal, stdVal, splits, zH, yClim)

    print("Done! Processed data saved to:", outDir)


if __name__ == "__main__":
    main()