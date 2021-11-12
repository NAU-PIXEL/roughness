"""Helpers for Diviner data."""
from pathlib import Path
import numpy as np
import xarray as xr
import rasterio as rio
import roughness.helpers as rh
import roughness.config as cfg

DIVINER_WLS = [
    7.8,  # C3
    8.25,
    8.55,
    (13 + 23) / 2,
    (25 + 41) / 2,
    (50 + 100) / 2,
    (100 + 400) / 2,  # C9
]  # [microns]


def lev4hourly2xr(
    fgrds,
    ext=None,
    savefile=None,
    tres=None,
    interp_method=None,
    interp_wrap=False,
):
    """
    Return xarray of Diviner lev4 hourly grd tiles with time resolution tres.

    If interp_method: Interpolate missing values along tloc axis with method
    specified (e.g. 'nearest', 'linear', 'cubic', see xr.interpolate_na methods).
    Linear seems to produce fewest sharp artifacts.

    If savefile: save result to netCDF (.nc) file for quick I/O into xarray,
    e.g. with xr.open_dataarray(savefile).

    Parameters
    ----------
    fgrds ()
    tres (num): time resolution
    interp_method (str): interpolation method (see xarray.interpolate_na)
    interp_wrap (bool): interpolate around time axis (wrap around 0/24h)
    ext (list): extent of ROI
    savefile (str): Path to save result to netCDF (.nc) file
    """
    if not tres:
        # Assume 8 bands (ch3 - ch9 + tbol), diurnal [24 h], infer tres [hr]
        tres = 24 / (len(fgrds) / 8)  # [hr]
    grids = []
    bands = []
    for i, fgrd in enumerate(fgrds):
        _, band, ind = Path(fgrd).stem.split("-")
        tloc = (int(ind) - 1) * tres  # tloc in [0, 24) h

        # Read in .grd and set its local time to tloc
        grid = xr.open_rasterio(fgrds[i]).sel(band=1)
        grid["tloc"] = tloc
        grids.append(grid)

        # Concatenate all local time of this band into single xr.DataArray
        if tloc == 24 - tres:
            diurnal_band = xr.concat(grids, dim="tloc")
            diurnal_band["band"] = band
            bands.append(diurnal_band)
            grids = []

    # Concatenate all bands into single 4D xr.DataArray (lon, lat, tloc, band)
    out = xr.concat(bands, dim="band")
    out = out.rename("Temperature")
    out = out.sortby(
        "y", ascending=False
    )  # TODO: bug doesn't always fix yflip

    # Convert x, y coords to lat, lon
    if ext is not None:
        lon, lat = rh.xy2lonlat_coords(out.x, out.y, ext)
        out = out.rename({"x": "lon", "y": "lat"})
        out = out.assign_coords(lon=lon, lat=lat)

    if interp_method is not None:
        # Interpolate missing points on tloc
        if interp_wrap:
            # Cludgey handling of tloc edges. Wrap full dataset around midnight then interp
            postmidnight = out.sel(tloc=slice(0, 12 - tres))
            postmidnight["tloc"] = np.arange(24, 36, tres)
            premidnight = out.sel(tloc=slice(12, 24 - tres))
            premidnight["tloc"] = np.arange(-12, 0, tres)
            out = xr.concat([premidnight, out, postmidnight], dim="tloc")
        out = out.interpolate_na("tloc", method=interp_method).sel(
            tloc=slice(0, 24 - tres)
        )

    if savefile is not None:
        out.to_netcdf(savefile)
    return out


def xarr2geotiff(xarr, savefile, crs=cfg.MOON2000_ESRI):
    """Write 3D xarray (bands, y, x) to geotiff with rasterio."""
    count, height, width = xarr.shape
    band_indices = np.arange(count) + 1
    transform = None
    if "transform" in xarr.attrs:
        transform = tuple(xarr.attrs["transform"])
    profile = {
        "driver": "GTiff",
        "height": height,
        "width": width,
        "count": count,
        "dtype": str(xarr.dtype),
        "transform": transform,
        "crs": crs,
    }
    with rio.open(savefile, "w", **profile) as dst:
        dst.write(xarr.values, band_indices)
        for i in dst.indexes:
            band_name = str(xarr.band[i - 1].values)
            dst.update_tags(i, band_name)
    if Path(savefile).exists():
        print("Wrote to", savefile)
    else:
        print("Failed to write to", savefile)


def fit_poly_daytime(Tday, deg=2):
    """
    Return polynomial fit of order deg to daytime temperature data.
    """
    tloc = np.linspace(6, 18, len(Tday) + 1)[:-1]
    nan = np.isnan(Tday)
    pfit = np.polyfit(tloc[~nan], Tday[~nan], deg)
    fit_func = np.poly1d(pfit)
    fit = fit_func(tloc)
    return fit


def smooth_daytime(T_xarr, savefile=None):
    """
    Return smoothed daytime T from 6 AM to 6 PM with 2nd order polyfit.
    """
    tres = T_xarr.tloc[1].values - T_xarr.tloc[0].values
    Tday = T_xarr.sel(tloc=np.arange(6, 18, tres))
    Tsmooth = xr.apply_ufunc(
        fit_poly_daytime,
        Tday,
        input_core_dims=[["tloc"]],
        output_core_dims=[["tloc"]],
        vectorize=True,
    )
    if savefile:
        Tsmooth.to_netcdf(savefile)
    return Tsmooth


def add_wls_diviner(xarr):
    """
    Return xarr with wavelength coordinate (drop tbol).
    """
    out = xarr.where(xarr.band != "tb", drop=True)
    out = out.assign_coords(wavelength=("band", DIVINER_WLS))
    return out


def load_div_lev4(
    roi,
    ext,
    savefile="T.nc",
    smoothday=False,
    invert_y=False,
    load_cached=True,
    divdir=cfg.DATA_DIR_DIVINER,
):
    """
    Return Diviner lev4 data as xarray.
    """
    roi_str = roi.replace("'", "_").lower()
    savepath = Path(divdir) / roi_str / savefile
    if load_cached and savepath.exists():
        return xr.open_dataarray(savepath)

    print("Loading Diviner lev4 data for", roi)
    dirpath = Path(divdir) / roi_str
    saveraw = None if smoothday else savepath
    fgrds = [f.as_posix() for f in Path(dirpath).rglob("*.grd")]
    out = lev4hourly2xr(fgrds, ext, saveraw, interp_method=None)
    if smoothday:
        T_smooth = smooth_daytime(out, savepath)
        out = T_smooth

    # Sometimes Diviner data inverted about y - only need to fix once
    if invert_y:
        out = out.close()
        tmp = xr.load_dataarray(savepath)  # load imports file and closes it
        tmp = tmp.assign_coords(lat=tmp.lat[::-1])  # invert lat
        tmp.to_netcdf(savepath)
        out = xr.open_dataarray(savepath)
    return out
