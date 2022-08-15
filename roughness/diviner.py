"""Helpers for Diviner data."""
from functools import lru_cache
from pathlib import Path
import numpy as np
import pandas as pd
import xarray as xr
import roughness.config as cfg
import roughness.helpers as rh
import roughness.emission as re


DIV_C = ("c3", "c4", "c5", "c6", "c7", "c8", "c9")
DIV_WL_MIN = np.array([7.55, 8.10, 8.38, 13, 25, 50, 100])  # [microns]
DIV_WL_MAX = np.array([8.05, 8.40, 8.68, 23, 41, 99, 400])  # [microns]
DIV_WL1 = np.array([0.0001, 8.075, 8.40, 13, 25, 50, 100])
DIV_WL2 = np.array([8.075, 8.40, 13, 25, 50, 100, 500])
DIV_TMIN = np.array([190, 190, 180, 95, 60, 40, 0])
ITERS = np.array([5, 5, 6, 10, 16, 24, 56])
FILT_SCALE = np.array(
    [1.0005, 1.0003, 1.0010, 1.0018, 1.00600, 0.71939, 0.79358]
)
FDIV_T2R = "/home/ctaiudovicic/projects/roughness/tmp/div_t2r.txt"
FDIV_FILT = "/nfs/data/josh/home/projects/diviner/diviner_filter_functions.hdf"


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
    for fgrd in fgrds:
        _, band, ind = Path(fgrd).stem.split("-")
        tloc = (int(ind) - 1) * tres  # tloc in [0, 24) h

        # Read in .grd and set its local time to tloc
        grid = xr.open_rasterio(fgrd).sel(band=1)
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


def add_wls_diviner(xarr, bands=("t3", "t4", "t5", "t6", "t7", "t8", "t9")):
    """
    Return xarr with wavelength coordinate.
    """
    wl = get_diviner_wls().values
    out = xarr.sel(band=slice("t3", "t9")).assign_coords(
        wavelength=("band", wl)
    )
    out = out.reindex(band=bands)  # Drops unneeded bands
    return out


def get_diviner_wls(bres=1):
    """
    Return diviner c3-c9 wavelength arr with band_res values in each bandpass.
    """
    if bres == 1:
        wls = np.mean((DIV_WL_MIN, DIV_WL_MAX), axis=0)
    else:
        wzip = zip(DIV_WL_MIN, DIV_WL_MAX)
        wls = np.array(
            [np.linspace(w1, w2, bres) for w1, w2 in wzip]
        ).flatten()
    return rh.wl2xr(wls)


def div_integrated_rad(emission, units="W/m^2/sr/um"):
    """
    Return integrated radiance of diviner bands.
    """
    out = []
    for dwlmin, dwlmax in zip(DIV_WL_MIN, DIV_WL_MAX):
        cemiss = emission.sel(wavelength=slice(dwlmin, dwlmax))
        wlmin = cemiss.wavelength.min().values
        wlmax = cemiss.wavelength.max().values
        out.append(cemiss.integrate("wavelength").values / (wlmax - wlmin))
    wls = get_diviner_wls().values
    return rh.spec2xr(out, wls, units=units)


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


def div_tbol(divbt, wl1=DIV_WL1, wl2=DIV_WL2, tmin=DIV_TMIN, iters=ITERS):
    """
    Return tbol from Diviner C3-C9 brightness temperatures.
    """

    def f(wl, bt, n_iter):
        """CFST iteration helper. n_iter small, so loop is faster"""
        v = 1.43883e4 / (wl * bt)
        tot = 0
        for w in range(1, n_iter + 1):
            tot += (
                w**-4
                * np.e ** -(w * v)
                * (((w * v + 3) * w * v + 6) * w * v + 6)
            )
        return tot

    def cfst(bt, wl1, wl2, n_iter=100):
        """
        Return fraction of Planck radiance between wl1 and wl2 at bt.

        Translated from JB, via DAP via Houghton 'Physics of Atmospheres'
        """
        return 1.53989733e-1 * (f(wl2, bt, n_iter) - f(wl1, bt, n_iter))

    vcfst = np.vectorize(cfst, otypes=[float])

    # Backfill values below tmin from the next band over (C9->C3)
    # Convenient to use pandas but overkill, maybe do numba func
    temp = (
        pd.Series(np.where(divbt >= tmin, divbt, np.nan))
        .fillna(method="bfill")
        .values
    )

    # Convert to radiance, weight by cfst, sum and convert back to temperature
    rad = re.get_rad_sb(temp) * vcfst(temp, wl1, wl2, iters)
    tbol = re.get_temp_sb(np.nansum(rad))
    return tbol


@lru_cache(1)
def load_div_filters(
    fdiv_filt=FDIV_FILT, scale=True, wlunits=True, bands=DIV_C
):
    """
    Return Diviner filter functions
    """
    hdf = xr.load_dataset(fdiv_filt)
    wn = hdf.xaxis.values.squeeze()
    data = hdf.data.squeeze().values
    data = data.reshape(data.shape[1], data.shape[0])  # reads wrong axis order
    div_filt = xr.DataArray(
        data,
        name="Normalized Response",
        dims=["wavenumber", "band"],
        coords={"band": list(bands), "wavenumber": wn},
    )
    if scale:
        div_filt = div_filt * FILT_SCALE[np.newaxis, :]
    if wlunits:
        div_filt.coords["wavelength"] = 10000 / div_filt.wavenumber
        div_filt = div_filt.swap_dims({"wavenumber": "wavelength"})
    return div_filt


@lru_cache(1)
def load_div_t2r(fdiv_t2r=FDIV_T2R):
    """
    Return Diviner temperature to radiance lookup table.
    """
    return pd.read_csv(fdiv_t2r, index_col=0, header=0, delim_whitespace=True)


def divfilt_rad(wnrad, div_filt=None):
    """
    Return radiance of Diviner bands (convolve rad with Diviner filter funcs).
    """
    if div_filt is None:
        div_filt = load_div_filters()
    filtered = div_filt * wnrad  # Must both have wavelength coord
    return filtered.sum(dim="wavelength") * 2


def divrad2bt(divrad, fdiv_t2r=FDIV_T2R):
    """
    Return brightness temperatures from Diviner radiance (e.g. from div_filt).
    """
    t2r = load_div_t2r(fdiv_t2r)  # cached lookup
    rad = divrad.values

    # Mask negative values or those larger than max in each column of t2r
    mask = rad < t2r.max(axis=0)

    # Nanargmin gives the closest index (i.e. brightness temperature)
    return np.nanargmin(np.abs(t2r - np.where(mask, rad, -1)), axis=0)


def divrad2bt_interp(divrad, fdiv_t2r=FDIV_T2R):
    """
    Return brightness temperatures from Diviner radiance (e.g. from div_filt).
    """
    t2r = load_div_t2r(fdiv_t2r)  # cached lookup
    rad = divrad.values
    mask = (0 < rad) & (rad < t2r.max(axis=0))
    rad = np.where(mask, rad, -1)
    dbt = np.zeros(len(divrad.band.values))
    for i, band in enumerate(divrad.band.values):
        # Interpolate lookup to convert radiance to brightness temperature
        dbt[i] = np.interp(
            rad[i],  # desired x-coords (radiance)
            t2r[band].values,  # actual x-coords (radiance)
            t2r.index.values,  # actual y-coords (temperature)
        )
    return dbt


def emission2tbol_xr(emission, wls=None, div_filt=None):
    """Return tbol from input emission array at wls."""
    if wls is None:
        wls = emission.wavelength
    elif not isinstance(wls, xr.DataArray):
        wls = rh.wl2xr(wls)
    wnrad = re.wlrad2wnrad(wls, emission)
    divrad = divfilt_rad(wnrad, div_filt)
    tbol = xr.zeros_like(emission.isel(wavelength=0).drop("wavelength"))
    dim0, dim1 = tbol.dims  # Generic so that x,y or lat,lon work in any order
    for i in range(len(tbol[dim0])):
        for j in range(len(tbol[dim1])):
            div_bts = divrad2bt(divrad.isel({dim0: i, dim1: j}))
            tbol[i, j] = div_tbol(div_bts)
    return tbol
