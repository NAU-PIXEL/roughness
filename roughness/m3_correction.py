"""M3-specific functions for testing"""
import itertools as it
from functools import lru_cache
import json
from datetime import datetime
from pathlib import Path
import multiprocessing as mp
import warnings
import numpy as np
import pandas as pd
import xarray as xr
import rasterio as rio
import spiceypy
import roughness.emission as re
import roughness.helpers as rh
from roughness.config import FALBEDO, DATA_DIR, DATA_DIR_M3

# JSON
with open(DATA_DIR_M3 / "m3_wls.json", "r", encoding="utf8") as _fname:
    _M3_WLS = json.load(_fname)
    GWL = _M3_WLS["GWL"]
    TWL = _M3_WLS["TWL"]

# Constants
NCPUS = max(mp.cpu_count() - 1, 1)  # Number of CPUs available (at least 1)
M3PATH = Path("/work/ctaiudovicic/data") / "m3"  # M3 PDS volume
M3ANCPATH = DATA_DIR_M3 / "M3_ancillary_files"
M3ALTPHOTOM = DATA_DIR_M3 / "M3_ancillary_files" / "alt_photom_tables"
KWIN = {"usgs": 2, "dask": False, "xoak": False}  # get_m3xr()
DROP_ATTRS = (
    "long_name",
    "wavelength",
    "target_fwhm",
    "target_wavelengths",
    "global_Channel_Number",
    "Target_Channel_Number",
)
# Spice
SPICEPATH = DATA_DIR / "m3" / "spice"
METAKR = SPICEPATH / "moon_tloc.tm"
KERNELPATH = SPICEPATH / "kernels"

# Silence rasterio georeferencing warnings
warnings.filterwarnings("ignore", category=rio.errors.NotGeoreferencedWarning)


def m3_refl(basename, ext, wls=None, kwin=KWIN, kwre={}):
    """
    Return M3 reflectance using the Tai Udovicic et al (2022) roughness
    thermal correction (updated from Bandfield et al., 2018).
    Uses ray-casting generated shadow_lookup table, KRC temperature table and
    applies standard M3 processing steps (empirical photometric correction).

    """
    chunks = kwin.pop("chunks", False)
    kwin["dask"] = False
    obs = subset_m3xr_ext(get_m3xr(basename, "obs", **kwin), ext)
    alb = get_albedo_feng(obs.lon, obs.lat)
    tloc = np.mean(get_tloc_range_spice(basename, ext[0], ext[1]))
    wls = get_m3_wls(basename) if wls is None else wls
    wls = rh.wl2xr(wls)

    # Get rough emission
    i, iaz, e, eaz, g = get_m3_geom_xr(obs, None)  # TODO: dem
    args = (i, iaz, e, eaz, alb, wls, tloc)
    if chunks:
        # emission = xr.map_blocks(get_emission_xr, i, args, kwre, template=(i*wls))
        # emission = emission.load()  # Compute emission on chunks with dask
        chunks["wavelength"] = chunks.pop("band")
        emission = xr.zeros_like(i * wls).chunk(chunks)
        emission = map_blocks(
            emission_xr_helper, emission, *args, ind_dims=i.dims, **kwre
        )
    else:
        emission = get_emission_xr(*args, **kwre)

    # Remove emission from rad and convert to I/F
    rad = subset_m3xr_ext(get_m3xr(basename, "rad", **kwin), ext)
    L_sun = get_solar_irradiance_xr(basename, obs)
    photom = get_m3photom(i, e, g, basename)
    refl = re.get_rad_factor(rad, L_sun, emission, 1) * photom
    refl.attrs = rad.attrs
    return refl, emission


def emission_xr_helper(ind, args, kwargs):
    """Helper function for emission_xr."""
    # ind is (wls, x, y), drop wls dim and slice other args by (x, y)
    # ind = tupl[i for i in ind if not (i.start == 0 and i.stop == len(args[4]))]
    args = [a[ind] if i < 5 else a for i, a in enumerate(args)]
    # kwargs = [k[ind] if isinstance(k, xr.DataArray) else k for k in kwargs]
    return get_emission_xr(*args, **kwargs)


def map_blocks(func, outda, *args, ind_dims=None, **kwargs):
    """Map func to blocks of args."""
    # Magic to get dask chunk slices as list of multiple indices
    if ind_dims is None:
        ind_dims = outda.dims
    cranges = []
    for dim, chunks in zip(outda.dims, outda.chunks):
        if dim in ind_dims:
            cedges = np.r_[[0], np.cumsum(chunks)]
            cranges.append(zip(cedges[:-1], cedges[1:]))
    inds = [
        tuple(slice(*ind) for ind in inds) for inds in it.product(*cranges)
    ]

    # Run in parallel
    with mp.Pool(NCPUS) as pool:
        args = zip(inds, it.repeat(args), it.repeat(kwargs))
        out = pool.starmap(func, args)
    # Get results
    for i, ind in enumerate(inds):
        # func(ind, args, kwargs)
        outda.data[ind] = out[i]
    return outda


def get_emission_xr(inc, iaz, em, eaz, alb, wls, tloc, **kwargs):
    """Return emission [W/(m^2 um)]."""
    emission = xr.zeros_like(inc * wls)
    dorig = emission.dims
    dim0, dim1 = [dim for dim in inc.dims if dim != "band"]
    emission = emission.transpose(dim0, dim1, "wavelength")
    for j in range(len(emission[dim0])):
        for k in range(len(emission[dim1])):
            ind = {dim0: j, dim1: k}
            geom = [da.isel(ind).values for da in [inc, iaz, em, eaz]]
            albedo = alb.isel(ind).values
            lat = inc.isel(ind).lat.values
            tparams = {"tloc": tloc, "albedo": albedo, "lat": lat}
            emission[j, k] = re.rough_emission_lookup(
                geom, wls, tparams=tparams, **kwargs
            )
    # Transpose back to original axis order
    emission = emission.transpose(*dorig)
    return emission


# def m3_refl_geotif(
#     m3_basename,
#     rad_L1,
#     obs,
#     wl=None,
#     dem=None,
#     rms=18,
#     albedo=None,
#     emissivity=None,
#     rerad=False,
#     rerad_albedo=0.12,
#     flookup=FLOOKUP,
#     tlookup=TLOOKUP,
#     cast_shadow_time=0.05,
#     nprocs=NCPUS,
# ):
#     """
#     Return M3 reflectance using the roughness thermal correction (Bandfield et
#     al., 2018). Uses ray-casting generated shadow_lookup table. Applies
#     standard M3 photometric correction.

#     """
#     # Get wavelength, viewing geometry, photometry from M3 files
#     if wl is None:
#         wl = get_m3_wls(rad_L1)
#     solar_dist = get_solar_dist(m3_basename, obs).values  # TODO: check
#     sun_thetas, sun_azs, sc_thetas, sc_azs, phases = get_m3_geom(obs, dem)

#     # Compute albedo from raw (uncorrected) radiance
#     if albedo is None:
#         i, _, e, _, g = get_m3_geom(obs, dem)
#         albedo = get_m3albedo(rad_L1, obs, i, e, g, m3_basename, amax=0.4)
#     if not isinstance(albedo, xr.DataArray):
#         albedo = np.ones(rad_L1.shape[:2]) * albedo

#     # Get roughness emission for each pixel from rad eq on subpixel facets
#     # Single process for loops
#     if nprocs == 1:
#         emission = xr.zeros_like(rad_L1).reindex(
#             {"wavelength": wl}, method="pad"
#         )
#         for i, lat in enumerate(emission.lat.values):
#             print(f"line {i}/{emission.lat.size}", end="\r", flush=True)
#             for j, lon in enumerate(emission.lon.values):
#                 if isinstance(albedo, xr.DataArray):
#                     alb = float(albedo.sel(lon=lon, lat=lat))
#                 else:
#                     alb = albedo[i, j]
#                 geom = [
#                     c[i, j]
#                     for c in (
#                         sun_thetas.values,
#                         sun_azs.values,
#                         sc_thetas.values,
#                         sc_azs.values,
#                     )
#                 ]
#                 tloc = tlocs.values[i, j]
#                 tparams = {"tloc": tloc, "albedo": alb, "lat": lat}
#                 emission[i, j] = re.rough_emission_lookup(
#                     geom,
#                     wl,
#                     rms,
#                     tparams,
#                     rerad,
#                     rerad_albedo,
#                     flookup,
#                     tlookup,
#                     emissivity,
#                     cast_shadow_time,
#                 )
#     else:
#         emission = get_emission_mp(
#             rad_L1.lat.values,
#             sun_thetas.values,
#             sun_azs.values,
#             sc_thetas.values,
#             sc_azs.values,
#             wl,
#             rms,
#             albedo.values,
#             tlocs.values,
#             rerad,
#             rerad_albedo,
#             flookup,
#             tlookup,
#             emissivity,
#             cast_shadow_time,
#             nprocs,
#         )
#         emission = xr.DataArray(
#             emission,
#             coords=[rad_L1.lat, rad_L1.lon, wl],
#             dims=["lat", "lon", "wavelength"],
#         )

#     # Remove emission from Level 1 radiance and convert to I/F
#     rad_emission = emission.interp(wavelength=rad_L1.wavelength)
#     # Try passing in rough emission with emiss=1 or try converting to 3um BT
#     # rad_emission = re.bbr(3um_bt)
#     i_f = re.get_rad_factor(
#         rad_L1,
#         L_sun,
#         rad_emission,
#     )  # emiss=None to get from data
#     # i_f = (rad - emission) / L_sun  # When emissivity is used in rough
#     refl = i_f * photom  # TODO:photom
#     return refl, emission, L_sun


# def get_emission_mp(
#     lats,
#     sun_thetas,
#     sun_azs,
#     sc_thetas,
#     sc_azs,
#     wl,
#     rms,
#     albedo,
#     tlocs,
#     rerad,
#     rerad_albedo,
#     flookup,
#     tlookup,
#     emiss,
#     cast_shadow_time,
#     procs=NCPUS,
# ):
#     """Return emission parallelized across image"""
#     geom = zip(
#         sun_thetas.ravel(), sun_azs.ravel(), sc_thetas.ravel(), sc_azs.ravel()
#     )
#     geom = np.array(list(geom)).reshape(*sun_thetas.shape, 4)

#     xinds, yinds = np.nonzero(sun_thetas < 90)
#     args = (
#         (
#             geom[i, j],
#             wl,
#             rms,
#             {"albedo": albedo[i, j], "lat": lats[j], "tloc": tlocs[i, j]},
#             rerad,
#             rerad_albedo,
#             flookup,
#             tlookup,
#             emiss,
#             cast_shadow_time,
#         )
#         for i, j in zip(xinds, yinds)
#     )
#     with mp.Pool(processes=procs) as pool:
#         # Note: this parallel is inefficient, not much work for one pixel
#         emission_arr = np.array(pool.map(rough_emission_helper, args))
#     out_shape = (*sun_thetas.shape, len(wl))
#     emission = np.zeros(out_shape)
#     emission[xinds, yinds] = emission_arr
#     return emission


# def rough_emission_helper(args):
#     """Unpack args for rough emission lookup."""
#     return re.rough_emission_lookup(*args)


def get_m3_geom(obs, dem=None):
    """Return geometry array from M3 observation file and optional dem."""
    if dem is None:
        dem_theta = np.radians(obs[:, :, 7])
        dem_az = np.radians(obs[:, :, 8])
    else:
        dem_theta = np.arctan(dem[:, :, 0])
        dem_az = dem[:, :, 1]
    sun_az = np.radians(obs[:, :, 0])
    sun_theta = np.radians(obs[:, :, 1])
    sc_az = np.radians(obs[:, :, 2])
    sc_theta = np.radians(obs[:, :, 3])

    # Compute angles in degrees
    i, e, g, _ = rh.get_ieg(
        dem_theta, dem_az, sun_theta, sun_az, sc_theta, sc_az
    )
    sun_az = np.rad2deg(sun_az)
    sc_az = np.rad2deg(sc_az)
    return i, sun_az, e, sc_az, g


def get_m3_geom_xr(obs, dem=None):
    """Return geometry from M3 observation file and optional dem as xarray."""
    sun_az = obs.sel(band=1)  # To-sun azimuth [deg]
    sun_z = obs.sel(band=2)  # To-sun zenith [deg]
    sc_az = obs.sel(band=3)  # To-spacecraft azimuth [deg]
    sc_z = obs.sel(band=4)  # To-spacecraft zenith [deg]
    if dem is None:
        # Pull cos(i) relative to dem and phase from obs
        i = np.rad2deg(np.arccos(obs.sel(band=10)))  # Incidence [deg]
        g = obs.sel(band=5)  # Phase [deg]

        # Compute e relative to dem
        dem_sl = obs.sel(band=8)  # Facet slope [deg]
        dem_az = obs.sel(band=9)  # Facet azimuth [deg]
        dem_cart = rh.sph2cart_xr(np.deg2rad(dem_sl), np.deg2rad(dem_az))
        sc_cart = rh.sph2cart_xr(np.deg2rad(sc_z), np.deg2rad(sc_az))
        e = rh.get_angle_between_xr(dem_cart, sc_cart, True)
    else:
        # Compute i,e,g relative to dem
        dem_sl = np.arctan(dem.slope)  # [radians]
        dem_az = dem.aspect  # [radians]
        args = [np.deg2rad(a) for a in [sun_z, sun_az, sc_z, sc_az]]
        i, e, g, _ = rh.get_ieg_xr(dem_sl, dem_az, *args)
    return i, sun_az, e, sc_az, g


def get_m3_tloc(obs, dem=None, lat=0):
    """
    Return average local time of M3 observation.
    """
    return rh.inc_to_tloc(*get_avg_sun_inc_az(obs, dem), lat)


def get_avg_sun_inc_az(obs, dem=None):
    """
    Return average solar incidence and azimuth from obs file.
    """
    sun_thetas, sun_azs, *_ = get_m3_geom(obs, dem)
    return np.mean(sun_thetas), np.mean(sun_azs)


def get_m3_wls(img):
    """
    Return wavelength array for an M3 image [microns].
    """
    if is_targeted(img):
        wls = TWL
    else:
        wls = GWL
    return np.array(wls) / 1000  # [nm] -> [microns]


def get_solar_dist(basename, obs):
    """
    Get the solar distance subtracted off of M3 obs band 6.
    """
    obs_header = get_m3_headers(basename, imgs="obs")
    sdist = 1
    try:
        params = parse_envi_header(obs_header)
        description = "".join(params["description"])
        i = description.find("(au)")
        sdist = float(description[i + 6 : i + 16])
    except FileNotFoundError as e:
        w = f"Could not get sdist for {obs_header}: {str(e)}"
        print(w)
    if isinstance(obs, xr.DataArray):
        solar_dist = obs.sel(band=6).drop("band") + sdist
    else:
        solar_dist = obs[:, :, 5] + sdist
    return solar_dist


@lru_cache(2)
def get_solar_spectrum(basename, m3ancpath=M3ANCPATH):
    """Return m3 solar spectrum [W/(m^2 um)]."""
    if is_targeted(basename):
        fss = Path(m3ancpath) / "targeted" / "M3T20110224_RFL_SOLAR_SPEC.TAB"
    else:
        fss = Path(m3ancpath) / "M3G20110224_RFL_SOLAR_SPEC.TAB"
    ss_raw = np.genfromtxt(fss).T[:, :, np.newaxis]
    ss = np.moveaxis(ss_raw[1:2], 1, 2)
    return ss


def get_solar_spectrum_xr(basename, m3ancpath=M3ANCPATH):
    """Return m3 solar spectrum [W/(m^2 um)]."""
    ss = xr.DataArray(
        get_solar_spectrum(basename, m3ancpath).squeeze(),
        coords=[("wavelength", get_m3_wls(basename))],
    )
    return ss


def get_solar_irradiance_xr(basename, obs):
    """Return m3 solar spectrum normalized by solar distance [W/(m^2 um)]."""
    solar_dist = get_solar_dist(basename, obs)
    solar_spec = get_solar_spectrum_xr(is_targeted(basename))
    L_sun = re.get_solar_irradiance(solar_spec, solar_dist)
    return L_sun


def get_m3photom(i, e, g, basename, newph=True, polish=False):
    """
    Return photometric correction factor, used to correct M3 radiance to
    i=30, e=0, g=30 using the M3 photometric correction.
    """
    # Limb darkening breaks down at high inc / em angle (Besse et al., 2013)
    # if i > 85 or e > 85:
    #     return np.ones_like(g) * np.nan

    # Get M3 photometry table interpolated to phase values
    targeted = is_targeted(basename)
    if isinstance(g, xr.DataArray):
        photom = get_photom_xr(targeted, newph)
        photom30 = photom.sel(phase=30)
        photom_g = photom.interp(phase=g)
    else:
        phase, photom = get_photom(targeted, newph)
        photom30 = photom[np.argmin(np.abs(phase - 30))]
        photom_g = np.interp(g, phase[:, 0], photom[:, 0])
    norm_photom = photom30 / photom_g

    # limb darkening Lommel-Selliger (Eq. 3, Besse et al., 2013)
    i, e = np.radians(i), np.radians(e)
    ld = np.cos(i) / (np.cos(i) + np.cos(e))
    ld30 = np.cos(np.pi / 6) / (np.cos(np.pi / 6) + np.cos(0))  # i=30, e=0
    norm_ld = ld30 / ld

    # Get M3 spectral polish
    if polish:
        iswarm = get_m3_warm(basename)
        spec_polish = get_polish(targeted, iswarm)
    else:
        spec_polish = 1

    photom_corr = norm_ld * norm_photom * spec_polish  # * np.pi
    return photom_corr


# def get_m3photom_np(
#     inc,
#     em,
#     phase,
#     newph=True,
#     polish=False,
#     m3ancpath=M3ANCPATH,
#     targeted=False,
# ):
#     """
#     Return photometric correction factor, used to correct M3 radiance to
#     i=30, e=0, g=30 using the M3 photometric correction.
#     """
#     # Get M3 photometry table
#     dimx, dimy = inc.shape
#     a_phase, alpha = get_photom(targeted, newph)
#     dimz = alpha.shape[2]
#     alpha_out = np.zeros((dimx, dimy, dimz))
#     for i in range(dimz):
#         # interpolate alpha table to shape of phase
#         alpha_out[:, :, i] = np.interp(phase, a_phase[:, 0], alpha[:, 0, i])

#     norm_alpha = alpha[30] / alpha_out
#     cinc = np.cos(np.radians(inc))
#     cem = np.cos(np.radians(em))
#     cos30 = np.cos(np.radians(30))
#     angle = (cos30 / (cos30 + 1)) / (cinc / (cinc + cem))

#     # Get M3 spectral polish
#     spec_polish = 1
#     if polish:
#         iswarm = get_m3_warm(basename)
#         spec_polish = get_polish(targeted, iswarm, m3ancpath)
#     photom_corr = (
#         np.pi * norm_alpha * spec_polish * angle[:, :, np.newaxis]
#     )  # * angle_arr
#     return photom_corr


@lru_cache(maxsize=3)
def get_photom(targeted, newph=True, m3ancpath=M3ANCPATH, m3apt=M3ALTPHOTOM):
    """
    Return solar spectrum and alpha photometry tables as arrays.
    """
    if targeted:
        fphotom = Path(m3apt) / "M3T20120120_RFL_F_ALPHA_HIL.TAB"
    elif newph:
        # Newer photometric correction fixes 2.5-3 micron photometry (default)
        fphotom = Path(m3apt) / "M3G20120120_RFL_F_ALPHA_HIL.TAB"
    else:
        # Old photometric correction (artifacts > 2.5 microns)
        fphotom = Path(m3ancpath) / "M3G20111109_RFL_F_ALPHA_HIL.TAB"

    photom = np.genfromtxt(fphotom, skip_header=1)[:, np.newaxis, :]
    phase = photom[:, :, 0].astype(int)
    photom = photom[:, :, 1:]
    return phase, photom


def get_photom_xr(targeted, *args, **kwargs):
    """Return xarray m3 photom"""
    phase, photom = get_photom(targeted, *args, **kwargs)
    photom = xr.DataArray(
        photom.squeeze(),
        coords=[
            ("phase", phase.squeeze()),
            ("wavelength", get_m3_wls(targeted)),
        ],
    )
    return photom


@lru_cache(maxsize=4)
def get_polish(targeted, iswarm=False, m3ancpath=M3ANCPATH):
    """
    Return polish based on whether the detector was warm or cold.

    Returns
    -------
    polish_arr (np 1x1xN array)
    """
    if iswarm and targeted:
        fp = Path(m3ancpath) / "targeted" / "M3T20111020_RFL_STAT_POL_2.TAB"
    elif iswarm and not targeted:
        fp = Path(m3ancpath) / "M3G20110830_RFL_STAT_POL_2.TAB"
    elif not iswarm and targeted:
        fp = Path(m3ancpath) / "targeted" / "M3T20111020_RFL_STAT_POL_1.TAB"
    elif not iswarm and not targeted:
        fp = Path(m3ancpath) / "M3G20110830_RFL_STAT_POL_1.TAB"
    polish_table = np.loadtxt(fp)
    polish_arr = polish_table[:, 2].reshape(1, 1, len(polish_table))
    return polish_arr


def get_polish_xr(targeted, *args, **kwargs):
    """Return xarray polish"""
    polish = get_polish(targeted, *args, **kwargs).squeeze()
    polish = xr.DataArray(
        polish,
        coords=[("wavelength", get_m3_wls(targeted))],
    )
    return polish


def get_m3albedo(rad, obs, i, e, g, basename, wlmin=0.55, wlmax=2, amax=0.4):
    """
    Return albedo (Bandfield et al. 2018)
    """
    # Add factor to change M3 derived albedo to thermal albedo - JB
    factor = 2 - np.cos(np.deg2rad(i))

    # Get solar irradiance
    L_sun = get_solar_irradiance_xr(basename, obs)
    photom = get_m3photom(i, e, g, basename)
    refl = re.get_rad_factor(rad, L_sun) * photom

    # Clip noisy M3 channels
    if isinstance(refl, xr.DataArray):
        refl = refl.sel(wavelength=slice(wlmin, wlmax))
        L_sun = L_sun.sel(wavelength=slice(wlmin, wlmax))
        albedo = (refl * L_sun).sum("wavelength") / L_sun.sum("wavelength")
    # refl = refl[:, :, bmin:]
    # L_sun = L_sun[:, :, bmin:]
    # albedo = np.sum(refl * L_sun, axis=2) / np.sum(L_sun, axis=2)
    albedo = albedo * factor
    albedo.clip(max=amax)  # limit exteme albedos
    return albedo


def is_targeted(m3img):
    """
    Return True if m3img has 256 bands (targeted mode) rather than 85.
    """
    if isinstance(m3img, bool):
        return m3img
    if isinstance(m3img, str):
        return m3img.startswith("M3T")  # Check M3G or M3T
    m3img = np.atleast_1d(m3img)
    return m3img.shape[-1] == 256  # Check if shape 85 or 256


def get_m3_headers(
    basename,
    dirpath=M3PATH,
    flatdir=False,
    imgs=("rad", "ref", "obs", "sup", "loc"),
    usgs=0,
    usgspath="",
    ext=".HDR",
):
    """
    Return dict of paths to M3 header files for an observation.

    See Also
    --------
    https://astrogeology.usgs.gov/maps/
            moon-mineralogy-mapper-geometric-data-restoration

    Examples
    --------
    >>> get_m3_headers("M3G20090609T183254", dirpath="/path/to/m3")
    {'rad': '/path/to/m3/CH1M3_0003/DATA/20090415_20090816/200906/L1B/
             M3G20090609T183254_V03_RDN.HDR',
    'ref': '/path/to/m3/CH1M3_0004/DATA/20090415_20090816/200906/L2/
            M3G20090609T183254_V01_RFL.HDR',
    'obs': '/path/to/m3/CH1M3_0003/DATA/20090415_20090816/200906/L1B/
            M3G20090609T183254_V03_OBS.HDR',
    'sup': '/path/to/m3/CH1M3_0004/DATA/20090415_20090816/200906/L2/
            M3G20090609T183254_V01_SUP.HDR',
    'loc': '/path/to/m3/CH1M3_0003/DATA/20090415_20090816/200906/L1B/
            M3G20090609T183254_V03_LOC.HDR'}
    >>> get_m3_headers("M3G20090609T183254", dirpath="/path/to/imgs",
                       flatdir=True)
    {'rad': '/path/to/imgs/M3G20090609T183254_V03_RDN.HDR',
    'ref': '/path/to/imgs/M3G20090609T183254_V01_RFL.HDR',
    'obs': '/path/to/imgs/M3G20090609T183254_V03_OBS.HDR',
    'sup': '/path/to/imgs/M3G20090609T183254_V01_SUP.HDR',
    'loc': '/path/to/imgs/M3G20090609T183254_V03_LOC.HDR'}
    >>> get_m3_headers("M3G20090609T183254", dirpath="m3path", usgs=1,
                        usgspath="usgspath")
    {'rad': 'm3path/CH1M3_0003/DATA/20090415_20090816/200906/L1B/
             M3G20090609T183254_V03_RDN.HDR',
    'ref': 'm3path/CH1M3_0004/DATA/20090415_20090816/200906/L2/
            M3G20090609T183254_V01_RFL.HDR',
    'obs': 'usgspath/boardman2014/M3G20090609T183254_V03_OBS.HDR',
    'sup': 'm3path/CH1M3_0004/DATA/20090415_20090816/200906/L2/
            M3G20090609T183254_V01_SUP.HDR',
    'loc': 'usgspath/boardman2014/M3G20090609T183254_V03_LOC.HDR'}
    """
    single_img = False
    if isinstance(imgs, str):
        single_img = True
        imgs = [imgs]

    img2name = {
        "rad": "RDN",
        "ref": "RFL",
        "obs": "OBS",
        "sup": "SUP",
        "loc": "LOC",
        "lbl": "L1B",
    }

    if flatdir:
        # All images in toplevel directory at dirpath
        l1b = dirpath
        l2 = dirpath
    else:
        # Full m3 volume located at dirpath
        l1b = get_m3_directory(basename, dirpath, "L1B")
        l2 = get_m3_directory(basename, dirpath, "L2")

    # Get full path for each img
    headers = {}
    for img in imgs:
        _ext = ext
        if img in ("rad", "loc", "obs", "lbl"):
            version = "V03"
            level = l1b
            if img == "lbl":
                _ext = ".LBL"
        elif img in ("ref", "sup"):
            version = "V01"
            level = l2
        else:
            raise ValueError(f"Unknown image type: {img}")
        fname = "_".join((basename, version, img2name[img] + _ext))
        headers[img] = Path(level) / fname

    # USGS corrected loc and obs
    if usgs:
        if not usgspath:
            usgspath = Path(dirpath) / "USGS"
        if usgs == 1:
            # Step 1 usgs correction (2014)
            usgsdir = "boardman2014"
            loc = basename + "_V03_LOC.HDR"
            obs = basename + "_V03_OBS.HDR"
        elif usgs == 2:
            # Step 2 usgs correction (2017)
            usgsdir = "boardman2017"
            loc = basename + "_V03_L1B_LOC_USGS.hdr"
            obs = basename + "_V03_L1B_OBS.HDR"
        else:
            raise ValueError("Invalid usgs step (must be 1 or 2)")
        headers["loc"] = Path(usgspath) / usgsdir / loc
        headers["obs"] = Path(usgspath) / usgsdir / obs
    if single_img:
        headers = headers[imgs[0]]
    return headers


def parse_envi_header(fname):
    """
    Return dictionary of parameters in the envi header.

    TODO: port to open planetary package
    """
    params = {}
    with open(fname, encoding="utf8") as f:
        key, value = "", ""
        line = f.readline()
        while line:
            if isinstance(value, list):
                if "}" in line:
                    value.append(  # pylint: disable=E1101
                        line.split("}")[0].strip()
                    )
                    params[key] = list(filter(None, value))
                    value = ""
                else:
                    value.append(line.strip(" \t\n,"))
            elif "=" in line:
                key, value = line.split("=")
                key, value = key.strip(), value.strip()
                if value == "{":
                    value = []
                elif "DESCRIPTION" in key and line.count('"') == 1:
                    # Parse multi-line description
                    values = [value]
                    line = f.readline()
                    while line and line.count('"') != 1:
                        values.append(line.strip())
                        line = f.readline()
                    values.append(line.strip())
                    params[key] = " ".join(values)
                else:
                    params[key] = value.strip('"')
            line = f.readline()
    return params


def get_m3_directory(basename, dirpath="", level="L2"):
    """
    Return PDS-like path to M3 observation, given basename and dirpath to m3
    root directory.
    """
    # Build M3 data path as list, join at the end
    m3dir = Path(dirpath)

    if level == "L1B":
        m3dir = m3dir / "CH1M3_0003"
    elif level == "L2":
        m3dir = m3dir / "CH1M3_0004"
    m3dir = m3dir / "DATA"

    # Get optical period
    opt = get_m3_op(basename)
    if opt[2] == "1":
        m3dir = m3dir / "20081118_20090214"
    elif opt[2] == "2":
        m3dir = m3dir / "20090415_20090816"

    m3dir = m3dir / basename[3:9] / level
    return m3dir


def get_m3_date(basename):
    """Return date object representing date of M3 observation"""
    return datetime.strptime(basename[3:].replace("T", ""), "%Y%m%d%H%M%S")


def get_m3_op(basename):
    """Return optical period of m3 observation based on date of observation"""

    def date_in_range(date, start, end):
        start = datetime.strptime(start, "%Y%m%d")
        end = datetime.strptime(end, "%Y%m%d")
        return start <= date < end

    date = get_m3_date(basename)
    if date_in_range(date, "20081118", "20090125"):
        opt = "OP1A"
    elif date_in_range(date, "20090125", "20090215"):
        opt = "OP1B"
    elif date_in_range(date, "20090415", "20090428"):
        opt = "OP2A"
    elif date_in_range(date, "20090513", "20090517"):
        opt = "OP2B"
    elif date_in_range(date, "20090520", "20090623"):
        opt = "OP2C1"
    elif date_in_range(date, "20090623", "20090722"):
        opt = "OP2C2"
    elif date_in_range(date, "20090722", "20090817"):
        opt = "OP2C3"
    else:
        msg = f"Optical period for {basename} not found. Check img name."
        raise ValueError(msg)
    return opt


def get_m3_warm(basename):
    """Return True if m3 observation was warm (Besse et al., 2013)."""
    return get_m3_op(basename) in ("OP1A", "OP2B", "OP2C1", "OP2C2")


def read_m3_image(hdr_path, ymin=0, ymax=None, xmin=0, xmax=None):
    """
    Return np.array() with subset of m3 image with rasterio. Assumes the
    image is in the same directory as the header in hdr_path.
    """
    ext = ".IMG" if hdr_path[-1].isupper() else ".img"
    fname = hdr_path[:-4] + ext
    with rio.open(fname) as src:
        rwindow = rio.windows.Window.from_slices(
            (ymin, ymax), (xmin, xmax), width=src.width, height=src.height
        )
        img = rio2xy(src.read(window=rwindow))

    return img


def readm3geotif_xr(fname, ext):
    """
    Read M3 geotiff and return xarray
    """
    da = xr.open_dataarray(fname)
    if "wavelength" in da.attrs:
        wls = np.array(da.attrs["long_name"], dtype=float) / 1000
        da = da.assign_coords(wavelength=("band", wls))
        da = da.swap_dims({"band": "wavelength"}).drop("band")
        del da.attrs["wavelength"]
        del da.attrs["long_name"]
    elif "long_name" in da.attrs:
        da = da.assign_coords(labels=("band", np.array(da.attrs["long_name"])))
        del da.attrs["long_name"]

    da = da.rename({"x": "lon", "y": "lat"}).sortby("lat")
    da = da.sel(lon=slice(ext[0], ext[1]), lat=slice(ext[2], ext[3]))
    return da.transpose("lat", "lon", *set(da.dims).difference({"lat", "lon"}))


def read_m3_geotiff(fname, ext, pad=False):
    """
    Return M3 image from file f with extent ext using rasterio.

    Parameters
    ----------
    f (str): path to file
    ext (tuple): (latmin, latmax, lonmin, lonmax)
    pad (bool): pad image to match extent of ext
    """
    left, right, bot, top = ext
    with rio.open(fname) as src:
        w = rio.windows.from_bounds(left, bot, right, top, src.transform)
        img = rio2xy(src.read(window=w))
        if pad:
            img = rio_pad(img, ext, w, src.transform)
    return img


def ext2window(ext, transform):
    """Return rasterio window of data bounded by ext in dataset with transform."""
    # Location of upper left corner of data (pixel offset)
    row_off, col_off = rio.transform.rowcol(transform, ext[0], ext[3])
    # Size of window (in pixels)
    xres = transform[0]
    yres = -transform[4]
    width = int(round((ext[1] - ext[0]) / xres))
    height = int(round((ext[3] - ext[2]) / yres))
    w = rio.windows.Window(col_off, row_off, width, height)
    return w


def read_m3_geotiff_ext(fname, ext):
    """Read M3 geotiff with ext (minlon, maxlon, minlat, maxlat)"""
    with rio.open(fname) as src:
        w = ext2window(ext, src.transform)
        img = src.read(window=w)
        img = rio2xy(img)
        img = rio_pad(img, ext, w, src.transform)
        if src.count == 1 and len(img.shape) > 2:
            img = img[:, :, 0]
    return img


def m3_size_est(basename, ext):
    """Return estimates number of pixels from M3img name and extent."""
    mpp_res = get_m3_res(basename)
    ppd_res = mpp2ppd(mpp_res, lat=np.mean(ext[2:]))
    xsize = ppd_res * (ext[1] - ext[0])
    ysize = ppd_res * (ext[3] - ext[2])
    zsize = len(get_m3_wls(basename))
    pixels = int(xsize * ysize)
    data_size = pixels * zsize * 8 / 1e6  # MB assuming 8 bit float
    return pixels, data_size


def get_m3_res(basename):
    """
    Return the M3 spatial resolution in m/pixel which depends on the
    operational mode (Global or Target) and the spacecraft altitude (doubled
    in OP2C).
    """
    mode = basename[2]
    if mode == "G":
        res = 140  # m/pixel
    elif mode == "T":
        res = 70  # m/pixel
    else:
        raise ValueError("Basename must begin with M3G or M3T.")
    if get_m3_op(basename) in ("OP2C1", "OP2C2", "OP2C3"):
        res *= 2
    return res


def mpp2ppd(mpp, lat=0, Rmoon=1737.4):
    """Convert lunar meters / pixel to pixels / degree resolution at lat."""
    circum_eq = 2 * np.pi * Rmoon * 1000  # m
    circum_lat = circum_eq * np.cos(lat)
    ppd = (circum_lat / mpp) / 360
    return ppd


def rio_pad(arr, ext, w, transform, fill_value=np.nan):
    """Zero-pad a rasterio windowed read array to shape of full extent"""
    if len(arr.shape) == 2:
        arr = arr[:, :, np.newaxis]
    arr_width, arr_height, bands = arr.shape
    res = transform[0]  # deg/pixel
    wbounds = rio.transform.array_bounds(w.height, w.width, transform)
    ext_bounds = (ext[0], ext[2], ext[1], ext[3])
    # Compute bounds of full extent output image
    ext_width = int(round((ext_bounds[2] - ext_bounds[0]) / res))  # pixels
    ext_height = int(round((ext_bounds[3] - ext_bounds[1]) / res))  # pixels
    # Get coords of window in extent sized image
    ymax = int(round((ext_bounds[3] - wbounds[3]) / res))
    xmin = int(round((wbounds[0] - ext_bounds[0]) / res))
    # Init extent-sized output
    out = np.full((ext_width, ext_height, bands), fill_value)
    if (ymax > 0) and (xmin > 0):
        out[xmin : xmin + arr_width, ymax : ymax + arr_height, :] = arr
    elif ymax > 0:
        out[:arr_width, ymax : ymax + arr_height, :] = arr
    elif xmin > 0:
        out[xmin : xmin + arr_width, :arr_height, :] = arr
    else:
        out[:arr_width, :arr_height, :] = arr
    return out


def get_albedo_feng(lons, lats):
    """Return albedo interpolated from Feng et al. (2020) at lons, lats."""
    return read_albedo_map_feng().interp(lon=lon180(lons), lat=lats)


def read_albedo_map_feng(plot=False, falbedo=FALBEDO, amin=0.04, amax=0.22):
    """
    Return normal bolometric bond albedo albedo map from Feng et al. (2020).

    https://doi.org/10.5281/zenodo.3575481
    """
    df = pd.read_csv(falbedo, header=None, delimiter=r"\s+")
    lats = np.linspace(90, -90, len(df))
    lon = np.linspace(-180, 180, len(df.columns))
    albmap = xr.DataArray(df, coords=[lats, lon], dims=["lat", "lon"])
    albmap.name = "albedo"
    if plot:
        albmap.plot.imshow(size=5, aspect=len(albmap.lon) / len(albmap.lat))
    return albmap.clip(amin, amax)


# Read M3 rois
def read_m3_rois(roi_path, sort_by_size=False):
    """Return dict of rois from roi_path."""
    with open(roi_path, encoding="utf8") as f:
        prio = json.load(f)
    if sort_by_size:
        size = []
        for k, v in prio.items():
            ext, *basenames = v
            size.append([m3_size_est(basenames[0], ext)[1], k])
        prio = {k: prio[k] for _, k in sorted(size)}
    return prio


# Read M3 PDS data with xarray (rioxarray, dask, xoak)
def get_m3xr(basename, img, usgs=2, dask=False, chunks=None, xoak=False):
    """
    Return m3 img (rad/ref/obs/sup/loc) as lazy loaded xarray.
    """
    hdr = get_m3_headers(basename, imgs=img, usgs=usgs)
    fimg = Path(hdr).with_suffix(".IMG")
    if not fimg.exists():
        fimg = Path(hdr).with_suffix(".img")
    if dask and chunks is None:
        # Chunk along y (lines), never bands (efficient spectral operations)
        # x (samples) usually 304 or 608 so chunking along x unnecessary
        chunks = {"band": None, "y": "auto", "x": None}
    chunks = None
    da = xr.open_dataarray(
        fimg,
        engine="rasterio",
        parse_coordinates=True,
        chunks=chunks,
        drop_variables=["spatial_ref"],
        default_name=img,
    )

    if img in ("rad", "ref"):
        wls = get_m3_wls(basename)
        da = (
            da.assign_coords(wavelength=("band", wls))
            .swap_dims({"band": "wavelength"})
            .drop("band")
        )

    # Not all attrs useful, some don't parse well from envi header
    da.attrs = {k: v for k, v in da.attrs.items() if k not in DROP_ATTRS}
    da.attrs["basename"] = basename
    da.attrs["fimg"] = fimg.as_posix()

    # Set up lat / lon coords from loc backplane (if loc this is redundant)
    if img == "loc":
        return da
    loc = get_m3xr(basename, "loc", usgs, dask=False, xoak=False)
    da = da.assign_coords(
        {
            "lat": (("y", "x"), loc.sel(band=2).data),
            "lon": (("y", "x"), loc.sel(band=1).data),
        }
    )
    da.attrs["floc"] = loc.attrs["fimg"]

    if xoak:
        da = index_m3xoak(da)
    return da


def subset_m3xr_ext(da, ext):
    """
    Return m3 da subsetted to lat/lon ext [minlon, maxlon, minlat, maxlat].
    """
    warnings.filterwarnings("ignore")  # Dask chunk performance warning
    subda = da.where(
        (lon360(ext[0]) <= da.lon)
        & (da.lon <= lon360(ext[1]))
        & (ext[2] <= da.lat)
        & (da.lat <= ext[3]),
        drop=True,
    )
    if subda.chunks is not None:
        chunks = {d: None for d in subda.dims}
        chunks["y"] = "auto"
        subda.chunk(chunks)
    return subda


def index_m3xoak(ds, opts=None):
    """Return ds with 2D xoak index set from loc file."""
    # Default kdtree opts (from timing global data and 300x1000 pix output)
    if opts is None:
        opts = dict(leafsize=512, balanced_tree=False, compact_nodes=False)
    ds.xoak.set_index(["lat", "lon"], "scipy_kdtree", **opts)
    return ds


def sel_m3xoak(ds, lats, lons):
    """
    Return xarray of m3 data at lats, lons using xoak index.
    """
    if ds.xoak.index is None:
        raise ValueError("Must have xoak index set (see index_m3_xoak).")

    # Make (lat, lon) xoak selector
    sel = xr.Dataset(
        {
            "lat": (("y", "x"), np.tile(lats, (len(lons), 1))),
            "lon": (("y", "x"), np.tile(lons, (len(lats), 1)).T),
        }
    )
    return ds.xoak.sel(lat=sel.lat, lon=sel.lon)


# Spiceypy helpers (getting local solar time from the Moon)
def furnsh_kernels(kernels_path=KERNELPATH):
    """
    Return spiceypy kernels for meta_path.
    """
    kernels = Path(kernels_path)
    if kernels.suffix != ".tm":
        # Get all kernels, exlude meta kernels ('.tm')
        kernels = [
            k.as_posix() for k in kernels.rglob("*") if k.suffix != ".tm"
        ]
    spiceypy.furnsh(kernels)  # Else assume path is a .tm


def m3basename2utc(basename, fmt="%Y-%m-%dT%H:%M:%S"):
    """Return UTC time from m3 basename."""
    return get_m3_date(basename).strftime(fmt)


def utc2lstmoon(utc, lon):
    """
    Return lunar local solar time at utc [yyyy-mm-ddThh:mm:ss] and lon [deg].
    Uses spiceypy. Moon is NAIF ID 301.
    """
    if spiceypy.ktotal("ALL") == 0:
        furnsh_kernels()
    et = spiceypy.str2et(utc)
    lst = spiceypy.spiceypy.et2lst(et, 301, np.deg2rad(lon), "PLANETOGRAPHIC")
    return lst


def lst2tloc(lst):
    """Return decimal local time [hr] from lst [hr,min,sec] (spice et2lst)."""
    return lst[0] + lst[1] / 60 + lst[2] / 3600


def get_tloc_range_spice(basename, minlon, maxlon=None):
    """
    Return M3 min and max local time using spice.
    Gets start and end times from label file if found, else takes timestamp
    from the basename. In future could read actual times from TIM files
    (but range usually <0.1 lst for a given image).
    """
    lons = [minlon, maxlon] if maxlon else [minlon]
    # Try to read time from M3 label file, if exists, else infer from basename
    try:
        lbl = get_m3_headers(basename, imgs="lbl")
        meta = parse_envi_header(lbl)
        utcs = [meta["START_TIME"], meta["STOP_TIME"]]
    except FileNotFoundError:
        utcs = [m3basename2utc(basename)]
    tlocs = []
    for utc in utcs:
        for lon in lons:
            tlocs.append(lst2tloc(utc2lstmoon(utc, lon)))
    return min(tlocs), max(tlocs)


# def convert_axes(src, dst, nax=3):
#     """Return axis order from src or dst."""
#     axes = {
#         'xy': [0, 1, 2],
#         'np': [1, 0, 3],
#         'rio': [2, 1, 0]
#     }
# return axes[src], axes[dst]
def lon360(lon):
    """Convert lon to (0, 360)."""
    return (lon + 360) % 360


def lon180(lon):
    """Convert lon to (-180, 180)."""
    return (lon + 180) % 360 - 180


def rio2xy(arr):
    """Swap axes from [(bands), rows, columns] to [x, y, (bands)]."""
    if len(arr.shape) == 3:
        arr = np.moveaxis(arr, [0, 1, 2], [2, 1, 0])
    elif len(arr.shape) == 2:
        arr = np.moveaxis(arr, 0, 1)
    return arr


def xy2np(arr):
    """Swap axes from [x, y, (bands)] to [rows, columns, (bands)]."""
    if len(arr.shape) == 3:
        arr = np.moveaxis(arr, [0, 1, 2], [1, 0, 2])
    elif len(arr.shape) == 2:
        arr = np.moveaxis(arr, 0, 1)
    return arr


def wl_to_ind(wl, wavelengths):
    """Return index of wl in wavelengths."""
    ind = np.argmin(np.abs(np.array(wavelengths) - wl))
    return ind, wavelengths[ind]


def m3_wl_to_ind(wl, img):
    """
    Get index and nearest wavelength of img given wl in nanometers.
    """
    return wl_to_ind(wl, get_m3_wls(img))


def get_dem_xr(fslope, fazimuth, felevation):
    """Return DEM as xarray from fslope, fazimuth, felevation"""
    s = xr.open_dataarray(fslope).sel(band=1).drop("band")
    a = xr.open_dataarray(fazimuth).sel(band=1).drop("band")
    e = xr.open_dataarray(felevation).sel(band=1).drop("band")
    s.name = "Slope (rad)"
    a.name = "Aspect (rad)"
    e.name = "Elevation"
    dem = xr.Dataset({"slope": s, "aspect": a, "elevation": e})
    dem = dem.rename({"x": "lon", "y": "lat"})
    return dem


def get_dem_geotiff(
    fslo, fazim, felev, ext, width=None, height=None, resampling="nearest"
):
    """
    Return DEM geotiff from fslo, fazim, felev geotiff files.
    """
    left, right, bot, top = ext
    rs = getattr(rio.enums.Resampling, resampling)
    with rio.open(fslo) as s, rio.open(fazim) as a, rio.open(felev) as e:
        w = rio.windows.from_bounds(left, bot, right, top, s.transform)
        if width is None:
            width = int(w.width)
        if height is None:
            height = int(w.height)
        outshape = (1, height, width)
        slo = s.read(window=w, out_shape=outshape, resampling=rs)
        azim = a.read(window=w, out_shape=outshape, resampling=rs)
        elev = e.read(window=w, out_shape=outshape, resampling=rs)
    dem = rio2xy(np.vstack([slo, azim, elev]))
    return dem


# Spectra
def get_spec(arr, wlmin=0, wlmax=1e5, wls=None):
    """Return the wavelengths and spectrum of a 1D arr (1 pixel) of M3 data
    Optionally specify min and max wavelengths to crop spectrum"""
    if wls is None:
        wls = get_m3_wls(arr)
    bmin, wlmin = wl_to_ind(wlmin, wls)
    bmax, wlmax = wl_to_ind(wlmax, wls)
    wls = wls[bmin : bmax + 1]
    spec = arr[bmin : bmax + 1]
    return wls, spec


def get_avg_spec(img, wlmin=0, wlmax=1e5, wls=None):
    """Return the wavelengths and mean spectrum within the specified img array.
    Optionally specify min and max wavelengths to crop spectrum."""
    arr = np.nanmean(img, axis=(0, 1))  # Average along z (spectral) axis
    return get_spec(arr, wlmin, wlmax, wls)


def sel_nearest(da, coord, x):
    """Return nearest coord value of each x (or index of x) in xarr."""
    scalar = np.isscalar(x)
    x = [x] if scalar else x
    nearest = []
    for val in x:
        nearest.append(da[coord].sel({coord: val}, method="nearest").values)
    return nearest[0] if scalar else nearest


def lin_contin(spec, wl1=2.4, wl2=2.6):
    """Return slope and intercept of linear continuum between wl1 and wl2."""
    wl1, wl2 = sel_nearest(spec, "wavelength", (wl1, wl2))
    slope = (spec.sel(wavelength=wl2) - spec.sel(wavelength=wl1)) / (wl2 - wl1)
    intercept = spec.sel(wavelength=wl1) - slope * wl1
    return slope, intercept


def contin_remove(spec, wl1=2.4, wl2=2.6, divide=True):
    """Remove linear continuum fit between wl1 and wl2 from spec."""
    slope, intercept = lin_contin(spec, wl1, wl2)
    contin = slope * spec.wavelength + intercept
    if divide:
        return spec / contin
    return spec - contin


def ohibd(spec, wl1=2.69673, wl2=2.93627):
    """Return 3um Integrated Band Depth from spec between (wl1, wl2)."""
    wl1, wl2 = sel_nearest(spec, "wavelength", (wl1, wl2))
    spec_oh = spec.sel(wavelength=slice(wl1, wl2))
    ibd = (1 - spec_oh).integrate("wavelength")
    dwl = wl2 - wl1
    return ibd / dwl
