"""M3-specific functions for testing"""
import time
import os.path as op
import numpy as np
import pandas as pd
import xarray as xr
import rasterio as rio
import roughness.emission as re
import roughness.helpers as rh
from roughness.config import M3GWL as GWL
from roughness.config import M3TWL as TWL
from roughness.config import FLOOKUP, FALBEDO, TLOOKUP

# Constants
PATH = "/work/ctaiudovicic/data/"
M3PATH = f"{PATH}m3/"
M3ANCPATH = f"{PATH}M3_ancillary_files/"
M3ALTPHOTOM = f"{PATH}M3_ancillary_files/alt_photom_tables/"


def m3_refl(
    m3_basename,
    rad_L1,
    obs,
    wl=None,
    dem=None,
    rms=18,
    albedo=None,
    emiss=None,
    rerad=False,
    flookup=FLOOKUP,
    tlookup=TLOOKUP,
    toffset=0,
):
    """
    Return M3 reflectance using the roughness thermal correction (Bandfield et
    al., 2018). Uses ray-casting generated shadow_lookup table. Applies
    standard M3 photometric correction.

    """
    # Get wavelength, viewing geometry, photometry from M3 files
    if wl is None:
        wl = get_m3_wls(rad_L1)
    solar_dist = get_solar_dist(m3_basename, obs)
    sun_thetas, sun_azs, sc_thetas, sc_azs, phases, _ = get_m3_geom(obs, dem)
    photom = get_m3photom(
        sun_thetas,
        sc_thetas,
        phases,
        polish=True,
        targeted=is_targeted(rad_L1),
    )
    # Get solar irradiance
    solar_spec = get_solar_spectrum(is_targeted(rad_L1))
    L_sun = re.get_solar_irradiance(solar_spec, solar_dist[:, :, np.newaxis])

    # Compute albedo from raw (uncorrected) radiance
    if albedo is None:
        refl_raw = re.get_rad_factor(rad_L1, 0, L_sun) * photom
        albedo = get_m3albedo(sun_thetas, refl_raw, solar_spec)
    if not isinstance(albedo, xr.DataArray):
        albedo = np.ones(rad_L1.shape[:2]) * albedo
    # geom = (sun_thetas, sun_azs, sc_thetas, sc_azs)
    # emission = re.rough_emission_lookup(
    #     geom, wl, rms, albedo, rad_L1.lat, tloc, rerad, flookup, tlookup
    # )
    # Get roughness emission for each pixel from rad eq on subpixel facets
    emission = xr.zeros_like(rad_L1).reindex({"wavelength": wl}, method="pad")
    for i, lon in enumerate(emission.lon):
        print(f"{i}/{emission.lon.size}")
        for j, lat in enumerate(emission.lat):
            # TODO: split into function, parallelize
            if isinstance(albedo, xr.DataArray):
                alb = albedo.sel(lon=lon, lat=lat)
            else:
                alb = albedo[i, j]
            geom = [c[i, j] for c in (sun_thetas, sun_azs, sc_thetas, sc_azs)]
            emission[i, j] = re.rough_emission_lookup(
                geom, wl, rms, alb, lat, rerad, flookup, tlookup
            )
            # emission[i, j] = re.rough_emission_eq(
            #   geom, wl, rms, alb, emiss, solar_dist[i, j], rerad, flookup
            # )

    # Remove emission from Level 1 radiance and convert to I/F
    rad_emission = emission.interp(wavelength=rad_L1.wavelength)
    i_f = re.get_rad_factor(rad_L1, rad_emission, L_sun, emiss)
    refl = i_f * photom
    return refl, emission


def get_m3_geom(obs, dem=None):
    """Return geometry array from M3 observation file and optional dem."""
    if dem is None:
        dem_theta = np.radians(obs[:, :, 7])
        dem_az = np.radians(obs[:, :, 8])
    else:
        dem_theta = np.arctan(dem[:, :, 0])
        dem_az = dem[:, :, 1]
    sun_theta = np.radians(obs[:, :, 1])
    sun_az = np.radians(obs[:, :, 0])
    sc_theta = np.radians(obs[:, :, 3])
    sc_az = np.radians(obs[:, :, 2])

    # Compute angles in degrees
    i, e, g, sun_sc_az = rh.get_ieg(
        dem_theta, dem_az, sun_theta, sun_az, sc_theta, sc_az
    )
    sun_az = np.rad2deg(sun_az)
    sc_az = np.rad2deg(sc_az)
    return i, sun_az, e, sc_az, g, sun_sc_az


def get_m3_tloc(obs, dem=None):
    """
    Return average local time of M3 observation.
    """
    return rh.inc_to_tloc(*get_avg_sun_inc_az(obs, dem))


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
    obs_fname = get_m3_headers(basename, imgs="obs")["obs"]
    ext = ".HDR" if obs_fname[-3:].isupper() else ".hdr"
    obs_header = obs_fname[:-4] + ext
    sdist = 1
    try:
        params = parse_envi_header(obs_header)
        description = "".join(params["description"])
        i = description.find("(au)")
        sdist = float(description[i + 6 : i + 16])
    except FileNotFoundError as e:
        w = "Could not get sdist for {}: {}".format(obs_fname, str(e))
        print(w)
    solar_dist = obs[:, :, 5] + sdist
    return solar_dist


def get_solar_spectrum(targeted, m3ancpath=M3ANCPATH):
    """Return m3 solar spectrum [W/(m^2 um)]."""
    if targeted:
        fss = m3ancpath + "targeted/M3T20110224_RFL_SOLAR_SPEC.TAB"
    else:
        fss = m3ancpath + "M3G20110224_RFL_SOLAR_SPEC.TAB"
    ss_raw = np.genfromtxt(fss).T[:, :, np.newaxis]
    ss = np.moveaxis(ss_raw[1:2], 1, 2)
    return ss


def get_m3photom(
    inc,
    em,
    phase,
    isalpha=True,
    polish=False,
    m3ancpath=M3ANCPATH,
    targeted=False,
):
    """
    Return photometric correction factor, used to correct M3 radiance to
    i=30, e=0, g=30 using the M3 photometric correction.
    """
    # Get M3 photometry table
    dimx, dimy = inc.shape
    a_phase, alpha = get_photom(targeted, isalpha)
    dimz = alpha.shape[2]
    alpha_out = np.zeros((dimx, dimy, dimz))
    for i in range(dimz):
        # interpolate alpha table to shape of phase
        alpha_out[:, :, i] = np.interp(phase, a_phase[:, 0], alpha[:, 0, i])

    norm_alpha = alpha[30] / alpha_out
    cinc = np.cos(np.radians(inc))
    cem = np.cos(np.radians(em))
    cos30 = np.cos(np.radians(30))
    angle = (cos30 / (cos30 + 1)) / (cinc / (cinc + cem))

    # Get M3 spectral polish
    spec_polish = 1
    if polish:
        spec_polish = get_polish(polish, targeted, m3ancpath)
    photom_corr = (
        np.pi * norm_alpha * spec_polish * angle[:, :, np.newaxis]
    )  # * angle_arr
    return photom_corr


def get_photom(targeted, alpha, m3ancpath=M3ANCPATH, m3apt=M3ALTPHOTOM):
    """
    Return solar spectrum and alpha photometry tables as arrays.
    """
    if not targeted:
        if not alpha:
            alpha_file = m3ancpath + "M3G20111109_RFL_F_ALPHA_HIL.TAB"
        else:
            # Newer photometric correction fixes 2.5-3 micron photometry -
            # this should generally be applied by default
            alpha_file = m3apt + "M3G20120120_RFL_F_ALPHA_HIL.TAB"
    else:
        alpha_file = m3apt + "M3T20120120_RFL_F_ALPHA_HIL.TAB"

    al = np.genfromtxt(alpha_file, skip_header=1)[:, np.newaxis, :]
    a_phase = al[:, :, 0]
    al = al[:, :, 1:]
    return a_phase, al


def get_polish(polish, targeted, m3ancpath=M3ANCPATH):
    """
    Return polish based on whether the detector was warm or cold.

    Returns
    -------
    polish_arr (np 1x1xN array)
    """
    # Polish file paths
    coldpolish = op.join(m3ancpath, "M3G20110830_RFL_STAT_POL_1.TAB")
    coldpolish_targeted = op.join(
        m3ancpath, "targeted", "M3T20111020_RFL_STAT_POL_1.TAB"
    )
    warmpolish = op.join(m3ancpath, "M3G20110830_RFL_STAT_POL_2.TAB")
    warmpolish_targeted = op.join(
        m3ancpath, "targeted", "M3T20111020_RFL_STAT_POL_2.TAB"
    )

    # Cold detector (20090119-20090428; 20090712-20090817)
    if (
        (polish == 1)
        or (20090119 <= polish <= 20090428)
        or (20090712 <= polish <= 20090817)
    ):
        fpolish = coldpolish_targeted if targeted else coldpolish
    # Warm detector warm (20081118-20090118; 20090513-20090710)
    elif (
        (polish == 0)
        or (1 < polish <= 20090118)
        or (20090513 <= polish <= 20090710)
    ):
        fpolish = warmpolish_targeted if targeted else warmpolish
    else:
        raise ValueError("Cannot find polish file.")
    polish_table = np.loadtxt(fpolish)
    polish_arr = polish_table[:, 2].reshape(1, 1, len(polish_table))
    return polish_arr


def get_m3albedo(sun_thetas, refl, solar_spec):
    """
    Return albedo (Bandfield et al. 2018)
    """
    # Add factor to change M3 derived albedo to thermal albedo - JB
    factor = 2 - np.cos(np.deg2rad(sun_thetas))

    # Clip noisy M3 channels
    refl = refl[:, :, 3:]
    solar_spec = solar_spec[:, :, 3:]
    albedo = np.sum(refl * solar_spec, axis=2) / np.sum(solar_spec, axis=2)
    albedo = albedo * factor
    albedo.clip(max=0.4)  # limit exteme albedos
    return albedo


def is_targeted(m3img):
    """
    Return True if m3img has 256 bands (targeted mode) rather than 85.
    """
    m3img = np.atleast_1d(m3img)
    return m3img.shape[-1] == 256


def get_m3_headers(
    basename,
    dirpath=M3PATH,
    flatdir=False,
    imgs=("rad", "ref", "obs", "sup", "loc"),
    usgs=0,
    usgspath="",
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
    if isinstance(imgs, str):
        imgs = [imgs]

    IMGS = {
        "rad": "RDN",
        "ref": "RFL",
        "obs": "OBS",
        "sup": "SUP",
        "loc": "LOC",
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
        if img in ("rad", "loc", "obs"):
            f = "_".join((basename, "V03", IMGS[img] + ".HDR"))
            headers[img] = op.join(l1b, f)
        elif img in ("ref", "sup"):
            f = "_".join((basename, "V01", IMGS[img] + ".HDR"))
            headers[img] = op.join(l2, f)

    # USGS corrected loc and obs
    if usgs:
        if not usgspath:
            usgspath = dirpath
        if usgs == 1:
            # Step 1 usgs correction (2014)
            usgsdir = "boardman2014"
            locext = "_V03_LOC.HDR"
            obsext = "_V03_OBS.HDR"
        elif usgs == 2:
            # Step 2 usgs correction (2017)
            usgsdir = "boardman2017"
            locext = "_V03_L1B_LOC_USGS.hdr"
            obsext = "_V03_L1B_OBS.HDR"
        else:
            raise ValueError("Invalid usgs step (must be 1 or 2)")
        headers["loc"] = op.join(usgspath, usgsdir, basename + locext)
        headers["obs"] = op.join(usgspath, usgsdir, basename + obsext)

    return headers


def parse_envi_header(fname):
    """
    Return dictionary of parameters in the envi header.

    TODO: port to open planetary package
    """
    params = {}
    with open(fname) as f:
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
                key, value = "".join(line.split()).split("=")
                if value == "{":
                    value = []
                else:
                    params[key] = value
            line = f.readline()
    return params


def get_m3_directory(basename, dirpath="", level="L2"):
    """
    Return PDS-like path to M3 observation, given basename and dirpath to m3
    root directory.
    """

    # Build M3 data path as list, join at the end
    path = [dirpath]

    if level == "L1B":
        path.append("CH1M3_0003")
    elif level == "L2":
        path.append("CH1M3_0004")

    path.append("DATA")

    # Get optical period
    opt = get_m3_op(basename)
    if opt[2] == "1":
        path.append("20081118_20090214")
    elif opt[2] == "2":
        path.append("20090415_20090816")

    path.append(basename[3:9])
    path.append(level)

    return op.join(*path)


def get_date(basename):
    """Return date object representing date of M3 observation"""
    return time.strptime(basename[3:11], "%Y%m%d")


def get_m3_op(basename):
    """Return optical period of m3 observation based on date of observation"""

    def date_in_range(date, start, end):
        start = time.strptime(start, "%Y%m%d")
        end = time.strptime(end, "%Y%m%d")
        return start <= date < end

    date = get_date(basename)
    if date_in_range(date, "20081118", "20090125"):
        opt = "OP1A"
    elif date_in_range(date, "20090125", "20090215"):
        opt = "OP1B"
    elif date_in_range(date, "20090415", "20090428"):
        opt = "OP2A"
    elif date_in_range(date, "20090513", "20090517"):
        opt = "OP2B"
    elif date_in_range(date, "20090520", "20090816"):
        opt = "OP2C"
    else:
        msg = f"Optical period for {basename} not found. Check img name."
        raise ValueError(msg)
    return opt


def read_m3_image(hdr_path, ymin=0, ymax=None, xmin=0, xmax=None):
    """
    Return np.array() with subset of m3 image with rasterio. Assumes the
    image is in the same directory as the header in hdr_path.
    """
    ext = ".IMG" if hdr_path[-1].isupper() else ".img"
    f = hdr_path[:-4] + ext
    with rio.open(f) as src:
        rwindow = rio.windows.Window.from_slices(
            (ymin, ymax), (xmin, xmax), width=src.width, height=src.height
        )
        img = rio2xy(src.read(window=rwindow))

    return img


def read_m3_geotiff(f, ext, pad=False):
    """
    Return M3 image from file f with extent ext using rasterio.

    Parameters
    ----------
    f (str): path to file
    ext (tuple): (latmin, latmax, lonmin, lonmax)
    pad (bool): pad image to match extent of ext
    """
    left, right, bot, top = ext
    with rio.open(f) as src:
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


def read_m3_geotiff_ext(f, ext):
    """Read M3 geotiff with ext (minlon, maxlon, minlat, maxlat)"""
    with rio.open(f) as src:
        w = ext2window(ext, src.transform)
        img = src.read(window=w)
        img = rio2xy(img)
        img = rio_pad(img, ext, w, src.transform)
        if src.count == 1 and len(img.shape) > 2:
            img = img[:, :, 0]
    return img


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


# def convert_axes(src, dst, nax=3):
#     """Return axis order from src or dst."""
#     axes = {
#         'xy': [0, 1, 2],
#         'np': [1, 0, 3],
#         'rio': [2, 1, 0]
#     }
# return axes[src], axes[dst]


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
