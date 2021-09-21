"""M3-specific functions for testing"""
import time
import os.path as op
import numpy as np
import roughness.emission as re
import roughness.helpers as rh

# Constants
PATH = "/work/ctaiudovicic/data/"
M3PATH = f"{PATH}m3/"
M3ANCPATH = f"{PATH}M3_ancillary_files/"
M3ALTPHOTOM = f"{PATH}M3_ancillary_files/alt_photom_tables/"
SHADOW_LOOKUP = f"{PATH}shade_lookup_4D.npy"

# M3 channel wavelengths
GWL = (
    460.99,
    500.92,
    540.84,
    580.76,
    620.69,
    660.61,
    700.54,
    730.48,
    750.44,
    770.40,
    790.37,
    810.33,
    830.29,
    850.25,
    870.21,
    890.17,
    910.14,
    930.10,
    950.06,
    970.02,
    989.98,
    1009.95,
    1029.91,
    1049.87,
    1069.83,
    1089.79,
    1109.76,
    1129.72,
    1149.68,
    1169.64,
    1189.60,
    1209.57,
    1229.53,
    1249.49,
    1269.45,
    1289.41,
    1309.38,
    1329.34,
    1349.30,
    1369.26,
    1389.22,
    1409.19,
    1429.15,
    1449.11,
    1469.07,
    1489.03,
    1508.99,
    1528.96,
    1548.92,
    1578.86,
    1618.79,
    1658.71,
    1698.63,
    1738.56,
    1778.48,
    1818.40,
    1858.33,
    1898.25,
    1938.18,
    1978.10,
    2018.02,
    2057.95,
    2097.87,
    2137.80,
    2177.72,
    2217.64,
    2257.57,
    2297.49,
    2337.42,
    2377.34,
    2417.26,
    2457.19,
    2497.11,
    2537.03,
    2576.96,
    2616.88,
    2656.81,
    2696.73,
    2736.65,
    2776.58,
    2816.50,
    2856.43,
    2896.35,
    2936.27,
    2976.20,
)
TWL = (
    446.02,
    456.00,
    465.98,
    475.96,
    485.95,
    495.93,
    505.91,
    515.89,
    525.87,
    535.85,
    545.83,
    555.81,
    565.79,
    575.77,
    585.76,
    595.74,
    605.72,
    615.70,
    625.68,
    635.66,
    645.64,
    655.62,
    665.60,
    675.58,
    685.56,
    695.55,
    705.53,
    715.51,
    725.49,
    735.47,
    745.45,
    755.43,
    765.41,
    775.39,
    785.37,
    795.36,
    805.34,
    815.32,
    825.30,
    835.28,
    845.26,
    855.24,
    865.22,
    875.20,
    885.18,
    895.17,
    905.15,
    915.13,
    925.11,
    935.09,
    945.07,
    955.05,
    965.03,
    975.01,
    984.99,
    994.97,
    1004.96,
    1014.94,
    1024.92,
    1034.90,
    1044.88,
    1054.86,
    1064.84,
    1074.82,
    1084.80,
    1094.78,
    1104.77,
    1114.75,
    1124.73,
    1134.71,
    1144.69,
    1154.67,
    1164.65,
    1174.63,
    1184.61,
    1194.59,
    1204.58,
    1214.56,
    1224.54,
    1234.52,
    1244.50,
    1254.48,
    1264.46,
    1274.44,
    1284.42,
    1294.40,
    1304.38,
    1314.37,
    1324.35,
    1334.33,
    1344.31,
    1354.29,
    1364.27,
    1374.25,
    1384.23,
    1394.21,
    1404.19,
    1414.18,
    1424.16,
    1434.14,
    1444.12,
    1454.10,
    1464.08,
    1474.06,
    1484.04,
    1494.02,
    1504.00,
    1513.99,
    1523.97,
    1533.95,
    1543.93,
    1553.91,
    1563.89,
    1573.87,
    1583.85,
    1593.83,
    1603.81,
    1613.80,
    1623.78,
    1633.76,
    1643.74,
    1653.72,
    1663.70,
    1673.68,
    1683.66,
    1693.64,
    1703.62,
    1713.60,
    1723.59,
    1733.57,
    1743.55,
    1753.53,
    1763.51,
    1773.49,
    1783.47,
    1793.45,
    1803.43,
    1813.41,
    1823.40,
    1833.38,
    1843.36,
    1853.34,
    1863.32,
    1873.30,
    1883.28,
    1893.26,
    1903.24,
    1913.22,
    1923.21,
    1933.19,
    1943.17,
    1953.15,
    1963.13,
    1973.11,
    1983.09,
    1993.07,
    2003.05,
    2013.03,
    2023.01,
    2033.00,
    2042.98,
    2052.96,
    2062.94,
    2072.92,
    2082.90,
    2092.88,
    2102.86,
    2112.84,
    2122.82,
    2132.81,
    2142.79,
    2152.77,
    2162.75,
    2172.73,
    2182.71,
    2192.69,
    2202.67,
    2212.65,
    2222.63,
    2232.62,
    2242.60,
    2252.58,
    2262.56,
    2272.54,
    2282.52,
    2292.50,
    2302.48,
    2312.46,
    2322.44,
    2332.42,
    2342.41,
    2352.39,
    2362.37,
    2372.35,
    2382.33,
    2392.31,
    2402.29,
    2412.27,
    2422.25,
    2432.23,
    2442.22,
    2452.20,
    2462.18,
    2472.16,
    2482.14,
    2492.12,
    2502.10,
    2512.08,
    2522.06,
    2532.04,
    2542.03,
    2552.01,
    2561.99,
    2571.97,
    2581.95,
    2591.93,
    2601.91,
    2611.89,
    2621.87,
    2631.85,
    2641.83,
    2651.82,
    2661.80,
    2671.78,
    2681.76,
    2691.74,
    2701.72,
    2711.70,
    2721.68,
    2731.66,
    2741.64,
    2751.63,
    2761.61,
    2771.59,
    2781.57,
    2791.55,
    2801.53,
    2811.51,
    2821.49,
    2831.47,
    2841.45,
    2851.44,
    2861.42,
    2871.40,
    2881.38,
    2891.36,
    2901.34,
    2911.32,
    2921.30,
    2931.28,
    2941.26,
    2951.25,
    2961.23,
    2971.21,
    2981.19,
    2991.17,
)


def m3_refl(
    m3_basename, rad_L1, obs, dem=None, shadow_lookup_path=SHADOW_LOOKUP
):
    """
    Return M3 reflectance using the roughness thermal correction (Bandfield et
    al., 2018). Uses ray-casting generated shadow_lookup table. Applies
    standard M3 photometric correction.

    TODO: finish testing this
    """
    # Get wavelength, viewing geometry, photometry from M3 files
    rms = 18  # Assumed rms roughness value of Moon
    albedo = 0.12  # Assumed hemispherical broadband albedo
    emiss = 0.985  # Assumed emissivity
    wl = get_m3_wls(rad_L1) / 1000  # nm -> um
    solar_dist = get_solar_dist(m3_basename)
    sun_thetas, sun_azs, sc_thetas, sc_azs, phases, _ = get_m3_geom(obs, dem)
    photom = get_m3photom(sun_thetas, sc_thetas, phases)

    # Get rougness emission for each pixel from rad eq on subpixel facets
    emission = np.zeros(rad_L1.shape)
    for i in range(emission.shape[0]):
        for j in range(emission.shape[1]):
            # TODO: split into function
            geom = [c[i, j] for c in (sun_thetas, sun_azs, sc_thetas, sc_azs)]
            emission[i, j] = re.rough_emission_eq(
                geom, wl, rms, albedo, emiss, solar_dist, shadow_lookup_path
            )

    # Get solar irradiance
    solar_spec = get_solar_spectrum(is_targeted(rad_L1))
    L_sun = re.get_solar_irradiance(solar_spec, solar_dist)

    # Remove emission from Level 1 radiance and convert to I/F
    i_f = re.get_rad_factor(rad_L1, emission, L_sun)
    refl = i_f * photom

    # TODO: New refl way higher, how come? emission func? I/F step?
    return refl


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

    i, e, g, sun_sc_az = rh.get_ieg(
        dem_theta, dem_az, sun_theta, sun_az, sc_theta, sc_az
    )

    return i, sun_az, e, sc_az, g, sun_sc_az


def get_m3_wls(img):
    """
    Get the wavelength array for an M3 image
    """
    if is_targeted(img):
        wls = TWL
    else:
        wls = GWL
    return np.array(wls)


def get_solar_dist(basename):
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
    return sdist


def get_solar_spectrum(targeted, m3ancpath=M3ANCPATH):
    """Return m3 solar spectrum"""
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
    # https://astrogeology.usgs.gov/maps/moon-mineralogy-mapper-geometric-data-restoration
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
    """
    params = {}
    with open(fname) as f:
        key, value = "", []
        line = f.readline()
        while line:
            if isinstance(value, list):
                if "}" in line:
                    value.append(line.split("}")[0].strip())
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
