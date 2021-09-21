"""Roughness helper functions"""
import os
import contextlib
from pathlib import Path
import numpy as np
import numpy.f2py
import xarray as xr
import jupytext
from . import config as cfg
from . import __version__

# Line of sight helpers
def lookup2xarray(lookups):
    """
    Convert list of default lookups to xarray.DataSet.

    Parameters
    ----------
    lookups (array): Lookup tables (e.g. from make_los_table).

    Returns
    -------
    xarray.DataArray: xarray.DataArray with labeled dims and coords.
    """
    names = cfg.LUT_NAMES
    longnames = cfg.LUT_LONGNAMES
    for i, lut in enumerate(lookups):
        # Make DataArray
        da = np2xr(lut, name=names[i])
        da.attrs["long_name"] = longnames[i]

        # Add DataArray to DataSet
        if i == 0:
            ds = da.to_dataset()
        else:
            ds[names[i]] = da
    return ds


def np2xr(arr, dims=None, coords=None, name=None, cnames=None, cunits=None):
    """
    Convert numpy array to xarray.DataArray.

    Parameters
    ----------
    array (np.array): Numpy array to convert to xarray.DataArray.
    dims (list of str): Dimensions name of each param (default: index order).
    coords (list of arr): Coordinate arrays of each dim.
    name (str): Name of xarray.DataArray (default: None)
    cnames (list of str): Coordinate names of each param.
    cunits (list of str): Coordinate units of each param (default: deg).

    Returns
    -------
    xarray.DataArray
    """
    ndims = len(arr.shape)
    if dims is None:
        dims = cfg.LUT_DIMS[:ndims]
    if coords is None:
        coords = get_lookup_coords(*arr.shape)[:ndims]
    if cnames is None:
        cnames = cfg.LUT_DIMS_LONGNAMES[:ndims]
    if cunits is None:
        cunits = ["deg"] * ndims

    # Make xarray-readable (dim, coord) pairs
    coords_xr = list(zip(dims, coords))

    # Make DataArray
    da = xr.DataArray(arr, coords=coords_xr, name=name)
    for i, coord in enumerate(da.coords):
        da.coords[coord].attrs["long_name"] = cnames[i]
        da.coords[coord].attrs["units"] = cunits[i]
    return da


def wl2xr(arr, units="microns"):
    """Return wavelength numpy array as xarray."""
    da = xr.DataArray(arr, coords=[("wavelength", arr)])
    da.coords["wavelength"].attrs["long_name"] = "Wavelength"
    da.coords["wavelength"].attrs["units"] = units
    return da


def wn2xr(arr, units="cm^-1"):
    """Return wavenumber numpy array as xarray."""
    da = xr.DataArray(arr, coords=[("wavenumber", arr)])
    da.coords["wavelength"].attrs["long_name"] = "Wavenumber"
    da.coords["wavelength"].attrs["units"] = units
    return da


def get_lookup_coords(nrms=10, ninc=10, naz=36, ntheta=45):
    """
    Return coordinate arrays corresponding to number of elements in each axis.

    Return lookup axes in the following order with ranges:
        rms: [0, 50] degrees
        inc: [0, 90] degrees
        az: [0, 360] degrees
        theta: [0, 90] degrees

    Parameters
    ----------
    nrms (int): Number of RMS slopes in [0, 50) degrees.
    ninc (int): Number of incidence angles in [0, 90) degrees.
    naz (int): Number of facet azimuth bins in [0, 360) degrees.
    ntheta (int): Number of facet slope bins in [0, 90) degrees.

    Return
    ------
    lookup_coords (tuple of array): Coordinate arrays (rms, cinc, az, theta)
    """
    rms_coords = np.linspace(0, 50, nrms, endpoint=False)
    cinc_coords = np.linspace(1, 0, ninc, endpoint=True)  # [0, 90) degrees
    azim_coords = np.linspace(0, 360, naz, endpoint=False)
    slope_coords = np.linspace(0, 90, ntheta, endpoint=False)
    inc_coords = np.rad2deg(np.arccos(cinc_coords))

    return (rms_coords, inc_coords, azim_coords, slope_coords)


def facet_grids(los_table, units="degrees"):
    """
    Return 2D grids of surface facet slope and azimuth angles of los_table.

    Assumes los_table axes are (az, theta) with ranges:
        az: [0, 360] degrees
        theta: [0, 90] degrees

    Parameters
    ----------
    los_table (arr): Line of sight table (dims: az, theta)
    units (str): Return grids in specified units ('degrees' or 'radians')

    Return
    ------
    thetas, azs (tuple of 2D array): Coordinate grids of facet slope and az
    """
    if isinstance(los_table, xr.DataArray):
        az_arr = los_table.az.values
        theta_arr = los_table.theta.values
    else:
        naz, ntheta = los_table.shape
        _, _, az_arr, theta_arr = get_lookup_coords(naz=naz, ntheta=ntheta)
    if units == "radians":
        az_arr = np.radians(az_arr)
        theta_arr = np.radians(theta_arr)

    # Make coordinate grids
    theta_grid, az_grid = np.meshgrid(theta_arr, az_arr)
    if isinstance(los_table, xr.DataArray):
        theta_grid = xr.ones_like(los_table) * theta_grid
        theta_grid.name = f"theta [{units}]"
        az_grid = xr.ones_like(los_table) * az_grid
        az_grid.name = f"azimuth [{units}]"
    return theta_grid, az_grid


def get_facet_bins(naz=36, ntheta=45):
    """
    Return az, theta bin arrays of los_table.

    Assumes los_table axes are (az, theta) with ranges:
        az: [0, 360] degrees
        theta: [0, 90] degrees

    Parameters
    ----------
    los_table (array): Line of sight table (dims: az, theta)
    units (str): Return grids in specified units ('degrees' or 'radians')

    Return
    ------
    thetas, azs (tuple of 2D array): Bin edges of facet slope and az
    """
    azim_coords = np.linspace(0, 360, naz + 1)
    slope_coords = np.linspace(0, 90, ntheta + 1)
    return (azim_coords, slope_coords)


# File I/O helpers
def rm_regex(dirpath, regex):
    """Remove all files matching regex in dirpath."""
    dirpath = Path(dirpath)
    for f in dirpath.rglob(regex):
        f.unlink()


def fname_with_demsize(filename, demsize):
    """
    Return filename with demsize appended to the end.

    Parameters
    ----------
    fname (str or Path): Filename to append to.
    demsize (int): Length of dem in pixels.

    Returns
    -------
    fname_with_demsize (str): New filename with new demsize appended.
    """
    filename = Path(filename)
    return filename.with_name(f"{filename.stem}_s{demsize}{filename.suffix}")


def versions_match(version_a, version_b, precision=2):
    """
    Check if semantic versions match to precision (default 2).

    Examples
    --------
    >>> versions_match('1.0.0', '1.2.3', precision=1)
    True
    >>> versions_match('1.2.0', '1.2.3', precision=2)
    True
    >>> versions_match('1.2.3', '1.2.3', precision=3)
    True
    """
    va_split = version_a.split(".")
    vb_split = version_b.split(".")
    for i in range(precision):
        if va_split[i] != vb_split[i]:
            return False
    return True


def check_data_updated():
    """Check if data version is up to date."""
    data_version = get_data_version()
    if data_version is None or not versions_match(data_version, __version__):
        print("WARNING: The roughness/data folder is not up to date!")
        print("Update to the newest lookup tables with -d flag.")
        return False
    return True


def get_data_version(data_version_file=cfg.FDATA_VERSION):
    """Get data version in data/data_version.txt."""
    data_version = None
    try:
        with open(data_version_file, "r") as f:
            data_version = f.readline().strip()
    except FileNotFoundError:
        pass
    return data_version


def set_data_version(data_version_file=cfg.FDATA_VERSION):
    """Set data version in data/data_version.txt."""
    with open(data_version_file, "w") as f:
        f.write(__version__)


@contextlib.contextmanager
def change_working_directory(path):
    """Change working directory and revert to previous on exit."""
    prev_cwd = Path.cwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(prev_cwd)


# Fortran helpers
def compile_lineofsight(los_f90=cfg.FLOS_F90, verbose=False):
    """Compile lineofsight module with numpy f2py. Error if no compiler."""
    with open(los_f90) as f:
        src = f.read()
    failed = numpy.f2py.compile(
        src, "lineofsight", verbose=verbose, source_fn=los_f90
    )
    if failed:
        msg = "Cannot compile Fortran raytracing code. Please ensure you have \
               a F90 compatible Fortran compiler (e.g. gfortran) installed."
        raise ImportError(msg)


def import_lineofsight(fortran_dir=cfg.FORTRAN_DIR):
    """Import lineofsight module. Compile first if not found."""
    # pragma pylint: disable=import-outside-toplevel
    lineofsight = None
    with change_working_directory(fortran_dir):
        try:
            from .fortran import lineofsight
        except (ImportError, ModuleNotFoundError):
            try:
                compile_lineofsight()
                from .fortran import lineofsight
            except (ImportError, ModuleNotFoundError):
                msg = "Cannot compile lineofsight FORTRAN module. Please \
                       ensure you have a F90 compatible Fortran compiler (e.g.\
                       gfortran) installed."
                print(msg)
    return lineofsight


def build_jupyter_notebooks(nbpath=cfg.EXAMPLES_DIR):
    """Build Jupyter notebooks using Jupytext."""
    print("Setting up Jupyter notebooks")
    for nb_py in nbpath.rglob("*.py"):
        fout = nb_py.with_suffix(".ipynb")
        if fout.exists():
            print(f"Skipping existing notebook {fout}")
        else:
            jupytext.write(jupytext.read(nb_py), fout)
            print(f"Wrote {fout}")


# Geometry helpers
def get_surf_geometry(ground_zen, ground_az, sun_zen, sun_az, sc_zen, sc_az):
    """
    Return local i, e, g, azimuth give input viewing geometry assuming all
    input zeniths and azimuths are in radians.
    """
    ground = sph2cart(ground_zen, ground_az)
    sun = sph2cart(sun_zen, sun_az)
    sc = sph2cart(sc_zen, sc_az)
    inc = get_local_inc(ground, sun)
    em = get_local_em(ground, sc)
    phase = get_local_phase(sun, sc)
    az = get_local_az(ground, sun, sc)
    return (inc, em, phase, az)


def get_ieg(ground_zen, ground_az, sun_zen, sun_az, sc_zen, sc_az):
    """
    Return local solar incidence, spacecraft emission and phase angle (i, e, g)
    and azimuth given input viewing geometry.

    Input zeniths and azimuths in radians.
    """
    ground = sph2cart(ground_zen, ground_az)
    sun = sph2cart(sun_zen, sun_az)
    sc = sph2cart(sc_zen, sc_az)
    inc = get_local_inc(ground, sun)
    em = get_local_em(ground, sc)
    phase = get_local_phase(sun, sc)
    az = get_local_az(ground, sun, sc)
    return (inc, em, phase, az)


def safe_arccos(arr):
    """Return arccos, restricting output to [-1, 1] without raising Exception."""
    if (arr < -1).any() or (arr > 1).any():
        print("Invalid cosine input; restrict to [-1,1]")
        arr[arr < -1] = -1
        arr[arr > 1] = 1
    return np.arccos(arr)


def get_azim(ground_sc_ground, ground_sun_ground):
    """
    Return the azimuth arccos(dot product of the spacecraft and sun vectors)
    """
    dot_azim = np.degrees(
        np.arccos(np.sum(ground_sc_ground * ground_sun_ground, axis=2))
    )
    oob = dot_azim[(dot_azim < 0) * (dot_azim > 180)]
    if oob.any():
        w = f"Azimuth {oob} outside (0, 180); Setting to 0"
        print(w)
        dot_azim[(dot_azim < 0) * (dot_azim > 180)] = 0
    return dot_azim


def get_local_inc(ground, sun):
    """
    Return solar incidence angle of each pixel given local topography. Assumes
    inputs are same size and shape and are in 3D Cartesian coords (ixjxk).
    """
    cos_inc = element_dot(ground, sun)
    inc = np.degrees(safe_arccos(cos_inc))
    inc[inc > 89.999] = 89.999
    return inc


def get_local_em(ground, sc):
    """
    Return emergence angle of each pixel given local topography. Assumes
    inputs are same size and shape and are in 3D Cartesian coords (ixjxk).
    """
    cos_em = element_dot(ground, sc)
    em = np.degrees(safe_arccos(cos_em))
    return em


def get_local_phase(sun, sc):
    """
    Return phase angle of each pixel given local topography. Assumes
    inputs are same size and shape and are in 3D Cartesian coords (ixjxk).
    """
    cos_phase = element_dot(sc, sun)
    phase = np.degrees(safe_arccos(cos_phase))
    return phase


def get_local_az(ground, sun, sc):
    """
    Return azimuth angle of the spacecraft with respect to the sun and local
    slope. Assumes inputs are same size and shape and are in 3D
    Cartesian coords (ixjxk).
    """
    sc_rel_ground = element_norm(element_triple_cross(ground, sc, ground))
    sun_rel_ground = element_norm(element_triple_cross(ground, sun, ground))
    cos_az = element_dot(sc_rel_ground, sun_rel_ground)
    az = np.degrees(safe_arccos(cos_az))
    az[(az < 0) * (az > 180)] = 0
    return az


def inc_to_tloc(inc, az):
    """
    Convert solar incidence and az to decimal local time (in 6-18h).

    Parameters
    ----------
    inc: (float)
        Solar incidence in degrees (0, 90)
    az: (str)
        Solar azimuth in degrees (0, 360)
    """
    inc = inc.copy()
    if isinstance(az, np.ndarray):
        inc[az < 180] *= -1
    elif az < 180:
        inc *= -1
    coinc = 90 + inc  # (-90, 90) -> (0, 180)
    tloc = 6 * coinc / 90 + 6  # (0, 180) -> (6, 18)
    return tloc


def tloc_to_inc(tloc):
    """
    Convert decimal local time to solar incidence (equator only).
    """
    coinc = (tloc - 6) * 90 / 6  # (6, 18) -> (0, 180)
    inc = coinc - 90  # (0, 180) -> (-90, 90)
    return inc


# Linear algebra
# def element_az_elev(v1, v2):
#     """
#     Return azimuth and elevation of v2 relative to v1.
#
#    untested
#     """
#     v = v2 - v1
#     az = np.degrees(np.arctan2(v[:, :, 0], v[:, :, 1]))
#     elev = np.degrees(np.arctan2(v[:, :, 2], np.sqrt(v[:, :, 0]** 2 + v[:, :, 1]**2)))
#     return az, elev


def element_cross(A, B):
    """
    Return element-wise cross product of two 3D arrays in cartesian coords.
    """
    out = np.zeros_like(A)
    out[:, :, 0] = A[:, :, 1] * B[:, :, 2] - A[:, :, 2] * B[:, :, 1]
    out[:, :, 1] = A[:, :, 2] * B[:, :, 0] - A[:, :, 0] * B[:, :, 2]
    out[:, :, 2] = A[:, :, 0] * B[:, :, 1] - A[:, :, 1] * B[:, :, 0]
    return out


def element_dot(A, B):
    """Return element-wise dot product of two 3D arr in Cartesian coords."""
    return np.sum(A * B, axis=2)


def element_norm(A):
    """Return input array of vectors normalized to length 1."""
    mag = np.sqrt(np.sum(A ** 2, axis=2))
    return A / mag[:, :, np.newaxis]


def element_triple_cross(A, B, C):
    """Return element-wise triple cross product of three 3D arr in Cartesian"""
    return (
        B * (element_dot(A, C))[:, :, np.newaxis]
        - C * (element_dot(A, B))[:, :, np.newaxis]
    )


# Coordinate transformations
def cart2pol(x, y):
    """
    Convert ordered coordinate pairs from Cartesian (X,Y) to polar (r,theta).

    Parameters
    ----------
    X: X component of ordered Cartesian coordinate pair.
    Y: Y component of ordered Cartesian coordinate pair.

    Returns
    -------
    r: Distance from origin.
    theta: Angle, in radians.
    """
    r = np.sqrt(x ** 2 + y ** 2)
    theta = np.arctan2(y, x)
    return (r, theta)


def pol2cart(rho, phi):
    """
    Convert ordered coordinate pairs from polar (r,theta) to Cartesian (X,Y).

    Parameters
    ----------
    r: Distance from origin.
    theta: Angle, in radians.

    Returns
    -------
    X: X component of ordered Cartesian coordinate pair.
    Y: Y component of ordered Cartesian coordinate pair.
    """
    x = rho * np.cos(phi)
    y = rho * np.sin(phi)
    return (x, y)


def sph2cart(theta, phi, radius=1):
    """
    Convert from spherical (theta, phi, r) to cartesian (x, y, z) coordinates.

    Theta and phi must be specified in radians and returned units correspond to
    units of r, default is unitless (r=1 is unit vector). Returns vectors along
    z-axis (e.g., if theta and phi are int/float return 1x1x3 vector; if theta
    and phi are MxN arrays return 3D array of vectors MxNx3).

    Parameters
    ----------
    theta (num or array): Polar angle [rad].
    phi (num or array): Azimuthal angle [rad].
    radius (num or array): Radius (default=1).

    Returns
    -------
    cartesian (array): Cartesian vector(s), same shape as theta and phi.
    """
    return np.dstack(
        [
            radius * np.sin(theta) * np.cos(phi),
            radius * np.sin(theta) * np.sin(phi),
            radius * np.cos(theta),
        ]
    ).squeeze()


def xy2lonlat_coords(x, y, extent):
    """Convert x,y coordinates to lat,lon coordinates."""
    lon = np.linspace(extent[0], extent[1], len(x))
    lat = np.linspace(extent[3], extent[2], len(y))
    return lon, lat
