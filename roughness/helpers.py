"""Roughness helper functions"""
import os
import contextlib
from pathlib import Path
import numpy as np
import numpy.f2py
import jupytext
from . import config as cfg
from . import __version__

# Line of sight helpers
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
    cinc_coords = np.linspace(1, 0, ninc, endpoint=False)  # [0, 90) degrees
    azim_coords = np.linspace(0, 360, naz, endpoint=False)
    slope_coords = np.linspace(0, 90, ntheta, endpoint=False)
    inc_coords = np.rad2deg(np.arccos(cinc_coords))

    return (rms_coords, inc_coords, azim_coords, slope_coords)


def get_facet_grids(los_table, units="degrees"):
    """
    Return 2D grids of surface facet slope and azimuth angles of los_table.

    Assumes los_table axes are (az, theta) with ranges:
        az: [0, 360] degrees
        theta: [0, 90] degrees

    Parameters
    ----------
    los_table (array): Line of sight table (dims: az, theta)
    units (str): Return grids in specified units ('degrees' or 'radians')

    Return
    ------
    thetas, azs (tuple of 2D array): Coordinate grids of facet slope and az
    """
    naz, ntheta = los_table.shape
    _, _, az_arr, theta_arr = get_lookup_coords(naz=naz, ntheta=ntheta)
    if units == "radians":
        az_arr = np.radians(az_arr)
        theta_arr = np.radians(theta_arr)

    return np.meshgrid(theta_arr, az_arr)


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
    with change_working_directory(fortran_dir):
        try:
            from .fortran import lineofsight
        except (ImportError, ModuleNotFoundError):
            compile_lineofsight()
            from .fortran import lineofsight
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
