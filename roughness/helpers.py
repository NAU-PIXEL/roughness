"""Roughness helper functions"""
from pathlib import Path
import numpy as np
import numpy.f2py

FLOS_F90 = Path(__file__).parent / "lineofsight.f90"


# Compile Fortran
def compile_lineofsight(los_f90=FLOS_F90, verbose=False):
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
