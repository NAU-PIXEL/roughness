"""Main module."""
from functools import lru_cache
import importlib.resources as pkg_resources
import numpy as np
from numpy import sin, cos, tan, exp, pi

with pkg_resources.path("data", "los_lookup.npy") as p:
    LOS_LOOKUP = p.as_posix()


def get_shadow_table(rms, sun_theta, sun_az, los_lookup=LOS_LOOKUP):
    """
    Return probability that each facet is in shadow given (sun_that, sun_az)
    on a surface with rms roughness.

    Parameters
    ----------
    rms_slope (num): Root-mean-square slope of roughness [deg]
    sun_theta (num or array): Solar inclination angle [deg]
    sun_az (num or array): Solar azimuth [deg]
    los_lookup (str or 4D array): Path to file or los lookup array

    Returns
    -------
    shadow_prob (2D array): Shadow probability table (dims: az, theta)
    """
    if isinstance(los_lookup, str):
        los_lookup = load_los_lookup(los_lookup)
    # Invert to get P(shadowed) since los_lookup defaults to P(illuminated)
    shadow_table = 1 - get_los_table(rms, sun_theta, sun_az, los_lookup)
    return shadow_table


def get_view_table(rms, sc_theta, sc_az, los_lookup=LOS_LOOKUP):
    """
    Return probability that each facet is observed from (sc_theta, sc_az)
    on a surface with rms roughness.

    Parameters
    ----------
    rms_slope (num): Root-mean-square slope of roughness [deg]
    sc_theta (num or array): Spacecraft emission angle [deg]
    sc_az (num or array): Spacecraft azimuth [deg]
    los_lookup (str or 4D array): Path to file or los lookup array

    Returns
    -------
    view_prob (2D array): View probability table (dims: az, theta)
    """
    if isinstance(los_lookup, str):
        los_lookup = load_los_lookup(los_lookup)
    view_table = get_los_table(rms, sc_theta, sc_az, los_lookup)

    # Normalize to get overall probability of each facet being observed
    view_table /= np.sum(view_table)
    return view_table


def get_los_table(rms, theta, az, los_lookup):
    """
    Return 2D line of sight probability table of rms rough surface facets in
    the line of sight of (theta, az).

    Output los_table is interpolated from ray-casting derived los_lookup.
    See make_los_table.py to produce a compatible los_lookup file.

    Parameters
    ----------
    rms (num): Root-mean-square surface theta describing roughness [deg]
    theta (num or array): Inclination angle [deg]
    az (num or array): Azimuth [deg]
    los_lookup (4D array): Los lookup (dims: rms, cinc, az, theta)

    Returns
    -------
    los_table (2D array): Line of sight probability table (dims: az, theta)
    """
    cinc = 1 - cos(np.radians(theta))

    # Interpolate nearest values in lookup to (rms, cinc)
    rms_slopes, cincs, azs, _ = get_lookup_coords(los_lookup)
    rms_table = interp_lookup(rms, rms_slopes, los_lookup)
    cinc_table = interp_lookup(cinc, cincs, rms_table)

    # Rotate table from az=270 to az
    los_table = rotate_az_lookup(cinc_table, az, azs)
    return los_table


@lru_cache(maxsize=1)  # Cache 1 los_lookup
def load_los_lookup(path=LOS_LOOKUP):
    """
    Return line of sight lookup at path.

    Parameters
    ----------
    path (optional: str): Path to los lookup file containing 4D numpy array

    Returns
    -------
    los_lookup (4D array): Shadow lookup (dims: rms, cinc, az, theta)
    """
    return np.load(path, allow_pickle=True)


def get_lookup_coords(los_lookup):
    """
    Return coordinate arrays corresponding to each dimension of loslos lookup.

    Assumes los_lookup axes are in the following order with ranges:
        rms: [0, 50] degrees
        cinc: [0, 1]
        az: [0, 360] degrees
        theta: [0, 90] degrees

    Parameters
    ----------
    los_lookup (4D array): Shadow lookup (dims: rms, cinc, az, theta)

    Return
    ------
    lookup_coords (tuple of array): Coordinate arrays (rms, cinc, az, theta)
    """
    nrms, ninc, naz, ntheta = los_lookup.shape
    rms_coords = np.linspace(0, 50, nrms)
    inc_coords = np.linspace(0, 90, ninc)  # (0, 90) degrees
    az_coords = np.linspace(0, 360, naz)
    theta_coords = np.linspace(0, 90, ntheta)

    return (rms_coords, inc_coords, az_coords, theta_coords)


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
    az_arr = np.linspace(0, 360, naz)
    theta_arr = np.linspace(0, 90, ntheta)
    if units == "radians":
        az_arr = np.radians(az_arr)
        theta_arr = np.radians(theta_arr)

    return np.meshgrid(theta_arr, az_arr)


def interp_lookup(xvalue, xcoords, values):
    """
    Return values array interpolated on first axis to xvalue, given xcoords.

    Uses linear interpolation, reduces the dimensionality of values by 1.
    The first axis of values must be the same length as xcoords.

    Parameters
    ----------
    xvalue (num): X-coordinate at which to interpolate.
    xcoords (1D array): Array of X-coordinates of the first dim of values.
    values (ndarray): Array of values to interpolate.

    Returns
    -------
    interp_values ((n-1)darray): Values[x] (interpolated if necessary)
    """
    # If x outside arr range, use closest edge value
    if xvalue <= xcoords[0]:
        return values[0]
    if xvalue >= xcoords[-1]:
        return values[-1]

    # Find nearest indices
    i_right = np.searchsorted(xcoords, xvalue)
    i_left = i_right - 1

    # Find relative distance and interpolate
    dist = (xvalue - xcoords[i_left]) / (xcoords[i_right] - xcoords[i_left])
    interp_values = (1 - dist) * values[i_left] + dist * values[i_right]
    return interp_values


def rotate_az_lookup(los_table, target_az, azcoords, az0=270):
    """
    Return los_table rotated from az0 to nearest target_az in azcoords.

    Given az0 must match solar az used to produce los_lookup (default az0=270).

    Parameters
    ----------
    los_table (array): Line of sight probability table (dims: az, theta)
    target_az (num): Target azimuth [deg]
    azcoords (array): Coordinate array of azimuths in los_table [deg]
    az0 (optional: num): Solar azimuth used to derive loslos lookup [deg]

    Returns
    -------
    rotated_los_table (array): los_table rotated on az axis (dims: az, theta)
    """
    ind0 = np.argmin(np.abs(azcoords - az0))
    ind = np.argmin(np.abs(azcoords - target_az))
    az_shift = ind0 - ind
    return np.concatenate((los_table[az_shift:], los_table[:az_shift]))


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
            radius * sin(theta) * cos(phi),
            radius * sin(theta) * sin(phi),
            radius * cos(theta),
        ]
    ).squeeze()


def slope_dist(theta, theta0, dist="rms"):
    """
    Return slope probability distribution for rough fractal surfaces. Computes
    distributions parameterized by mean slope (theta0) using Shepard 1995 (rms)
    or Hapke 1984 (tbar) roughness.

    Parameters
    ----------
    theta (array): Angles to compute the slope distribution at [rad].
    theta0 (num): Mean slope angle (fixed param in RMS / theta-bar eq.) [rad].
    dist (str): Distribution type (rms or tbar)

    Returns
    -------
    slopes (array): Probability of a slope occurring at each theta.
    """
    theta = np.atleast_1d(theta)
    # If theta0 is 0, slope_dist is undefined. Set to 0 everywhere.
    if theta0 == 0:
        return np.zeros_like(theta)

    # Calculate Gaussian RMS slope distribution (Shepard 1995)
    if dist == "rms":
        slopes = (tan(theta) / tan(theta0) ** 2) * exp(
            -tan(theta) ** 2 / (2 * tan(theta0) ** 2)
        )

    # Calculate theta-bar slope distribution (eqn. 44, Hapke 1984)
    elif dist == "tbar":
        A = 2 / (pi * tan(theta0) ** 2)
        B = pi * tan(theta0) ** 2
        slopes = A * exp(-tan(theta) ** 2 / B) * sin(theta) / cos(theta) ** 2
    else:
        raise ValueError("Invalid distribution specified (accepts rms, tbar).")
    # Normalize to convert to probability
    slopes = slopes / np.sum(slopes)
    return slopes
