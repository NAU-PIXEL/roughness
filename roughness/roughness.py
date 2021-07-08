"""Main module."""
import os
from functools import lru_cache
import numpy as np
from numpy import sin, cos, tan, exp, pi

ROUGHNESS_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SHADOW_LOOKUP = os.path.join(ROUGHNESS_DIR, "data", "shadow_fraction_4D.npy")


def get_shadow_prob(
    rms, sun_theta, sun_az, sc_theta, sc_az, shadow_lookup=SHADOW_LOOKUP
):
    """
    Return shadow probability of each facet, given input viewing geometry and
    rms roughness.

    Parameters
    ----------
    rms_slope (num): Root-mean-square slope of roughness [deg]
    sun_theta (num or array): Solar inclination angle(s) [deg]
    sun_az (num or array): Solar azimuth(s) [deg]
    sc_theta (num or array): Spacecraft emission angle(s) [deg]
    sc_az (num or array): Spacecraft azimuth(s) [deg]
    shadow_lookup (optional: str or 4D array): Path to file or shadow lookup

    Returns
    -------
    shadow_prob (2D array): Shadow probability table (dims: az, theta)
    """
    if isinstance(shadow_lookup, str):
        shadow_lookup = load_shadow_lookup(shadow_lookup)
    elif (
        not isinstance(shadow_lookup, np.ndarray)
        or len(shadow_lookup.shape) != 4
    ):
        raise ValueError("Invalid shadow_lookup.")
    shadow_table = get_shadow_table(rms, sun_theta, sun_az, shadow_lookup)
    shadow_prob = correct_shadow_table(rms, sc_theta, sc_az, shadow_table)
    return shadow_prob


def correct_shadow_table(rms_slope, sc_theta, sc_az, shadow_table):
    """
    Return shadow table corrected for roughness and viewing geometry.

    Parameters
    ----------
    rms_slope (num): Root-mean-square slope of roughness [deg]
    sc_theta (num or array): Spacecraft emission angle(s) [deg]
    sc_az (num or array): Spacecraft azimuth(s) [deg]
    shadow_table (array): Shadow fraction table (dims: az, theta)

    Returns
    -------
    shadow_prob (array): Shadow probability table (dims: az, theta)
    """
    # Get P(theta|rms): rough surface probability distribution
    facet_theta, facet_az = get_facet_grids(shadow_table, "radians")
    slope_prob = slope_dist(facet_theta, np.radians(rms_slope))

    # Get P(visible|spacecraft): probability of observing facets from sc vector
    facet_cartesian = sph2cart(facet_theta, facet_az)
    sc_cartesian = sph2cart(np.radians(sc_theta), np.radians(sc_az))
    visible_prob = view_correction(facet_cartesian, sc_cartesian)

    # Correct shadow table by P(theta|rms), P(visible|spacecraft)
    shadow_prob = shadow_table * slope_prob * visible_prob
    return shadow_prob


def view_correction(facet_cartesian, sc_cartesian):
    """
    Return the fraction of surface facets (theta_grid, az_grid) visible from
    the viewing direction (sc_theta, sc_az). Visible faction is 1 - projection
    of the surface facets onto a plane orthogonal to the viewing vector.

    Facets oriented > 90 degrees away from the viewing direction are set to 0.

    Parameters
    ----------
    facet_cartesian (MxNx3 array): Grid of facet vectors in cartesian coords.
    sc_cartesian (len 3 arrary): Spacecraft view vector in cartesian coords.

    Returns
    -------
    visible (MxN array): Probability that facets are visible from spacecraft.
    """
    projection = proj_orthogonal(facet_cartesian, sc_cartesian)

    # Invert projection (facets pointed toward sc have smallest proj)
    visible = 1 - np.linalg.norm(projection, axis=2)

    # Zero out any facets oriented away from the sc direction (angle > 90)
    cos_angle = np.dot(facet_cartesian, sc_cartesian)
    visible[cos_angle < 0] = 0

    return visible


def get_shadow_table(rms_slope, sun_theta, sun_az, shadow_lookup):
    """
    Return shadow fraction of surface facets interpolated from shadow lookup.

    Parameters
    ----------
    rms_slope (num): Root-mean-square slope of roughness [deg]
    sun_theta (num or array): Solar inclination angle(s) [deg]
    sun_az (num or array): Solar azimuth(s) [deg]
    shadow_lookup (4D array): Shadow lookup (dims: rms, cinc, az, theta)

    Returns
    -------
    shadow_table (2D array): Shadow fraction table (dims: az, theta)
    """
    cinc = 1 - cos(np.radians(sun_theta))

    # Interpolate nearest values in lookup to (rms_slope, cinc)
    rms_slopes, cincs, azs, _ = get_lookup_coords(shadow_lookup)
    rms_table = interp_lookup(rms_slope, rms_slopes, shadow_lookup)
    cinc_table = interp_lookup(cinc, cincs, rms_table)

    # Rotate table from az=270 to sun_az
    shadow_table = rotate_az_lookup(cinc_table, sun_az, azs)
    return shadow_table


@lru_cache(maxsize=1)  # Cache 1 shadlow lookup table
def load_shadow_lookup(path=SHADOW_LOOKUP):
    """
    Return shadow lookup at path.

    Parameters
    ----------
    path (optional: str): Path to shadow lookup file containing 4D numpy array

    Returns
    -------
    shadow_lookup (4D array): Shadow lookup (dims: rms, cinc, az, theta)
    """
    return np.load(path, allow_pickle=True)


def get_lookup_coords(shadow_lookup):
    """
    Return coordinate arrays corresponding to each dimension of shadow lookup.

    Assumes shadow_lookup axes are in the following order with ranges:
        rms: [0, 50] degrees
        cinc: [0, 1]
        az: [0, 360] degrees
        theta: [0, 90] degrees

    Parameters
    ----------
    shadow_lookup (4D array): Shadow lookup (dims: rms, cinc, az, theta)

    Return
    ------
    lookup_coords (tuple of array): Coordinate arrays (rms, cinc, az, theta)
    """
    nrms, ncinc, naz, ntheta = shadow_lookup.shape
    rms_coords = np.linspace(0, 50, nrms)
    ncinc_coords = np.linspace(1, 0, ncinc)  # (0, 90) degrees
    az_coords = np.linspace(0, 360, naz)
    theta_coords = np.linspace(0, 90, ntheta)

    return (rms_coords, ncinc_coords, az_coords, theta_coords)


def get_facet_grids(shadow_table, units="degrees"):
    """
    Return 2D grids of surface facet slope and azimuth angles of shadow_table.

    Assumes shadow_table axes are (az, theta) with ranges:
        az: [0, 360] degrees
        theta: [0, 90] degrees

    Parameters
    ----------
    shadow_table (array): Shadow fraction table (dims: az, theta)
    units (str): Return grids in specified units ('degrees' or 'radians')

    Return
    ------
    thetas, azs (tuple of 2D array): Coordinate grids of facet slope and az
    """
    naz, ntheta = shadow_table.shape
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


def rotate_az_lookup(shadow_table, sun_az, azcoords, az0=270):
    """
    Return shadow_table rotated from az0 to nearest sun_az in azcoords.

    Since shadow_lookup derived at one solar az (default az0=270), we rotate
    shadow_table to get shadows with respect to sun_az.

    Parameters
    ----------
    shadow_table (array): Shadow fraction table (dims: az, theta)
    sun_az (num): Solar azimuth (target azimuth for rotation)
    azcoords (array): Coordinate array of azimuths in shadow_table
    az0 (optional: num): Solar azimuth used to derive shadow lookup

    Returns
    -------
    rotated_shadow_table (array): Shadow fraction table rotated on az axis
    """
    ind0 = np.argmin(np.abs(azcoords - az0))
    ind = np.argmin(np.abs(azcoords - sun_az))
    az_shift = ind0 - ind
    return np.concatenate((shadow_table[az_shift:], shadow_table[:az_shift]))


def proj_orthogonal(a, b):
    """
    Return projection of vector(s) a onto plane orthogonal to vector b.

    Vectors a and b must be in Cartesian coordinates. Vector a may be a 1D
    array of shape (3,) or a 3D array of shape (NxMx3). Vector b must be a 1D
    array of shape (3,).

    Parameters
    ----------
    a (len 3 vec or MxNx3 array): Vector(s) to project.
    b (len 3 vec): Orthogonal vector defining the plane to project a onto

    Returns
    -------
    proj_a_onto_norm_plane (array): Projection of a onto plane orthogonal to b
    """
    bnorm = np.sqrt(sum(b ** 2))
    proj_a_onto_b = (np.dot(a, b[:, None]) / bnorm ** 2) * b
    proj_a_onto_norm_plane = a - proj_a_onto_b
    return proj_a_onto_norm_plane


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
