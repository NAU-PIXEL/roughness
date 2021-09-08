"""Main module."""
from functools import lru_cache
import numpy as np
from numpy import sin, cos, tan, exp, pi
import xarray as xr
from . import helpers as rh
from .config import FLOOKUP, AZ0


def get_table_xarr(params, dims=None, da="losprob", los_lookup=FLOOKUP):
    """
    Return interpolated xarray.DataArray of los_lookup table at given params.

    Parameters
    ----------
    params (list): List of values to query in los_lookup.
    dims (list of str): Dimensions name of each param (default: index order).
    da (str): DataArray in xarray.DataSet to query (default: losprob)
    los_lookup (path or xarray): Path to file or los lookup array.

    Returns
    -------
    table (xarray.DataArray): Table of los_lookup values
    """
    # Load los_lookup
    if not isinstance(los_lookup, (xr.Dataset, xr.DataArray)):
        los_lookup = load_los_lookup(los_lookup)

    # Get desired DataArray from los_lookup DataSet
    if isinstance(los_lookup, xr.Dataset):
        los_lookup = los_lookup[da]

    # Get table dimensions
    if dims is None:
        dims = list(los_lookup.dims)

    # Get table coordinates
    coords = {dims[i]: params[i] for i in range(len(params))}

    # Rotate los_lookup to target azimuth
    if "az" in coords:
        los_lookup = rotate_az_lookup(los_lookup, coords["az"])

    # Get table values
    table = los_lookup.interp(coords, method="linear")
    return table


def get_shadow_table(rms, sun_theta, sun_az, los_lookup=FLOOKUP):
    """
    Return probability that each facet is in shadow given (sun_that, sun_az)
    on a surface with rms roughness.

    Parameters
    ----------
    rms_slope (num): Root-mean-square slope of roughness [deg]
    sun_theta (num or array): Solar inclination angle [deg]
    sun_az (num or array): Solar azimuth [deg]
    los_lookup (path or 4D array): Path to file or los lookup array

    Returns
    -------
    shadow_prob (2D array): Shadow probability table (dims: az, theta)
    """
    # Invert to get P(shadowed) since los_lookup defaults to P(illuminated)
    shadow_table = 1 - get_los_table(rms, sun_theta, sun_az, los_lookup)
    return shadow_table


def get_view_table(rms, sc_theta, sc_az, los_lookup=FLOOKUP):
    """
    Return probability that each facet is observed from (sc_theta, sc_az)
    on a surface with rms roughness.

    Parameters
    ----------
    rms_slope (num): Root-mean-square slope of roughness [deg]
    sc_theta (num or array): Spacecraft emission angle [deg]
    sc_az (num or array): Spacecraft azimuth [deg]
    los_lookup (path or 4D array): Path to file or los lookup array

    Returns
    -------
    view_prob (2D array): View probability table (dims: az, theta)
    """
    view_table = get_los_table(rms, sc_theta, sc_az, los_lookup)

    # Normalize to get overall probability of each facet being observed
    view_table /= np.nansum(view_table)
    return view_table


def get_los_table(rms, inc, az=270, los_lookup=FLOOKUP):
    """
    Return 2D line of sight probability table of rms rough surface facets in
    the line of sight of (theta, az).

    Output los_table is interpolated from ray-casting derived los_lookup.
    See make_los_table.py to produce a compatible los_lookup file.

    Parameters
    ----------
    rms (num): Root-mean-square surface theta describing roughness [deg]
    inc (num or array): Inclination angle [deg]
    az (num or array): Azimuth [deg]
    los_lookup (path or 4D array): Los lookup (dims: rms, cinc, az, theta)

    Returns
    -------
    los_table (2D array): Line of sight probability table (dims: az, theta)
    """
    # If los_lookup is a path, load it
    try:
        _ = los_lookup.shape
    except AttributeError:
        los_lookup = load_los_lookup(los_lookup)
    # Interpolate nearest values in lookup to (rms, cinc)
    rms_slopes, incs, azs, _ = rh.get_lookup_coords(*los_lookup.shape)
    rms_table = interp_lookup(rms, rms_slopes, los_lookup)
    inc_table = interp_lookup(inc, incs, rms_table)

    # Rotate table from az=270 to az
    los_table = rotate_az_lookup(inc_table, az, azs)
    return los_table


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


def rotate_az_lookup(los_table, target_az, az0=None):
    """
    Return los_table rotated from az0 to nearest target_az in azcoords.

    Given az0 must match solar az used to produce los_lookup (default az0=270).

    Parameters
    ----------
    los_table (array): Line of sight probability table (dims: az, theta)
    target_az (num): Target azimuth [deg]
    azcoords (array): Coordinate array of azimuths in los_table [deg]
    az0 (optional: num): Solar azimuth used to derive los_lookup [deg]

    Returns
    -------
    rotated_los_table (array): los_table rotated on az axis (dims: az, theta)
    """
    if az0 is None:
        az0 = float(los_table.attrs.get("az0", AZ0))
    azcoords = los_table.az.values
    ind0 = np.argmin(np.abs(azcoords - az0))
    ind = np.argmin(np.abs(azcoords - target_az))
    az_shift = ind - ind0
    return los_table.roll(az=az_shift, roll_coords=False)


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


# File I/O
@lru_cache(maxsize=1)
def load_los_lookup(path=FLOOKUP):
    """
    Return line of sight lookup at path.

    Parameters
    ----------
    path (optional: str): Path to los lookup file containing 4D numpy array

    Returns
    -------
    los_lookup (4D array): Shadow lookup (dims: rms, cinc, az, theta)
    """
    try:
        ext = path.suffix
    except AttributeError:
        ext = "." + path.split(".")[-1]
    if ext == ".nc":
        los_lookup = xr.open_dataset(path)
    elif ext == ".npy":
        los_lookup = np.load(path)
    else:
        raise ValueError("Invalid los_lookup file format.")
    return los_lookup
