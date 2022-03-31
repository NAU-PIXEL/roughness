"""Main module."""
from functools import lru_cache
import numpy as np
from numpy import sin, cos, tan, exp, pi
import xarray as xr
from . import helpers as rh
from .config import FLOOKUP, AZ0


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
    return 1 - get_los_table(rms, sun_theta, sun_az, los_lookup, "prob")


def get_view_table(rms, sc_theta, sc_az, weighted=True, los_lookup=FLOOKUP):
    """
    Return probability that each facet is observed from (sc_theta, sc_az)
    on a surface with rms roughness. If weighted is True, the probability is
    weighted by the cosine of the angle between the facet and view direction.

    Parameters
    ----------
    rms_slope (num): Root-mean-square slope of roughness [deg]
    sc_theta (num or array): Spacecraft emission angle [deg]
    sc_az (num or array): Spacecraft azimuth [deg]
    weighted (bool): Whether to weight by cos(angle) between facet and view
    los_lookup (path or 4D array): Path to file or los lookup array

    Returns
    -------
    view_prob (2D array): View probability table (dims: az, theta)
    """
    view = get_los_table(rms, sc_theta, sc_az, los_lookup, "prob")
    if weighted:
        view *= get_view_weights(view.theta, view.az, sc_theta, sc_az)
    return view


def get_view_weights(facet_theta, facet_az, sc_theta, sc_az):
    """
    Return weighting factor for surface facets (facet_theta, facet_az) as
    viewed from (sc_theta, sc_az) as cos(angle) between facet and view vectors.

    Computes cos(angle) between the vectors as cartesian dot product.

    Parameters
    ----------
    facet_theta (2D array): Surface facet thetas [deg]
    facet_az (2D array): Surface facet azimuths [deg]
    sc_theta (num or array): Spacecraft emission angle [deg]
    sc_az (num or array): Spacecraft azimuth [deg]
    """
    ft, fa, st, sa = map(np.deg2rad, [facet_theta, facet_az, sc_theta, sc_az])
    facet_cart = rh.sph2cart(*np.meshgrid(ft, fa))
    sc_cart = rh.sph2cart(st, sa)
    return rh.element_dot(facet_cart, sc_cart)


def get_facet_weights(rms, inc, los_lookup=FLOOKUP):
    """
    Return the weighted probability of each facet occurring given an rms rough
    surface facets in the line of sight of (theta, az).

    Parameters
    ----------
    rms (num): Root-mean-square surface theta describing roughness [deg]
    inc (num or array): Inclination angle [deg]
    los_lookup (path or xarray): Los lookup xarray or path to file.

    Returns
    -------
    weights (2D array): Weighted probability table that sums to 1.
    """
    tot = get_table_xarr([rms, inc], ["rms", "inc"], None, "total", los_lookup)
    if rms <= tot.theta.values[1]:
        tot = xr.zeros_like(tot)
        tot.loc[{"theta": 0}] = 1
    return tot / tot.sum(skipna=True)


def get_los_table(rms, inc, az=None, los_lookup=FLOOKUP, da="prob"):
    """
    Return 2D line of sight probability table of rms rough surface facets in
    the line of sight of (inc, az).

    Output los_table is interpolated from ray-casting derived los_lookup.
    See make_los_table.py to produce a compatible los_lookup file.

    Parameters
    ----------
    rms (num): Root-mean-square surface theta describing roughness [deg]
    inc (num or array): Inclination angle [deg]
    az (num or array): Azimuth [deg]
    los_lookup (path or xarray): Los lookup xarray or path to file.
    da (str): DataArray to query ['total', 'los', 'prob'] (default: prob)

    Returns
    -------
    los_table (2D array): Line of sight probability table (dims: az, theta)
    """
    # Format rms and inc to pass to get_table_xarr
    los_table = get_table_xarr((rms, inc), ("rms", "inc"), az, da, los_lookup)
    return los_table


def get_table_xarr(params, dims=None, az=None, da=None, los_lookup=FLOOKUP):
    """
    Return interpolated xarray.DataArray of los_lookup table at given params.

    Parameters
    ----------
    params (list): List of values to query in los_lookup.
    dims (list of str): Dimensions name of each param (default: index order).
    az (num): Azimuth of observation to query (default: az0).
    da (str): DataArray in xarray.DataSet to query (default: prob)
    los_lookup (path or xarray): Path to file or los lookup array.

    Returns
    -------
    table (xarray.DataArray): Table of los_lookup values
    """
    if los_lookup is None:
        los_lookup = FLOOKUP
    if isinstance(los_lookup, np.ndarray):
        los_lookup = rh.np2xr(los_lookup)

    # Load los_lookup if necessary and get desired DataArray
    if not isinstance(los_lookup, (xr.Dataset, xr.DataArray)):
        los_lookup = load_los_lookup(los_lookup)
    if da is not None and isinstance(los_lookup, xr.Dataset):
        los_lookup = los_lookup[da]

    # Get table dimensions if not named
    if dims is None:
        dims = list(los_lookup.dims)

    # Get table coordinates
    coords = {dims[i]: params[i] for i in range(len(params))}

    # Rotate los_lookup to target azimuth (else leave in terms of az0)
    if az is not None:
        los_lookup = rotate_az_lookup(los_lookup, az)

    # Get table values
    table = los_lookup.interp(coords, method="linear")
    return table


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


def slope_dist(theta, theta0, dist="rms", units="rad"):
    """
    Return slope probability distribution for rough fractal surfaces. Computes
    distributions parameterized by mean slope (theta0) using Shepard 1995 (rms)
    or Hapke 1984 (tbar) roughness.

    Parameters
    ----------
    theta (array): Angles to compute the slope distribution at [rad].
    theta0 (num): Mean slope angle (fixed param in RMS / theta-bar eq.) [rad].
    dist (str): Distribution type (rms or tbar)
    units (str): Units of theta and theta0 (deg or rad)

    Returns
    -------
    slopes (array): Probability of a slope occurring at each theta.
    """
    theta = np.atleast_1d(theta)
    if units == "deg":
        theta = np.radians(theta)
        theta0 = np.radians(theta0)

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
    path (optional: str): Path to los lookup file containing xarray.Dataset

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
