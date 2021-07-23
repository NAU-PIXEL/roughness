"""Module for calculating emitted radiance."""
from functools import lru_cache
import importlib.resources as pkg_resources
import numpy as np
import sys
from . import roughness as r

with pkg_resources.path("data", "Bennu_EQ1_KRC_temp_table.npy") as p:
    TEMP_TABLE = p.as_posix()

def get_temperature_table(lat, lst, temp_lookup=TEMP_TABLE):
    """
    Return slope/azimuth temperature distribution for a given
    latitude and local solar time.

    Parameters
    ----------
    lat (num): Latitude [deg]
    lst (num or array): Local Solar Time [deg]
    temp_lookup (str or 4D array): Path to file or temp lookup array

    Returns
    -------
    temp_table (2D array): Temperature table (dims: az, theta)
    """
    if isinstance(temp_lookup, str):
        temp_lookup = load_temperature_table(temp_lookup)

    # Interpolate nearest values in lookup to (rms, cinc)
    lats, lsts, azs, _ = get_temperature_coords(*temp_lookup.shape)
    lat_table = r.interp_lookup(lat, lats, temp_lookup)
    lst_table = r.interp_lookup(lst, lsts, lat_table)

    return lst_table

def get_temperature_coords(nlat=19, nlst=97, naz=37, ntheta=46):
    """
    Return coordinate arrays corresponding to number of elements in each axis.
    Return lookup axes in the following order with ranges:
        lat: [-90, 90] degrees
        lst: [0, 24] hours
        az: [0, 360] degrees
        theta: [0, 90] degrees
    Parameters
    ----------
    nlat (int): Latitude [-90, 90] degrees.
    nlst (int): Local solar time in [0, 24] hours.
    naz (int): Number of facet azimuth bins in [0, 360] degrees.
    ntheta (int): Number of facet slope bins in [0, 90] degrees.
    Return
    ------
    lookup_coords (tuple of array): Coordinate arrays (lat, lst, az, theta)
    """
    lat_coords = np.linspace(-90, 90, nlat, endpoint=True)
    lst_coords = np.linspace(0, 24, nlst, endpoint=True)  # [0, 90) degrees
    azim_coords = np.linspace(0, 360, naz, endpoint=True)
    slope_coords = np.linspace(0, 90, ntheta, endpoint=True)

    return (lat_coords, lst_coords, azim_coords, slope_coords)

# File I/O
@lru_cache(maxsize=1)
def load_temperature_table(path=TEMP_TABLE):
    """
    Return temperature lookup table at path.
    Parameters
    ----------
    path (optional: str): Path to 4D numpy temperature lookup table
    Returns
    -------
    temperature_table (4D array): Temperature lookup (dims: lat, lst, az, theta)
    """
    return np.load(path, allow_pickle=True)
