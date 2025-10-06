"""IO module for netcdf lookup tables (LUT)."""

from functools import lru_cache
import numpy as np
import xarray as xr


def rotate_az_lookup(los_table, target_az, az0=0):
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
    
    azcoords = los_table.az.values
    targ = (target_az+az0)%360
    az_shift = np.argmin(np.abs(azcoords - targ))
    return los_table.roll(az=az_shift, roll_coords=False)


def get_LUT(inLUT,params,var=None,rotate=None,az0=0):

    if not isinstance(inLUT, xr.Dataset):
        lut = load_XarrayDS(inLUT)
    else:
        lut = inLUT
    
    lut_vars = [i for i in lut.data_vars]
    if var is None:
        var = lut_vars[0]
    lut = lut[var]

    for key in params:
        if key not in lut.coords.keys():
            raise ValueError(
                "Supplied parameters do not match LUT coordinates. "
            )
        
    if rotate is not None:
        lut = rotate_az_lookup(lut, rotate, az0)
        
    selparams = {k: v for k, v in params.items() if v in lut.coords[k]}
    intparams = {k: v for k, v in params.items() if k not in selparams}
    out = lut.sel(selparams).interp(intparams, method = "linear").to_numpy()
    
    return out

def get_tiled_theta(inLUT):
    if not isinstance(inLUT, xr.Dataset):
        lut = load_XarrayDS(inLUT)
    else:
        lut = inLUT
    
    theta = lut.theta.values
    thetas = np.tile(theta,(36,1))
    return thetas

@lru_cache(maxsize=2)
def load_XarrayDS(inpath):
    out = xr.open_dataset(inpath)
    return out 