"""Main module."""
import numpy as np
from numpy import sin, cos, tan, arccos, exp, pi

def visible_slopes(theta, azimuth, slope, slope_type='rms', nslopes=90):
    """
    Return probability of observering slopes from 0-90 degrees given a rough
    surface (slope, slope_type), corrected for viewing geometery (theta, phi).

    Parameters
    ----------
    theta (num): To-spacecraft emission angle [deg].
    azimuth (num): To-spacecraft azimuth angle from North [deg].
    slope (num): Mean slope of distribution [deg].
    slope_type (str): Slope distribution (rms or tbar).
    nslopes (num): Number of slopes to sample in 0-90 [deg].

    Returns
    -------
    P(slope) (arr, len=nslopes): Probability of viewing each slope in [0-90]
    """
    # Grid of all possible surface vectors (theta 0-90, az 0-360)
    res = nslopes/90
    theta_arr = np.radians(np.arange(0, 90, res))
    azimuth_arr = np.radians(np.arange(0, 360, res))
    theta_surf, azimuth_surf = np.meshgrid(theta_arr, azimuth_arr)
    
    # Spacecraft unit vector
    theta_sc = np.radians(theta) 
    azimuth_sc = np.radians(azimuth)

    # Cartesian dot product gives cos(angle) between vectors
    cartesian_sc = sph2cart(theta_sc, azimuth_sc)
    cartesian_surf = sph2cart(theta_surf, azimuth_surf)
    dot = np.dot(cartesian_surf, cartesian_sc) 
    # dot[dot > 1] = 1

    # Angle > 90 deg means surface is oriented away from spacecraft
    # Theses surfaces are not visible, so set to nan
    angle = arccos(dot)
    angle[angle >= (np.pi/2)] = np.nan

    # Get raw slope distribution and correct for viewing geometry
    slopes = slope_dist(theta_surf, slope, slope_type)
    vis_slopes = slopes * cos(angle) / cos(theta_sc)
    # vis_slopes[vis_slopes < 0] = 0

    return vis_slopes


def sph2cart(theta, phi, r=1):
    """
    Convert from spherical (theta, phi, r) to cartesian (x, y, z) coordinates.

    Theta and phi must be specified in radians and returned units correspond to 
    units of r, default is unitless (r=1 is unit vector). Returns vectors along 
    z-axis (e.g., if theta and phi are int/float return 1x1x3 vector; if theta 
    and phi are MxN arrays return 3D array of vectors MxNx3).

    Parameters
    ----------
    theta (num or arr): Polar angle [rad].
    phi (num or arr): Azimuthal angle [rad].
    r (num or arr): Radius (default=1).

    Returns
    -------
    MxNx3 arr: Vectors converted to cartesian coordinates.
    """
    return np.dstack([
        r * sin(theta) * cos(phi),  # x
        r * sin(theta) * sin(phi),  # y
        r * cos(theta)  # z
    ]).squeeze()




def slope_dist(theta, theta0, dist='rms'):
    """
    Return slope probability distribution for rough fractal surfaces. Computes
    distributions parameterized by mean slope (theta0) using Shepard 1995 (rms) 
    or Hapke 1984 (tbar) roughness.

    Parameters
    ----------
    theta (arr): Angles to compute the slope distribution at [rad].
    theta0 (num): Mean slope angle (fixed param in RMS / theta-bar eq.) [rad].
    dist (str): Distribution type (rms or tbar)

    Returns
    -------
    slopes (arr): Probability of a slope occurring at each theta.
    """
    # Calculate Gaussian RMS slope distribution (Shepard 1995)
    if dist == 'rms': 
        slopes = ((tan(theta) / tan(theta0)**2) * 
                  exp(-tan(theta)**2 / (2 * tan(theta0)**2)))
    elif dist == 'rmsnorm':
        slopes = (1/(2*pi*tan(theta0)**2 * cos(theta)**3) * 
                  exp(-tan(theta)**2 / (2 * tan(theta0)**2)))

    # Calculate theta-bar slope distribution (Hapke 1984)
    elif dist == 'tbar':
        slopes = ((2 * sin(theta) / (pi * tan(theta0)**2 * cos(theta)**2)) *
                 exp((-tan(theta)**2) / (pi * tan(theta0)**2)))
    else:
        raise ValueError('Invalid distribution specified (accepts rms, tbar).')
    return slopes
