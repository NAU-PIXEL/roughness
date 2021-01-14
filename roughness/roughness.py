"""Main module."""
import numpy as np
from numpy import sin, cos, tan, exp, pi


def visible_slopes(theta, azimuth, slope, dist="rms", ntheta=90, naz=360):
    """
    Return probability of observering slopes from 0-90 degrees given a rough
    surface (slope, slope_type), corrected for viewing geometery (theta, phi).

    Parameters
    ----------
    theta (num): To-spacecraft emission angle [deg].
    azimuth (num): To-spacecraft azimuth angle from North [deg].
    slope (num): Mean slope of distribution [deg].
    slope_type (str): Slope distribution (rms or tbar).
    ntheta (num): Number of surface slope thetas to sample range [0-90) deg.
    naz (num): Number of surface slope azimuths to sample range [0-360) deg.

    Returns
    -------
    P(slope) (arr, len=nslopes): Probability of viewing each slope in [0-90]
    """
    # Grid of all possible surface vectors (theta 0-90, az 0-360)
    theta_arr = np.radians(np.arange(0, 90, 90 / ntheta))
    azimuth_arr = np.radians(np.arange(0, 360, 360 / naz))
    theta_surf, azimuth_surf = np.meshgrid(theta_arr, azimuth_arr)
    cartesian_surf = sph2cart(theta_surf, azimuth_surf)

    # Spacecraft unit vector
    theta, azimuth = (np.radians(theta), np.radians(azimuth))
    cartesian_sc = sph2cart(theta, azimuth)

    # Cartesian dot product gives cos(angle) between vectors
    cos_surf_sc = np.dot(cartesian_surf, cartesian_sc)

    # Negative cos(angle) means surface oriented away from sc, set to 0
    cos_surf_sc[cos_surf_sc < 0] = 0
    # cos_surf_sc[cos_surf_sc > 1] = 1  # shouldn't be possible

    # Get raw slope distribution and correct for viewing geometry
    slopes = slope_dist(theta_surf, slope, dist)
    vis_slopes = slopes * cos_surf_sc / cos(theta_surf)
    # vis_slopes[vis_slopes < 0] = 0

    return vis_slopes


def sph2cart(theta, phi, radius=1):
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
    radius (num or arr): Radius (default=1).

    Returns
    -------
    MxNx3 arr: Vectors converted to cartesian coordinates.
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
    theta (arr): Angles to compute the slope distribution at [rad].
    theta0 (num): Mean slope angle (fixed param in RMS / theta-bar eq.) [rad].
    dist (str): Distribution type (rms or tbar)

    Returns
    -------
    slopes (arr): Probability of a slope occurring at each theta.
    """
    # Calculate Gaussian RMS slope distribution (Shepard 1995)
    if dist == "rms":
        slopes = (tan(theta) / tan(theta0) ** 2) * exp(
            -tan(theta) ** 2 / (2 * tan(theta0) ** 2)
        )

    # Calculate theta-bar slope distribution (Hapke 1984)
    elif dist == "tbar":
        slopes = (
            2 * sin(theta) / (pi * tan(theta0) ** 2 * cos(theta) ** 2)
        ) * exp((-tan(theta) ** 2) / (pi * tan(theta0) ** 2))
    else:
        raise ValueError("Invalid distribution specified (accepts rms, tbar).")
    return slopes
