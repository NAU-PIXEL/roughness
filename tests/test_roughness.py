"""Test roughness module."""
import pytest
import numpy as np
from roughness import __author__, __email__
import roughness.roughness as rn
import roughness.helpers as rh
from roughness.config import LUT_NAMES


def test_get_shadow_table_nadir():
    """Test get_shadow_table() with nadir sun vectorn."""
    rms = 20
    sun_theta, sun_az = (0, 270)  # solar inclination angle
    actual = np.nansum(rn.get_shadow_table(rms, sun_theta, sun_az))
    expected = 0
    np.testing.assert_equal(actual, expected)


def test_get_shadow_table_custom_sl():
    """Test get_shadow_table() with supplied shadow_lookup array."""
    sl = np.ones((3, 6, 5, 4))  # in line of sight everywhere
    actual = rn.get_shadow_table(0, 0, 0, sl).values
    expected = np.zeros(sl.shape[-2:])
    np.testing.assert_equal(actual, expected)


def test_get_view_table_nadir():
    """Test get_view_table() with nadir spacecraft vector."""
    rms = 20
    sc_theta, sc_az = (0, 270)  # spacecraft emisssion angle
    actual = np.nansum(rn.get_view_table(rms, sc_theta, sc_az))
    expected = 792
    np.testing.assert_equal(actual, expected)


def test_get_los_table():
    """Test get_los_table"""
    mock_lookup = np.arange(10 * 10 * 36 * 45).reshape(10, 10, 36, 45)
    actual = rn.get_los_table(0, 0, 270, los_lookup=mock_lookup)
    expected = np.arange(36 * 45).reshape(36, 45)
    np.testing.assert_array_equal(actual, expected)


def test_load_los_lookup():
    """Test load_los_lookup from default"""
    lookup = rn.load_los_lookup()
    for name in LUT_NAMES:
        assert name in lookup


def test_rotate_az_lookup():
    """Test rotate_az_lookup"""
    shadow_table = np.arange(10).reshape(5, 2)
    az_coords = np.array([0, 90, 180, 270, 360])
    coords = (az_coords, np.array([0, 45]))
    shadow_table = rh.np2xr(shadow_table, dims=("az", "theta"), coords=coords)

    # No shift (target_az = 270)
    actual = rn.rotate_az_lookup(shadow_table, 270).values
    expected = shadow_table
    np.testing.assert_equal(actual, expected)

    # Shift 90 degrees W
    actual = rn.rotate_az_lookup(shadow_table, 180).values
    expected = np.array([[2, 3], [4, 5], [6, 7], [8, 9], [0, 1]])
    np.testing.assert_equal(actual, expected)

    # Shift 90 degrees E
    actual = rn.rotate_az_lookup(shadow_table, 360).values
    expected = np.array([[8, 9], [0, 1], [2, 3], [4, 5], [6, 7]])
    np.testing.assert_equal(actual, expected)


def test_project_info():
    """Test __author__ value."""
    assert __author__ == "Christian J. Tai Udovicic"
    assert __email__ == "cj.taiudovicic@gmail.com"


def test_slope_dist_zero():
    """Test slope_dist with theta0==0"""
    thetas = np.radians((0, 10, 20))
    theta0 = 0
    actual = rn.slope_dist(thetas, theta0)
    expected = [0, 0, 0]
    np.testing.assert_array_equal(actual, expected)


def test_slope_dist_rms():
    """Test slope_dist for RMS slope distribution."""
    thetas = np.radians((0, 10, 20))
    theta0 = np.radians(20)
    actual = rn.slope_dist(thetas, theta0, "rms")
    expected = [0, 0.415304, 0.584696]
    np.testing.assert_array_almost_equal(actual, expected)


def test_slope_dist_tbar():
    """Test slope_dist for theta-bar slope distribution."""
    thetas = np.radians((0, 10, 20))
    theta0 = np.radians(20)
    actual = rn.slope_dist(thetas, theta0, "tbar")
    expected = [0, 0.370978, 0.629022]
    np.testing.assert_array_almost_equal(actual, expected)


def test_slope_dist_unknown():
    """Test slope_dist fails for unknown slope distribution."""
    with pytest.raises(ValueError):
        rn.slope_dist(0, 1, "???")
