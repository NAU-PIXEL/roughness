"""Test roughness module."""
import pytest
import numpy as np
from roughness import __author__, __email__
import roughness.roughness as r


def test_get_shadow_table_nadir():
    """Test get_shadow_table() with nadir sun vector."""
    rms = 20
    sun_theta, sun_az = (0, 270)  # solar inclination angle
    actual = np.nansum(r.get_shadow_table(rms, sun_theta, sun_az))
    expected = 0
    np.testing.assert_equal(actual, expected)


def test_get_shadow_table_custom_sl():
    """Test get_shadow_table() with supplied shadow_lookup array."""
    sl = np.ones((3, 6, 5, 4))  # in line of sight everywhere
    actual = r.get_shadow_table(0, 0, 0, sl)
    expected = np.zeros(sl.shape[-2:])
    np.testing.assert_equal(actual, expected)


def test_get_view_table_nadir():
    """Test get_view_table() with nadir spacecraft vector."""
    rms = 20
    sc_theta, sc_az = (0, 270)  # spacecraft emisssion angle
    actual = np.nansum(r.get_view_table(rms, sc_theta, sc_az))
    expected = 1
    np.testing.assert_equal(actual, expected)


def test_get_los_table():
    """Test get_los_table"""
    mock_lookup = np.arange(10 * 10 * 36 * 45).reshape(10, 10, 36, 45)
    actual = r.get_los_table(0, 0, 270, mock_lookup)
    expected = np.arange(36 * 45).reshape(36, 45)
    np.testing.assert_array_equal(actual, expected)


def test_load_los_lookup():
    """Test load_los_lookup"""
    # Test load from disk
    lookup = r.load_los_lookup()
    assert lookup.shape == (10, 10, 36, 45)


def test_interp_lookup():
    """Test interp_lookup"""

    values = np.arange(24).reshape(4, 3, 2)
    xcoords = np.array([0, 1, 2, 3])

    # Test x in xcoords
    xvalue = 1
    actual = r.interp_lookup(xvalue, xcoords, values)
    expected = values[1]
    np.testing.assert_equal(actual, expected)

    # Test x < min(xcoords)
    xvalue = -1
    actual = r.interp_lookup(xvalue, xcoords, values)
    expected = values[0]
    np.testing.assert_equal(actual, expected)

    # Test x > max(xcoords)
    xvalue = 4
    actual = r.interp_lookup(xvalue, xcoords, values)
    expected = values[-1]
    np.testing.assert_equal(actual, expected)

    # Test x half way between two xcoords
    xvalue = 1.5
    actual = r.interp_lookup(xvalue, xcoords, values)
    expected = (values[1] + values[2]) / 2
    np.testing.assert_equal(actual, expected)

    # Test x a third of the way between two xcoords
    xvalue = 2 / 3
    actual = r.interp_lookup(xvalue, xcoords, values)
    expected = (1 / 3) * values[0] + (2 / 3) * values[1]
    np.testing.assert_almost_equal(actual, expected)


def test_rotate_az_lookup():
    """Test rotate_az_lookup"""
    shadow_table = np.arange(10).reshape(5, 2)
    az_coords = np.array([0, 90, 180, 270, 360])

    # No shift (target_az = 270)
    actual = r.rotate_az_lookup(shadow_table, 270, az_coords)
    expected = shadow_table
    np.testing.assert_equal(actual, expected)

    # Shift 90 degrees W
    actual = r.rotate_az_lookup(shadow_table, 180, az_coords)
    expected = np.array([[2, 3], [4, 5], [6, 7], [8, 9], [0, 1]])
    np.testing.assert_equal(actual, expected)

    # Shift 90 degrees E
    actual = r.rotate_az_lookup(shadow_table, 360, az_coords)
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
    actual = r.slope_dist(thetas, theta0)
    expected = [0, 0, 0]
    np.testing.assert_array_equal(actual, expected)


def test_slope_dist_rms():
    """Test slope_dist for RMS slope distribution."""
    thetas = np.radians((0, 10, 20))
    theta0 = np.radians(20)
    actual = r.slope_dist(thetas, theta0, "rms")
    expected = [0, 0.415304, 0.584696]
    np.testing.assert_array_almost_equal(actual, expected)


def test_slope_dist_tbar():
    """Test slope_dist for theta-bar slope distribution."""
    thetas = np.radians((0, 10, 20))
    theta0 = np.radians(20)
    actual = r.slope_dist(thetas, theta0, "tbar")
    expected = [0, 0.370978, 0.629022]
    np.testing.assert_array_almost_equal(actual, expected)


def test_slope_dist_unknown():
    """Test slope_dist fails for unknown slope distribution."""
    with pytest.raises(ValueError):
        r.slope_dist(0, 1, "???")
