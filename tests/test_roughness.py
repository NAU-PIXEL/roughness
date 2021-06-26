"""Test module for roughness."""
import pytest
import numpy as np
from roughness import __author__, __email__
import roughness.roughness as r


def test_get_shadow_table_nadir():
    """Test get_shadow_table() with nadir sun vector."""
    rms = 20
    sun_theta, sun_az = (0, 270)  # solar inclination angle
    actual = r.get_shadow_table(rms, sun_theta, sun_az)
    expected = np.zeros(actual.shape)
    np.testing.assert_equal(actual, expected)


def test_get_shadow_table_custom_sl():
    """Test get_shadow_table() with supplied shadow_lookup array."""
    sl = np.ones((3, 6, 5, 4))  # 1 everywhere
    actual = r.get_shadow_table(0, 0, 0, sl)
    expected = np.ones(sl.shape[-2:])
    np.testing.assert_equal(actual, expected)


def test_get_view_table_nadir():
    """Test get_view_table() with nadir spacecraft vector."""
    rms = 20
    sc_theta, sc_az = (0, 270)  # spacecraft emisssion angle
    actual = r.get_view_table(rms, sc_theta, sc_az)
    expected = np.ones(actual.shape)
    np.testing.assert_equal(actual, expected)


def test_get_los_table():
    """Test get_los_table"""
    mock_lookup = np.arange(11 * 11 * 37 * 46).reshape(11, 11, 37, 46)
    actual = r.get_los_table(0, 0, 270, mock_lookup)
    expected = np.arange(37 * 46).reshape(37, 46)
    np.testing.assert_array_equal(actual, expected)


def test_load_los_lookup():
    """Test load_los_lookup"""
    # Test load from disk
    lookup = r.load_los_lookup()
    assert lookup.shape == (11, 11, 37, 46)


def test_get_lookup_coords():
    """Test get_lookup_coords"""
    sl = np.ones((3, 6, 5, 4))
    rms, inc, az, theta = r.get_lookup_coords(sl)
    np.testing.assert_almost_equal(rms, [0, 25, 50])
    np.testing.assert_almost_equal(inc, [0.0, 18.0, 36.0, 54.0, 72.0, 90.0])
    np.testing.assert_almost_equal(az, [0, 90, 180, 270, 360])
    np.testing.assert_almost_equal(theta, [0, 30, 60, 90])


def test_get_facet_grids():
    """Test get_facet_grids"""
    table = np.ones((3, 2))
    actual = r.get_facet_grids(table)
    expected = [
        np.array([[0, 90], [0, 90], [0, 90]]),
        np.array([[0, 0], [180, 180], [360, 360]]),
    ]
    np.testing.assert_almost_equal(actual, expected)


def test_get_facet_grids_radians():
    """Test get_facet_grids with radian units"""
    table = np.ones((3, 2))
    actual = r.get_facet_grids(table, units="radians")
    expected = [
        np.array([[0, np.pi / 2], [0, np.pi / 2], [0, np.pi / 2]]),
        np.array([[0, 0], [np.pi, np.pi], [2 * np.pi, 2 * np.pi]]),
    ]
    np.testing.assert_almost_equal(actual, expected)


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


def test_sph2cart_1d():
    """Test sph2cart with numeric arguments."""
    actual = r.sph2cart(np.radians(45), np.radians(45), 1)
    expected = np.array((0.5, 0.5, np.sqrt(0.5)))
    np.testing.assert_array_almost_equal(actual, expected)


def test_sph2cart_2d():
    """Test sph2cart with 2D array arguments."""
    theta = np.array(([30, 45], [45, 30]))
    azimuth = np.array(([30, 45], [30, 45]))
    actual = r.sph2cart(np.radians(theta), np.radians(azimuth), 1)
    expected = np.sqrt(
        np.array(
            [
                [[0.1875, 0.0625, 0.75], [0.25, 0.25, 0.5]],
                [[0.375, 0.125, 0.5], [0.125, 0.125, 0.75]],
            ]
        )
    )
    np.testing.assert_array_almost_equal(actual, expected)


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
