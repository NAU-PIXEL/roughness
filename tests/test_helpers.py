"""Test helpers module."""
import numpy as np
import roughness.helpers as rh


def test_sph2cart_1d():
    """Test sph2cart with numeric arguments."""
    actual = rh.sph2cart(np.radians(45), np.radians(45), 1)
    expected = np.array((0.5, 0.5, np.sqrt(0.5)))
    np.testing.assert_array_almost_equal(actual, expected)


def test_sph2cart_2d():
    """Test sph2cart with 2D array arguments."""
    theta = np.array(([30, 45], [45, 30]))
    azimuth = np.array(([30, 45], [30, 45]))
    actual = rh.sph2cart(np.radians(theta), np.radians(azimuth), 1)
    expected = np.sqrt(
        np.array(
            [
                [[0.1875, 0.0625, 0.75], [0.25, 0.25, 0.5]],
                [[0.375, 0.125, 0.5], [0.125, 0.125, 0.75]],
            ]
        )
    )
    np.testing.assert_array_almost_equal(actual, expected)


def test_get_lookup_coords():
    """Test get_lookup_coords"""
    rms, inc, az, theta = rh.get_lookup_coords(5, 2, 4, 3)
    np.testing.assert_almost_equal(rms, [0, 10, 20, 30, 40])
    np.testing.assert_almost_equal(inc, [0, 60])
    np.testing.assert_almost_equal(az, [0, 90, 180, 270])
    np.testing.assert_almost_equal(theta, [0, 30, 60])


def test_facet_grids():
    """Test facet_grids"""
    table = np.ones((3, 2))
    actual = rh.facet_grids(table)
    expected = [
        np.array([[0, 45], [0, 45], [0, 45]]),
        np.array([[0, 0], [120, 120], [240, 240]]),
    ]
    np.testing.assert_almost_equal(actual, expected)


def test_facet_grids_radians():
    """Test facet_grids with radian units"""
    table = np.ones((2, 2))
    actual = rh.facet_grids(table, units="radians")
    expected = [
        np.array([[0, np.pi / 4], [0, np.pi / 4]]),
        np.array([[0, 0], [np.pi, np.pi]]),
    ]
    np.testing.assert_almost_equal(actual, expected)
