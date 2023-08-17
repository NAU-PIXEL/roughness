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
    rms, inc, az, theta = rh.get_lookup_coords(4, 2, 4, 3)
    np.testing.assert_almost_equal(rms, [0, 15, 30, 45])
    np.testing.assert_almost_equal(inc, [0, 90])
    np.testing.assert_almost_equal(az, [45, 135, 225, 315])
    np.testing.assert_almost_equal(theta, [15, 45, 75])


def test_facet_grids():
    """Test facet_grids"""
    table = np.ones((2, 3))
    actual = rh.facet_grids(table)
    expected = (
        np.array([[15.0, 45.0, 75.0], [15.0, 45.0, 75.0]]),
        np.array([[90.0, 90.0, 90.0], [270.0, 270.0, 270.0]]),
    )
    np.testing.assert_almost_equal(actual, expected)

def test_get_inc_az_from_subspacecraft():
    """Test get_inc_az_from_subspacecraft"""
    actual = rh.get_inc_az_from_subspacecraft(0,0,45,45)
    expected = (60,35.26)
    np.testing.assert_array_almost_equal(actual, expected, 2)

    actual = rh.get_inc_az_from_subspacecraft(0,0,-45,-45)
    expected = (60,215.26)
    np.testing.assert_array_almost_equal(actual, expected, 2)

    actual = rh.get_inc_az_from_subspacecraft(-10,0,-55,45)
    expected = (60,324.73)
    np.testing.assert_array_almost_equal(actual, expected, 2)