"""Test module for roughness."""
import pytest
import numpy as np
from roughness import __author__, __email__
from roughness.roughness import sph2cart, slope_dist, visible_slopes


def test_project_info():
    """Test __author__ value."""
    assert __author__ == "Christian Tai Udovicic"
    assert __email__ == "cj.taiudovicic@gmail.com"


def test_sph2cart_1d():
    """Test sph2cart with numeric arguments."""
    actual = sph2cart(np.radians(45), np.radians(45), 1)
    expected = np.array((0.5, 0.5, np.sqrt(0.5)))
    np.testing.assert_array_almost_equal(actual, expected)


def test_sph2cart_2d():
    """Test sph2cart with 2D array arguments."""
    theta = np.array(([30, 45], [45, 30]))
    azimuth = np.array(([30, 45], [30, 45]))
    actual = sph2cart(np.radians(theta), np.radians(azimuth), 1)
    expected = np.sqrt(
        np.array(
            [
                [[0.1875, 0.0625, 0.75], [0.25, 0.25, 0.5]],
                [[0.375, 0.125, 0.5], [0.125, 0.125, 0.75]],
            ]
        )
    )
    np.testing.assert_array_almost_equal(actual, expected)


def test_slope_dist_rms():
    """Test slope_dist for RMS slope distribution."""
    actual = slope_dist((0, 10, 20), 20, "rms")
    expected = np.array([0, 0.12421774, 0.27111624])
    np.testing.assert_array_almost_equal(actual, expected)


def test_slope_dist_tbar():
    """Test slope_dist for theta-bar slope distribution."""
    actual = slope_dist((0, 10, 20), 20, "tbar")
    expected = np.array([0, -0.09569568, 0.50721859])
    np.testing.assert_array_almost_equal(actual, expected)


def test_slope_dist_unknown():
    """Test slope_dist fails for unknown slope distribution."""
    with pytest.raises(ValueError):
        slope_dist((0, 10, 20), 20, "???")


def test_visible_slopes_shape():
    """Test visible_slopes output shape."""
    assert visible_slopes(0, 0, 20).shape == (360, 90)
    assert visible_slopes(0, 0, 20, ntheta=41, naz=183).shape == (183, 41)


def test_visible_slopes_zero_emiss():
    """Test visible_slopes equal slope dist at 0 emission angle."""
    actual = visible_slopes(0, 0, 20, ntheta=3, naz=2)
    theta = (0, np.pi / 6, np.pi / 3)
    sdist = slope_dist(theta, 20, "rms")
    expected = np.array((sdist, sdist))
    np.testing.assert_array_almost_equal(actual, expected)


def test_visible_slopes_high_emiss():
    """Test visible_slopes at high emission angle."""
    actual = visible_slopes(60, 0, 20, ntheta=3, naz=2)
    expected = np.array([[0, 0.1115790, 0.51290283], [0, 0, 0]])
    np.testing.assert_array_almost_equal(actual, expected)
