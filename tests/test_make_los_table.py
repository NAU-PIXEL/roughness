"""Test make_los_table.py"""
import sys
import numpy as np
import pytest
import roughness.make_los_table as mlt


@pytest.mark.skipif(
    "linux" not in sys.platform, reason="Raytrace onlycompiles on Linux"
)
def test_make_los_table_small():
    """Test make_los_table()."""
    rms_inc_az_theta = (2, 2, 1, 2)
    thresh = 0
    size = 100
    np.random.seed(0)
    actual = mlt.make_los_table(*rms_inc_az_theta, thresh, size, write=False)
    expected = [
        np.array([[[[0, 0]], [[0, 0]]], [[[9806, 194]], [[9806, 194]]]]),
        np.array([[[[0, 0]], [[0, 0]]], [[[9806, 194]], [[7792, 127]]]]),
        np.array(
            [
                [[[np.nan, np.nan]], [[np.nan, np.nan]]],
                [[[1, 1]], [[0.7946155, 0.6546392]]],
            ]
        ),
    ]
    np.testing.assert_array_equal(actual["total"].values, expected[0])
    np.testing.assert_array_equal(actual["los"].values, expected[1])
    np.testing.assert_array_almost_equal(actual["prob"].values, expected[2])


def test_make_zsurf():
    """Test make_zsurf()."""
    n = 4  # Size of x, y
    np.random.seed(0)
    actual = mlt.make_zsurf(n, fout="test", write=False)
    expected = np.array(
        [
            [1.22621598, -0.30610008, 0.8652084, 0.34876805],
            [0.18977297, 0.14626959, -1.16397329, -0.05466952],
            [1.01859442, -0.22372572, -3.11001879, 0.18105776],
            [-0.16313815, -0.03342096, 1.13733848, -0.05817911],
        ]
    )

    np.testing.assert_array_almost_equal(actual, expected)


def test_make_zsurf_hsmall():
    """Test make_zsurf()."""
    n = 4  # Size of x, y
    H = 0  # Hurst exponent
    np.random.seed(0)
    actual = mlt.make_zsurf(n, H, fout="test", write=False)
    expected = np.array(
        [
            [1.22621598, -0.30610008, 0.8652084, 0.34876805],
            [0.18977297, 0.14626959, -1.16397329, -0.05466952],
            [1.01859442, -0.22372572, -3.11001879, 0.18105776],
            [-0.16313815, -0.03342096, 1.13733848, -0.05817911],
        ]
    )
    np.testing.assert_array_almost_equal(actual, expected)


def test_make_zsurf_scale_factors():
    """Test make_zsurf_scale_factors()."""
    zsurf = np.array(
        [
            [1.22621598, -0.30610008, 0.8652084, 0.34876805],
            [0.18977297, 0.14626959, -1.16397329, -0.05466952],
            [1.01859442, -0.22372572, -3.11001879, 0.18105776],
            [-0.16313815, -0.03342096, 1.13733848, -0.05817911],
        ]
    )
    rms_arr = np.array([0, 25, 50, 75])
    actual = mlt.make_scale_factors(zsurf, rms_arr, write=False)
    expected = np.array([0, 0.28466134, 0.90154845, 3.75400359])
    np.testing.assert_array_almost_equal(actual, expected)
