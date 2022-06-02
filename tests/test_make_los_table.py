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
    rmss = np.array([0, 45])
    incs = np.array([0, 90])
    naz, ntheta = (1, 2)
    thresh = 1
    size = 100
    # TODO: mock this to avoid randomness
    np.random.seed(0)
    zsurf = mlt.make_zsurf(size, write=False)
    actual = mlt.make_los_table(zsurf, rmss, incs, naz, ntheta, thresh)
    shape = actual.total.values.shape
    expected_total = np.array(
        [[1, 4917, 1, 4917], [np.nan, np.nan, np.nan, np.nan]]
    ).reshape(shape)
    expected_los = np.array(
        [[1, 4917, 1, 0], [np.nan, np.nan, np.nan, np.nan]]
    ).reshape(shape)
    expected_prob = np.array(
        [[1, 1, 1, 0], [np.nan, np.nan, np.nan, np.nan]]
    ).reshape(shape)
    np.testing.assert_array_equal(actual["total"].values, expected_total)
    np.testing.assert_array_equal(actual["los"].values, expected_los)
    np.testing.assert_array_almost_equal(actual["prob"].values, expected_prob)


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
