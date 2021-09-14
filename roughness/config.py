"""Roughness configuration."""
from pathlib import Path
from . import __version__

# Defaults
VERSION = __version__
LUT_NAMES = ("total", "los", "prob")
LUT_LONGNAMES = ("Total facets", "LOS facets", "P(LOS)")
LUT_DIMS = ("rms", "inc", "az", "theta")
LUT_DIMS_LONGNAMES = (
    "RMS Roughness",
    "Incidence Angle",
    "Facet Azimuth Angle",
    "Facet Slope Angle",
)
NRMS = 10  # Number of RMS slopes in [0, 50) degrees.
NINC = 10  # Number of incidence angles in [0, 90) degrees.
NAZ = 36  # Number of facet azimuth bins in [0, 360) degrees.
NTHETA = 45  # Number of facet slope bins in [0, 90) degrees.
THRESHOLD = 25  # Minimum number of valid facets to do binning, else nan
DEMSIZE = 10000  # Size length of square dem in pixels.
AZ0 = 270  # Azimuth of LOS [degrees].

# Paths (dirs)
ROOT_DIR = Path(__file__).parents[1]
ROUGHNESS_DIR = ROOT_DIR / "roughness"
DATA_DIR = ROOT_DIR / "data"
EXAMPLES_DIR = ROOT_DIR / "examples"
FIGURES_DIR = ROOT_DIR / "figures"
FORTRAN_DIR = ROUGHNESS_DIR / "fortran"

# Paths (setup_roughness.py)
FDATA_VERSION = DATA_DIR / ".data_version.txt"

# Paths (config.py)
FLOS_F90 = FORTRAN_DIR / "lineofsight.f90"

# Paths (make_los_table.py)
FZSURF = DATA_DIR / "zsurf.npy"
FZ_FACTORS = DATA_DIR / "zsurf_scale_factors.npy"
FLOOKUP = DATA_DIR / "default_lookup.nc"
FTOT_FACETS = DATA_DIR / "total_facets_4D.npy"
FLOS_FACETS = DATA_DIR / "los_facets_4D.npy"
FLOS_LOOKUP = DATA_DIR / "los_lookup_4D.npy"

# Paths (examples)
FIG_SLOPE_DIST_SHEP = FIGURES_DIR / "slope_dist_shepard.png"
FIG_SLOPE_DIST_HAPKE = FIGURES_DIR / "slope_dist_hapke.png"
FIG_SLOPE_DIST_GSURF = FIGURES_DIR / "slope_dist_gaussian_surf.png"
FIG_SLOPE_DIST_VIS_RMS = FIGURES_DIR / "slope_dist_vis_rms.png"
FIG_SLOPE_DIST_VIS_THETA = FIGURES_DIR / "slope_dist_vis_theta.png"

# Plotting (plot_roughness.py)


# Help strings (setup_roughness.py)
HNRMS = "Number of rms slope bins in [0, 55) deg (default: 10)."
HNINC = "Number of incidence bins in [0, 90) deg (default: 10)."
HNAZ = "Number of facet azimuth bins in [0, 360) deg (default: 36)."
HNTHETA = "Number of facet slope (theta) bins in [0, 90) deg (default: 45)."
HTHRESH = "Threshold number of facets to compute los fraction (default: 25)."
HDSIZE = "Side length of synthetic DEM in pixels (default: 10000)."
HRAZ = "Raytracing line of sight azimuth in degrees (default: 270)"
