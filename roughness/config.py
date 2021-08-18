"""Roughness configuration."""
from pathlib import Path

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
FTOT_FACETS = DATA_DIR / "total_facets_4D.npy"
FLOS_FACETS = DATA_DIR / "los_facets_4D.npy"
FLOS_LOOKUP = DATA_DIR / "los_lookup_4D.npy"

# Paths (examples)
FIG_SLOPE_DIST_SHEP = FIGURES_DIR / "slope_dist_shepard.png"
FIG_SLOPE_DIST_HAPKE = FIGURES_DIR / "slope_dist_hapke.png"
FIG_SLOPE_DIST_GSURF = FIGURES_DIR / "slope_dist_gaussian_surf.png"
FIG_SLOPE_DIST_VIS_RMS = FIGURES_DIR / "slope_dist_vis_rms.png"
FIG_SLOPE_DIST_VIS_THETA = FIGURES_DIR / "slope_dist_vis_theta.png"

# Help strings (setup_roughness.py)
HNRMS = "Number of rms slope bins in [0, 55) deg (default: 10)."
HNINC = "Number of incidence bins in [0, 90) deg (default: 10)."
HNAZ = "Number of facet azimuth bins in [0, 360) deg (default: 36)."
HNTHETA = "Number of facet slope (theta) bins in [0, 90) deg (default: 45)."
HTHRESH = "Threshold number of facets to compute los fraction (default: 25)."
HDSIZE = "Side length of synthetic DEM in pixels (default: 10000)."
HRAZ = "Raytracing line of sight azimuth in degrees (default: 270)"
