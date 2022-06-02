"""Roughness configuration."""
import json
from pathlib import Path
from . import __version__

# Constants
SC = 1360  # Solar constant at 1 au [W/m^2]
SB = 5.6703271e-8  # Stephan Boltzmann constant [W m^-2 K^-4]
CCM = 2.99792458e10  # Speed of light in vacuum [cm/s]
KB = 1.38064852e-23  # Boltzmann constant [J / K]
HC = 6.62607004e-34  # Planck constant [J s]

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
RMSS = (0, 1, 3, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50)
INCS = (0, 2, 5, 10, 20, 30, 45, 60, 70, 80, 85, 88, 90)
NRMS = 10  # Number of RMS slopes in [0, 50) degrees.
NINC = 10  # Number of incidence angles in [0, 90) degrees.
NAZ = 36  # Number of facet azimuth bins in [0, 360) degrees.
NTHETA = 45  # Number of facet slope bins in [0, 90) degrees.
MIN_FACETS = 25  # Minimum number of valid facets to do binning, else nan
DEMSIZE = 12000  # Size length of square dem in pixels.
AZ0 = 270  # Azimuth of LOS [degrees].

# Paths (dirs)
ROOT_DIR = Path(__file__).parents[1]
ROUGHNESS_DIR = ROOT_DIR / "roughness"
DATA_DIR = ROOT_DIR / "data"
DATA_DIR_M3 = DATA_DIR / "m3"
DATA_DIR_DIVINER = DATA_DIR / "Diviner" / "lev4"
EXAMPLES_DIR = ROOT_DIR / "examples"
FIGURES_DIR = ROOT_DIR / "figures"
FORTRAN_DIR = ROUGHNESS_DIR / "fortran"
TEST_DIR = ROOT_DIR / "tests"

# Paths (setup_roughness.py)
FDATA_VERSION = DATA_DIR / ".data_version.txt"

# Paths (config.py)
FLOS_F90 = FORTRAN_DIR / "lineofsight.f90"

# Paths (make_los_table.py)
FZSURF = DATA_DIR / "zsurf.npy"
FZ_FACTORS = DATA_DIR / "zsurf_scale_factors.npy"
FLOOKUP = DATA_DIR / "default_lookup.nc"
FALBEDO = DATA_DIR / "Feng_et_al_2020_Lunar_bolometric_Bond_albedo.txt"
TLOOKUP = DATA_DIR / "temp_lookup.nc"


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

# JSON
with open(DATA_DIR_M3 / "m3_wls.json", "r", encoding="utf8") as f:
    M3_WLS = json.load(f)
    M3GWL = M3_WLS["GWL"]
    M3TWL = M3_WLS["TWL"]

# Georeferencing
MOON2000_ESRI = "ESRI:104903"
