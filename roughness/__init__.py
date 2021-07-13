"""roughness namespace."""
# Import functions into main roughness namespace (e.g. roughness.func())
from .helpers import get_lookup_coords
from .make_los_table import (
    make_los_table,
    FLOS_LOOKUP,
    FLOS_FACETS,
    FTOT_FACETS,
)
from .roughness import (
    get_shadow_table,
    get_view_table,
    get_los_table,
    load_los_lookup,
)

try:
    from importlib_metadata import PackageNotFoundError, version
except ImportError:  # pragma: no cover
    from importlib.metadata import PackageNotFoundError, version

__author__ = "Christian J. Tai Udovicic"
__email__ = "cj.taiudovicic@gmail.com"

# Used to automatically set version number from github actions
# as well as not break when being tested locally
try:
    __version__ = version(__package__)
except PackageNotFoundError:  # pragma: no cover
    __version__ = "0.0.0"
