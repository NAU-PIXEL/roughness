"""roughness namespace."""

from importlib_metadata import PackageNotFoundError, version

__author__ = "Christian Tai Udovicic"
__email__ = "cj.taiudovicic@gmail.com"

# Used to automatically set version number from github actions
# as well as not break when being tested locally
try:
    __version__ = version(__package__)
except PackageNotFoundError:  # pragma: no cover
    __version__ = "0.0.0"
