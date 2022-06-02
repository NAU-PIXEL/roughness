"""Init roughness."""
# Cludge: Fix path to projlib for rioxarray (must import roughness first)
import os
import site

pdb = os.path.join(site.getsitepackages()[0], "pyproj/proj_dir/share/proj")
pdb = os.path.abspath(pdb)
if os.environ["PROJ_LIB"] != pdb:
    os.environ["PROJ_LIB"] = pdb

# Automatically set version number with github actions, don't break local tests
try:
    from importlib_metadata import PackageNotFoundError, version
except ImportError:  # pragma: no cover
    from importlib.metadata import PackageNotFoundError, version
try:
    __version__ = version(__package__)
except PackageNotFoundError:  # pragma: no cover
    __version__ = "0.0.0"

__author__ = "Christian J. Tai Udovicic"
__email__ = "cj.taiudovicic@gmail.com"
