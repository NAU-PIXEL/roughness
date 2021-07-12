"""Roughness helper functions"""
import os
import numpy.f2py


def compile_lineofsight(vebose=False):
    """Compile lineofsight module with numpy f2py. Error if no compiler."""
    cwd = os.path.abspath(os.path.dirname(__file__)) + os.sep
    fsrc = f"{cwd}lineofsight.f90"
    with open(fsrc) as f:
        src = f.read()
    failed = numpy.f2py.compile(
        src, "lineofsight", verbose=verbose, source_fn=fsrc
    )
    if failed:
        msg = "Cannot compile Fortran raytracing code. Please ensure you have \
               a F90 compatible Fortran compiler (e.g. gfortran) installed."
        raise ImportError(msg)
