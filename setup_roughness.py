"""Init roughness - run first time using roughness or after updating."""
import argparse
from roughness import __version__
from roughness.make_los_table import make_default_los_table
import roughness.config as cfg
import roughness.helpers as rh


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--nrms", "-r", type=int, help=cfg.HNRMS)
    parser.add_argument("--ninc", "-i", type=int, help=cfg.HNINC)
    parser.add_argument("--naz", "-a", type=int, help=cfg.HNAZ)
    parser.add_argument("--ntheta", "-t", type=int, help=cfg.HNTHETA)
    parser.add_argument("--threshold", "-o", type=int, help=cfg.HTHRESH)
    parser.add_argument("--demsize", "-s", type=int, help=cfg.HDSIZE)
    parser.add_argument("--az0", "-z", type=int, help=cfg.HRAZ)
    parser.add_argument("--default", "-d", action="store_true")
    parser.add_argument("--nolos", "-n", action="store_true")
    args = parser.parse_args()
    kwargs = {k: v for k, v in vars(args).items() if v is not None}

    # Sync Jupyter notebooks with jupytext
    rh.build_jupyter_notebooks()

    # Check if we should skip making los table
    nolos = kwargs.pop("nolos", False)
    if nolos:
        pass
    else:
        # Make line of sight lookup table
        default = kwargs.pop("default", False)
        if default or not cfg.FLOOKUP.exists():
            print("Generating default los table.")
            make_default_los_table()
            rh.set_data_version()
        # elif rh.check_data_updated():
        # _ = make_los_table(**kwargs)
        else:
            print("No changes made")
