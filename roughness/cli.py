"""
Command-line interface for running MoonPIES.

Usage:
    roughness [options]

Options:
    --download  Download data from Zenodo
    --version   Show version
    --help      Show help info

Examples:
    roughness --download
"""
import argparse
import subprocess
from roughness import __version__
from roughness.config import DATA_DIR, ZENODO_DOI, FDATA_VERSION


def run():
    """Command-line interface for roughness."""
    # Get optional random seed and cfg file from cmd-line args
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-d",
        "--download",
        default=False,
        action="store_true",
        help="download data from Zenodo",
    )
    parser.add_argument(
        "-v",
        "--version",
        action="version",
        version=f"%(prog)s {__version__}",
    )

    # Run the cli
    args = parser.parse_args()
    if args.download:
        cmdargs = ("-R", "1", "-o", DATA_DIR, ZENODO_DOI)
        r = subprocess.call(["zenodo_get", *cmdargs])
        if r == 0:  # If successful, update data version
            # Store wget call in FDATA_VERSION (urls change if data changes)
            cmdargs = ("-w", FDATA_VERSION, "-o", DATA_DIR, ZENODO_DOI)
            subprocess.call(["zenodo_get", *cmdargs])


if __name__ == "__main__":
    run()
