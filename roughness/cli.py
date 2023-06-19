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
import os
import argparse
import subprocess
from roughness import __version__, config


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
        os.chdir(config.DATA_DIR)
        subprocess.call(["zenodo_get", "-R", "1", config.ZENODO_DOI])


if __name__ == "__main__":
    run()
