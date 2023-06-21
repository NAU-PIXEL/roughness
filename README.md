<h1 align="center">Roughness</h1>

<div align="center">
  <strong>Illumination of rough planetary surfaces.</strong>
</div>

<div align="center">
  <span>
  <!-- PYPI version -->
  <!-- <a href="https://badge.fury.io/py/roughness">
    <img src="https://badge.fury.io/py/roughness.svg"
      alt="PYPI version" />
  </a> -->
 <!-- Test Coverage -->
  <!-- <a href="https://codecov.io/github/choojs/choo">
    <img src="https://img.shields.io/codecov/c/github/choojs/choo/master.svg?style=flat-square"
      alt="Test Coverage" />
  </a> -->
  <!-- Zenodo DOI -->
  <a href="https://zenodo.org/badge/latestdoi/328820617"><img src="https://zenodo.org/badge/328820617.svg" alt="DOI">
  <!-- Code Quality and Tests -->
  </a>
  <a href="https://github.com/NAU-PIXEL/roughness/actions/workflows/code_quality_checks.yml"><img src="https://github.com/NAU-PIXEL/roughness/actions/workflows/code_quality_checks.yml/badge.svg" alt="Code Quality and Tests">
  <!-- Docs -->
  </a>
  <a href="https://nau-pixel.github.io/roughness/"><img src="https://github.com/NAU-PIXEL/roughness/actions/workflows/docs_publish.yml/badge.svg" alt="Documentation">
  <!-- Code Style Black -->
  </a>
  <a href="https://github.com/psf/black"><img src="https://img.shields.io/badge/code%20style-black-000000.svg" alt="Code Style: Black" />
  </a>
  </span>
</div>

A python package for predicting the thermal emission from anisothermal rough planetary surfaces.

## Documentation

See full documentation at [nau-pixel.github.io/roughness](https://nau-pixel.github.io/roughness/)

See usage examples at [nau-pixel.github.io/roughness/examples](https://nau-pixel.github.io/roughness/examples/)

## Installation

To clone and run the package, you'll need [Git](https://git-scm.com) and [Poetry](https://python-poetry.org/docs/) installed on your computer.

```bash
# Clone this repository
$ git clone git@github.com:NAU-PIXEL/roughness.git

# Enter the repository
$ cd roughness

# Install dependencies into a venv with poetry
$ poetry install

# Run setup script (may take awhile)
$ poetry run python setup_roughness.py

# Now you can open a Jupyter server...
$ poetry run python jupyter notebook

# or activate the venv directly from the terminal...
$ poetry shell
$ python

# or activate the venv from your favorite IDE
# The venv is located at ~/roughness/.venv/bin/python
```

## Contribute

This package is a work in progress. We appreciate any and all contributions in the form of bug reports & feature requests on our [issues](https://github.com/NAU-PIXEL/roughness/issues) page, or as pull requests (see [contributing guide](https://github.com/NAU-PIXEL/roughness/tree/main/CONTRIBUTING.md) for more details).

## References and citation

This package is adapted from code by the late Dr. J. L. Bandfield. You can read more about the first iterations of this code in [Bandfield et al. (2015)](https://doi.org/10.1016/j.icarus.2014.11.009) and [Bandfield et al. (2018)](https://doi.org/10.1038/s41561-018-0065-0).

Please cite this software using the DOI of the latest version provided on [Zenodo](https://doi.org/10.5281/zenodo.5498089).

## License

[MIT](https://github.com/NAU-PIXEL/roughness/tree/main/LICENSE). Learn more [here](https://tldrlegal.com/license/mit-license).

Copyright (c) 2023, Christian J. Tai Udovicic
