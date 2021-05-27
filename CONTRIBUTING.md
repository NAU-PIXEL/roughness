# Contributing

Contributions are welcome, and they are greatly appreciated! Every little bit
helps, and credit will always be given.

You can contribute in many ways:

## Types of Contributions

### Report Bugs

Report bugs at the [issues board](https://github.com/nau-pixel/roughness/issues).

If you are reporting a bug, please include:

* Your operating system name and version.
* Any details about your local setup that might be helpful in troubleshooting.
* Detailed steps to reproduce the bug.

### Fix Bugs

Look through the GitHub issues for bugs. Anything tagged with "bug" and "help
wanted" is open to whoever wants to implement it.

### Implement Features

Look through the GitHub issues for features. Anything tagged with "enhancement"
and "help wanted" is open to whoever wants to implement it.

### Write Documentation

roughness could always use more documentation, whether as part of the
official roughness docs, in docstrings, or even on the web in blog posts,
articles, and such.

### Submit Feedback

The best way to send feedback is to file an issue at [issues board](https://github.com/nau-pixel/roughness/issues).

If you are proposing a feature:

* Explain in detail how it would work.
* Keep the scope as narrow as possible, to make it easier to implement.
* Remember that this is a volunteer-driven project, and that contributions
  are welcome :)

## Setting up a dev environment

First, make sure you have cloned the repo using the instructions in the [README](https://github.com/NAU-PIXEL/roughness#installation) and have run:

`poetry install`

Next, install [pre-commit](https://pre-commit.com/) with:

`poetry run pre-commit install`

Pre-commit will automatically format your code each time you `git commit` and make sure your style will pass checks before pushing to GitHub.

**NOTE:** If the pre-commit checks fail, your changes will not be committed and you will have to `git add` and `git commit` again. To skip pre-commit (e.g. to snapshot a work in progress), use `git commit --no-verify`.

Next, check that your development environment is working with:

`poetry run pytest`

All roughness code will be automatically formatted to the [black](https://black.readthedocs.io/en/stable/) standard upon commit. To run `black` manually:

`poetry run black roughness/ tests/`

Roughness also checks style with [pylint](https://www.pylint.org/). You can run pylint with:

`poetry run pylint roughness/ tests/`

All tests, formatting and linting are checked when you open a GitHub pull request. Make sure to run the above steps before opening a pull request to make sure your contribution is in good shape!

## Contributing examples and Jupyter

Roughness uses [jupytext](https://jupytext.readthedocs.io/en/latest/index.html) to keep track of Jupyter notebooks in Git. Jupytext pairs `.ipynb` files with `.py` files, allowing us to collaborate on Jupyter notebooks via simpler python files. When you first checkout `roughness`, you will need to build Jupyter notebooks in the examples folder locally by running:

`poetry run jupytext --sync examples/*`

Once the notebooks are built, you can edit them in Jupyter as you would normally. Edited examples notebooks will automatically be converted back to their python versions when you make a new commit. The above `--sync` command can also be run at any time to manually sync notebook changes.

**Note:**: All raw `.ipynb` must be converted with `jupytext` before being added to the repository.
