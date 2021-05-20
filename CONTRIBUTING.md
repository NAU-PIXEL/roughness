# Contributing

Contributions are welcome, and they are greatly appreciated! Every little bit
helps, and credit will always be given.

You can contribute in many ways:

## Types of Contributions

### Report Bugs

Report bugs at https://github.com/nau-pixel/roughness/issues.

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

The best way to send feedback is to file an issue at https://github.com/nau-pixel/roughness/issues.

If you are proposing a feature:

* Explain in detail how it would work.
* Keep the scope as narrow as possible, to make it easier to implement.
* Remember that this is a volunteer-driven project, and that contributions
  are welcome :)

## Contributing Code

First, make sure you have a environment set up with `poetry` using the instructions in the [README](https://github.com/NAU-PIXEL/roughness).

To check that your installation is working, you can run all of the tests from the main `roughness` directory with:

`poetry run pytest`

Roughness uses [black](https://black.readthedocs.io/en/stable/) to format python code. this can be done automatically by running:

`poetry run black roughness/ tests/`

Roughness also does style checking with pylint. You can run pylint with:
  
`poetry run pylint roughness/ tests/`

All tests, black formatting and linting are checked automatically when you open a GitHub pull request. You will see if your PR whether or not these tests passed. It is good practice to run the above steps locally before every push to make sure everything looks good!