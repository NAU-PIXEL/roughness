# Roughness Example Notebooks

Example usage of the `roughness` package. Browse the Jupyter notebook examples in the official documentation (COMING SOON).

## Advanced Users

If you have cloned this repo locally, you can generate local versions of these jupyter notebooks using `jupytext` which is included with the `poetry` dev environment (follow instructions in the roughness [README](../README.md) to set up a poetry environment). Alternatively, you can `pip install jupytext` (not recommended).

From a shell, navigate to the top-level `roughness` directory and run the following command (note the `*`).

```bash
poetry run jupytext --sync ./examples/*
```

This command will build Jupyter notebooks from the jupytext `.py` files in the examples directory which can then be explored and run as any other Jupyter notebook.

If you intend to update a notebook and contribute it back to the repository (e.g. in a pull request), simply run the above `jupytext --sync` command to update the `.py` files and then add and commit the `.py` files (all `.ipynb` files and changes are ignored by default).
