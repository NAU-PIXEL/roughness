"""Helpers for plotting roughness tables and arrays."""
from matplotlib import cm
import matplotlib.pyplot as plt

# Configure plots
plt.style.use("dark_background")
CMAP = cm.get_cmap("magma").copy()
CMAP.set_bad(color="gray")  # set nan color

CMAP_R = cm.get_cmap("magma_r").copy()
CMAP_R.set_bad(color="gray")  # set nan color


def plot_slope_az_table(table, cmap_r=False, clabel=None, ax=None):
    """
    Plot 2D (facet slope vs facet azimuth) lineofsight table.

    Parameters
    ----------
    table (ndarray): 2D array of slope vs azimuth.
    cmap_r (bool): Use reverse colormap.
    clabel (str): Label for colorbar.
    ax (matplotlib.axes.Axes): Axes to plot on (default: generate new ax).

    """
    if ax is None:
        _, ax = plt.subplots(figsize=(6, 6))
    p = ax.imshow(
        table,
        extent=(0, 90, 360, 0),
        aspect=90 / 360,
        cmap=CMAP_R if cmap_r else CMAP,
        interpolation="none",
    )
    ax.set_title("Facet Slope vs. Facet Azimuth")
    ax.set_xlabel("Facet slope angle [deg]")
    ax.set_ylabel("Facet azimuth angle [deg]")
    ax.figure.colorbar(p, ax=ax, shrink=0.8, label=clabel)
    return ax
