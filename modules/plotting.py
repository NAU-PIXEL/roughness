"""Helpers for plotting roughness tables and arrays."""
import numpy as np
import xarray as xr
from matplotlib import cm
import matplotlib.pyplot as plt


# Configure plots
# plt.style.use("dark_background")
# plt.style.use("seaborn-paper")
CMAP = cm.get_cmap("magma").copy()
CMAP.set_bad(color="gray")  # set nan color

CMAP_R = cm.get_cmap("magma_r").copy()
CMAP_R.set_bad(color="gray")  # set nan color


def plot_slope_az_table(
    table,
    cmap_r=False,
    clabel=None,
    title=None,
    ax=None,
    proj=None,
    vmin=None,
    vmax=None,
    extent=None,
    **kwargs,
):
    """
    Plot a 2D line-of-sight table with facet slope vs facet azimuth.

    Parameters:
        table (ndarray): A 2D array of slope vs azimuth.
        cmap_r (bool, optional): Use a reverse colormap. Default is False.
        clabel (str, optional): Label for the colorbar. Default is an empty
            string.
        ax (matplotlib.axes.Axes, optional): Axes to plot on. If not provided, 
            a new Axes object will be generated.
        proj (str, optional): The projection to use. Valid values are 'polar'
            and None. Default is None.
        vmin (float, optional): The minimum value for the colorbar. Default is
            None.
        vmax (float, optional): The maximum value for the colorbar. Default is
            None.
    """
    if ax is not None and proj is not None and ax.name != proj:
        msg = f"Ax type {ax.name} must match projection, {proj}."
        raise ValueError(msg)
    if ax is None:
        _, ax = plt.subplots(figsize=(6, 6), subplot_kw={"projection": proj})
    if title is None:
        title = "Facet Slope vs. Azimuth"
    if vmin is None:
        vmin = np.nanmin(table)
    if vmax is None:
        vmax = np.nanmax(table)
    if extent is None:
        extent = (0, 90, 0, 360)
    if isinstance(table, (xr.DataArray, xr.Dataset)):
        table = table.transpose("az", "theta")
        extent = (
            table.theta.min(),
            table.theta.max(),
            table.az.min(),
            table.az.max(),
        )

    if proj == "polar":
        # Define polar coords R ~ slope, Theta ~ azimuth
        r = np.linspace(*extent[:2], table.shape[1]+1)
        theta = np.linspace(*extent[2:], table.shape[0]+1)
        R, Theta = np.meshgrid(r, theta)
        p = ax.pcolormesh(
            np.deg2rad(Theta),
            R,
            table,
            cmap=CMAP_R if cmap_r else CMAP,
            vmin=vmin,
            vmax=vmax,
            shading="auto",
            rasterized=True,
            **kwargs,
        )
        ax.set_theta_zero_location("N")
        ax.set_theta_direction(-1)
        ax.set_rlabel_position(15)
        ax.grid("on", lw=0.1, c="k")
        ax.figure.colorbar(p, ax=ax, shrink=0.8, label=clabel)
    else:
        p = ax.imshow(
            table,
            extent=extent,
            aspect=(extent[1] - extent[0]) / (extent[3] - extent[2]),
            cmap=CMAP_R if cmap_r else CMAP,
            vmin=vmin,
            vmax=vmax,
            interpolation="none",
            **kwargs,
        )
        ax.set_xlabel("Facet slope angle [deg]")
        ax.set_ylabel("Facet azimuth angle [deg]")
        ax.figure.colorbar(p, ax=ax, shrink=0.8, label=clabel)
    ax.set_title(title)
    return p
