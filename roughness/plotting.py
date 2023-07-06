"""Helpers for plotting roughness tables and arrays."""
import numpy as np
import xarray as xr
from matplotlib import cm
import matplotlib.pyplot as plt
import roughness.m3_correction as m3

# Configure plots
# plt.style.use("dark_background")
plt.style.use("seaborn-paper")
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
        extent = (0, 90, 360, 0)
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
        r = np.linspace(*extent[:2], table.shape[1])
        theta = np.linspace(*extent[2:], table.shape[0])
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


# M3 plotting
def m3_imshow(
    img,
    xmin=None,
    xmax=None,
    ymin=None,
    ymax=None,
    wl=750,
    ax=None,
    title=None,
    cmap="gray",
    **kwargs,
):
    """
    Use imshow to display an M3 image. Specify wavelength, wl, to show the
    closest channel to that wavelength.

    Parameters:
        img (ndarray): M3 image.
        xmin, xmax, ymin, ymax (int): The x and y index ranges to plot.
        wl (int, optional): Wavelength to show, gets nearest (default: 750nm).
        ax (matplotlib.axes.Axes, optional): Axes to plot on. If not provided, 
            a new Axes object will be generated.
        title (str, optional): Plot title (default: '')
        cmap (str, optional): Colormap (default: "gray").
    """
    band = 0 if len(img.shape) > 2 else None
    if ax is None:
        _, ax = plt.subplots()
    if len(img.shape) > 2 and img.shape[2] > 1:
        band, wl = m3.m3_wl_to_ind(wl, img)
        if title is None:
            title = f"M3 img: band {band+1} ({wl} nm)"
    p = ax.imshow(img[xmin:xmax, ymin:ymax, band].T, cmap=cmap, **kwargs)
    ax.set_title(title)
    return p


def m3_spec(img, fmt="-", wls=None, ax=None, title=None, **kwargs):
    """
    Plot M3 spectrum.

    Parameters:
        img (ndarray): M3 image.
        fmt (str, optional): Plot format (default: '-' for line).
        wls (ndarray, optional): Wavelengths to plot (default: all).
        ax (matplotlib.axes.Axes, optional): Axes to plot on. If not provided, 
            a new Axes object will be generated.
        title (str, optional): Plot title (default: '')
    """
    if title is None:
        title = "M3 spectrum"
    if ax is None:
        _, ax = plt.subplots()
    if len(img.shape) > 2 and img.shape[2] > 1:
        m3wls, spec = m3.get_avg_spec(img, wls=wls)
    else:
        m3wls, spec = m3.get_spec(img, wls=wls)
    if wls is None:
        wls = m3wls
    ax.plot(wls, spec, fmt, **kwargs)
    ax.set_title(title)
    ax.set_xlabel("Wavelength [microns]")
    return ax


def dem_imshow(
    dem,
    xmin=None,
    xmax=None,
    ymin=None,
    ymax=None,
    band="slope",
    ax=None,
    title="",
    cmap="viridis",
    **kwargs,
):
    """
    Use imshow to display a stacked DEM image (slope, azimuth, elevation).

    Parameters:
        dem (ndarray): A 3D stacked DEM (slope, azimuth, elevation).
        xmin, xmax, ymin, ymax (int): The x and y index ranges to plot.
        band (str or tuple): The band to display. Valid values are 'slope', 
            'azim', 'elev', or a tuple of (0, 1, 2) respectively.
        ax (matplotlib.axes.Axes, optional): Axes to plot on. If not provided, 
            a new Axes object will be generated.
        title (str, optional): Plot title (default: '')
        cmap (str, optional): Colormap (default: "gray").
    """
    zmap = {"slope": 0, "azim": 1, "elev": 2, 0: "slope", 1: "azim", 2: "elev"}
    if len(dem.shape) > 2 and dem.shape[2] > 1:
        if isinstance(band, str) and band in zmap:
            zind = zmap[band]
        elif isinstance(band, int) and band in (0, 1, 2):
            zind = band
        else:
            raise ValueError(f"Unknown band {band} specified.")
        if title is None:
            title = f"DEM {zmap[zind]}"
    else:
        zind = None
    if ax is None:
        _, ax = plt.subplots()
    p = ax.imshow(dem[xmin:xmax, ymin:ymax, zind].T, cmap=cmap, **kwargs)
    ax.set_title(title)
    return p
