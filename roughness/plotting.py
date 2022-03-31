"""Helpers for plotting roughness tables and arrays."""
import numpy as np
from matplotlib import cm
import matplotlib.pyplot as plt
import roughness.m3_correction as m3

# Configure plots
plt.style.use("dark_background")
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
    **kwargs,
):
    """
    Plot 2D (facet slope vs facet azimuth) lineofsight table.

    Parameters
    ----------
    table (ndarray): 2D array of slope vs azimuth.
    cmap_r (bool): Use reverse colormap.
    clabel (str): Label for colorbar.
    ax (matplotlib.axes.Axes): Axes to plot on (default: generate new ax).
    proj (str): Projection to use ('polar', None) (default: None).
    vmin (float): Minimum value for colorbar (default: None).
    vmax (float): Maximum value for colorbar (default: None).
    """
    if ax is not None and ax.name != proj:
        msg = f"Ax type {ax.name} must match projection, {proj}."
        msg += " Redefine ax with projection or remove ax or projection from args."
        raise ValueError(msg)
    if ax is None:
        _, ax = plt.subplots(figsize=(6, 6), subplot_kw={"projection": proj})
    if title is None:
        title = "Facet Slope vs. Azimuth"
    if vmin is None:
        vmin = np.nanmin(table)
    if vmax is None:
        vmax = np.nanmax(table)

    if proj == "polar":
        # Define polar coords R ~ slope, Theta ~ azimuth
        r = np.linspace(0, 90, table.shape[1])
        theta = np.linspace(0, 360, table.shape[0])
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
            extent=(0, 90, 360, 0),
            aspect=90 / 360,
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
    Use imshow to display an M3 image. Specify wavelength, wl, to show a
    the closest channel to that wavelength.

    Parameters
    ----------
    img (ndarray): M3 image.
    xmin (int): Minimum x index to plot.
    xmax (int): Maximum x index to plot.
    ymin (int): Minimum y index to plot.
    ymax (int): Maximum y index to plot.
    wl (int): Wavelength to show (default: 750).
    ax (matplotlib.axes.Axes): Axes to plot on (default: generate new ax).
    title (str): Plot title (default: None).
    cmap (str): Colormap (default: "gray").
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

    Parameters
    ----------
    img (ndarray): M3 image.
    fmt (str): Plot format (default "-" for line).
    wls (ndarray): Wavelengths to plot (default: all).
    ax (matplotlib.axes.Axes): Axes to plot on (default: generate new ax).
    title (str): Plot title (default: None).
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
    title=None,
    cmap="viridis",
    **kwargs,
):
    """
    Use imshow to display a image from a stacked dem (slope, azim, elev).

    Parameters
    ----------
    dem (ndarray): 3D stacked dem (slope, azim, elev).
    xmin (int): Minimum x index to plot.
    xmax (int): Maximum x index to plot.
    ymin (int): Minimum y index to plot.
    ymax (int): Maximum y index to plot.
    band (str): 'slope', 'azim', 'elev' or (0, 1, 2), respectively.
    ax (matplotlib.axes.Axes): Axes to plot on (default: generate new ax).
    title (str): Title for plot.
    cmap (str): Colormap to use.
    """
    zmap = {"slope": 0, "azim": 1, "elev": 2, 0: "slope", 1: "azim", 2: "elev"}
    if len(dem.shape) > 2 and dem.shape[2] > 1:
        if isinstance(band, str) and band in zmap:
            zind = zmap[band]
        elif isinstance(band, int) and band in (0, 1, 2):
            zind = band
        else:
            raise ValueError(f"Unknown band {band} specified.")
    else:
        zind = None
    if title is None:
        title = f"DEM {zmap[zind]}"
    if ax is None:
        _, ax = plt.subplots()
    p = ax.imshow(dem[xmin:xmax, ymin:ymax, zind].T, cmap=cmap, **kwargs)
    ax.set_title(title)
    return p
