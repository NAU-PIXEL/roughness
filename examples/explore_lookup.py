# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.11.2
#   kernelspec:
#     display_name: 'Python 3.9.4 64-bit (''.venv'': poetry)'
#     name: python3
# ---

# %% [markdown]
# # Explore shadowing lookup tables
#
# Roughness parameterizes shadowing conditions using raytracing lookup tables (see `make_los_table.py`). The lookup table takes solar geometry (inc, az) and surface roughness (RMS) as input and returns the probability that a surface facet of a particular orientation (slope, az) is in shadow.
#
# Below is an interactive visualization of the lookup tables on polar plots where facet orientation is the azimuth (as degrees from North) and facet slope increases radially from flat at the center (0 degrees) to high slopes on the margins (90=vertical).

# %%
import numpy as np
import matplotlib.pyplot as plt
import ipywidgets as ipyw
from ipywidgets import interact
from roughness import roughness as rn
from roughness import plotting as rp
from roughness.config import FLOOKUP

# Load lookups (losfrac, Num los facets, Num facets total)
lookup = rn.load_los_lookup(FLOOKUP)

# Get coord arrays and interactive plot sliders for rms, inc, az
rmss = lookup.rms.values
incs = lookup.inc.values
azs = lookup.az.values
slopes = lookup.theta.values
rms_slider = ipyw.IntSlider(
    20,
    min=rmss.min(),
    max=rmss.max(),
    step=5,
    description="RMS",
    continuous_update=False,
)
inc_slider = ipyw.IntSlider(
    30,
    min=incs.min(),
    max=incs.max(),
    step=5,
    description="Solar inc",
    continuous_update=False,
)
az_slider = ipyw.IntSlider(
    260,
    min=azs.min(),
    max=azs.max(),
    step=15,
    description="Solar az",
    continuous_update=False,
)


# %%
@interact
def plot(rms=rms_slider, inc=inc_slider, az=az_slider):
    fig = plt.figure(figsize=(10, 8))
    ax0 = fig.add_subplot(221, projection="polar")
    ax1 = fig.add_subplot(222)
    ax2 = fig.add_subplot(223, projection="polar")
    ax3 = fig.add_subplot(224)

    # Total facet plot
    table = rn.get_los_table(rms, inc, az, lookup, "total")
    title = f"Binned facet counts (RMS={rms}$^o$)"
    rp.plot_slope_az_table(
        table,
        clabel="Facet count",
        title=title,
        ax=ax0,
        proj="polar",
        vmin=0,
        vmax=None,
    )

    # Slope distribution
    # slopedist = rn.get_facet_weights(rms, inc, lookup).mean('az')
    slopedist = table.mean("az")
    slopedist /= slopedist.sum()
    thetas = slopedist.theta.values
    ax1.plot(thetas, slopedist, label="Roughness")
    thetas = thetas[~np.isnan(slopedist)]
    ax1.plot(thetas, rn.slope_dist(thetas, rms, "rms", "deg"), label="Shepard")
    ax1.plot(thetas, rn.slope_dist(thetas, rms, "tbar", "deg"), label="Hapke")
    ax1.set_xlabel("Facet slope (degrees)")
    ax1.set_ylabel("Normalized bin counts")
    ax1.set_title(f"Normalized slope distribution")
    ax1.legend()

    # Shadowed fraction
    table = rn.get_shadow_table(rms, inc, az, lookup)
    title = f"Shadow probability (RMS={rms}$^o$, inc={inc}$^o$, az={az}$^o$)"
    rp.plot_slope_az_table(
        table,
        cmap_r=True,
        clabel="P(shadowed)",
        title=title,
        ax=ax2,
        proj="polar",
        vmin=0,
        vmax=1,
    )

    # # Shadowing probability vs. theta
    thetas = table.theta.values
    smin = table.interp(az=az).values
    smax = table.interp(az=(az + 180) % 360).values
    smean = table.mean(dim="az").values
    ax3.plot(thetas, smax, label="Max")
    ax3.plot(thetas, smean, label="Mean")
    ax3.plot(thetas, smin, label="Min")
    ax3.set_ylim(0, 1)
    ax3.set_xlabel("Facet slope (degrees)")
    ax3.set_ylabel("Shadow probability")
    ax3.set_title("Shadow probability vs. facet slope")
    ax3.legend()
