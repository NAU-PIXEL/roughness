# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.11.2
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Interactive plots to explore line of sight / shadowing tables
#
# ## Setup

# %%
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
import ipywidgets as ipyw
from ipywidgets import interact
from roughness import roughness as r
from roughness import make_los_table as mlt

plt.style.use("dark_background")
vis_cmap = cm.get_cmap("magma").copy()
vis_cmap.set_bad(color="gray")  # set nan color

shadow_cmap = cm.get_cmap("magma_r").copy()
shadow_cmap.set_bad(color="gray")  # set nan color

# Load lookups (losfrac, Num los facets, Num facets total)
loslookup = r.load_los_lookup(mlt.FLOS_LOOKUP)
losfacets = r.load_los_lookup(mlt.FLOS_FACETS)
totalfacets = r.load_los_lookup(mlt.FTOT_FACETS)

# Generic plot setup for 4D line of sight tables
def plot_table(table, title, cmap, clabel, ax):
    """Plot 4D los-like table."""
    p = ax.imshow(
        table,
        extent=(0, 90, 0, 360),
        aspect=90 / 360,
        cmap=cmap,
        interpolation="none",
        # vmin=0,
        # vmax=1
    )
    ax.set_title(title)
    ax.set_xlabel("Facet slope angle [deg]")
    ax.set_ylabel("Facet azimuth angle [deg]")
    ax.figure.colorbar(p, ax=ax, shrink=0.8, label=clabel)
    return ax


# %% [markdown]
# ## Shadow table

# %%
# Get coord arrays and interactive plot sliders for rms, inc, az
rmss, incs, azs, slopes = r.get_lookup_coords(loslookup)
rms_slider = ipyw.IntSlider(20, min=rmss.min(), max=rmss.max(), step=1)
inc_slider = ipyw.IntSlider(30, min=incs.min(), max=incs.max(), step=1)
az_slider = ipyw.IntSlider(270, min=azs.min(), max=azs.max(), step=15)


@interact
def plot_shadow_table(rms=rms_slider, inc=inc_slider, az=az_slider):
    """Plot shadowing conditions at rms, inc, az."""
    f, ax = plt.subplots(figsize=(6, 6))
    shadow_table = r.get_shadow_table(rms, inc, az, loslookup)
    title = "Fraction of facets shadowed in slope / az bin"
    clabel = "P(shadowed)"
    ax = plot_table(shadow_table, title, shadow_cmap, clabel, ax)


# %% [markdown]
# ## View table

# %%
# Get coord arrays and interactive plot sliders for rms, inc, az
rmss, incs, azs, slopes = r.get_lookup_coords(loslookup)
rms_slider = ipyw.IntSlider(20, min=rmss.min(), max=rmss.max(), step=1)
inc_slider = ipyw.IntSlider(30, min=incs.min(), max=incs.max(), step=1)
az_slider = ipyw.IntSlider(270, min=azs.min(), max=azs.max(), step=15)


@interact
def plot_view_table(rms=rms_slider, inc=inc_slider, az=az_slider):
    """Plot shadowing conditions at rms, inc, az."""
    f, ax = plt.subplots(figsize=(6, 6))
    view_table = r.get_view_table(rms, inc, az, loslookup)
    title = "Norm prob of visible facets in slope / az bin"
    clabel = "P(visible)/sum(visible)"
    ax = plot_table(view_table, title, vis_cmap, clabel, ax)


# %% [markdown]
# ## Line of sight facets

# %%
# Get coord arrays and interactive plot sliders for rms, inc, az
rmss, incs, azs, slopes = r.get_lookup_coords(losfacets)
rms_slider = ipyw.IntSlider(20, min=rmss.min(), max=rmss.max(), step=1)
inc_slider = ipyw.IntSlider(30, min=incs.min(), max=incs.max(), step=1)
az_slider = ipyw.IntSlider(270, min=azs.min(), max=azs.max(), step=15)


@interact
def plot_los_facets(rms=rms_slider, inc=inc_slider, az=az_slider):
    """Plot shadowing conditions at rms, inc, az."""
    f, ax = plt.subplots(figsize=(6, 6))
    los_facet_table = r.get_los_table(rms, inc, az, losfacets)
    title = "Line of sight facet count in slope / az bin"
    clabel = "N(lineofsight)"
    ax = plot_table(los_facet_table, title, vis_cmap, clabel, ax)


# %% [markdown]
# ## Total facets

# %%
# Get coord arrays and interactive plot sliders for rms, inc, az
rmss, incs, azs, slopes = r.get_lookup_coords(totalfacets)
rms_slider = ipyw.IntSlider(20, min=rmss.min(), max=rmss.max(), step=1)
inc_slider = ipyw.IntSlider(30, min=incs.min(), max=incs.max(), step=1)
az_slider = ipyw.IntSlider(270, min=azs.min(), max=azs.max(), step=15)


@interact
def plot_total_facets(rms=rms_slider, inc=inc_slider, az=az_slider):
    """Plot shadowing conditions at rms, inc, az."""
    f, ax = plt.subplots(figsize=(6, 6))
    total_facet_table = r.get_los_table(rms, inc, az, totalfacets)
    title = "Total facet count in slope / az bin"
    clabel = "N(total)"
    ax = plot_table(total_facet_table, title, vis_cmap, clabel, ax)


# %% [markdown]
# ## All four

# %%
# Get coord arrays and interactive plot sliders for rms, inc, az
rmss, incs, azs, slopes = r.get_lookup_coords(loslookup)
rms_slider = ipyw.IntSlider(20, min=rmss.min(), max=rmss.max(), step=1)
inc_slider = ipyw.IntSlider(30, min=incs.min(), max=incs.max(), step=1)
az_slider = ipyw.IntSlider(270, min=azs.min(), max=azs.max(), step=15)

titles = [
    "Fraction of facets shadowed in slope / az bin",
    "Norm prob of visible facets in slope / az bin",
    "Line of sight facet count in slope / az bin",
    "Total facet count in slope / az bin",
]
clabels = [
    "P(shadowed)",
    "P(visible)/sum(visible)",
    "N(lineofsight)",
    "N(total)",
]


@interact
def plot_all_tables(rms=rms_slider, inc=inc_slider, az=az_slider):
    """Plot shadowing conditions at rms, inc, az."""
    f, axs = plt.subplots(2, 2, figsize=(14, 12))
    tables = [
        r.get_shadow_table(rms, inc, az, loslookup),
        r.get_view_table(rms, inc, az, loslookup),
        r.get_los_table(rms, inc, az, losfacets),
        r.get_los_table(rms, inc, az, totalfacets),
    ]

    for i, ax in enumerate(axs.flatten()):
        cmap = shadow_cmap if i == 0 else vis_cmap
        ax = plot_table(tables[i], titles[i], vis_cmap, clabels[i], ax)
