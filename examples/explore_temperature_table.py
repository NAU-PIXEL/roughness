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
# # Interactive plots to explore temperature table
#
# ## Setup

# %%
# %reset -sf
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
import ipywidgets as ipyw
from ipywidgets import interact
import emission as em

plt.style.use("dark_background")
temp_cmap = cm.get_cmap("magma").copy()
temp_cmap.set_bad(color="gray")  # set nan color

# Load lookups (losfrac, Num los facets, Num facets total)
templookup = em.load_temperature_table()


# Generic plot setup for 4D line of sight tables
def plot_table(table, title, cmap, clabel, ax):
    """Plot 4D los-like table."""
    p = ax.imshow(
        table,
        extent=(0, 90, 360, 0),
        aspect=90 / 360,
        cmap=cmap,
        interpolation="none",
        vmin=templookup.min(),
        vmax=templookup.max()
    )
    ax.set_title(title)
    ax.set_xlabel("Facet slope angle [deg]")
    ax.set_ylabel("Facet azimuth angle [deg]")
    ax.figure.colorbar(p, ax=ax, shrink=0.8, label=clabel)
    return ax


# %% [markdown]
# ## Temperature table

# %%
# Get coord arrays and interactive plot sliders for rms, inc, az
lats, lsts, azs, slopes = em.get_temperature_coords(*templookup.shape)
lat_slider = ipyw.IntSlider(0, min=lats.min(), max=lats.max(), step=1)
lst_slider = ipyw.IntSlider(12, min=lsts.min(), max=lsts.max(), step=1)

@interact
def plot_temp_table(lat=lat_slider, lst=lst_slider):
    """Plot temperature for slope/azim at varying lat/local time"""
    f, ax = plt.subplots(figsize=(10, 10))
    temp_table = em.get_temperature_table(lat, lst, templookup)
    title = "Temperature of surface at each slope / az bin"
    clabel = "T(K)"
    ax = plot_table(temp_table, title, temp_cmap, clabel, ax)
