# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.11.2
#   kernelspec:
#     display_name: 'Python 3.7.7 64-bit (''.venv'': poetry)'
#     name: python3
# ---

# %% [markdown]
# # Interactive plots to explore line of sight / shadowing tables
#
# ## Setup

# %%
import matplotlib.pyplot as plt
import ipywidgets as ipyw
from ipywidgets import interact
from roughness import plotting as rp
from roughness import roughness as rn
from roughness import make_los_table as mlt

# Load lookups (losfrac, Num los facets, Num facets total)
lookup = rn.load_los_lookup(mlt.FLOOKUP)

# Get coord arrays and interactive plot sliders for rms, inc, az
rmss = lookup.rms.values
incs = lookup.inc.values
azs = lookup.az.values
slopes = lookup.theta.values
rms_slider = ipyw.IntSlider(20, min=rmss.min(), max=rmss.max(), step=1)
inc_slider = ipyw.IntSlider(30, min=incs.min(), max=incs.max(), step=1)
az_slider = ipyw.IntSlider(270, min=azs.min(), max=azs.max(), step=15)


# %% [markdown]
# ## Shadow table

# %%
@interact
def plot_shadow_table(rms=rms_slider, inc=inc_slider, az=az_slider):
    """Plot shadowing conditions at rms, inc, az."""
    shadow_table = rn.get_shadow_table(rms, inc, az, lookup)
    clabel = "P(shadowed)"
    ax = rp.plot_slope_az_table(shadow_table, True, clabel)
    ax.set_title("Fraction of facets shadowed in slope / az bin")


# %% [markdown]
# ## View table

# %%
@interact
def plot_shadow_table(rms=rms_slider, inc=inc_slider, az=az_slider):
    """Plot shadowing conditions at rms, inc, az."""
    view_table = rn.get_view_table(rms, inc, az, lookup)
    clabel = "P(visible)"
    ax = rp.plot_slope_az_table(view_table, True, clabel)
    ax.set_title("Fraction of facets visible in slope / az bin")


# %% [markdown]
# ## Total facets

# %%
@interact
def plot_shadow_table(rms=rms_slider, inc=inc_slider, az=az_slider):
    """Plot shadowing conditions at rms, inc, az."""
    total_facet_table = rn.get_los_table(rms, inc, az, lookup, "prob")
    clabel = "Total facets"
    ax = rp.plot_slope_az_table(total_facet_table, True, clabel)
    ax.set_title("Total facet count in slope / az bin")


# %% [markdown]
# ## Line of sight facets

# %%
@interact
def plot_shadow_table(rms=rms_slider, inc=inc_slider, az=az_slider):
    """Plot shadowing conditions at rms, inc, az."""
    los_facet_table = rn.get_los_table(rms, inc, az, lookup, "los")
    clabel = "LOS facets"
    ax = rp.plot_slope_az_table(los_facet_table, True, clabel)
    ax.set_title("Line of sight facet count in slope / az bin")


# %%
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
    _, axs = plt.subplots(2, 2, figsize=(12, 10))
    tables = [
        rn.get_shadow_table(rms, inc, az, lookup),
        rn.get_view_table(rms, inc, az, lookup),
        rn.get_los_table(rms, inc, az, lookup, "los"),
        rn.get_los_table(rms, inc, az, lookup, "total"),
    ]

    for i, ax in enumerate(axs.flatten()):
        cmap_r = i == 0
        ax = rp.plot_slope_az_table(tables[i], cmap_r, clabels[i], ax)
        ax.set_title(titles[i])


# %%
