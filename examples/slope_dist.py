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
# # Slope distributions
#
# Several examples of slope distributions computed from analytic equations and raytracing binning code. Most plots show probablility of a surface slope occurring vs facet slope in [0, 90] degrees (where 0 degrees is a flat, level facet).

# %%
import numpy as np
import matplotlib.pyplot as plt
from roughness import config as cfg
from roughness import roughness as rn
from roughness import helpers as rh

plt.style.use("dark_background")
SAVEFIGS = False
lookup = rn.load_los_lookup(cfg.FLOOKUP)

# %% [markdown]
# ## RMS (Shepard 1995)
#
# Analytical RMS slope distribution using equations from Shepard (1995).

# %%
theta = np.arange(90)
rms_arr = (5, 10, 20, 30, 40, 50)

plt.figure()
for rms in rms_arr:
    p_theta = rn.slope_dist(np.radians(theta), np.radians(rms), "rms")
    plt.plot(theta, p_theta, label=f"RMS$={rms}^o$")
plt.title("Shepard gaussian slope distribution vs RMS")
plt.ylabel("$P(\\theta)$")
plt.xlabel("Facet $\\theta$ angle [deg]")
plt.xlim(0, 90)
plt.ylim(0, 0.15)
plt.legend(ncol=2)
if SAVEFIGS:
    plt.savefig(cfg.FIG_SLOPE_DIST_SHEP, dpi=300)

# %% [markdown]
# ## Theta-bar (Hapke 1984)
#
# Analytical slope distributions using equations from Hapke (1984).

# %%
theta = np.arange(90)
tbar_arr = (5, 10, 20, 30, 40, 50)

plt.figure()
for tbar in tbar_arr:
    p_theta = rn.slope_dist(np.radians(theta), np.radians(tbar), "tbar")
    label = "$\\bar{\\theta}=$"
    plt.plot(theta, p_theta, label=label + f"${tbar}^o$")
plt.title("Hapke gaussian slope distributions vs theta-bar")
plt.ylabel("$P(\\theta)$")
plt.xlabel("Facet $\\theta$ angle [deg]")
plt.xlim(0, 90)
plt.ylim(0, 0.15)
plt.legend(ncol=2)
if SAVEFIGS:
    plt.savefig(cfg.FIG_SLOPE_DIST_HAPKE, dpi=300)

# %% [markdown]
# ## Slope distributions from lineofsight lookup tables

# %%
rms_arr = (5, 10, 20, 30, 40, 50)

plt.figure()
for rms in rms_arr:
    facet_table = rn.get_los_table(rms, 0, 270, lookup, "total")
    p_theta = np.sum(facet_table, axis=0) / np.nansum(facet_table)
    plt.plot(facet_table.theta, p_theta, label=f"RMS$={rms}^o$")
plt.title("Synthetic gaussian surface RMS rough slope distributions")
plt.ylabel("$P(\\theta)$")
plt.xlabel("Facet $\\theta$ angle [deg]")
plt.xlim(0, 90)
plt.ylim(0, None)
plt.legend(ncol=2)
if SAVEFIGS:
    plt.savefig(cfg.FIG_SLOPE_DIST_GSURF, dpi=300)

# %% [markdown]
# ## Visible facets from lineofsight lookup vs RMS
#
# Viewing azimuth is particularly important for higher RMS values and shows the distinction between syn-facing slopes (surface facets oriented towards the spacecraft) and anti-facing slopes (surface facets orented away from the spacecraft).

# %%
azs = lookup.az.values
theta = lookup.theta.values
rms_arr = (5, 10, 20, 30, 40, 50)
sc_theta = 60
sc_az = 270

plt.figure()
for rms in rms_arr:
    # Compute prob of totalfacets being visible from sc_az
    facet_table = rn.get_los_table(rms, sc_theta, sc_az, lookup, "total")
    view_table = rn.get_view_table(rms, sc_theta, sc_az, lookup)
    vis_facet_table = facet_table * view_table

    # Get syn facets facing sc_az and anti facets 180 degrees from sc_az
    p_theta270 = vis_facet_table[np.argmin(np.abs(azs - 270))]
    p_theta90 = vis_facet_table[np.argmin(np.abs(azs - 90))]

    # Normalize and plot both curves
    p_theta270 = p_theta270 / np.nansum(p_theta270)
    p_theta90 = p_theta90 / np.nansum(p_theta90)
    (line,) = plt.plot(theta, p_theta270, label=f"RMS={rms}$^o$")
    plt.plot(theta, p_theta90, ls=":", lw=3, c=line.get_color())
plt.title(f"Visible slopes vs RMS (with view angle $\\theta$={sc_theta}$^o$)")
plt.ylabel("$P(\\theta)$")
plt.xlabel("Facet $\\theta$ angle [deg]")
plt.xlim(0, 90)
plt.ylim(0, None)
# Make label lines for legend
plt.plot([0, 1e-3], [0, 0], "w-", label="syn facets")
plt.plot([0, 1e-3], [0, 0], "w:", lw=3, label="anti facets")
plt.legend()
if SAVEFIGS:
    plt.savefig(cfg.FIG_SLOPE_DIST_VIS_RMS, dpi=300)

# %% [markdown]
# ## Visible facets from lineofsight lookup vs view angle
#
# Viewing angle (spacecraft emission angle) mainly affects anti-facing slopes (surface facets orented away from the spacecraft), but has little effect on syn-facing slopes (surface facets oriented towards the spacecraft), even at high roughness.

# %%
az = lookup.az.values
theta = lookup.theta.values
sc_thetas = (0, 15, 30, 45, 60, 75)
rms = 40
sc_az = 270

plt.figure()
for sc_theta in sc_thetas:
    # Compute prob of totalfacets being visible from sc_az
    facet_table = rn.get_los_table(rms, sc_theta, sc_az, lookup, "total")
    view_table = rn.get_view_table(rms, sc_theta, sc_az, lookup)
    vis_facet_table = facet_table * view_table

    # Get syn facets facing sc_az and anti facets 180 degrees from sc_az
    p_theta270 = vis_facet_table[np.argmin(np.abs(azs - 270))]
    p_theta90 = vis_facet_table[np.argmin(np.abs(azs - 90))]

    # Normalize and plot both curves
    p_theta270 = p_theta270 / np.nansum(p_theta270)
    p_theta90 = p_theta90 / np.nansum(p_theta90)
    (line,) = plt.plot(
        theta, p_theta270, label=f"view $\\theta$={sc_theta}$^o$"
    )
    plt.plot(theta, p_theta90, ls=":", lw=3, c=line.get_color())
plt.title(f"Visible slopes vs view angle (with RMS={rms}$^o$)")
plt.ylabel("$P(\\theta)$")
plt.xlabel("Facet $\\theta$ angle [deg]")
plt.xlim(0, 90)
plt.ylim(0, None)

# Make label lines for legend
plt.plot([0, 1e-3], [0, 0], "w-", label="syn facets")
plt.plot([0, 1e-3], [0, 0], "w:", lw=3, label="anti facets")
plt.legend()
if SAVEFIGS:
    plt.savefig(cfg.FIG_SLOPE_DIST_VIS_THETA, dpi=300)
