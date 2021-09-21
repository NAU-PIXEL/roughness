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
# # Generate visible slope distribution from los lookup table
#
# Using a ray-casting line of sight lookup table, return a table of probabilities that a given surface facet (az, slope) is in shadow. The probabilities are interpolated from the lookup table which includes ray-casting runs from a variety of synthetic Gaussian surfaces with varying RMS slopes and local solar incidence angles.
#
# The inputs to the function are:
#
# - `rms_slope`: Root-mean-square slope of a Gaussian slope distribution of surface facets
# - `sun_theta`: To-sun incidence angle w.rn.t. the surface
# - `sun_az`: To-sun azimuth angle w.rn.t. the surface
# - `sc_theta`: To-spacecraft exitance angle w.rn.t. the surface
# - `sc_az`: To-spacecraft azimuth angle w.rn.t. the surface
# - (optional) `los_lookup_path`: Path to custom 4D los lookup file
#
# The default lookup table `los_lookup_4D.npy` predicts shadowing across:
#
# - RMS slopes from 0 to 50 degrees (in 5 degree increments)
# - cos(solar incidence angle) from 0 to 1 (in 0.1 increments)
#
# The table returns a 2D grid of probabilities of slope shadowing corresponding to the following range of facet slope azimuth and angle combinations:
#
# - Slope azimuths measured from North from 0 to 360 degrees (in 10 degree increments)
# - Slope angles measured from a level surface from 0 to 90 degrees (in 2 degree increments)
#
# The table interpolates between the supplied RMS slope (`rms_slope`) and solar incidence angle (`sun_theta`) and returns the 2D grid of shadow probabilities as a function of azimuth and slope angle.
#
# Since shadow lookup was generated with solar azimuth = 270 degrees, we rotate the table such that the shadowing of the returned facets are specified with respect to the supplied solar azimuth (`sun_az`).
#
# Since the observed shadowing distribution will vary based on the observation direction, the shadow facet table is also projected onto the plane perpendicular to the vector of the viewing direction supplied (`sc_theta`, `sc_az`). For example, at 0 phase angle (spacecraft and sun vectors are the same), all shadows from the point of view of the spacecraft are hidden.

# %%
import numpy as np
import matplotlib.pyplot as plt
import ipywidgets as ipyw
from ipywidgets import interact
from roughness import roughness as rn
from roughness import helpers as rh
from roughness import plotting as rp

lookup = rn.load_los_lookup()

# %%
rmss = lookup.rms.values
incs = lookup.inc.values
azs = lookup.az.values
rms_slider = ipyw.IntSlider(20, min=rmss.min(), max=rmss.max(), step=1)
inc_slider = ipyw.IntSlider(30, min=incs.min(), max=incs.max(), step=1)
az_slider = ipyw.IntSlider(270, min=azs.min(), max=azs.max(), step=15)


@interact
def plot_shadow_table(rms=rms_slider, inc=inc_slider, az=az_slider):
    """Plot shadowing conditions at rms, inc, az."""
    shadow_table = rn.get_shadow_table(rms, inc, az, lookup)
    ax = rp.plot_slope_az_table(shadow_table, clabel="P(shadowed)")
    ax.set_title("Fraction of facets shadowed at (slope / az)")


# %%
# Get raw shadow table with solar incidence angle below
rms_slope = 20
sun_theta = 60
sun_az = 270

shadow_table = rn.get_shadow_table(rms_slope, sun_theta, sun_az, lookup)
ax = rp.plot_slope_az_table(shadow_table, clabel="P(shadowed)")
ax.set_title(f"Facet shadow fraction RMS={rms_slope}$^o$ inc={sun_theta}$^o$")

# %%
# Sanity check on facet vectors (slope in 0-90, az in 0-360)
theta_surf, azimuth_surf = rh.facet_grids(shadow_table)
cartesian_surf = rh.sph2cart(np.radians(theta_surf), np.radians(azimuth_surf))

fig = plt.figure()
ax = plt.axes(projection="3d")
ax.view_init(45, 60)

nrms, ninc, nslope, naz = lookup.prob.shape
X, Y, Z = cartesian_surf.reshape((nslope * naz, 3)).T + 1e-8
O = np.zeros(len(X))  # origin

# Color by az angle
c = np.arctan2(Y, X)
# Flatten and normalize
c = (c.ravel() - c.min()) / c.ptp()
# Repeat for each body line and two head lines
c = np.concatenate((c, np.repeat(c, 2)))
# Colormap
c = plt.cm.hsv(c)
# Subsample because there are a lot of vectors
s = slice(None, None, 11)
q = ax.quiver(
    O[s],
    O[s],
    O[s],
    X[s],
    Y[s],
    Z[s],
    colors=c[s],
    length=0.08,
    normalize=True,
)


# %%
# Transcibed the "correction" steps from TI_Scripts. This
# part should do the view angle correction but... looks weird
sc_theta = 20
sc_az = 270
theta, az = (np.radians(30), np.radians(sc_az))

cartesian_sc = rh.sph2cart(theta, az)
cos_surf_sc = np.dot(cartesian_surf, cartesian_sc)

angle = np.arccos(cos_surf_sc)
angle[angle > np.pi / 2] = np.pi / 2
corr = (1 / np.cos(theta_surf)) * np.cos(angle)

fig = plt.figure()
ax = plt.axes(projection="3d")
ax.view_init(30, 70)

ax.plot_surface(theta_surf, azimuth_surf, corr, cmap="viridis")

# %%
# Get shadow table from lookup
rms_slope = 20
solar_az = 270

fig = plt.figure(figsize=(15, 8))
sun_thetas = (0, 30, 60)
for i, solar_inc in enumerate(sun_thetas):
    shadow_table = rn.get_shadow_table(rms_slope, solar_inc, solar_az, lookup)
    ax = fig.add_subplot(1, len(sun_thetas), i + 1, projection="3d")
    ax.view_init(35, -45)
    ax.plot_surface(
        theta_surf,
        azimuth_surf,
        1 - shadow_table,
        cmap="viridis",
        vmin=0,
        vmax=1,
    )
    ax.set_title(
        f"RMS={rms_slope}$^o$ solar_inc={solar_inc}$^o$ (az={solar_az}$^o$N)"
    )
    ax.set_xlabel("Facet slope ($^o$)")
    ax.set_ylabel("Facet azimuth ($^o$)")
    ax.set_zlabel("P(shadowed)")
    ax.set_zlim(0, 1)

# plt.savefig('./shadow_tables.png', dpi=300)

# %%
# Compare rms slopes
solar_inc = 60
solar_az = 270

fig = plt.figure(figsize=(15, 8))
rms_slopes = (0, 20, 40)
for i, rms_slope in enumerate(rms_slopes):
    shadow_table = rn.get_shadow_table(rms_slope, solar_inc, solar_az, lookup)
    ax = fig.add_subplot(1, len(sun_thetas), i + 1, projection="3d")
    ax.view_init(35, 135)
    ax.plot_surface(
        theta_surf, azimuth_surf, shadow_table, cmap="viridis", vmin=0, vmax=1
    )
    ax.set_title(
        f"RMS={rms_slope}$^o$ solar_inc={solar_inc}$^o$ (az={solar_az}$^o$N)"
    )
    ax.set_xlabel("Facet slope ($^o$)")
    ax.set_ylabel("Facet azimuth ($^o$)")
    ax.set_zlabel("P(shadowed)")
    ax.set_zlim(0, 1)

# plt.savefig('./shadow_tables.png', dpi=300)

# %%
# Get probability of certain facets, given rms slope dist
slope_correction = rn.slope_dist(np.radians(theta_surf), np.radians(rms_slope))

fig = plt.figure()
ax = plt.axes(projection="3d")
ax.view_init(45, -75)
ax.plot_surface(theta_surf, azimuth_surf, slope_correction, cmap="viridis")
