# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.12.0
#   kernelspec:
#     display_name: 'Python 3.7.7 64-bit (''.venv'': poetry)'
#     name: python3
# ---

# %% [markdown]
# # Rough emitted radiance spectra
#
# Using `roughness` to predict a rough emitted radiance spectrum from a rough surface, given a particular incidence (sun) vector and emission (spacecraft) vector.

# %%
import numpy as np
import matplotlib.pyplot as plt
from roughness import roughness as rn
from roughness import helpers as rh
from roughness import emission as re
from roughness import plotting as rp

# %% [markdown]
# ## Lunar example - no roughness
#
# Let's assume we're observing the lunar surface at local noon (so we expect no shadowing due to roughness). Our spacecraft is nadir-pointing (emission angle of 0).
#
# We want to use the following vectors to define our sun and spacecraft vectors:
#
# - sun_theta = 0 (noontime sun)
# - sun_az = 0 (arbitrary when nadir, choose $0^\circ$ N)
# - sc_theta = 0 (nadir-pointing spacecraft)
# - sc_az = 0 (arbitrary when nadir, choose $0^\circ$ N)
#
# We will also assume our surface has $rms=30^\circ$ RMS roughness.
#
# We can then predict facet shadowing conditions using the sun vector (`sun_theta`, `sun_az`) using `get_shadow_table()`. We expect there to be no shadows:

# %%
sun_theta = sun_az = sc_theta = sc_az = 0  # [degrees]
rms = 30  # [degrees]

shadow_table = rn.get_shadow_table(rms, sun_theta, sun_az)
ax = rp.plot_slope_az_table(
    shadow_table, True, clabel="P(shadowed)", vmin=0, vmax=1
)
plt.show()

# %% [markdown]
# Next, we can compute the distribution of facets visible from the spacecraft vector (`sc_theta`, `sc_az`) using `get_view_table()`. Looking nadir, all facets should have the same probability of being in view.

# %%
view_table = rn.get_view_table(rms, sc_theta, sc_az)
ax = rp.plot_slope_az_table(view_table, clabel="P(visible)", vmin=0, vmax=1)
plt.show()

# %% [markdown]
# To compute the rough emitted radiance spectrum, we must:
#
# - compute the Planck radiance emitted from each facet at its temperature
# - weight each facet by its probability of being visible from the spacecraft
#
# We will assume that at local noon, all facets have sufficient time to thermally equillibrate and will all have $T \approx 390 K$. We will call the temperature of illuminated facets `temp_illum`.

# %%
wavelengths = np.linspace(1, 80, 1000)  # [microns]
temp_illum = 390  # [K]
rad_illum = re.bbrw(wavelengths, temp_illum)

plt.plot(wavelengths, rad_illum)
plt.show()

# %% [markdown]
# ## Lunar example with roughness
#
# Now let's assume we're observing the lunar surface at local 4 PM (sun_theta=$60^\circ$) with a nadir pointing spacecraft.
#
# - sun_theta = 60
# - sun_az = 270 (Sun from the west in the afternoon)
# - sc_theta = 0
# - sc_az = 0
#
# Once again, our surface has $rms=30^\circ$ RMS roughness.

# %%
sun_theta = 60  # [degrees]
sun_az = 270  # [degrees]
sc_theta = sc_az = 0  # [degrees]
rms = 30  # [degrees]

shadow_table = rn.get_shadow_table(rms, sun_theta, sun_az)
print(f"Mean shadow_table:{np.nanmean(shadow_table):.2%}")
ax = rp.plot_slope_az_table(
    shadow_table, True, clabel="P(shadowed)", vmin=0, vmax=1
)
plt.show()

# %% [markdown]
# Now that we have some facets shadowed, we will have a multi-component emitted radiance spectrum. The simplest is to treat all sunlit facets as a "hot" temperature and all shadowed facets as a "cold" temperature.

# %%
wls = np.linspace(1, 80, 1000)  # [microns]
temp_illum = 390  # [K]
temp_shadow = 200  # [K]
shadow_wls = shadow_table.expand_dims({"wavelength": wls}, axis=2)

rad_rough = re.emission_2component(wls, temp_illum, temp_shadow, shadow_wls)

for i in range(rad_rough.shape[0]):
    for j in range(rad_rough.shape[1]):
        plt.plot(wavelengths, rad_rough[i, j])
plt.show()

# %%
# rms = 18  # Assumed rms roughness value of Moon
albedo = 0.12  # Assumed hemispherical broadband albedo
emiss = 0.985  # Assumed emissivity

solar_dist = 0.98

sun_theta = 40  # [degrees]
sun_az = 270  # [degrees]
sc_theta = 0
sc_az = 90  # [degrees]
rms = 0  # [degrees]

shadow_table = rn.get_shadow_table(rms, sun_theta, sun_az)

# %%
import xarray as xr

wl_arr = np.linspace(1, 50, 100)  # [microns]
wls = xr.DataArray(wl_arr, coords={"wavelength": wl_arr})
rerad = True
flookup = None
# wls = wls.broadcast_like(shadow_table)
# # wls

# %%
f_theta, f_az = rh.facet_grids(shadow_table)
cinc = re.cos_facet_sun_inc(f_theta, f_az, sun_theta, sun_az)

facet_inc = np.degrees(np.arccos(cinc))
albedo_array = re.get_albedo_scaling(facet_inc, albedo)
rad_illum = re.get_emission_eq(cinc, albedo_array, emiss, solar_dist)
rad0 = rad_illum[0, 0]  # Radiance of level surface facet
if rerad:
    alb0 = albedo_array[0, 0]  # Albedo of level surface facet
    rad_illum += re.get_reradiation(
        cinc, f_theta, albedo, emiss, solar_dist, rad0, alb0
    )
temp_illum = re.get_temp_sb(rad_illum)

# Compute temperatures for shadowed facets (JB approximation)
temp_shade = re.get_shadow_temp(sun_theta, sun_az, temp_illum[0, 0])

# Compute 2 component emission of illuminated and shadowed facets
emission_table = re.emission_2component(
    wls, temp_illum, temp_shade, shadow_table
)

# Correct emission for viewing geometry (emission observed by sc)
view_table = rn.get_view_table(rms, sc_theta, sc_az, flookup)
vis_emission_table = emission_table * view_table

# Weight each facet by its probability of occurrance, sum over all facets
facet_weights = rn.get_facet_weights(rms, sun_theta, flookup)
rough_emission = re.emission_spectrum(vis_emission_table, facet_weights)

# %%
tmax = temp_illum.max().values
tmin = temp_illum.where(temp_illum > 0, drop=True).min().values
tshade = temp_shade
p = vis_emission_table.sel(az=270).plot.line(
    x="wavelength", add_legend=False, ls=":"
)
plt.plot(wls, rough_emission, "-", lw=2, label="Rough emission")
plt.plot(wls, re.bbrw(wls, tmax), "r--", label=f"Tmax   : {tmax:.0f} K")
plt.plot(wls, re.bbrw(wls, tmin), "y--", label=f"Tmin   : {tmin:.3f} K")
plt.plot(wls, re.bbrw(wls, tshade), "b--", label=f"Tshadow: {tshade:.0f} K")
# plt.plot(wls, rad_total, 'w-', lw=3, label='total')
plt.title("Radiance of visible slopes and weighted rough emission")
plt.legend()
plt.show()

# %%
facet_btemp_table = re.btempw(wls, vis_emission_table)
tmean = facet_btemp_table.where(facet_btemp_table > 0, drop=True).mean().values

p = facet_btemp_table.sel(az=270).plot.line(
    x="wavelength", add_legend=False, ls=":"
)
plt.plot(wls, re.btempw(wls, rough_emission), "-", lw=2, label="Rough BT")
plt.axhline(
    temp_illum.max(),
    c="r",
    ls="--",
    label=f"Tmax: {temp_illum.max().values:.0f} K",
)
plt.axhline(tmean, c="b", ls="--", label=f"Tavg: {tmean:.0f} K")
plt.title("Facet and rough brightness temperature vs. wavelength")
plt.legend(loc="upper right")
plt.xscale("log")
plt.show()
