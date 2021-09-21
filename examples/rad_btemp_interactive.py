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

# %%
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import ipywidgets as ipyw
from ipywidgets import interact
from roughness import roughness as rn
from roughness import helpers as rh
from roughness import emission as re
from roughness import plotting as rp


wl_arr = np.logspace(0, 2, 300)  # [microns]
wls = xr.DataArray(wl_arr, coords={"wavelength": wl_arr})

albedo = 0.12  # Assumed hemispherical broadband albedo
emiss = 0.985  # Assumed emissivity
solar_dist = 1
rms = 18
rms_arr = (0, 6, 12, 18, 24)

sun_theta = 40  # [degrees]
sun_az = 270  # [degrees]
sc_theta = 0
sc_az = 90  # [degrees]
rms = 18  # [degrees]

st_slider = ipyw.IntSlider(20, min=0, max=90, step=10)
sa_slider = ipyw.IntSlider(270, min=0, max=360, step=90)
sct_slider = ipyw.IntSlider(0, min=0, max=90, step=10)
sca_slider = ipyw.IntSlider(270, min=0, max=360, step=90)


# %%
@interact
def plot_re(
    sun_theta=st_slider,
    sun_az=sa_slider,
    sc_theta=sct_slider,
    sc_az=sca_slider,
):
    """Plot rough emission"""
    f, ax = plt.subplots(2, figsize=(8, 6), sharex=True)
    geom = (sun_theta, sun_az, sc_theta, sc_az)
    for rms in rms_arr:
        remiss = re.rough_emission_eq(
            geom, wls, rms, albedo, emiss, solar_dist, True
        )
        temiss = re.btempw(wls, remiss)
        ax[0].semilogx(wls, remiss, label=f"RMS={rms}")
        btemp = re.btempw(wls, remiss)
        ax[1].semilogx(wls, btemp)
    ax[0].legend()
    #     ax[1].set_ylim(-40, 5)
    plt.show()


# %%
def value_at_diviner_channels(xarr):
    """Return value of xarr at each diviner channel."""
    dwls = [
        3,
        7.8,
        8.25,
        8.55,
        (13 + 23) / 2,
        (25 + 41) / 2,
        (50 + 100) / 2,
        (100 + 400) / 2,
    ]  # [microns]
    return xarr.interp({"wavelength": dwls})


# %%
@interact
def plot_rad(
    sun_theta=st_slider,
    sun_az=sa_slider,
    sc_theta=sct_slider,
    sc_az=sca_slider,
):
    """Plot rough radiance and max, shadow facet rad."""
    shadow_table = rn.get_shadow_table(rms, sun_theta, sun_az)
    facet_theta, facet_az = rh.facet_grids(shadow_table)
    cinc = re.cos_facet_sun_inc(facet_theta, facet_az, sun_theta, sun_az)

    facet_inc = np.degrees(np.arccos(cinc))
    albedo_array = re.get_albedo_scaling(facet_inc, albedo)
    rad_illum = re.get_emission_eq(cinc, albedo_array, emiss, solar_dist)
    rad_illum += re.get_reradiation(
        cinc,
        facet_theta,
        albedo,
        emiss,
        solar_dist,
        rad_illum[0, 0],
        albedo_array[0, 0],
    )
    temp_illum = re.get_temp_sb(rad_illum)
    temp0 = temp_illum[0, 0]  # Temperature of level surface facet

    # Compute temperatures for shadowed facets (Josh approximation)
    temp_shade = re.get_shadow_temp(sun_theta, sun_az, temp0)
    facet_emission_table = re.emission_2component(
        wls, temp_illum, temp_shade, shadow_table
    )

    #     view_table = rn.get_view_table(rms, sc_theta, sc_az)
    #     facet_emission_table = facet_emission_table * view_table
    weighted_emission = facet_emission_table * rn.get_facet_weights(
        rms, sun_theta
    )

    rough_emission = weighted_emission.sum(dim=("az", "theta"))
    tmax = re.get_temp_sb(rad_illum.max().values)

    p = facet_emission_table.sel(az=270).plot.line(
        x="wavelength", add_legend=False, ls=":"
    )
    print(value_at_diviner_channels(rough_emission))
    plt.plot(wls, rough_emission, "-", lw=2, label="rough")
    plt.plot(wls, re.bbrw(wls, tmax), "r--", label=f"Tmax   : {tmax:.0f} K")
    #     plt.plot(wls, re.bbrw(wls, temp_shade), 'b--', label=f'Tshadow: {temp_shade:.0f} K')
    plt.legend()
    plt.xscale("log")
    plt.show()


# %%
@interact
def plot_bt(
    sun_theta=st_slider,
    sun_az=sa_slider,
    sc_theta=sct_slider,
    sc_az=sca_slider,
):
    """Plot rough radiance and max, shadow facet rad."""
    f = plt.figure(figsize=(10, 5))
    shadow_table = rn.get_shadow_table(rms, sun_theta, sun_az)
    facet_theta, facet_az = rh.facet_grids(shadow_table)
    cinc = re.cos_facet_sun_inc(facet_theta, facet_az, sun_theta, sun_az)

    facet_inc = np.degrees(np.arccos(cinc))
    albedo_array = re.get_albedo_scaling(facet_inc, albedo)
    rad_illum = re.get_emission_eq(cinc, albedo_array, emiss, solar_dist)
    rerad = re.get_reradiation(
        cinc,
        facet_theta,
        albedo,
        emiss,
        solar_dist,
        rad_illum[0, 0],
        albedo_array[0, 0],
    )
    rad_illum += rerad
    temp_illum = re.get_temp_sb(rad_illum)
    temp0 = temp_illum[0, 0]  # Temperature of level surface facet

    # Compute temperatures for shadowed facets (Josh approximation)
    temp_shade = re.get_shadow_temp(sun_theta, sun_az, temp0)
    facet_emission_table = re.emission_2component(
        wls, temp_illum, temp_shade, shadow_table
    )

    view_table = rn.get_view_table(rms, sc_theta, sc_az)
    facet_emission_table = facet_emission_table * view_table
    weighted_emission = facet_emission_table * rn.get_facet_weights(
        rms, sun_theta
    )

    rough_emission = weighted_emission.sum(dim=("az", "theta"))

    tmax = temp_illum.max().values
    trough = re.btempw(wls, rough_emission)
    facet_btemps = re.btempw(wls, facet_emission_table)
    tavg = facet_btemps.mean().values
    print(value_at_diviner_channels(trough) - trough.interp(wavelength=8.25))

    facet_btemp_table = re.btempw(wls, facet_emission_table)
    p = facet_btemp_table.mean("az").plot.line(
        x="wavelength", add_legend=False, ls=":"
    )
    plt.plot(wls, trough, "-", lw=2, label="rough")
    plt.axhline(tmax, c="r", ls="--", label=f"Tmax   : {tmax:.0f} K")
    plt.axhline(tavg, c="y", ls="--", label=f"Tavg   : {tavg:.0f} K")
    #     plt.axhline(temp_shade, c='b', ls='--', label=f'Tshadow: {temp_shade:.0f} K')
    plt.xscale("log")
    plt.ylim(300, 400)
    plt.legend()
    plt.show()
