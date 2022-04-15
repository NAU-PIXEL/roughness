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
import matplotlib as mpl
import matplotlib.pyplot as plt
import ipywidgets as ipyw
from ipywidgets import interact
from roughness import roughness as rn
from roughness import diviner as rd
from roughness import helpers as rh
from roughness import emission as re
from roughness import plotting as rp

# tlookup = "/home/ctaiudovicic/projects/MiscDebris/data/220319_temp_table.nc"
tlookup = "/home/ctaiudovicic/projects/roughness/data/temp_lookup.nc"


def emission2divbt(wls, emission):
    wnrad = re.wlrad2wnrad(wls, emission)
    div_bt = rd.divrad2bt(rd.divfilt_rad(wnrad))
    tbol = rd.div_tbol(div_bt)
    return div_bt, tbol


div_filt = rd.load_div_filters()
wls = div_filt.wavelength
div_wls = rd.get_diviner_wls()

# albedo = 0.12  # Assumed hemispherical broadband albedo
# emiss = 0.95  # Assumed emissivity
# solar_dist = 1
# rms = 18
rms_arr = (0, 5, 10, 20, 30)

# sun_theta = 40  # [degrees]
# sun_az = 270  # [degrees]
# sc_theta = 0
# sc_az = 90  # [degrees]
# rms = 18  # [degrees]

# st_slider = ipyw.IntSlider(20, min=0, max=90, step=10, description="Solar inc", continuous_update=False)
# sa_slider = ipyw.IntSlider(270, min=0, max=360, step=45, description="Solar az", continuous_update=False)
tloc_slider = ipyw.IntSlider(
    12, min=6, max=18, step=1, description="Time", continuous_update=False
)
sct_slider = ipyw.IntSlider(
    0,
    min=0,
    max=90,
    step=10,
    description="Spacecraft inc",
    continuous_update=False,
)
sca_slider = ipyw.IntSlider(
    270,
    min=0,
    max=360,
    step=45,
    description="Spacecraft az",
    continuous_update=False,
)
lat_slider = ipyw.IntSlider(
    0, min=-85, max=85, step=5, description="Latitude", continuous_update=False
)
rerad_box = ipyw.Checkbox(
    value=False, description="Reradiation", disabled=False
)

rms_slider = ipyw.IntSlider(
    20, min=0, max=50, step=15, description="RMS", continuous_update=False
)
albedo_slider = ipyw.FloatSlider(
    0.12,
    min=0.045,
    max=0.22,
    step=0.05,
    description="Albedo",
    continuous_update=False,
)

cmap = plt.cm.get_cmap("viridis")
norm = mpl.colors.Normalize(vmin=min(rms_arr), vmax=max(rms_arr))
colors = [cmap(norm(r)) for r in rms_arr]


# %%
@interact
def plot_re(
    tloc=tloc_slider,
    albedo=albedo_slider,
    sc_theta=sct_slider,
    sc_az=sca_slider,
    lat=lat_slider,
    rerad=rerad_box,
):
    """Plot rough emission"""
    f, ax = plt.subplots(2, figsize=(8, 6), sharex=True)
    sun_theta, sun_az = rh.tloc_to_inc(tloc, lat, True)
    geom = (sun_theta, sun_az, sc_theta, sc_az)
    for r, rms in enumerate(rms_arr):
        emission = re.rough_emission_lookup(geom, wls, rms, albedo, lat, rerad)
        btemp = re.btempw(wls, re.mrad2cmrad(emission))
        div_bt, tbol = emission2divbt(wls, emission)
        ax[0].semilogx(wls, emission, c=colors[r], label=f"RMS={rms}$^o$")
        ax[1].semilogx(wls, btemp, c=colors[r])
        ax[1].semilogx(div_wls, div_bt, "o", c=colors[r])
        ax[1].axhline(
            tbol, c=colors[r], linestyle="--", label="$T_{bol}$={tbol:.2f}"
        )

    ax[0].legend()
    ax[0].set_title(
        f"Rough rad and bt {sun_theta = :.2f}$^o$, {sun_az = :.2f}$^o$"
    )
    ax[0].set_ylabel("Rough Emission [W/m$^2$]")
    ax[1].set_ylabel("Brightness Temperature [K]")
    ax[1].set_xlabel("Wavelength [$\mu m$]")
    plt.show()


# %%
@interact
def plot_re(
    rms=rms_slider,
    albedo=albedo_slider,
    tloc=tloc_slider,
    sc_theta=sct_slider,
    sc_az=sca_slider,
    lat=lat_slider,
    rerad=rerad_box,
):
    """Plot shadow and temp tables"""
    fig = plt.figure(figsize=(10, 8))
    ax0 = fig.add_subplot(221, projection="polar")
    ax1 = fig.add_subplot(222, projection="polar")
    ax2 = fig.add_subplot(223, projection="polar")
    ax3 = fig.add_subplot(224, projection="polar")
    sun_theta, sun_az = rh.tloc_to_inc(tloc, lat, True)
    geom = (sun_theta, sun_az, sc_theta, sc_az)
    # fig.suptitle(f"Solar inc={sun_theta:.2f}$^o$, Solar az={sun_az:.2f}$^o$)")
    # Shadow table
    shadow_table = rn.get_shadow_table(rms, sun_theta, sun_az)
    title = "Shadow probability (rms, sun_theta, sun_az)"
    rp.plot_slope_az_table(
        shadow_table,
        cmap_r=True,
        clabel="P(shadowed)",
        title=title,
        ax=ax0,
        proj="polar",
        vmin=0,
        vmax=1,
    )

    # Temp table
    temp_table = re.get_temp_table(tloc, albedo, lat)
    tflat = temp_table.sel(theta=0, az=0)
    temp_table = temp_table.interp_like(shadow_table)

    title = f"Temperature (tloc, albedo, lat)"
    rp.plot_slope_az_table(
        temp_table,
        clabel="Temperature [K]",
        title=title,
        ax=ax1,
        proj="polar",
    )

    # Shadow temp table
    tshade = re.get_shadow_temp(sun_theta, sun_az, tflat)
    temp_shade = re.get_shadow_temp_table(temp_table, sun_theta, sun_az, tflat)
    title = (
        f"Shadow temp (tflat={tflat.values:.2f}K, tshade={tshade.values:.2f}K)"
    )
    rp.plot_slope_az_table(
        temp_shade,
        clabel="Temperature [K]",
        title=title,
        ax=ax2,
        proj="polar",
    )
    plt.show()


# %%
