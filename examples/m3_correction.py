# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.11.2
#   kernelspec:
#     display_name: 'Python 3.7.3 64-bit (''enthalpy-dev'': conda)'
#     language: python
#     name: python37364bitenthalpydevconda07243e6a90064f55a3b050dd2a1302a7
# ---

# %% [markdown]
# # Test m3_refl correction
#

# %%
import os

os.chdir("/home/ctaiudovicic/projects/roughness")
import numpy as np
import matplotlib.pyplot as plt
from roughness import roughness as r
from roughness import m3_correction as m3
import enthalpy.io as io
import enthalpy.plotting as ep

# %%
pdem = "/work/ctaiudovicic/data/Kaguya/"
fdem = [
    pdem + f"lola_kaguya_dem_{im}cyl_128.tif" for im in ("slope_", "azim_", "")
]

tifdir = "/work/team_moon/data/warped/210202/"
basename = "M3G20090209T212512"  # 'M3G20090419T050945
# ext = (-56.35, -55.5, 8, 10)
ext = (-56.35, -56, 8.8, 9)

# Read rad image and dem
hrds = io.get_m3_headers(basename)
rad = io.read_m3_geotiff(tifdir + basename + "_rad.tif", ext)
obs = io.read_m3_geotiff(tifdir + basename + "_obs.tif", ext)
width, height = rad.shape[:2]
dem = io.get_dem_geotiff(*fdem, ext, width, height, "bilinear")

if True:
    f, ax = plt.subplots(1, 3)
    ep.m3_imshow(rad, ax=ax[0])
    ep.dem_imshow(obs[:, :, 7:], ax=ax[1])
    ep.dem_imshow(dem, ax=ax[2])

# %%
import enthalpy.enthalpy as ee

old_savefile = "tmp_corr.npy"
fimgs = [tifdir + basename + im + ".tif" for im in ("_rad", "_obs", "_sup")]
old_corr = ee.tcorr_geotiff(fimgs, basename, ext, old_savefile)

# %%
old_orig, old_ref = old_corr["ref_orig_corr"], old_corr["ref_corr"]
srad, sobs, sdem = rad[:10, :10], obs[:10, :10], dem[:10, :10]
rad_corr = m3.m3_refl(basename, srad, sobs, sdem)
# rad_corr = m3.m3_refl(basename, rad, obs, dem)

if True:
    f, axs = plt.subplots(2, 2)
    ax = axs.flatten()
    ep.m3_imshow(rad_corr, wl=2900, ax=ax[0])
    ep.m3_imshow(old_orig, wl=2900, ax=ax[1])
    ep.m3_imshow(old_ref, wl=2900, ax=ax[2])
    ep.dem_imshow(sdem, ax=ax[3])

# %%
# TODO: New refl way higher, how come? emission or I/F step?
(rad_corr - old_ref)[5, 5, :]

# %% [markdown]
# ## Step thru m3_refl

# %%
import multiprocessing as mp
from enthalpy.enthalpy import (
    m3photom,
    tremove,
    get_albedo,
    T_pred_mp_helper,
    T_predict_eq,
    bbrw,
    btemp,
    btempw,
    wl2wn,
    T_eq,
    tremove,
    rad_eq,
    correction,
    bbr,
)
import enthalpy.classes as ec
import enthalpy.helpers as h
from numpy import newaxis as nax

m3img = ec.M3Image(basename)
fimgs = [tifdir + basename + im + ".tif" for im in ("_rad", "_obs", "_sup")]
imgs = [io.read_m3_geotiff(f, ext)[:10, :10, :] for f in fimgs]
imgs[1][:, :, 5] += io.get_sdist(m3img.hdrs.obs)  # fix obs sdist

# Get DEM resampled to img size
fdem = (m3img.dem_slope_path, m3img.dem_azim_path, m3img.dem_elev_path)
width, height = imgs[0].shape[:2]
dem = io.get_dem_geotiff(*fdem, ext, width, height, "bilinear")

# Args to pass to m3_refl_CTU()
rad, obs, sup = imgs
shade_lookup = m3img.sl
slope = m3img.slope
isalpha = m3img.alpha
polish = m3img.polish
emiss = m3img.emiss
m3ancpath = m3img.m3ancpath
full_output = False
chunk = ""
procs = mp.cpu_count() - 1

# %%
# def m3_refl_CTU():
# Constants
targeted = h.is_targeted(rad)
dimx, dimy, dimz = rad.shape

# Wavelength array
wl = h.get_m3_wls(rad) / 1000  # [um]
# Ground vector norm to local slope (polar)
if dem is not None and dem.any():
    ground_zen = np.arctan(dem[:, :, 0])
    ground_az = dem[:, :, 1]
else:
    ground_zen = np.radians(obs[:, :, 7])
    ground_az = np.radians(obs[:, :, 8])
# Sun vector (polar)
sun_zen = np.radians(obs[:, :, 1])
sun_az = np.radians(obs[:, :, 0])
# Spacecraft vector (polar)
sc_zen = np.radians(obs[:, :, 3])
sc_az = np.radians(obs[:, :, 2])
# Solar distance and solar spectrum
sd = obs[:, :, 5]
ss = np.tile(h.get_solar_spectrum(targeted), (dimx, dimy, 1))

# Get inc, em, and az at each pixel, given topography
inc, em, phase, az = h.get_geometry(
    ground_zen, ground_az, sun_zen, sun_az, sc_zen, sc_az
)

# Do M3 temperature and photometric correction on rad, get albedo
photom = m3photom(inc, em, phase, isalpha, polish, m3ancpath, targeted)
ref_orig = tremove(rad, ss, sd) * photom
albedo = get_albedo(ref_orig, ss)


# T_predict_eq():
alb_arr = albedo
xinds, yinds = np.nonzero(rad[:, :, 5])
t_pred = np.zeros((dimx, dimy, dimz))
for i, j in zip(xinds, yinds):
    slope_dist = slope
    emission = em[i, j]
    azim = az[i, j]
    sdist = sd[i, j]
    lat = -inc[i, j]
    tloc = 12
    albedo = alb_arr[i, j]
    emiss = emiss
    shade_slope_dist = slope
    slope_type = "rms"
    ws = wl2wn(wl)
    shade_temp = 100
    shade_lookup = shade_lookup
    alb_scaling = 1
    # t_pred[i, j, :] = T_predict_eq(slope, em[i, j], az[i, j], sd[i, j], -inc[i, j], 12, albedo[i, j], emiss, slope, 'rms', wl2wn(wl), 100, shade_lookup, 1)['temp']
    if isinstance(emission, int) or isinstance(emission, float):
        emission = [emission]
    if isinstance(azim, int) or isinstance(azim, float):
        azim = [azim]

    sl = shade_lookup
    limit = 0

    solar_inc = h.tloc_to_inc(tloc, lat)
    solar_azim = h.tloc_to_az(tloc, lat)

    # Account for nan's because davinci is stupid
    if solar_azim > 360:
        if lat > 0:
            solar_azim = 180
        else:
            solar_azim = 0

    tmap_lookup = rad_eq(
        sdist,
        lat,
        tloc,
        albedo,
        emiss,
        sl=-1,
        az=-1,
        shade_temp=shade_temp,
        alb_scaling=alb_scaling,
    )

    if not (limit <= emission[0] < 85):
        w = f"Emission {emission[0]} out of bounds ({limit}, 85)"
        warnings.warn(w, RuntimeWarning)
        # return np.zeros(len(ws))
    else:
        corr_dict = correction(
            slope_dist,
            emission[0],
            azim[0],
            slope_type=slope_type,
            nangles=46,
            azstep=10,
        )

        corr = corr_dict["correction"]
        corr = np.concatenate((corr[-1:], corr))
        corr_dict["correction"] = corr
        if slope_dist >= 50:
            shade_lookup = sl["rms50"]
        else:
            weight = 1 - (slope_dist % 5) / 5
            table_ind = int(slope_dist - slope_dist % 5)
            key1 = "rms{:02d}".format(table_ind)
            key2 = "rms{:02d}".format(table_ind + 5)
            if isinstance(sl, str):
                print(sl)
            shade_lookup2 = sl[key1] * weight + sl[key2] * (1 - weight)

        cinc = 1 - np.cos(np.radians(solar_inc))
        table_ind = int(cinc // 0.1)
        weight = 1 - ((cinc * 10) - int(cinc * 10))  # 1 - remainder
        shade_table = (shade_lookup2[:, :, table_ind] * weight) + (
            shade_lookup2[:, :, table_ind + 1] * (1 - weight)
        )
        shift = int(round((solar_azim * 0.1) - 27))

        if solar_azim < 270:
            shift = 0 - shift
        elif solar_azim > 270:
            shift = 36 - shift

        if 0 < shift < 36:
            shade_table2 = np.concatenate(
                (shade_table[shift:36], shade_table[0:shift])
            )
            shade_table2 = np.concatenate((shade_table2, shade_table2[0:1]))

        dimx, dimy = tmap_lookup.shape
        if len(ws.shape) == 1:
            ws_arr = np.tile(ws[nax, nax, :], (dimx, dimy, 1))
        else:
            ws_arr = ws
        illum_arr = bbr(ws_arr, tmap_lookup[:, :, nax])
        if shade_slope_dist:
            bbr_shade = bbr(ws, shade_temp)
            shade_arr = bbr_shade
            tmap = (
                illum_arr * (1 - shade_table2[:, :, nax])
                + shade_arr * shade_table2[:, :, nax]
            )
        else:
            tmap = illum_arr

        tmap[tmap < 0] = 0
        tmap[tmap > 1000000] = 0

        temp = np.sum(tmap[:-1] * corr[:-1, :, nax], axis=(0, 1)) / np.sum(
            corr[:-1]
        )

        t_pred[i, j] = temp

        tmp = {}
        # tmp['solar_spec'] = ss
        # tmp['geom'] = (inc, em, phase, az)
        # tmp['photom'] = photom
        # tmp['albedo'] = albedo
        # tmp['t_pred'] = t_pred
        tmp["correction"] = corr_dict["correction"]
        tmp["shade_table"] = shade_table
        tmp["illum_arr"] = illum_arr
        tmp["shade_arr"] = shade_arr
        break

# dimx, dimy, _ = rad.shape
# # Derive emission due to roughness
# bbr_rough = bbrw(wl, btemp(wl2wn(wl), t_pred)) * 10000
# wl_last = np.tile(wl[-1], (dimx, dimy, 1))
# t_rough = btempw(wl_last, bbr_rough[:, :, -1:]/10000)
# wl_arr = np.tile(wl, (dimx, dimy, 1))

# # Get isothermal surface (essentially rms slope = 0) correction
# t_eq = T_eq(em, emiss, albedo, sd)
# bbr_isot = bbrw(wl_arr, np.tile(t_eq[:, :, nax], (1, 1, dimz)))*10000

# # Get M3 temps from the sup file
# if sup.shape[2] > 1:
#     t_sup = sup[:, :, 1]
# bbr_sup = bbrw(wl_arr, np.tile(t_sup[:, :, nax], (1, 1, dimz)))*10000

# # Do temperature and photometric correction on rad
# ref_orig_corr = tremove(rad, ss, sd, bbr_sup) * photom
# ref_isot_corr = tremove(rad, ss, sd, bbr_isot) * photom
# ref_corr = tremove(rad, ss, sd, bbr_rough) * photom

# # Return
# out = {}
# out['dem'] = dem
# out['ref_orig'] = ref_orig
# out['T_orig'] = t_sup
# out['ref_orig_corr'] = ref_orig_corr
# out['T_isot'] = t_eq
# out['ref_isot_corr'] = ref_isot_corr
# out['T_rough'] = t_rough[:, :, 0]
# out['ref_corr'] = ref_corr
# np.save(old_savefile, out)

# %%
temp

# %%
from roughness.m3_correction import (
    get_m3_wls,
    get_solar_dist,
    get_m3_geom,
    get_m3photom,
    SHADOW_LOOKUP,
)

m3_basename = basename
rad_L1 = rad
shadow_lookup_path = SHADOW_LOOKUP

rms = 18  # Assumed rms roughness value of Moon
albedo = 0.12  # Assumed hemispherical broadband albedo
emiss = 0.985  # Assumed emissivity
wl = get_m3_wls(rad_L1) / 1000  # nm -> um
solar_dist = get_solar_dist(m3_basename)
sun_thetas, sun_azs, sc_thetas, sc_azs, phases, _ = get_m3_geom(obs, dem)
photom = get_m3photom(sun_thetas, sc_thetas, phases, polish=True)

# TODO Tested up to here
emission = np.zeros(rad_L1.shape)
for i in range(emission.shape[0]):
    for j in range(emission.shape[1]):
        # Compute shadowing fraction of each pixel
        # Select shadow table based on rms, incidence and az
        sun_theta, sun_az = sun_thetas[i, j], sun_azs[i, j]
        sc_theta, sc_az = sc_thetas[i, j], sc_azs[i, j]
        shadow_table = r.get_shadow_table(
            rms, sun_theta, sun_az, shadow_lookup_path
        )
        naz, ntheta = shadow_table.shape

        # Produce sloped facets at same resolution as shadow_table
        facet_theta, facet_az = r.get_shadow_table_arrays(ntheta, naz)
        facet_cinc = r.get_facet_cinc(facet_theta, facet_az, sun_theta, sun_az)

        # Correct shadow table for viewing geometery
        vis_slopes = r.visible_slopes(sc_theta, sc_az, rms, "rms", ntheta, naz)
        shadow_frac = shadow_table * vis_slopes / np.sum(vis_slopes)

        # Compute temperatures of sunlit facets assuming radiative eq
        facet_inc = np.degrees(np.arccos(facet_cinc))
        albedo_array = r.get_albedo_scaling(facet_inc, albedo)
        rad_illum = r.get_emission_eq(
            facet_cinc, albedo_array, emiss, solar_dist
        )
        rad_dw = r.get_rad_downwelling(
            facet_cinc, facet_theta, rad_illum, albedo, emiss
        )
        temp_illum = r.get_temp_sb(rad_illum + rad_dw)

        # Compute temperatures for shadowed facets (Josh approximation)
        temp_shade = r.get_shadow_temp(sun_theta, sun_az, temp_illum[0, 0])

        # Compute 2 component emission of illuminated and shadowed facets

        facet_emission = r.get_emission_2component(
            wl,
            temp_illum[..., None],
            temp_shade[..., None],
            shadow_frac[..., None],
        )
        emission[i, j] = np.sum(facet_emission, axis=(0, 1))
        break

# %%
band = 30
# np.testing.assert_almost_equal(tmp['t_pred'][:,:,band], emission[:,:,band])
# np.testing.assert_almost_equal(tmp['correction'], vis_slopes)
np.testing.assert_almost_equal(tmp["shade_table"], shadow_table)


# %%
