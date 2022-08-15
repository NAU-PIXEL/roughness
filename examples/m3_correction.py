# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.13.8
#   kernelspec:
#     display_name: 'Python 3.9.4 64-bit (''.venv'': poetry)'
#     name: python3
# ---

# %% [markdown]
# # Test m3_refl correction
#

# %%
# TODO: check emission units
# Imports
from pathlib import Path

import numpy as np
import xarray as xr
import rasterio as rio
import matplotlib.pyplot as plt
from roughness import roughness as rn
from roughness import helpers as rh
from roughness import plotting as rp
from roughness import emission as re
from roughness import config as cfg
from roughness import m3_correction as m3

import warnings

warnings.filterwarnings("ignore", message="Mean of empty slice")
warnings.filterwarnings("ignore", message="Dataset has no geotransform")

# %%
# Functions
roi2ext = {
    "Reiner_Gamma_swirl": [-56.6, -55.6, 8, 10],
    "Theophilus_spinel": [26, 28, -14, -9],
    "SW_Mare_Humorum_LPD": [-46, -44, -26, -24],
    "Ina_IMP": [5, 6, 18, 19],
    "Sosigenes_IMP": [19, 20, 8, 9],
    "Sulpicius_Gallus_LPD": [10, 11, 20, 22],
    "Cauchy-5_dome": [37, 39, 6, 8],
    "Rima_Bode_LPD": [-4, -2, 10, 13],
}

diviner_wls = [
    7.8,  # C3
    8.25,
    8.55,
    (13 + 23) / 2,
    (25 + 41) / 2,
    (50 + 100) / 2,
    (100 + 400) / 2,  # C9
]  # [microns]


def lev4hourly2xr(
    fgrds,
    tres=None,
    interp_method=None,
    interp_wrap=False,
    ext=None,
    savefile=None,
):
    """
    Return xarray of Diviner lev4 hourly grd tiles with time resolution tres.

    If interp_method: Interpolate missing values along tloc axis with method
    specified (e.g. 'nearest', 'linear', 'cubic', see xr.interpolate_na methods).
    Linear seems to produce fewest sharp artifacts.

    If savefile: save result to netCDF (.nc) file for quick I/O into xarray,
    e.g. with xr.open_dataarray(savefile).

    TODO: wrap data around to make sure there are no gaps at start / end (midnight)
    """
    if not tres:
        # Assume ch3 to ch9 and tbol (8 bands), diurnal query [24 h], infer time res [hr]
        tres = 24 / (len(fgrds) / 8)  # [hr]
    grids = []
    bands = []
    for i in range(len(fgrds)):
        roi, band, ind = Path(fgrds[i]).stem.split("-")
        tloc = (int(ind) - 1) * tres  # tloc in [0, 24) h

        # Read in .grd and set its local time to tloc
        grid = xr.open_rasterio(fgrds[i]).sel(band=1)
        grid["tloc"] = tloc
        grids.append(grid)

        # Concatenate all local time of this band into single xr.DataArray
        if tloc == 24 - tres:
            diurnal_band = xr.concat(grids, dim="tloc")
            diurnal_band["band"] = band
            bands.append(diurnal_band)
            grids = []

    # Concatenate all bands into single 4D xr.DataArray (lon, lat, tloc, band)
    out = xr.concat(bands, dim="band")
    out = out.rename("Temperature")
    out = out.sortby(
        "y", ascending=False
    )  # TODO: bug doesn't always fix yflip

    # Convert x, y coords to lat, lon
    if ext is not None:
        lon, lat = rh.xy2lonlat_coords(out.x, out.y, ext)
        out = out.rename({"x": "lon", "y": "lat"})
        out = out.assign_coords(lon=lon, lat=lat)

    if interp_method is not None:
        # Interpolate missing points on tloc
        if interp_wrap:
            # Cludgey handling of tloc edges. Wrap full dataset around midnight then interp
            postmidnight = out.sel(tloc=slice(0, 12 - tres))
            postmidnight["tloc"] = np.arange(24, 36, tres)
            premidnight = out.sel(tloc=slice(12, 24 - tres))
            premidnight["tloc"] = np.arange(-12, 0, tres)
            out = xr.concat([premidnight, out, postmidnight], dim="tloc")
        out = out.interpolate_na("tloc", method=interp_method).sel(
            tloc=slice(0, 24 - tres)
        )

    if savefile is not None:
        out.to_netcdf(savefile)
    return out


def xarr2geotiff(xarr, savefile, crs=cfg.MOON2000_ESRI):
    """Write 3D xarray (bands, y, x) to geotiff with rasterio."""
    count, height, width = xarr.shape
    band_indices = np.arange(count) + 1
    transform = None
    if "transform" in xarr.attrs:
        transform = tuple(xarr.attrs["transform"])
    profile = {
        "driver": "GTiff",
        "height": height,
        "width": width,
        "count": count,
        "dtype": str(xarr.dtype),
        "transform": transform,
        "crs": crs,
    }
    with rio.open(savefile, "w", **profile) as dst:
        dst.write(xarr.values, band_indices)
        for i in dst.indexes:
            band_name = str(xarr.band[i - 1].values)
            dst.update_tags(i, band_name)
    if Path(savefile).exists():
        print("Wrote to", savefile)
    else:
        print("Failed to write to", savefile)
    return


def fit_poly_daytime(Tday):
    tloc = np.linspace(6, 18, len(Tday) + 1)[:-1]
    nan = np.isnan(Tday)
    pfit = np.polyfit(tloc[~nan], Tday[~nan], 4)
    fit_func = np.poly1d(pfit)
    fit = fit_func(tloc)
    return fit


def smooth_daytime(T_xarr, savefile=None):
    """Return smoothed daytime T from 6 AM to 6 PM with 3rd order polyfit."""
    tres = T_xarr.tloc[1].values - T_xarr.tloc[0].values
    Tday = T_xarr.sel(tloc=np.arange(6, 18, tres))
    Tsmooth = xr.apply_ufunc(
        fit_poly_daytime,
        Tday,
        input_core_dims=[["tloc"]],
        output_core_dims=[["tloc"]],
        vectorize=True,
    )
    if savefile:
        Tsmooth.to_netcdf(savefile)
    return Tsmooth


def add_wls_diviner(xarr):
    """
    Return xarr with wavelength coordinate (drop tbol).
    """
    out = xarr.where(xarr.band != "tb", drop=True)
    out = out.assign_coords(wavelength=("band", diviner_wls))
    return out


# %%
# Diviner setup
# ROI = 'SW_Mare_Humorum_LPD'
# ROI = 'Reiner_Gamma_swirl'
ROI = "Theophilus_spinel"
# ROI = 'Ina_IMP'

ROI_STR = ROI.replace("'", "_").lower()
dirpath = f"/work/ctaiudovicic/data/Diviner/lev4/{ROI_STR}/"
LOAD_CACHED = True
if not list(Path(dirpath).rglob("*.nc")):
    LOAD_CACHED = False  # must be False on 1st run

ext = roi2ext[ROI]
# import raw Diviner L4 data, load if cached
fT_raw = dirpath + "T_raw.nc"
if LOAD_CACHED and Path(fT_raw).exists():  # 1 s to read from disk
    T_raw = xr.open_dataarray(fT_raw)
else:  # For a 2 deg^2 ROI with tres=0.25h, this took 4 s
    fgrds = [f.as_posix() for f in Path(dirpath).rglob("*.grd")]
    T_raw = lev4hourly2xr(fgrds, interp_method=None, savefile=fT_raw, ext=ext)

# smooth daytime Diviner data with polynomial, load if cached
fT_smooth = dirpath + "T_smooth.nc"
if LOAD_CACHED and Path(fT_smooth).exists():  # 1 s to read from disk
    T_smooth = xr.open_dataarray(fT_smooth)
else:  # For a 2 deg^2 ROI with tres=0.25h, this took 70 s
    T_smooth = smooth_daytime(T_raw, fT_smooth)

# Find better way to fix this
invert_y = False
if invert_y:
    T_raw.close()
    T_raw = xr.load_dataarray(fT_raw)
    T_raw = T_raw.assign_coords(lat=T_raw.lat[::-1])
    T_raw.to_netcdf(fT_raw)

    T_smooth.close()
    T_smooth = xr.load_dataarray(fT_smooth)
    T_smooth = T_smooth.assign_coords(lat=T_smooth.lat[::-1])
    T_smooth.to_netcdf(fT_smooth)


# %%
f, ax = plt.subplots(1, 2, figsize=(8, 4))
day = slice(6, 18)
p = (
    T_raw.sel(tloc=day, band="t3")
    .isel(lon=10)
    .plot.line(x="tloc", add_legend=False, ax=ax[0])
)
ax[0].set_title("No interpolation")
p = (
    T_smooth.sel(band="t3")
    .isel(lon=10)
    .plot.line(x="tloc", add_legend=False, ax=ax[1])
)
ax[1].set_title("Smoothed (4th order poly fit)")
plt.show()

# %%
# Image plots of all bands
f, ax = plt.subplots(1, 2, figsize=(6, 6))
aspect = len(T_raw.lon) / len(T_raw.lat)
p = T_raw.sel(tloc=16, band="t4").plot.imshow(ax=ax[0])
q = T_smooth.sel(tloc=16, band="t4").plot.imshow(ax=ax[1])

# %%
# M3 setup
tifdir = "/work/team_moon/data/warped/210202/"
# basename = "M3G20090209T212512"  # RG
# basename = "M3G20090612T230542"  # RG
# basename = "M3G20090112T165924"  # SW humorum
basename = "M3G20090731T045352"  # theo
pdem = "/work/ctaiudovicic/data/Kaguya/"
fdem = [
    pdem + f"lola_kaguya_dem_slope_cyl_128.tif",
    pdem + f"lola_kaguya_dem_azim_cyl_128.tif",
    pdem + f"lola_kaguya_dem_cyl_128.tif",
]
# m3ext = (-56.25, -56.2, 8.55, 8.65) # RG
# m3ext = (-56.35, -56.1, 8.55, 8.68) # RG
# m3ext = (-45.26, -44.09, -25.65, -24.14)  # sw humorum
# m3ext = (26.65, 27.19, -13.47, -9.55)  # theo
m3ext = (26.9, 27, -11.3, -11.2)  # theo
rad = m3.read_m3_geotiff(tifdir + basename + "_rad.tif", m3ext)
obs = m3.read_m3_geotiff(tifdir + basename + "_obs.tif", m3ext)
width, height = rad.shape[:2]
dem = m3.get_dem_geotiff(*fdem, m3ext, width, height, "bilinear")

lon, lat = rh.xy2lonlat_coords(np.arange(width), np.arange(height), m3ext)
wls = m3.get_m3_wls(rad)
rad = xr.DataArray(
    rad,
    dims=["lon", "lat", "wavelength"],
    coords={"lat": lat, "lon": lon, "wavelength": wls},
)
albedo_map = m3.read_albedo_map_feng().interp_like(rad)

# Compute at extra wls to see emission at diviner wls
wl_all = np.concatenate([wls, diviner_wls])


# emission.isel(lat=2, lon=2).plot.line(x='wavelength')

# %%
# Compute ref correction
# refl, emission = m3.m3_refl(basename, rad, obs, wls, dem)
clat = np.mean(m3ext[2:4])
date = "2009-Jul-31"
# date = '2009-Feb-18'
emiss = None
rms = 18
refl, emission = m3.m3_refl(
    basename, rad, obs, wl_all, dem, rms, albedo_map, emiss, rerad=False
)

# %%
tloc = 9.25
f, ax = plt.subplots(1, 3, figsize=(8, 4))
rad.isel(wavelength=5).T.plot.imshow(ax=ax[0])
albedo_map.interp_like(rad).plot.imshow(ax=ax[1])
Tdiv = T_smooth.sel(
    lat=slice(m3ext[3], m3ext[2]), lon=slice(m3ext[0], m3ext[1])
)
q = Tdiv.sel(tloc=tloc, band="t4").plot.imshow(ax=ax[2])

# %%
geom = [41.965664, 78.30811, 4.89781, 57.546074]
sun_theta, sun_az, sc_theta, sc_az = geom
wl = np.concatenate([m3.get_m3_wls(rad), diviner_wls])
wl_xarr = rh.np2xr(wl, ["wavelength"], [wl])

rms = 30
albedo = 0.12
lat = -11.25
tloc = 9.118993123372396
date = "2009-Jul-31"
date2 = "2009-Feb-18"  # max seasonal
# TODO: date is not used

flookup = None
tlookup = None
# e = re.rough_emission_lookup(geom, wl, rms, albedo, clat, tloc, date)
# e
shadow_table = rn.get_shadow_table(rms, sun_theta, sun_az, flookup)
temp_illum = re.get_temp_table(tloc, albedo, lat, date, tlookup).interp_like(
    shadow_table
)
temp_shade = re.get_shadow_temp_table(
    tloc, albedo, lat, date, tlookup
).interp_like(shadow_table)
temp_illum2 = re.get_temp_table(tloc, albedo, lat, date2, tlookup).interp_like(
    shadow_table
)
temp_shade2 = re.get_shadow_temp_table(
    tloc, albedo, lat, date2, tlookup
).interp_like(shadow_table)

# %%
diurnal = re.get_temp_table(np.arange(1, 23), albedo, lat, date, tlookup)
p = diurnal.interp(az=270).plot.line(x="tloc", add_legend=False)

# %%
emission_table = re.emission_2component(
    wl_xarr, temp_illum, temp_shade, shadow_table
)
emission_table2 = re.emission_2component(
    wl_xarr, temp_illum2, temp_shade2, shadow_table
)

# %%
facet_weights = rn.get_facet_weights(rms, sun_theta, flookup)
view_weights = rn.get_view_table(rms, sc_theta, sc_az, flookup)
weight_table = facet_weights * view_weights
weight_table /= weight_table.sum()

# %%
(emission_table * weight_table).isel(wavelength=-6).plot.imshow()

# %%

rough_emission = re.emission_spectrum(emission_table, weight_table)
rough_emission_w = re.emission_spectrum(emission_table, facet_weights)
rough_emission_v = re.emission_spectrum(
    emission_table, view_weights / view_weights.sum()
)
rough_emission_raw = re.emission_spectrum(
    emission_table, 1 / np.prod(emission_table.shape[:2])
)

rough_emission.plot.line(ls="-.", x="wavelength", label="rough")
# rough_emission_w.plot.line(ls='--', x='wavelength', label='w rough')
# rough_emission_v.plot.line(ls=':', x='wavelength', label='v rough')
rough_emission_raw.plot.line(ls="-", x="wavelength", label="raw rough")
plt.legend()

# %%
rad_emission = rough_emission  # .interp(wavelength=rad.wavelength)
rbtemp = re.btempw(rad_emission.wavelength, rad_emission)
rbtemp.plot()

# %%
rad_emission = rough_emission_raw  # .interp(wavelength=rad.wavelength)
rbtemp = re.btempw(rad_emission.wavelength, rad_emission)
rbtemp.plot()

# %%
ktmp = (
    re.get_temp_table(tloc, albedo, lat, date, tlookup).sel(theta=0).mean("az")
)
ktmp

# %%
import pandas as pd

t = xr.load_dataarray(
    "/home/ctaiudovicic/projects/roughness/data/temp_lookup.nc"
)
t["gdate"] = pd.DatetimeIndex(t["gdate"].values)
t = t.sortby("gdate")
gdate = pd.to_datetime(date)
t_dates = t.interp(albedo=0.12, lat=clat, theta=0).sel(tloc=slice(8, 16))
t_now = t_dates.interp(gdate=gdate)

f, ax = plt.subplots()
t_dates["gdate"] = t_dates.gdate.dt.strftime("%b %Y")
t_dates.mean("az").plot.line(x="tloc", ax=ax)
t_now.mean("az").plot.line(ls="-.", x="tloc", ax=ax)
ax.set_title("Seasonal variation (current observation date is dash-dot)")
ax.axvline(tloc, ls="--", color="w")
ax.set_xlim(tloc - 0.5, tloc + 0.5)
plt.show()
# t.coords

# %%
dates = ["2008-Nov-18", "2009-Aug-18", "2009-Feb-18", "2009-May-18"]
ktmp1 = (
    re.get_temp_table(tloc, palb, plat, dates[3], tlookup)
    .sel(theta=0)
    .mean("az")
    .values
)
ktmp2 = (
    re.get_temp_table(tloc, palb, plat, dates[1], tlookup)
    .sel(theta=0)
    .mean("az")
    .values
)
print(ktmp1, ktmp2)

# %%
date = "2009-July-31"
# palb = 0.04
# plat=8.478
tlookup = None
ktmp = (
    re.get_temp_table(tloc, albedo_map, clat, date, tlookup)
    .sel(theta=0)
    .mean("az")
)

dtmp = add_wls_diviner(T_smooth)
# dtmp = dtmp.sel(lon=slice(None, -56.34))

dtmp = dtmp.sel(band="t7").interp(tloc=tloc).interp_like(ktmp)
print("Mean krc:", ktmp.mean().values)
print("Mean Diviner:", dtmp.mean().values)
# Plot and check
f, ax = plt.subplots()
(dtmp - ktmp).plot.imshow(vmin=-10, vmax=10, ax=ax, cmap="bwr")
plt.show()

# %%
# temp_shade.plot(col='gdate')
# temp_illum.plot(col='gdate')
wls = rh.np2xr(wl, ["wavelength"], [wl])
# wn = np.arange(14, 794, 2)
# wns = rh.np2xr(wn, ['wavenumber'], [wn])
rad_illum = re.bbrw(wls, temp_illum)
rad_shade = re.bbrw(wls, temp_shade)
rad_total = (1 - shadow_table) * rad_illum + shadow_table * rad_shade


# %%
f, ax = plt.subplots()
for i in range(1):  # range(len(rad_illum.az)):
    rad_illum.sel(wavelength=slice(0, 80)).isel(az=i).plot.line(
        ax=ax, x="wavelength", add_legend=False
    )
# facet_weights.plot.imshow()

# %%
wrad = facet_weights * rad_total
wemiss = wrad.sum(dim=("az", "theta"))
p = (
    re.btempw(wls, wemiss)
    .sel(wavelength=slice(3, 80))
    .plot.line(x="wavelength", add_legend=False)
)

# %%
f, ax = plt.subplots()
for i in range(1):  # range(len(rad_illum.az)):
    p = re.btempw(
        wls, rad_total.sel(wavelength=slice(3, 80)).isel(az=i)
    ).plot.line(ax=ax, x="wavelength", add_legend=False)
# facet_weights.plot.imshow()
wrad = facet_weights * rad_total
wemiss = wrad.sum(dim=("az", "theta"))
p = (
    re.btempw(wls, wemiss)
    .sel(wavelength=slice(3, 80))
    .plot.line(ax=ax, x="wavelength", add_legend=False)
)

# %%
# p=(rad_total).isel(az=-1).plot.line(x='wavelength', add_legend=False)
# p = re.btempw(wls, rad_total).isel(wavelength=-1).plot.line(x='theta', add_legend=False)
# p=(rad_illum).isel(az=-1).plot.line(x='wavelength', add_legend=False)
# p = re.btempw(wls, rad_illum).isel(wavelength=1).plot.imshow()
p = (rad_illum).isel(wavenumber=1).plot.imshow()

# %%
p = (rad_total).isel(az=-1).plot.line(x="wavelength", add_legend=False)

# %%
p = (
    re.btempw(wls, rad_total)
    .isel(wavelength=-1)
    .plot.line(x="theta", add_legend=False)
)

# %%
f, ax = plt.subplots()
e_spec = re.emission_spectrum(rad_total, facet_weights)
e_spec.plot(x="wavelength")
ax.plot(wls, re.bbrw(wls, 300), label="300")
ax.plot(wls, re.bbrw(wls, 350), label="350")
ax.legend()

# %%
f, ax = plt.subplots()
e_spec = re.emission_spectrum(rad_total, facet_weights)
re.btempw(wls, e_spec).plot(x="wavelength")
ax.plot(wls, re.btempw(wls, re.bbrw(wls, 300)), label="300")
ax.plot(wls, re.btempw(wls, re.bbrw(wls, 350)), label="350")
ax.legend()

# %%
facet_weights = rn.get_facet_weights(rms, 0, flookup)
p = facet_weights.plot.line(x="theta", add_legend=False)

# %%
re.btempw(rh.np2xr(wls, ["wavelength"], [wls]), e).plot.line(x="wavelength")


# %%
f, ax = plt.subplots()
aspect = len(rad.lon) / len(rad.lat)
# Spectra
emission.sel(lon=-56.15, method="nearest").plot.line(
    x="wavelength", add_legend=False, ax=ax
)
emission.sel(lon=-56.25, method="nearest").plot.line(
    x="wavelength", add_legend=False, ax=ax
)
emission.sel(lon=-56.35, method="nearest").plot.line(
    x="wavelength", add_legend=False, ax=ax
)

emission.sel(wavelength=3, method="nearest").T.plot.imshow(
    size=3, aspect=aspect
)

plt.show()

# %%
f, ax = plt.subplots()
aspect = len(rad.lon) / len(rad.lat)
# Spectra
refl.sel(lon=-56.15, method="nearest").plot.line(
    x="wavelength", add_legend=False, ax=ax
)
refl.sel(lon=-56.10, method="nearest").plot.line(
    x="wavelength", add_legend=False, ax=ax
)
# refl.sel(lon=-56.35, method='nearest').plot.line(x='wavelength', add_legend=False, ax=ax)

rad.sel(wavelength=3, method="nearest").T.plot.imshow(size=3, aspect=aspect)
refl.sel(wavelength=3, method="nearest").T.plot.imshow(size=3, aspect=aspect)

plt.show()

# %%
# Manual geom check
import roughness.helpers as rh

# Compute inc using dem
cx, cy = np.array(dem.shape[:2]) // 2
theta, az, _ = dem[cx, cy, :]
print("dem (theta,az):", np.degrees([theta, az]))
ground = rh.sph2cart(theta, az)
# print(ground)

# Local inc with dem
sun_az, sun_theta = np.radians(obs[cx, cy, :2])
print(np.degrees([sun_theta, sun_az]))
sun = rh.sph2cart(sun_theta, sun_az)
print(sun)
print()
inc = rh.get_local_inc(ground[None, None, :], sun[None, None, :])
print("Local inc (DEM):", np.degrees(np.arccos(fac_cinc)), inc)

# Compute inc using obs angles
fac_slope, fac_az = np.radians(obs[cx, cy, 7:9])
fac_cinc = obs[cx, cy, 9]
print("OBS Facet (slope, az)", np.degrees((fac_slope, fac_az)))
print("Local inc (OBS):", np.degrees(np.arccos(fac_cinc)))
# alt = obs[cx, cy, 6]
# ground = rh.sph2cart(fac_slope, fac_az)
# print(ground)

sun_az, sun_theta = np.radians(obs[cx, cy, :2])
print("sun (theta, az):", np.degrees((sun_theta, sun_az)))
# sdist = obs[cx, cy, 5] # +m3.get_solar_dist(basename)
# sdist_m = sdist * 1.496e+11  # [au] -> [m]
sun = rh.sph2cart(sun_theta, sun_az)
print(sun)
inc = rh.get_local_inc(ground[None, None, :], sun[None, None, :])
print("Local inc:", inc)

# %%
# Plot i, e, g images

# i, sun_az, e, sc_az, g, sun_sc_az = m3.get_m3_geom(obs, dem)
# f, ax = plt.subplots(1, 3, figsize=(12, 4))
# p = ax[0].imshow(i.T)
# ax[0].set_title('Incidence angle')
# p = ax[1].imshow(e.T)
# ax[1].set_title('Emission angle')
# p = ax[2].imshow(g.T)
# ax[2].set_title('Phase angle')
# f.colorbar(q, cax=plt.axes([0.95, 0.2, 0.05, 0.6]))

# %%
f, ax = plt.subplots()
left = refl[:5]
c_bright = refl[10:20, 5:20]
c_dark = refl[28:32, 5:16]
right = refl[35:]

# rp.m3_spec(rad, ax=ax)
rp.m3_spec(left, ax=ax, label="left")
rp.m3_spec(c_bright, ax=ax, label="crater illum")
rp.m3_spec(c_dark, ax=ax, label="crater shadow")
rp.m3_spec(right, ax=ax, label="right")
ax.legend()
plt.show()

# %%
f, ax = plt.subplots()
left = emission[:5]
c_bright = emission[10:20, 5:20]
c_dark = emission[28:32, 5:16]
right = emission[35:]

# rp.m3_spec(rad, ax=ax)
rp.m3_spec(left, ax=ax, label="left")
rp.m3_spec(c_bright, ax=ax, label="crater illum")
rp.m3_spec(c_dark, ax=ax, label="crater shadow")
rp.m3_spec(right, ax=ax, label="right")
ax.set_title("Emission")
ax.legend()
plt.show()

# %%
# Plot solar spectrum and L_sun
f, ax = plt.subplots()
pwls = m3.get_m3_wls(obs)[:-1]
solar_dist = np.mean(m3.get_solar_dist(basename, obs))
solar_spec = m3.get_solar_spectrum(m3.is_targeted(rad))
L_sun = re.get_solar_irradiance(solar_spec, solar_dist)
# rp.m3_spec(solar_spec, wls=pwls, ax=ax, label='solar spec')
rp.m3_spec(rad / L_sun, wls=pwls, ax=ax, label="rad/L_sun")
rp.m3_spec(
    (rad - 0.95 * emission) / L_sun, wls=pwls, ax=ax, label="rad-em /Lsun"
)
plt.legend()
plt.show()

# %%
# Manual I_F check

# sun_thetas, sun_azs, sc_thetas, sc_azs, phases, _ = m3.get_m3_geom(obs, dem)
# photom = m3.get_m3photom(sun_thetas, sc_thetas, phases, polish=True, targeted=m3.is_targeted(rad))
# sdist = m3.get_solar_dist(basename, obs)

# ref = rad*np.pi / (solar_spec/sdist[:, :, None]**2)
# rp.m3_spec(ref*photom)
# plt.show()

# %%
# BT vs wavelength
wl_xr = rh.np2xr(wl_all, ["wavelength"], [wl_all])
btemps = re.btempw(wl_xr, emission).mean(("lat", "lon"))
T_wl = add_wls_diviner(Tdiv)

f, ax = plt.subplots()
left = btemps
# c_bright = btemps[10:20, 5:20]
# c_dark = btemps[28:32, 5:16]
# right = btemps[35:]

# rp.m3_spec(rad, ax=ax)
# rp.m3_spec(left, wls=wl_all, ax=ax, label="rough")
# rp.m3_spec(c_bright, wls=wl_all, ax=ax, label='crater illum')
# rp.m3_spec(c_dark, wls=wl_all, ax=ax, label='crater shadow')
# rp.m3_spec(right, wls=wl_all, ax=ax, label='right')

T_wl.interp(tloc=tloc).mean("lat").plot.line(
    x="wavelength", ax=ax, add_legend=False
)
# rp.m3_spec(T_wl.sel(tloc=tloc, method='nearest').mean(('lat', 'lon')), wls=T_wl.wavelength, ax=ax, label='Diviner BT')
ax.legend()
ax.set_xlim(3, 200)
plt.show()

# %%
# Align M3 and Diviner BTs
# Reiner Gamma roi
btmp = btemps
# btmp = btemps.sel(lon=slice(-56.31, -56.19), lat=slice(8.5, 8.665))
# btmp = btmp.sel(lon=slice(None, -56.255))

dtmp = T_wl
# dtmp = T_wl.sel(lon=slice(-56.395, -56.26), lat=slice(8.565, 8.675))
# dtmp = dtmp.sel(lon=slice(None, -56.34))

# Plot and check
f, ax = plt.subplots(2)
btmp.isel(wavelength=-5).T.plot.imshow(ax=ax[0])
dtmp.sel(tloc=tloc, method="nearest").sel(band="t5").plot.imshow(ax=ax[1])
plt.show()

# %%
# aspect = len(T_wl.lon) / len(T_wl.lat)
# T_wl.sel(band='t3', lon=slice(-56.2, None), lat=slice(8, 9)).sel(tloc=8.25, method='nearest').plot.imshow(size=5, aspect=aspect)
# btmp = btmp.where(btemps > 200, drop=True)
for i, j in zip(range(len(btmp.lon)), range(len(btmp.lat))):
    rtemp = btmp.isel(lon=i, lat=j).values
    plt.plot(btmp.wavelength, rtemp, ".")
for i, j in zip(range(len(dtmp.lon)), range(len(dtmp.lat))):
    dtemp = dtmp.interp(tloc=tloc).isel(lon=i, lat=j).values
    # dtemp = dtmp.sel(tloc=tloc, method='nearest').isel(lon=i, lat=j).values
    plt.plot(dtmp.wavelength, dtemp, "rx")
plt.plot(btmp.wavelength, rtemp, ".", label="Rough BTs")
plt.plot(T_wl.wavelength, dtemp, "rx", label="Diviner BTs")
plt.axhline(t_now.interp(tloc=tloc).mean("az"), c="y", label="KRC 0$^o$ Temp")
plt.xscale("log")
plt.xlabel("Wavelength [microns]")
plt.ylabel("Brightness temperature [K]")
plt.xlim(1, 275)
plt.legend()
plt.show()

# %%
re.get_shadow_temp(sun_theta, sun_az, 350)

# %%
