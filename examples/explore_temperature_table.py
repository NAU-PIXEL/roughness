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
from roughness import roughness as r
from roughness import emission as em

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

# %%
# Needs to predict radiance:
# s/c azim,el
# solar azim,el
# los lookup table showing probabilites for both those cases
# azim/slope temperature table for given date, latitude, and local solar time.

# Needs to correct spectra:
# Solar spectrum & distance
# calibrated radiance observation
# 
# photometric correction factors


# %%
def bbr(XX,temp):
    """
    Calculate radiance in units of wavenumber, W/cm^2/sr/cm^-1, given an xaxis and a temperature.
    Xaxis can be a single value, an m x 1 array, or an m x n array where m is the number of spectral
    channels and n is the number of temperatures. If an m x n array is provided, the temps must be
    an n-sized vector.
    """

    # Define the constants for planck calculations bbr/btemp
    h=6.6260755e-34 # Planck constant
    c=2.99792458e10 # Speed of light in cm/s
    kb=1.38064852e-23 # Boltzmann constant
    
    # Combine constants into the First and Second radiation constants for easy calculation
    C1=2.*h*c*c # First radiation constant for cm
    C2=(h*c)/kb # Second radiation constant for cm
    
    # Calculate the blackbody radiance using the planck equation in per unit wavenumber form W/cm^2/sr/cm^-1
    rad=(C1*XX**3)/(np.exp(C2*XX/temp)-1.0)
    return rad
    
def btemp(XX,radiance):
    """
    Calculate brightness temperature given an xaxis and radiance in units of wavenumber, W/cm^2/sr/cm^-1.
    Xaxis can be a single value, an m x 1 array, or an m x n array where m is the number of spectral
    channels and n is the number of temperatures. If an m x 1 or m x n array is provided, the temps input 
    must be an equivalent sized array.
    """
    if(len(radiance.shape)==1):
        radiance=np.reshape(radiance,(-1,1))
        
    # Define the constants for planck calculations bbr/btemp
    h=6.6260755e-34 # Planck constant
    c=2.99792458e10 # Speed of light in cm/s
    kb=1.38064852e-23 # Boltzmann constant

    # Combine constants into the First and Second radiation constants for easy calculation
    C1=2.*h*c*c # First radiation constant for cm
    C2=(h*c)/kb # Second radiation constant for cm
    
    # Calculate the brightness temperature assuming radiance is in per unit wavenumber form W/cm^2/sr/cm^-1
    btemp=C2*XX/np.log(1.0+(C1*XX*XX*XX/radiance))
    return btemp


# %%
rms=20
inc=45
az=270
lat=45
lst=10

# Create the xaxis vector
XX = np.arange(start=100,stop=4000,step=10,dtype='double').reshape(-1,1)
print('X-axis vector',XX.shape)

# Create an array of xaxes as long as the flattened slope/azim tables.
XXarr = np.tile(XX,(36*45))
print('X-axis array',XXarr.shape)

# Read in our lookup tables
loslookup = r.load_los_lookup(r.LOS_LOOKUP)
templookup = em.load_temperature_table()

# Grab the shadow table for specific RMS, inc, az. Flatten it and reshape it to ease calc.
shadow_table = r.get_shadow_table(rms, inc, az, loslookup).flatten().reshape(1,-1)
print('Shadow Table',shadow_table.shape)

# Assume that this is a morning observation for now and grab the 0 slope temp for pre-dawn.
shadow_temp = em.get_temperature_table(lat,6,templookup)[0,0]
print('Shadow Temp (pre-dawn)',shadow_temp)

# Calculate the weighted radiance of shadowed regions
shadow_rad = bbr(XXarr,shadow_temp) * shadow_table
print('Shadowed Radiance',shadow_rad.shape)

# Make an illuminated facet table.
illum_table = 1 - shadow_table
print('Illuminated Table',illum_table.shape)

# Get the temperatures for slope/az at specific lat/lst. Flatten and reshape.
temp_table = em.get_temperature_table(lat, lst, templookup)[1:,1:].flatten().reshape(1,-1)
print('Temperature Table',temp_table.shape)

# Calculate the weighted illuminated surface radiances
illum_rad = bbr(XXarr,temp_table) * illum_table
print('Illuminated Radiance',illum_rad.shape)

# Add the weighted illuminated and shadowed radiances for each slope/azim.
total_rad = illum_rad + shadow_rad
print('Total Radiance Table',total_rad.shape)

# Uncomment for non-nadir view correction 
#view_table = r.get_view_table(rms, inc, az, loslookup)

total_table = r.load_los_lookup('/Users/habtop/Documents/GitHab.nosync/roughness/data/total_facets_4D.npy')
total_facets = r.get_shadow_table(rms, inc, az, total_table).flatten().reshape(1,-1)
print('Total Facets per Bin',total_facets.shape)

occurrence_probability = total_facets/np.nansum(total_facets)
print('Occurrence Probability',occurrence_probability.shape)

# Calculate the integrated radiance for the surface
integrated_rad = np.nansum(total_rad*occurrence_probability,axis=(1)).reshape(-1,1)
print('Integrated Radiance',integrated_rad.shape)


# %%
reftemp=255

fig,[ax1,ax2,ax3] = plt.subplots(3,1,figsize=(10,12),dpi=400,sharex=True)
for i in range (0,1620):
    ax1.plot(XX,total_rad[:,i],lw=0.5,color='w')
ax1.plot(XX,bbr(XX,reftemp),lw=2,color='r')
ax1.set_xlim(4000,100)
ax1.set_ylim(0,2e-5)
ax1.set_title('All Facet Radiance Spectra')
ax1.set_ylabel('Radiance\n  $(W/cm^2 \cdot sr \cdot cm^{-1})$')
ax1.set_xlabel('Wavenumber ($cm^{-1}$)')


ax2.plot(XX,integrated_rad,color='w',label=('Integrated radiance'))
ax2.plot(XX,bbr(XX,reftemp),lw=2,color='r',label=('Reference BB: '+str(reftemp)))
ax2.set_xlim(4000,100)
ax2.set_ylim(0,2e-5)
ax2.set_title('Integrated Rough Radiance Spectrum: RMS = '+str(rms))
ax2.set_ylabel('Radiance\n  $(W/cm^2 \cdot sr \cdot cm^{-1})$')
ax2.set_xlabel('Wavenumber ($cm^{-1}$)')
ax2.legend()

ax3.plot(XX,btemp(XX,integrated_rad),color='w')
ax3.set_xlim(4000,100)
ax3.set_ylim(200,300)
ax3.set_title('Rough Surface Brightness Temperature Spectrum')
ax3.set_ylabel('Brightness Temperature (K)')
ax3.set_xlabel('Wavenumber ($cm^{-1}$)')

fig.savefig('/Users/habtop/Desktop/bennu_first_light_radiance.jpg',format='jpg')
print()

# %%
