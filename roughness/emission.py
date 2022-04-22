"""Compute rough emission curves."""
from functools import lru_cache
import warnings
import numpy as np
import xarray as xr
import roughness.helpers as rh
import roughness.roughness as rn
from roughness.config import SB, SC, CCM, KB, HC, TLOOKUP


def rough_emission_lookup(
    geom,
    wls,
    rms,
    albedo,
    lat,
    rerad=True,
    flookup=None,
    tlookup=None,
    emissivity=1,
    cast_shadow_time=0.5,
):
    """
    Return rough emission spectrum given geom and params to tlookup.

    Parameters
    ----------
    geom (list of float): View geometry (sun_theta, sun_az, sc_theta, sc_az)
    wls (arr): Wavelengths [microns]
    rms (arr): RMS roughness [degrees]
    albedo (num): Hemispherical broadband albedo
    lat (num): Latitude [degrees]
    tloc (num): Local time past midnight [hours]
    rerad (bool): If True, include re-radiation from adjacent facets
    flookup (path or xarray): Facet lookup xarray or path to file.
    tlookup (path or xarray): Temperature lookup xarray or path to file.
    """
    # if date is not None:
    #     date = pd.to_datetime(date)
    sun_theta, sun_az, sc_theta, sc_az = geom
    tloc = rh.inc_to_tloc(sun_theta, sun_az, lat)
    if isinstance(wls, np.ndarray):
        wls = rh.wl2xr(wls)

    # Query los and temperature lookup tables
    shadow_table = rn.get_shadow_table(rms, sun_theta, sun_az, flookup)
    if not shadow_table.isnull().all():
        shadow_table = shadow_table.dropna("theta")

    # Get illum and shadowed facet temperatures
    temp_table = get_temp_table(tloc, albedo, lat, tlookup)
    tflat = temp_table.sel(theta=0, az=0)
    temp_table = temp_table.interp_like(shadow_table)
    temp_shade = get_shadow_temp_table(
        temp_table, sun_theta, sun_az, albedo, lat, tloc, cast_shadow_time
    )
    if rerad:
        rad_table = get_rad_sb(temp_table)
        rerad = get_reradiation_jb(
            sun_theta, rad_table.theta, tflat, albedo, emissivity
        )
        # rerad = get_reradiation_alt(rad_table.theta, tflat, emissivity)
        temp_table = get_temp_sb(rad_table + rerad)

    # Directional emissivity table
    # facet_theta, facet_az = rh.facet_grids(temp_table)
    # emissivity = get_emissivity_table(facet_theta, facet_az, sc_theta, sc_az)

    # Compute 2 component emission of illuminated and shadowed facets
    emission_table = emission_2component(
        wls, temp_table, temp_shade, shadow_table, emissivity
    )

    # Weight each facet by its prob of occurring and being visible from sc
    weights = rn.get_weight_table(rms, sun_theta, sc_theta, sc_az, flookup)

    weights = weights.interp_like(emission_table)
    weights = weights / weights.sum()
    rough_emission = emission_spectrum(emission_table, weights)
    return rough_emission


def get_shadow_temp_new(temp_table, albedo, lat, tloc, cast_shadow_time=0.5):
    """
    Return shadow temp scaled between current and dawn (coldest) temp.
    """
    if tloc < 12:
        tcold = get_temp_table(6, albedo, lat).interp_like(temp_table)
    else:
        tcold = get_temp_table(18, albedo, lat).interp_like(temp_table)
    return cast_shadow_time * tcold + (1 - cast_shadow_time) * temp_table


def get_shadow_temp_table(
    temp_table, sun_theta, sun_az, albedo, lat, tloc, cast_shadow_time
):
    """
    Return shadow temperature for each facet in temp_illum.

    Facets where solar inc > 90 degrees (shaded relief) retain the temp_table
    value since the sun has simply set in thermal model.

    Facets where solar inc < 90 degrees (cast shadow) are set to a shadow
    temperature following Bandfield et al. (2018).

    Parameters
    ----------
    tloc (num): Local time past midnight [hours].
    albedo (num): Local albedo.
    lat (num): Latitude [degrees].
    tlookup (path or xarray): Temperature lookup xarray or path to file.
    """
    tshadow = get_shadow_temp_new(
        temp_table, albedo, lat, tloc, cast_shadow_time
    )
    # Only change temp of facets < 90 degrees to sun (cinc > 0)
    # Otherwise set to min of JB cooling curve or temp_table temp if colder
    # tshadow = get_shadow_temp(sun_theta, sun_az, tflat)
    facet_theta, facet_az = rh.facet_grids(temp_table)
    cinc = rn.get_facet_cos_theta(facet_theta, facet_az, sun_theta, sun_az)
    temp_shadow = temp_table.copy()
    temp_shadow = xr.where(
        (cinc < 0) | (temp_shadow < tshadow), temp_shadow, tshadow
    )
    return temp_shadow


@lru_cache(maxsize=3)
def get_temp_table(tloc, albedo, lat, tlookup=None):
    """
    Return 2D temperature table of surface facets (inc, az).

    Parameters
    ----------
    date (str): Observation date (parsable with pandas.to_datetime).
    tloc (num): Local time past midnight [hours].
    albedo (num): Local albedo.
    lat (num): Latitude [degrees].
    los_lookup (path or xarray): Los lookup xarray or path to file.

    Returns
    -------
    temp_table (2D array): Facet temperatures (dims: az, theta)
    """
    if tlookup is None:
        tlookup = TLOOKUP
    params = [lat, albedo, tloc]  # lat symmetric about equator
    coords = ["lat", "albedo", "tloc"]
    # if date is not None:
    #     params.append(pd.to_datetime(date))
    #     # TODO : remove next 3 lines when xarray gdate is fixed
    #     tlookup = xr.open_dataarray(tlookup)
    #     tlookup["gdate"] = pd.DatetimeIndex(
    #         tlookup["gdate"].values
    #     ).sort_values()
    #     tlookup = tlookup.sortby("gdate")
    #     coords.append("gdate")

    temp_table = rn.get_table_xarr(
        params, coords, da="tsurf", los_lookup=tlookup
    )
    return temp_table


def rough_emission_eq(
    geom, wls, rms, albedo, emiss, solar_dist, rerad=False, flookup=None
):
    """
    Return emission spectrum corrected for subpixel roughness assuming
    radiative equillibrium. Uses ray-casting generated shadow_lookup table at
    sl_path.

    Parameters
    ----------
    geom (list of float): View geometry (sun_theta, sun_az, sc_theta, sc_az)
    wls (arr): Wavelengths [microns]
    rms (arr): RMS roughness [degrees]
    albedo (num): Hemispherical broadband albedo
    emiss (num): Emissivity
    solar_dist (num): Solar distance (au)
    rerad (bool): If True, include re-radiation from adjacent facets
    flookup (str): Los lookup path

    Returns
    -------
    rough_emission (arr): Emission spectrum vs. wls [W m^-2]
    """
    sun_theta, sun_az, sc_theta, sc_az = geom
    if isinstance(wls, np.ndarray):
        wls = rh.np2xr(wls, "wavelength", [wls])
    # If no roughness, return isothermal emission
    if rms == 0:
        return bb_emission_eq(sun_theta, wls, albedo, emiss, solar_dist)

    # Select shadow table based on rms, solar incidence and az
    shadow_table = rn.get_shadow_table(rms, sun_theta, sun_az, flookup)
    temp_illum = get_facet_temp_illum_eq(
        shadow_table, sun_theta, sun_az, albedo, emiss, solar_dist, rerad=rerad
    )

    # Compute temperatures for shadowed facets (JB approximation)
    temp0 = temp_illum[0, 0]  # Temperature of level surface facet
    temp_shade = get_shadow_temp(sun_theta, sun_az, temp0)

    # Compute 2 component emission of illuminated and shadowed facets
    emission_table = emission_2component(
        wls, temp_illum, temp_shade, shadow_table
    )

    # Weight each facet by its probability of occurrance
    weights = rn.get_weight_table(rms, sun_theta, sc_theta, sc_az, flookup)
    weights = weights / weights.sum()
    rough_emission = emission_spectrum(emission_table, weights)
    return rough_emission


def get_facet_temp_illum_eq(
    shadow_table, sun_theta, sun_az, albedo, emiss, solar_dist, rerad=True
):
    """Return temperatures of illuminated facets assuming rad eq"""
    # Produce sloped facets at same resolution as shadow_table
    f_theta, f_az = rh.facet_grids(shadow_table)
    cinc = rn.get_facet_cos_theta(f_theta, f_az, sun_theta, sun_az)
    # cinc = cos_facet_sun_inc(f_theta, f_az, sun_theta, sun_az)
    facet_inc = np.degrees(np.arccos(cinc))
    albedo_array = get_albedo_scaling(facet_inc, albedo)
    # Compute total radiance of sunlit facets assuming radiative eq
    rad_illum = get_emission_eq(cinc, albedo_array, emiss, solar_dist)
    if rerad:
        rad0 = rad_illum[0, 0]  # Radiance of level surface facet
        alb0 = albedo_array[0, 0]  # Albedo of level surface facet
        rad_illum += get_reradiation(
            cinc, f_theta, albedo, emiss, solar_dist, rad0, alb0
        )

    # Convert to temperature with Stefan-Boltzmann law
    facet_temp_illum = get_temp_sb(rad_illum)
    return facet_temp_illum


def bb_emission_eq(sun_theta, wls, albedo, emiss, solar_dist):
    """
    Return isothermal blackbody emission spectrum assuming radiative
    equilibrium.

    Parameters
    ----------
    sun_theta (num): Solar incidence angle [deg]
    wls (arr): Wavelengths [microns]
    albedo (num): Hemispherical broadband albedo
    emiss (num): Emissivity
    solar_dist (num): Solar distance (au)

    Return
    ------
    bb_emission (arr): Blackbody emission spectrum vs. wls [W m^-2]
    """
    cinc = np.cos(np.deg2rad(sun_theta))
    albedo_array = get_albedo_scaling(sun_theta, albedo)
    rad_isot = get_emission_eq(cinc, albedo_array, emiss, solar_dist)
    temp_isot = get_temp_sb(rad_isot)
    bb_emission = cmrad2mrad(bbrw(wls, temp_isot))
    return bb_emission


def cos_facet_sun_inc(surf_theta, surf_az, sun_theta, sun_az):
    """
    Return cos solar incidence of surface slopes (surf_theta, surf_az) given
    sun location (sun_theta, sun_az) using the dot product.

    If cinc < 0, angle b/t surf and sun is > 90 degrees. Limit these values to
    cinc = 0.

    Parameters
    ----------
    surf_theta (num or arr): Surface slope(s) [deg]
    surf_az (num or arr): Surface azimuth(s) [deg]
    sun_theta (num): Solar incidence angle [deg]
    sun_az (num): Solar azimuth from N [deg]
    """
    facet_vec = rh.sph2cart(np.deg2rad(surf_theta), np.deg2rad(surf_az))
    target_vec = rh.sph2cart(np.deg2rad(sun_theta), np.deg2rad(sun_az))
    cinc = (facet_vec * target_vec).sum(axis=2)
    cinc[cinc < 0] = 0
    if isinstance(surf_theta, xr.DataArray):
        cinc = xr.ones_like(surf_theta) * cinc
        cinc.name = "cos(inc)"
    return cinc


def get_albedo_scaling(inc, albedo=0.12, a=None, b=None, mode="vasavada"):
    """
    Return albedo scaled with solar incidence angle (eq A5; Keihm, 1984).

    Parameters a and b have been measured since Keihm (1984) and may differ for
    non-lunar bodies. Optionally supply a and b or select a mode:
        modes:
            'king': Use lunar a, b from King et al. (2020)
            'hayne': Use lunar a, b from Hayne et al. (2017)
            'vasavada': Use lunar a, b from Vasavada et al. (2012)
            else: Use original a, b from Keihm (1984)
    Note: if a or b is supplied, both must be supplied and supercede mode

    Parameters
    ----------
    albedo (num): Lunar average bond albedo (at inc=0)
    inc (num or arr): Solar incidence angle [deg]
    mode (str): Scale albedo based on published values of a, b
    """
    if a is not None and b is not None:
        pass
    elif mode == "king":
        # King et al. (2020)
        a = 0.0165
        b = -0.03625
    elif mode == "hayne":
        # Hayne et al. (2017)
        a = 0.06
        b = 0.25
    elif mode == "vasavada":
        # Vasavada et al. (2012)
        a = 0.045
        b = 0.14
    else:
        # Keihm et al. (1984)
        a = 0.03
        b = 0.14
    alb_scaled = albedo + a * (inc / 45) ** 3 + b * (inc / 90) ** 8
    return alb_scaled


def emission_spectrum(emission_table, weight_table=None):
    """
    Return emission spectrum as weighted sum of facets in emission_table.

    Weight_table must have same slope, az coords as emission_table.

    Parameters
    ----------
    emission_table (xr.DataArray): Table of radiance at each facet
    weight_table (xr.DataArray): Table of facet weights
    """
    if weight_table is None and isinstance(emission_table, xr.DataArray):
        weight_table = 1 / (len(emission_table.az) * len(emission_table.theta))
    elif weight_table is None:
        weight_table = 1 / np.prod(emission_table.shape[:2])

    if isinstance(emission_table, xr.DataArray):
        weighted = emission_table.weighted(weight_table.fillna(0))
        spectrum = weighted.sum(dim=("az", "theta"))
    else:
        weighted = emission_table * weight_table
        spectrum = np.sum(weighted, axis=(0, 1))
    return spectrum


def get_emission_eq(cinc, albedo, emissivity, solar_dist):
    """
    Return radiated emission assuming radiative equillibrium.

    Parameters
    ----------
    cinc (num): Cosine of solar incidence
    albedo (num): Hemispherical broadband albedo
    emissivity (num): Hemispherical broadband emissivity
    solar_dist (num): Solar distance (au)
    """
    f_sun = SC / solar_dist ** 2
    rad_eq = f_sun * cinc * (1 - albedo) / emissivity
    if isinstance(cinc, xr.DataArray):
        rad_eq.name = "Radiance [W m^-2]"
    return rad_eq


def get_reradiation_alt(facet_theta, t_flat, emiss=0.95):
    """
    Return emission incident on surf_theta from re-radiating adjacent level
    surfaces using lookup table. Modified from JB downwelling approximation.

    """
    # cinc = rn.get_facet_cos_theta(facet_theta, facet_az, sun_theta, sun_az)

    # Reflected radiance from sun to facet to each other facet?
    # (solar rad * facet albedo)/pi * cos(inc to each facet)

    # Emission from each facet to each other facet assuming lambertian
    # (sigT^4 * facet emissivity)/pi * cos(inc to each facet) * (1-albedo)
    # should be weighted by P(facet) - what is it scaled to? if we weight,
    #  we're presuming a total amount of rerad right?
    # rerad_table = get_rad_sb(temp_table) * emissivity / np.pi
    # rerad_table = rn.rotate_az_lookup(rerad_table, target_az=180, az0=0)

    # rerad_table *= (rerad_table.theta / 180) * albedo
    emiss = directional_emiss(90 - facet_theta)
    rerad_emitted = get_rad_sb(t_flat) * emiss
    rerad_absorbed = (facet_theta / 180) * rerad_emitted
    return rerad_absorbed


def directional_emiss(theta):
    """
    Return directional emissivity at angle theta.
    """
    trad = np.deg2rad(theta)
    return 0.993 - 0.0302 * trad - 0.0897 * trad ** 2


def get_reradiation_jb(
    sun_theta,
    facet_theta,
    t_flat,
    albedo=0.12,
    emiss=0.95,
    alb_scaling="vasavada",
):
    """
    Return emission incident on surf_theta from re-radiating adjacent level
    surfaces using JB downwelling approximation.
    """
    # emiss = 0.77
    facet_alb = get_albedo_scaling(
        facet_theta, albedo=albedo, mode=alb_scaling
    )
    rerad_emitted = get_rad_sb(t_flat) * emiss
    rerad_reflected = (
        SC
        * np.cos(np.deg2rad(sun_theta))
        * get_albedo_scaling(sun_theta, albedo, mode=alb_scaling)
    )
    rerad_absorbed = (facet_theta / 180) * (
        rerad_emitted + (1 - facet_alb) * rerad_reflected
    )
    return rerad_absorbed


def get_reradiation(
    cinc, surf_theta, albedo, emissivity, solar_dist, rad0, alb0=0.12
):
    """
    Return emission incident on surf_theta from re-radiating adjacent level
    surfaces. Approximation from JB modified from downwelling radiance.

    Parameters
    ----------
    cinc (num or arr): Cosine of solar incidence
    surf_theta (num or arr): Surface slope(s) [deg]
    surf_rad (num or arr): Surface radiance; e.g., from get_emission_eq()
    albedo (num or arr): Albedo (hemispherical if num, scaled by cinc if array)
    emissivity (num): Hemispherical broadband emissivity
    solar_dist (num): Solar distance (au)
    rad0 (num): Radiance of level surface facet
    alb0 (num): Albedo of level surface facet
    """
    # Albedo scaled by angle b/t facet and level surface (e.g. surf_theta)
    # TODO: Why scale by alb_level / 0.12?
    level_albedo = get_albedo_scaling(surf_theta, albedo) * alb0 / 0.12
    level_rad = get_emission_eq(cinc, level_albedo, emissivity, solar_dist)

    # Assuming infinite level plane visible from facet slope, the factor
    #  surf_theta / 180 gives incident reradiation
    rerad = (surf_theta / 180) * emissivity * (rad0 + alb0 * level_rad)

    if isinstance(rerad, xr.DataArray):
        rerad = rerad.clip(min=0)
    else:
        rerad[rerad < 0] = 0
    return rerad


def get_temp_sb(rad):
    """
    Return temperature assuming the Stefan-Boltzmann Law given radiance.

    Parameters
    ----------
    rad (num or arr): Radiance [W m^-2]
    """
    temp = (rad / SB) ** 0.25
    if isinstance(rad, xr.DataArray):
        temp.name = "Temperature [K]"
    return temp


def get_rad_sb(temp):
    """
    Return radiance assuming the Stefan-Boltzmann Law given temperature.

    Parameters
    ----------
    temp (num or arr): Temperature [K]
    """
    rad = SB * temp ** 4
    if isinstance(temp, xr.DataArray):
        rad.name = "Radiance [W m^-2]"
    return rad


def get_shadow_temp(sun_theta, sun_az, temp0):
    """
    Return temperature of shadows. Nominally 100 K less than the rad_eq
    surf_temp. At high solar incidence, scale shadow temperature by inc angle.
    Evening shadows are warmer than morning shadows. Tuned for lunar eqautor
    by Bandfield et al. (2015, 2018).

    Parameters
    ----------
    sun_theta (num): Solar incidence angle [deg]
    sun_az (num): Solar azimuth from N [deg]
    temp0 (num): Temperature of illuminated level surface facet [K]
    """
    shadow_temp_factor = 1
    if sun_theta >= 60 and 0 <= sun_az < 180:
        shadow_temp_factor = 1 - 0.6 * (sun_theta % 60) / 30
    elif sun_theta >= 60 and 180 <= sun_az < 360:
        shadow_temp_factor = 1 - 0.75 * (sun_theta % 60) / 30
    shadow_temp = temp0 - 100 * shadow_temp_factor
    return shadow_temp


def emission_2component(
    wavelength,
    temp_illum,
    temp_shadow,
    shadow_table,
    emissivity=1,
):
    """
    Return roughness emission spectrum at given wls given the temperature
    of illuminated and shadowed surfaces and fraction of the surface in shadow.

    Parameters
    ----------
    wavelength (num or arr): Wavelength(s) [microns]
    temp_illum (num or arr): Temperature(s) of illuminated fraction of surface
    temp_shade (num or arr): Temperature(s) of shadowed fraction of surface
    shadow_table (num or arr): Proportion(s) of surface in shadow
    """
    rad_illum = temp2rad(wavelength, temp_illum, emissivity)
    rad_shade = temp2rad(wavelength, temp_shadow, emissivity)
    rad_total = rad_illum * (1 - shadow_table) + rad_shade * shadow_table
    return rad_total


def get_emissivity_table(facet_theta, facet_az, sc_theta, sc_az):
    """
    Return emissivity table for given shadow_table and solar incidence angle.

    """
    cinc = rn.get_facet_cos_theta(facet_theta, facet_az, sc_theta, sc_az)
    # Bandfield et al. (2015) nighttime fit
    emissivity = 0.99 * cinc ** 0.14
    return emissivity


def temp2rad(wavelength, temp, emissivity=1):
    """
    Return "graybody" radiance at temperature and emissivity.
    """
    rad = bbrw(wavelength, temp) * emissivity
    return cmrad2mrad(rad)


def get_solar_irradiance(solar_spec, solar_dist):
    """
    Return solar irradiance from solar spectrum and solar dist
    assuming lambertian surface.

    Parameters
    ----------
    solar_spec (arr): Solar spectrum [W m^-2 um^-1] at 1 au
    solar_dist (num): Solar distance (au)
    """
    return solar_spec / (solar_dist ** 2 * np.pi)


def get_rad_factor(rad, emission, solar_irr, emissivity=None):
    """
    Return radiance factor (I/F) given observed radiance, modeled emission,
    solar irradiance and emissivity. If no emissivity is supplied,
    assume Kirchoff's Law (eq. 6, Bandfield et al., 2018).

    Parameters
    ----------
    rad (num or arr): Observed radiance [W m^-2 um^-1]
    emission (num or arr): Emission to remove from rad [W m^-2 um^-1]
    solar_irr (num or arr): Solar irradiance [W m^-2 um^-1]
    emissivity (num or arr): Emissivity (if None, assume Kirchoff)
    """
    if emissivity is None:
        # Assume Kirchoff's Law to compute emissivity
        emissivity = (rad - solar_irr) / (emission - solar_irr)
    return (rad - emissivity * emission) / solar_irr


def bbr(wavenumber, temp, radunits="wn"):
    """
    Return blackbody radiance at given wavenumber(s) and temp(s).

    Units='wn' for wavenumber: W/(m^2 sr cm^-1)
          'wl' for wavelength: W/(m^2 sr um)

    Translated from ff_bbr.c in davinci_2.22.

    Parameters
    ----------
    wavenumber (num or arr): Wavenumber(s) to compute radiance at [cm^-1]
    temp (num or arr): Temperatue(s) to compute radiance at [K]
    radunits (str): Return units in terms of wn or wl (wn: cm^-1; wl: um)
    """
    # Derive Planck radiation constants a and b from h, c, Kb
    a = 2 * HC * CCM ** 2  # [J cm^2 / s] = [W cm^2]
    b = HC * CCM / KB  # [cm K]

    if isinstance(temp, (float, int)):
        if temp <= 0:
            temp = 1
    elif isinstance(temp, xr.DataArray):
        temp = temp.clip(min=1)
    else:
        temp[temp < 0] = 1
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        rad = (a * wavenumber ** 3) / (np.exp(b * wavenumber / temp) - 1.0)
    if radunits == "wl":
        rad = wnrad2wlrad(wavenumber, rad)  # [W/(cm^2 sr um)]
    return rad


def bbrw(wavelength, temp, radunits="wl"):
    """
    Return blackbody radiance at given wavelength(s) and temp(s).

    Parameters
    ----------
    wavelength (num or arr): Wavelength(s) to compute radiance at [microns]
    temp (num or arr): Temperatue(s) to compute radiance at [K]
    """
    return bbr(wl2wn(wavelength), temp, radunits)


def btemp(wavenumber, radiance, radunits="wn"):
    """
    Return brightness temperature [K] from wavenumber and radiance.

    Translated from ff_bbr.c in davinci_2.22.
    """
    # Derive Planck radiation constants a and b from h, c, Kb
    a = 2 * HC * CCM ** 2  # [J cm^2 / s] = [W cm^2]
    b = HC * CCM / KB  # [cm K]
    if radunits == "wl":
        # [W/(cm^2 sr um)] -> [W/(cm^2 sr cm^-1)]
        radiance = wlrad2wnrad(wn2wl(wavenumber), radiance)
    T = (b * wavenumber) / np.log(1.0 + (a * wavenumber ** 3 / radiance))
    return T


def btempw(wavelength, radiance, radunits="wl"):
    """
    Return brightness temperature [K] at wavelength given radiance.

    """
    return btemp(wl2wn(wavelength), radiance, radunits=radunits)


def wnrad2wlrad(wavenumber, rad):
    """
    Convert radiance from units W/(cm2 sr cm-1) to W/(cm2 sr um).

    Parameters
    ----------
    wn (num or arr): Wavenumber(s) [cm^-1]
    wnrad (num or arr): Radiance in terms of wn [W/(cm^2 sr cm^-1)]
    """
    wavenumber_microns = wavenumber * 1e-4  # [cm-1] -> [um-1]
    rad_microns = rad * 1e4  # [1/cm-1] -> [1/um-1]
    return rad_microns * wavenumber_microns ** 2  # [1/um-1] -> [1/um]


def wlrad2wnrad(wl, wlrad):
    """
    Convert radiance from units W/(cm2 sr cm-1) to W/(cm2 sr um).
    """
    wn = 1 / wl  # [um] -> [um-1]
    wnrad = wlrad / wn ** 2  # [1/um] -> [1/um-1]
    return wnrad * 1e-4  # [1/um-1] -> [1/cm-1]


def wn2wl(wavenumber):
    """Convert wavenumber (cm-1) to wavelength (um)."""
    return 10000 / wavenumber


def wl2wn(wavelength):
    """Convert wavelength (um) to wavenumber (cm-1)."""
    return 10000 / wavelength


def cmrad2mrad(cmrad):
    """Convert radiance from units W/(cm2 sr cm-1) to W/(m2 sr cm-1)."""
    return cmrad * 1e4  # [W/(cm^2 sr um)] -> [W/(m^2 sr um)]


def mrad2cmrad(mrad):
    """Convert radiance from units W/(m2 sr cm-1) to W/(cm2 sr cm-1)."""
    return mrad * 1e-4  # [W/(m^2 sr um)] -> [W/(cm^2 sr um)]
