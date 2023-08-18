"""
This file contains the shadow lookup and temperature lookup classes that are
used by the roughness thermal model.
"""

from functools import cached_property
from pathlib import Path
import xarray as xr
from roughness.config import FLOOKUP, TLOOKUP
import roughness.emission as re


class RoughnessThermalModel:
    """Initialize a roughness thermophysical model object."""

    def __init__(self, shadow_lookup=FLOOKUP, temperature_lookup=TLOOKUP):
        if shadow_lookup == FLOOKUP and not Path(shadow_lookup).exists():
            raise FileNotFoundError(
                "Default shadow lookup table not found. Please run \
                `roughness -d` in the terminal to download it."
            )
        self.shadow_lookup = shadow_lookup
        self.temperature_lookup = temperature_lookup
        self.temperature_ds = xr.open_dataset(temperature_lookup)
        if not {"theta", "az"}.issubset(self.temperature_ds.dims):
            raise ValueError(
                f"Temperature lookup {self.temperature_lookup} \
                               must have dimensions az, theta"
            )
        
    def __repr__(self):
        return (f"RoughnessThermalModel(shadow_lookup='{self.shadow_lookup}'" +
                f", temperature_lookup='{self.temperature_lookup}')")

    def emission(
        self, sun_theta, sun_az, sc_theta, sc_az, wls, rms, tparams, **kwargs
    ):
        """
        Return rough surface emission at the given viewing geometry.

        Parameters:
            sun_theta,sun_az,sc_theta,sc_az (num): Viewing angles [degrees]
                (incidence, solar_azimuth, emergence, emergence_azimuth)
            wls (arr): Wavelengths [microns]
            rms (arr, optional): RMS roughness [degrees]. Default is 15.
            rerad (bool, optional): If True, include re-radiation from adjacent
                facets. Default is True.
            tparams (dict, optional): Dictionary of temp erature lookup parameters.
                Default is None.
            **kwargs

        Returns:
            (xr.DataArray): Rough emission spectrum.

        """
        if tparams.keys() != self.required_tparams:
            raise ValueError(f"Tparams must contain {self.required_tparams}")
        args = ((sun_theta, sun_az, sc_theta, sc_az), wls, rms, tparams)
        return re.rough_emission_lookup(*args, **kwargs)

    @cached_property
    def required_tparams(self):
        """Check which tparams are required by the TemperatureLookup."""
        return set(self.temperature_ds.dims) - {"theta", "az"}
        # dims = list(self.ds.dims)
        # try:
        #     dims.remove('theta')
        #     dims.remove('az')
        # except ValueError as e:
        #     raise ValueError(f"Facet dimensions 'theta' and 'az' must be in \
        #                    {self.temperature_lookup}") from e
        # return dims
