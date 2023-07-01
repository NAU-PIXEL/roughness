# Getting Started

The roughness python model requires 3 components to run:

- The package itself (see installation instructions)
- A raytrace shadowing lookup table (see [make_los_table](make_los_table.md))
- A temperature lookup table (see [make_temp_table](make_temp_table.md))

Basic versions of the generaic shadowing table and lunar temperature lookup table are provided on Zenodo (coming soon), but custom versions of each can be generated using the tools in the roughness package.

## Installation

The roughness package can be installed using pip:

```bash
pip install roughness
```

## Roughness workflow

The roughness thermophysical model requires 3 main components:

1. The `roughness` python` package
2. A raytracing shadowing lookup table
3. A surface temperature lookup table

The `roughness` package can fetch pre-generated shadowing and temperature lookup tables that were derived and validated for the Moon. Custom shadowing tables can be generated using the roughness package, and custom surface temperature tables can be generated with an arbitrary thermal model (but must be saved in xarray-readable format).

## Downloading the lookup tables

With roughness installed, download the pre-generated lunar shadowing and tempeature lookup tables with:

```bash
roughness -d
```

Calling this command again will check for updates and download new tables only if an update is available.

## Running the roughness thermophysical model

The roughness thermophysical model can be run from Python by importing the `RoughnessThermalModel` object from the roughness package and supplying the required geometry:

```python
from roughness import RoughnessThermalModel
rtm = RoughnessThermalModel()  # Init with default lookup tables
geometry = (30, 0, 60, 0)  # inc, inc_az, emit, emit_az [degrees]
wavelengths = np.arange(1, 100)  # [microns]
rms = 25  # RMS roughness [degrees]
print(rtm.required_tparams)
# {'albedo', 'lat', 'ls', 'tloc'}
tparams = {'albedo': 0.12, 'lat': 0, 'ls': 0, 'tloc': 12}  # temp table params
emission = rtm.emission(geometry, wavelengths, rms, tparams)  # returns xarray
emission.plot() # Plot emitted radiance vs. wavelength
```

For more examples, see the [examples](examples.md) page.
