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

## Downloading the lookup tables

With roughness installed, navigate to your desired data directory and download the pre-generated lookup tables from Zenodo with:

```bash
cd /path/to/roughness_data
roughness -d
```

## Running the roughness thermophysical model

The roughness thermophysical model can be run from Python by importing the roughness package and supplying the required geometry to the rough_emission function:

```python
import roughness
roughess.rough_emission(geometry, wavelengths, rms, tparams)
```

For more examples, see the [examples](examples.md) page.

## Roughness Primer: Types of roughness

### Gaussian RMS (Shepard et al., 1995)

RMS roughness is given by equation 13 in Shepard et al. (1995).

$P(\theta) =\frac{tan\theta}{\bar{tan\theta}^2} exp(\frac{-tan^2 \theta}{2 tan^2 \bar{\theta}})$

![RMS slope distributions](img/rms_slopes.png)

### Theta-bar (Hapke, 1984)

Theta-bar roughness is computed using equation 44 in Hapke (1984). It is given by:

$P(\theta) = A sec^2 \theta \ sin \theta \ exp(-tan^2 \theta / B tan^2 \bar{\theta})$

Where $A = \frac{2}{\pi tan^2 \bar{\theta}}$, and $B = \pi$. This reduces to:


$P(\theta) = \frac{2}{\pi tan^2 \bar{\theta}} \ sec^2 \theta \ sin \ \theta \ exp(\frac{-tan^2 \theta}{\pi tan^2 \bar{\theta}})$

$P(\theta) = \frac{2sin \theta}{\pi tan^2 \bar{\theta} \ cos^2 \theta} \ exp(\frac{-tan^2 \theta}{\pi tan^2 \bar{\theta}})$

![Theta-bar slope distributions](img/tbar_slopes.png)

### Accounting for viewing angle

The emission angle and azimuth the surface is viewed from will change the distribution of slopes observed.

For example, when viewed from the North (azimuth=0$^o$) and an emission angle of 30$^o$, we expect to see more North facing slopes than South facing, with the difference being more pronounced at higher roughness values.

![visible slope distributions](img/vis_slopes.png)
