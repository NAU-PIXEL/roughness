# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.11.2
#   kernelspec:
#     display_name: 'Python 3.9.4 64-bit (''.venv'': poetry)'
#     name: python3
# ---

# %% [markdown]
# # Make Line of Sight Lookup Table

# %%
import numpy as np
import matplotlib.pyplot as plt
import roughness.make_los_table as mlt
import roughness.plotting as rp

# Make zsurf to pass to make_los_table.
# The larger the better (but also slower). Default size is 10000.
# If you do write=True, it'll look for fout and only remake the 1st time
zsurf = mlt.make_zsurf(1000, write=True, fout="tmp.npy")
zsurf2 = zsurf.copy()
zsurf2[400:600, 400:600] = 10 * zsurf2.max()
# zsurf = mlt.make_zsurf(10000, write=True, fout='ben_zsurf_10000.npy')
rmss = np.arange(0, 31, 10)  # array of rms values to consider
incs = np.arange(0, 90, 10)  # array of incidence angles to raytrace at
lt = mlt.make_los_table(zsurf, rmss, incs, thresh=0)  # thresh 0 for debugging
lt2 = mlt.make_los_table(zsurf2, rmss, incs, thresh=0)
# lt = mlt.make_lost_table(zsurf, rmss, incs, thresh=100, fout='ben_los_table_10000.nc')

# Plot the rough surface
# plt.imshow(zsurf, cmap='gray')

# Plot the table interpolated to a certain rms and inc
rp.plot_slope_az_table(lt.prob.interp(rms=50, inc=86.5), proj="polar")

# Figure out where we have shadow probs and print how many facets were in those bins
# plt.Figure()
# facet_counts = lt.total.where(~lt.prob.isnull())
# facet_counts.sel(rms=40, inc=87.5, az=5).plot()
# plt.axhline(100, ls='--', c='r', label='100 facets')
# plt.legend()

# # Print total facets for solar inc vs. facet slope
# plt.Figure()
# lt.total.sel(rms=30, az=5, inc=slice(88, None)).plot()

# %%
from pathlib import Path

Path("tmp.npy").stem

# %%
plt.imshow(zsurf2)

# %%
rp.plot_slope_az_table(
    lt.sel(theta=slice(0, 60)).prob.interp(rms=30, inc=60), proj="polar"
)


# %%
delta = lt2.sel(theta=slice(0, 60)).prob.interp(rms=30, inc=60) - lt.sel(
    theta=slice(0, 60)
).prob.interp(rms=30, inc=60)
rp.plot_slope_az_table(delta, proj="polar")

# %%
rp.plot_slope_az_table(
    lt2.sel(theta=slice(0, 60)).prob.interp(rms=30, inc=60), proj="polar"
)


# %%
zsurf.shape

# %%
zsurf.max()

# %%
zsurf[400:600, 400:600] = zsurf.max()

# %%
plt.imshow(zsurf)

# %%
facet_counts = lt.total.where(~lt.prob.isnull())
facet_counts.sel(rms=40, inc=87.5, az=5).plot()
plt.axhline(100, ls="--", c="r", label="100 facets")
plt.legend()
plt.show()

# %%
# Print total facets for solar inc vs. facet slope
lt.total.sel(rms=30, az=5, inc=slice(88, None)).plot()

# %%
fsf = "/home/ctaiudovicic/projects/roughness/data/zsurf_scale_factors.npy"
sf = mlt.load_zsurf_scale_factors(fsf)
sf

# %%
dem = zsurf * sf[5]
rt = mlt.raytrace(dem, 60, 270)
plt.imshow(rt)
plt.colorbar()

# %%
in_buf = mlt.get_dem_los_buffer(dem, incs[5])
plt.imshow(in_buf)
plt.colorbar()

# %%
lt.total.interp(rms=20, inc=87).mean("az").plot()

# %%
