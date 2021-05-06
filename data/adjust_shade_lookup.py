"""C. J. Tai Udovicic May 5, 2021
This script mitigates a bug with the old shadow_lookup produced by J.B.
Old table had null values of ray-casting set to 0 (e.g. fully illuminated).
This causes very extreme slopes oriented away from sun to appear un-shadowed.
Here, we set reset all 0 values in the table to the value at 0 degree facets
New lookup is still bad... but less bad. 
Replace this script and the old lookups with new ray-cast lookup when ready.
"""
import os
import numpy as np
import matplotlib.pyplot as plt
import roughness.roughness as r

# Load old shadow_lookup
DATA_DIR = os.path.dirname(os.path.abspath(__file__))
sl = r.load_shadow_lookup(os.path.join(DATA_DIR, "shade_lookup_4D_old.npy"))
new_sl = sl.copy()

# Set shadow minimum to shadow fraction at theta = 0 (theoretical minimum)
for i, rms in enumerate(sl):
    for j, shadow_table in enumerate(rms):
        new_sl[i, j][new_sl[i, j] <= 0] = np.mean(new_sl[i, j, :, 0])

# Save new shadow_lookup
np.save(os.path.join(DATA_DIR, "shade_lookup_4D.npy"), new_sl)

# Plot comparison
fig = plt.figure(figsize=(9, 12))
theta_surf, azimuth_surf = r.get_facet_grids(
    shadow_table
)  # Surface slope/az grids
rms = 20  # rms roughness for comparison
sun_az = 270  # default solar azimuth

# Plot comparison at 3 different solar incidences
for i, sun_theta in enumerate((65, 75, 85)):
    shadow_table = r.get_shadow_table(rms, sun_theta, sun_az, sl)
    new_shadow_table = r.get_shadow_table(rms, sun_theta, sun_az, new_sl)

    # Old shadow lookup
    ax = fig.add_subplot(3, 2, 2 * i + 1, projection="3d")
    ax.view_init(35, -95)
    ax.plot_surface(theta_surf, azimuth_surf, shadow_table, cmap="viridis")
    ax.set_title(f"Old lookup (sun inc = {sun_theta}$^o$)")

    # New shadow lookup
    ax = fig.add_subplot(3, 2, 2 * i + 2, projection="3d")
    ax.view_init(45, -95)
    ax.plot_surface(theta_surf, azimuth_surf, new_shadow_table, cmap="viridis")
    ax.set_title(f"New lookup (sun inc = {sun_theta}$^o$)")
plt.show()
