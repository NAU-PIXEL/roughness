#!/usr/bin/env python3
import sys
import time
import numpy as np
import h5py
from roughness import roughness as r
from roughness import topoNshade as tns
from roughness import lineofsight as l

start_time = time.time()


################################################################################
############ Determine if the paths are provided, if so, assign them ###########
if len(sys.argv) < 1:
    print("WARNING: Missing input file!")
    print("Usage: make_shade_table.py <filepath>")
    print("    <filepath> - Path to the sim_slopes.v1.hdf as a string.")
    print(
        "     Returns - Script saves a shade table to the current directory."
    )
    sys.exit(1)

filepath = sys.argv[1]


def raytrace(dem, elev, azim):
    los = np.ones_like(dem)
    if elev == 0:
        los = los
    else:
        los = l.lineofsight(dem.flatten(), elev, azim, los)
    return los.T


# Load in the hdf DEM files that Josh used to generate the shadow table.
hf = h5py.File(filepath, "r")
# Grab the DEM, reshape, and transpose to make this identical to davinci.
data = np.array(hf["dem"][:]).reshape(10000, 10000).T
# Grab the multiplication tables
RMSmult = np.array(hf["mult_factors"]).reshape(10)

### Allocate the output arrays.
# RMS, inc
tot_shade = np.full((11, 11), np.nan)

# out will hold shadowed fraction per slope/azim bin
# bin_pop will hold the total slope/azim bin counts
# RMS, azimuth, slope, inc
# nrms, ncinc, naz, ntheta
out = np.full((11, 11, 37, 46), np.nan)
bin_pop = np.copy(out)
shade_pop = np.copy(out)

elems = 11 * 11 * 46
itnum = 0
############################################
### i loop varies the DEM RMS roughness. ###
for i in range(0, len(RMSmult)):
    # Multiplying by the RMS coefficient to get the proper roughness.
    dem = data * RMSmult[i]

    # Calculate the make shadow length so we can trim it from the western edge of the DEM.
    qbuffer = int(np.ceil(3.290527 * 2 * np.std(dem) * np.tan(np.deg2rad(84))))

    # Calculating facet slope and azim. This is a place where my method is very different from Josh's which is here:
    # slope=atand(((temp_dem[2:,1:9999]-temp_dem[1:9999,1:9999])^2+(temp_dem[1:9999,2:]-temp_dem[1:9999,1:9999])^2)^0.5)[buffer:]
    # azim=float((atan2(temp_dem[2:,1:9999]-temp_dem[1:9999,1:9999],-(temp_dem[1:9999,2:]-temp_dem[1:9999,1:9999]))+pi)/pi*180)[buffer:]
    slope, azim = tns.gradSlopeAspect(dem)
    slope = slope[:, qbuffer:-1]
    azim = azim[:, qbuffer:-1]

    #########################################################
    ### j loop varies incidence and regenerates shademap. ###
    for j in range(2, 11):
        # Calculate the incidence based on j. This is overly complicated and should be changed.
        inc = np.rad2deg(np.arccos((11 - j) * 0.1))

        # Call the fortran raytrace.
        tmpshade = raytrace(dem, inc, 270)

        # My raytrace sets illuminated areas to 1 and shadowed to 0. This is the opposite of Josh's
        # His sets shadowed areas to 1 and illuminated areas to -1 then later sets -1 to 0.
        shademap = tmpshade[:, qbuffer:-1]
        total_cells = shademap.shape[0] * shademap.shape[1]
        illum_cells = np.sum(shademap.flatten())
        shaded_cells = total_cells - illum_cells

        # tot_shade is just total shaded fraction as a function of incidence and roughness.
        tot_shade[i, j - 1] = shaded_cells / total_cells

        if tot_shade[i, j - 1] > 0:
            ##############################################################################
            ### k loop varies DEM slope and calculates azimuth bin shaded probability. ###
            for k in range(1, 46):
                # Use the value of k to calculate the bin edges.
                minslope = k * 2 - 3
                maxslope = k * 2 - 1

                # Make a temporary azim array that we can mess with.
                temp_azim = np.copy(azim)
                # If the facet slope is out of our bin range set it to -10 so we can ignore it later.
                temp_azim[slope < minslope] = -10
                temp_azim[slope >= maxslope] = -10

                # If there are azimuths that have not been nulled, do some binning.
                if np.max(temp_azim) > -10:
                    # Use histogram to make the azimuth bins for this slope bin.
                    azhist = np.histogram(
                        temp_azim.flatten(), range=(-15, 365), bins=38
                    )[0][1:38]

                    # Add 0-5 and 355-360 bins to create identical endpoints needed for extrapolation
                    azhist[0] = azhist[0] + azhist[36]
                    azhist[36] = azhist[0]

                    # Where the DEM is not shadowed, nullify it and recalculate azim bins to calc probability
                    temp_azim[shademap == 1] = -10
                    shadehist = np.histogram(
                        temp_azim.flatten(), range=(-15, 365), bins=38
                    )[0][1:38]
                    shadehist[0] = shadehist[0] + shadehist[36]
                    shadehist[36] = shadehist[0]

                    bin_pop[i, j - 1, :, k - 1] = azhist
                    shade_pop[i, j - 1, :, k - 1] = shadehist
                    out[i, j - 1, :, k - 1] = shadehist / azhist

                    itnum = itnum + 1
                    print(
                        round((itnum / elems) * 100, 2),
                        "%...........",
                        end="\r",
                        flush=True,
                    )

np.save(r.ROUGHNESS_DIR + "/data/total_shadow_fraction.npy", tot_shade)
np.save(r.ROUGHNESS_DIR + "/data/total_bin_population_4D.npy", bin_pop)
np.save(r.ROUGHNESS_DIR + "/data/shade_bin_population_4D.npy", shade_pop)
np.save(r.ROUGHNESS_DIR + "/data/shade_lookup_4D.npy", out)

print("--- %s seconds ---" % (time.time() - start_time))
