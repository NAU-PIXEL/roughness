"""Module containing all functions for generating a shadow lookup table. """
import os
import numpy as np
import lineofsight as l

ROUGHNESS_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(ROUGHNESS_DIR,"data")
ZSURF = os.path.join(ROUGHNESS_DIR, "data", "zsurf.npy")
ZSURF_FACTORS = os.path.join(ROUGHNESS_DIR, "data", "zsurf_scale_factors.npy")
CACHE = {}  # Cache lookup table(s) to reduce file I/O


def cart2pol(x, y):
    """
    Convert ordered coordinate pairs from Cartesian (X,Y) to polar (r,theta).

    Parameters
    ----------
    X: X component of ordered Cartesian coordinate pair.
    Y: Y component of ordered Cartesian coordinate pair.

    Returns
    -------
    r: Distance from origin.
    theta: Angle, in radians.
    """
    r = np.sqrt(x ** 2 + y ** 2)
    theta = np.arctan2(y, x)
    return (r, theta)


def pol2cart(rho, phi):
    """
    Convert ordered coordinate pairs from polar (r,theta) to Cartesian (X,Y).

    Parameters
    ----------
    r: Distance from origin.
    theta: Angle, in radians.

    Returns
    -------
    X: X component of ordered Cartesian coordinate pair.
    Y: Y component of ordered Cartesian coordinate pair.
    """
    x = rho * np.cos(phi)
    y = rho * np.sin(phi)
    return (x, y)

def make_conj_sym(inarray,n,phase=0):
    """
    Takes square mag/phase arrays and impose conjugate symmetry for ifft.

    Parameters
    ----------
    inarray: Square array that will be fed to ifft.
    n: Single axis dimension.
    phase: If the input is the phase portion of the complex FFT, set phase = 1.

    Returns
    -------
    out: inarray modified to have conjugate symmetry.
    """
    # Copy inarray for manipulation.
    out = np.copy(inarray)

    # Define critical indicies (e.g., DC, Nyquist positions)
    DC = 0
    Ny = int(n / 2)

    # Set the critical indices to 0.
    out[DC, DC] = 0
    out[DC, Ny] = 0
    out[Ny, Ny] = 0
    out[Ny, DC] = 0

    # Perform the Hermitian transpose. If input is phase, transpose is negative.
    if phase == 0:
        out[1:n, 1 : Ny] = np.rot90(out[1:n, Ny + 1 : n], 2)
        out[DC, 1 : Ny] = np.rot90(out[DC, Ny + 1 : n].reshape(-1, 1), 2).reshape(-1)
        out[Ny + 1 : n, DC] = np.rot90(out[1 : Ny, DC].reshape(-1, 1), 2).reshape(-1)
        out[Ny + 1 : n, Ny] = np.rot90(out[1 : Ny, Ny].reshape(-1, 1), 2).reshape(-1)

    elif phase == 1:
        out[1:n, 1 : Ny] = -np.rot90(out[1:n, Ny + 1 : n], 2)
        out[DC, 1 : Ny] = -np.rot90(out[DC, Ny + 1 : n].reshape(-1, 1), 2).reshape(-1)
        out[Ny + 1 : n, DC] = -np.rot90(out[1 : Ny, DC].reshape(-1, 1), 2).reshape(-1)
        out[Ny + 1 : n, Ny] = -np.rot90(out[1 : Ny, Ny].reshape(-1, 1), 2).reshape(-1)

    # Return symmetric array.
    return out

def make_zsurf(n=10000, H=0.8, qr=100):
    """
    Return a synthetic surface with self-affine roughness.
    Much of this is detailed in this paper:
    Tevis D B Jacobs et al 2017 Surf. Topogr.: Metrol. Prop. 5 013001
    https://doi.org/10.1088/2051-672X/aa51f8

    Parameters
    ----------
    n: Size in the square array.
    H: Hurst exponent.
    qr: Rolloff wavelength in units of q (wavevector; 1/m).

    Returns
    -------
    z(height) (size x size, arr): Artifical z surface with self-affine roughness.
    """
    # Make sure the dimensions are even.
    if n % 2:
        n = n - 1

    # Detemine image lengths. This implementation is finite but can be
    # modified for continuous wavevectors where q->infinity
    pixelwidth = 1 / n

    # Build the 1D wavevector, 1D FFT shift, unwrap to adjust for 2 pi phase.
    q = (np.unwrap(np.fft.fftshift(((2 * np.pi / n) * np.arange(n)))) - 2 * np.pi) / pixelwidth

    # Create 2D mesh of the wavevector and convert to polar coordinates.
    qxx, qyy = np.meshgrid(q, q)
    rho, phi = cart2pol(qxx, qyy)

    # Make a 2D PSD zeros matrix and fill it with the polar rho values
    # from the 2D wavevectors. First line account for the rolloff by making it
    # constant. This also properly accounts for Hurst exponent by raising the
    # wavevectors to (-2*(H+1))
    Cq = np.zeros((n, n))
    Cq[rho < qr] = qr ** (-2 * (H + 1))
    Cq[rho >= qr] = rho[rho >= qr] ** (-2 * (H + 1))

    # The center of the PSD is the ifft mean. We set this to 0.
    Cq[int(n / 2), int(n / 2)] = 0

    # This section imposes conjugate symmetry for  mag/phase input to ifft
    Bq = np.sqrt(Cq / (pixelwidth ** 2 / ((n * n) * ((2 * np.pi) ** 2))))
    Bq = make_conj_sym(Bq,n)

    # Here is where we introduce the randon gaussian portion.
    noise = np.random.rand(n, n)
    Phi = (2 * np.pi) * noise - np.pi
    Phi = make_conj_sym(Phi,n,phase=1)

    # Now we convert back to cartesian coordinates.
    a, b = pol2cart(Bq, Phi)

    # Time to invert the FFT and generate our topography
    # Need to allocate an empty complex array similar in size to our Bq.
    out = np.empty_like(Bq, dtype="complex128")

    # NumPy complex arrays allow definition of the real and imaginary components.
    out.real = a
    out.imag = b

    # Call the inverse FFT shift to put our low frequency stuff to the margins.
    out = np.fft.ifftshift(out)

    # And finally we call the inverse FFT.
    # Because we went through all that trouble to make the array symmetric this
    # should return a real, float array and make iFFT run faster.
    zsurf = np.fft.ifft2(out)

    # But it won't do that because of rounding so we call it real if close.
    zsurf = np.real_if_close(zsurf)

    # Because we are going to scale to various RMS slopes we want to normalize.
    norm_zsurf = (zsurf - np.mean(zsurf)) / np.std(zsurf)

    # Voila. A real array of height values with self-affine roughness.
    # Save it!
    np.save(os.path.join(DATA_DIR, "zsurf.npy"), norm_zsurf)

def load_zsurf(path=ZSURF):
    """
    Return zsurf at path and add to cache if not loaded previously.

    Parameters
    ----------
    path (optional: str): Path to zsurf numpy array

    Returns
    -------
    zsurf (2D array): : Artifical z surface with self-affine roughness.
    """
    if not os.path.exists(ZSURF):
        print('zsurf.npy not found. Making new surface.')
        make_zsurf()

    if path not in CACHE:
        CACHE[path] = np.load(path, allow_pickle=True)
    return CACHE[path]

def get_slope(dem):
    """
    Return the instantaneous slopefor a dem using NumPy's gradient method.
    Slope and aspect are determined using methods described in this paper:
      Ritter, Paul. "A vector-based slope and aspect generation algorithm."
      Photogrammetric Engineering and Remote Sensing 53, no. 8 (1987): 1109-1111

    Parameters
    ----------
    dem(height) (2D array): A two dimensional array of elevation data.

    Returns
    -------
    slope (2D array): A two dimensional array of slope data.
    """
    # We use the gradient function to return the dz_dx and dz_dy.
    dz_dy, dz_dx = np.gradient(dem)
    # Flatten the arrays to make them easily iterable for the aspect crap.
    # First I am storing the original dimensions so I can reshape upon return.
    dims = dem.shape
    dz_dy = dz_dy.flatten()
    dz_dx = dz_dx.flatten()

    # Slope is returned in degrees through simple calculation.
    # No windowing is done (e.g. Horn's method).
    slope = np.rad2deg(np.arctan(np.sqrt(dz_dx ** 2 + dz_dy ** 2)))

    return slope.reshape(dims)

def get_slope_aspect(dem):
    """
    Return the instantaneous slope and aspect for a dem. Both values is derived
    using NumPy's gradient method.
    Slope and aspect are determined using methods described in this paper:
      Ritter, Paul. "A vector-based slope and aspect generation algorithm."
      Photogrammetric Engineering and Remote Sensing 53, no. 8 (1987): 1109-1111

    Parameters
    ----------
    dem(height) (2D array): A two dimensional array of elevation data.

    Returns
    -------
    slope (2D array): A two dimensional array of slope data.
    aspect (2D array): A two dimensional array of aspect data.
    """
    # We use the gradient function to return the dz_dx and dz_dy.
    dz_dy, dz_dx = np.gradient(dem)
    # Flatten the arrays to make them easily iterable for the aspect crap.
    # First I am storing the original dimensions so I can reshape upon return.
    dims = dem.shape
    dz_dy = dz_dy.flatten()
    dz_dx = dz_dx.flatten()

    # Slope is returned in degrees through simple calculation.
    # No windowing is done (e.g. Horn's method).
    slope = np.rad2deg(np.arctan(np.sqrt(dz_dx ** 2 + dz_dy ** 2)))
    # Preallocate an array for the aspect.
    aspect = np.rad2deg(np.arctan(dz_dy / dz_dx))

    # Convert the arctan output to azimuth in degrees relative to North.

    # If no slope we don't need aspect. Set to 0 because 360 is North.
    aspect[slope ==0] = 0

    # Aspect is initially from -pi to pi clockwise about the x-axis.
    # We need to translate to clockwise about Y. To avoid dividing by 0
    # we're just going to explicitly set these to North or South based on
    # the sign of dz_dy.
    aspect[(dz_dx == 0) & (dz_dy < 0)] = 180
    aspect[(dz_dx == 0) & (dz_dy > 0)] = 360

    # Here we make our offsets to account for counterclockwise about
    # y-axis and 0-360.
    aspect[dz_dx > 0] = 270 - aspect[dz_dx > 0]
    aspect[dz_dx <= 0] = 90 - aspect[dz_dx <= 0]

    return slope.reshape(dims), aspect.reshape(dims)

def make_scale_factors():
    """
    Determine scale factors to impose a specific RMS slope to synthetic surface.

    Returns
    -------
    10 element vector of multiplicative coefficients from 5-50 degrees RMS slope
    """
    # Load the synthetic surface array from the numpy data type.
    zsurf = load_zsurf()

    # Create arrays for RMS coefficient/slope fitting.
    coeff=np.arange(0,105,5,dtype='f8')
    rawRMS=np.zeros_like(coeff)

    # Calculate RMS slope for all coeffcients.
    for index, value in enumerate(coeff):
        rawRMS[index] = get_rms(get_slope(zsurf*value))

    # Now fit the data with a high order polynomial.
    RMSmult = np.polynomial.Polynomial.fit(rawRMS,coeff,9)

    rough = np.arange(5,55,5)
    np.save(os.path.join(DATA_DIR, "zsurf_scale_factors.npy"), RMSmult(rough))

def load_zsurf_scale_factors(path=ZSURF_FACTORS):
    """
    Return scale factors at path.

    Parameters
    ----------
    path (optional: str): Path to zsurf numpy array

    Returns
    -------
    scale_factors (1D array): : Scale factors for zsurf.
    """
    if not os.path.exists(ZSURF_FACTORS):
        print('zsurf_scale_factors.npy not found. Calculating new factors.')
        make_scale_factors()

    return np.load(path)

def get_rms(y):
    """
    Compute the root mean square of an array, usually a Y array.

    Parameters
    ----------
    y (vector array): Array of values where the variance needs to be computes.

    Returns
    -------
    RMS (value): The RMS value.
    """
    return np.sqrt(np.mean(y ** 2.0))

def raytrace(dem,elev,azim):
    """
    Wrapper for preparing input to the lineofsight raytrace module.

    Parameters
    ----------
    dem: Input DEM to ray trace. 2D array where values are z-height.
    elev: Input vector elevation 0 is directly overhead.
    azim: Input vector azimuthal direction.

    Returns
    -------
    out: A 2D array with size of DEM where values are 0 or 1 where 0 == shadowed
    """
    # Make an input array where all values are illuminated
    los=np.ones_like(dem)
    # If elev == 0 everything is illuminated. No point in calling function.
    if (elev == 0):
        los=los
    else:
    # Passing the proper inputs to the fortran module.
        los=l.lineofsight(dem.flatten(),elev,azim,los)
    return los.T

def get_shadowed_slope_aspect(slope, azim, shademap, naz=36, ntheta=45):
    """
    Determine the shadowed percentage of facets for defined slope/aspect bins.

    Parameters
    ----------
    slope: Two dimensional array of slope data.
    azim: Two dimensional array of aspect data.
    shademap: The two dimensional output of raytrace.
    naz (int): Number of azimuth bins. Default=36
    ntheta (int): Number of slope bins. Default=45

    Returns
    -------
    slope (2D array): A two dimensional array of slope data.
    aspect (2D array): A two dimensional array of aspect data.
    """
    # Set up slope bin edges and steps.
    theta_step=90//ntheta
    theta_edge=np.arange(0,ntheta*theta_step,theta_step)

    # Make out output arrays. We want nan for now.
    out=np.full((naz,ntheta),np.nan)
    bin_pop=np.copy(out)
    shade_pop=np.copy(out)

    # Loop through slope bins.
    for slope_index , bin_val in enumerate(theta_edge):
        # Make a temporary azim array that we can mess with.
        all_azim=azim[(slope>=bin_val) & (slope<bin_val+theta_step)].flatten()

        # If there are azimuths that meet the criteria, do some binning.
        if len(all_azim)>0:
            # Use histogram to make the azimuth bins for this slope bin.
            azhist=np.histogram(all_azim,range=(0,360),bins=naz)[0]

            # Write out the full azimuth bin population.
            bin_pop[:,slope_index] = azhist

            # Find only shadowed facets in this slope bin.
            shade_azim=azim[(slope>bin_val) & (slope<bin_val+theta_step) & (shademap==0)].flatten()
            # Use histogram to make the azimuth bins for this slope bin.
            shadehist=np.histogram(shade_azim,range=(0,360),bins=naz)[0]

            # Write out the shaded azimuth bin population.
            shade_pop[:,slope_index] = shadehist

            # Write out the shadowed fraction.
            out[:,slope_index] = shadehist/azhist

    return bin_pop, shade_pop, out

def make_shade_table(nrms=10,ninc=18,naz=36,ntheta=45):
    """
    Generate shadow lookup table for a synthetic Gaussian surface with varying
    RMS slope and sloar incidence. Fraction is binned by facet slope/azimuth.
    Number of incidence angles and slope/azimuth bins can be adjusted.

    Parameters
    ----------
    nrms: Number of RMS slopes. Fixed value for now.
    ninc: Number of incidence Angles. Default=18
    naz: Number of azimuth bins. Default=36
    ntheta: Number of slope bins. Default=45

    Returns
    -------
    No values are returned. Three tables are saved in roughness data directory.
    total_bin_population_4D.npy: Number of facets in slope/azim bin.
    shade_bin_population_4D.npy: Number of shadowed facets in slope/azim bin.
    shadow_fraction_4D.npy: Fraction of shadowed facets per slope/azim bin.
    """
    # Indicies for reporting progress
    elems = nrms*ninc
    itnum = 0
    # Load in the synthetic DEM and scale factors
    zsurf=load_zsurf()
    mult=load_zsurf_scale_factors()

    # Make our output arrays.
    # Number of facets per slope/azim bin
    bin_pop=np.full((nrms,ninc,naz,ntheta),np.nan)
    # Number of shaded facets per slope/azim bin
    shade_pop=np.copy(bin_pop)
    # shade_pop/bin_pop
    shadow_frac=np.copy(bin_pop)

    # Setup our incidence angle steps
    inc_step=90//ninc
    inc=np.arange(inc_step,90+inc_step,inc_step)

    for rms_index, r in enumerate(mult):
        # Scale the dem to the prescirbed RMS slope.
        dem=zsurf*r

        for inc_index , i in enumerate(inc):
            # Run the ray trace. Always from West for now.
            tmpshade=raytrace(dem,i,270)

            # Find peak-to-peak of DEM
            elev_span=np.ceil(np.ptp(dem))

            # Calculate longest shadow for buffer.
            qbuffer=int(elev_span*np.tan(np.deg2rad(i)))

            # Calculate slope/aspect then trim buffer length from west edge.
            slope,azim=get_slope_aspect(dem)
            slope=slope[:,qbuffer:-1]
            azim=azim[:,qbuffer:-1]

            # Trim buffer for shademap. Point here is that
            # the west edge beyond the DEM can't cast a
            # shadow onto this area so we remove it.
            shademap=tmpshade[:,qbuffer:-1]

            # Determine the total shadowed fraction.
            total_cells=shademap.shape[0]*shademap.shape[1]
            illum_cells=np.sum(shademap.flatten())
            shaded_cells=total_cells-illum_cells

            tot_shade=shaded_cells/total_cells

            # If there are shadows, continue with binning.
            if tot_shade>0:
                bin_pop[rms_index,inc_index,:,:], \
                shade_pop[rms_index,inc_index,:,:], \
                shadow_frac[rms_index,inc_index,:,:] = get_shadowed_slope_aspect(slope,azim,shademap)

                itnum = itnum + 1
                print(round((itnum / elems) * 100, 2),"%...........",end="\r",flush=True,)

    np.save(os.path.join(DATA_DIR, "total_bin_population_4D.npy"), bin_pop)
    np.save(os.path.join(DATA_DIR, "shade_bin_population_4D.npy"), shade_pop)
    np.save(os.path.join(DATA_DIR, "shadow_fraction_4D.npy"), shadow_frac)
