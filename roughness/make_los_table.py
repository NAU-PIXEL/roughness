"""Generate line of sight lookup tables."""
import os
from functools import lru_cache
import numpy as np
import helpers as rh

# from roughness import helpers as rh

try:
    import lineofsight as l
except ModuleNotFoundError as e:
    rh.compile_lineofsight()
    import lineofsight as l

ROUGHNESS_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(ROUGHNESS_DIR, "data")

# Define output files
FZSURF = os.path.join(DATA_DIR, "zsurf.npy")
FZSURF_FACTORS = os.path.join(DATA_DIR, "zsurf_scale_factors.npy")
FTOT_FACETS = os.path.join(DATA_DIR, "total_facets_4D.npy")
FLOS_FACETS = os.path.join(DATA_DIR, "los_facets_4D.npy")
FLOS_PROB = os.path.join(DATA_DIR, "los_prob_4D.npy")


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


def make_conj_sym(inarray, n, phase=0):
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
        out[1:n, 1:Ny] = np.rot90(out[1:n, Ny + 1 : n], 2)
        out[DC, 1:Ny] = np.rot90(
            out[DC, Ny + 1 : n].reshape(-1, 1), 2
        ).reshape(-1)
        out[Ny + 1 : n, DC] = np.rot90(
            out[1:Ny, DC].reshape(-1, 1), 2
        ).reshape(-1)
        out[Ny + 1 : n, Ny] = np.rot90(
            out[1:Ny, Ny].reshape(-1, 1), 2
        ).reshape(-1)

    elif phase == 1:
        out[1:n, 1:Ny] = -np.rot90(out[1:n, Ny + 1 : n], 2)
        out[DC, 1:Ny] = -np.rot90(
            out[DC, Ny + 1 : n].reshape(-1, 1), 2
        ).reshape(-1)
        out[Ny + 1 : n, DC] = -np.rot90(
            out[1:Ny, DC].reshape(-1, 1), 2
        ).reshape(-1)
        out[Ny + 1 : n, Ny] = -np.rot90(
            out[1:Ny, Ny].reshape(-1, 1), 2
        ).reshape(-1)

    # Return symmetric array.
    return out


def make_zsurf(n=10000, H=0.8, qr=100, fout=FZSURF):
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
    fout (str): Path to write z surface.

    Returns
    -------
    zsurf (n x n arr): Artifical z surface with self-affine roughness.
    """
    # Try to load zsurf
    zsurf = load_zsurf(fout)
    if zsurf is not None and zsurf.shape == (n, n):
        print(f"Loaded zsurf from {fout}")
        return zsurf

    # Make sure the dimensions are even
    print(f"Making new z surface...")
    if n % 2:
        n = n - 1

    # Detemine image lengths. This implementation is finite but can be
    # modified for continuous wavevectors where q->infinity
    pixelwidth = 1 / n

    # Build the 1D wavevector, 1D FFT shift, unwrap to adjust for 2 pi phase.
    q = (
        np.unwrap(np.fft.fftshift(((2 * np.pi / n) * np.arange(n))))
        - 2 * np.pi
    ) / pixelwidth

    # Create 2D mesh of the wavevector and convert to polar coordinates.
    qxx, qyy = np.meshgrid(q, q)
    rho, _ = cart2pol(qxx, qyy)

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
    Bq = make_conj_sym(Bq, n)

    # Here is where we introduce the randon gaussian portion.
    noise = np.random.rand(n, n)
    Phi = (2 * np.pi) * noise - np.pi
    Phi = make_conj_sym(Phi, n, phase=1)

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
    np.save(fout, norm_zsurf)
    return norm_zsurf


@lru_cache(1)
def load_zsurf(path=FZSURF):
    """
    Return z surface with self-affine roughness at path.

    Parameters
    ----------
    path (optional: str): Path to zsurf numpy array

    Returns
    -------
    zsurf (2D array): Artifical z surface with self-affine roughness.
    """
    try:
        zsurf = np.load(path, allow_pickle=True)
    except FileNotFoundError:
        zsurf = None
    return zsurf


def get_slope_aspect(dem, get_aspect=True):
    """
    Return the instantaneous slope and aspect (if get_aspect) for dem.

    Both values derived using NumPy's gradient method by methods from:

    Ritter, Paul. "A vector-based slope and aspect generation algorithm."
    Photogrammetric Engineering and Remote Sensing 53, no. 8 (1987): 1109-1111

    Parameters
    ----------
    dem(height) (2D array): A two dimensional array of elevation data.
    get_aspect (bool): Whether to also compute and return aspect (default True)

    Returns
    -------
    slope (2D array): A two dimensional array of slope data.
    aspect (2D array, optional): A two dimensional array of aspect data.
    """
    # We use the gradient function to return the dz_dx and dz_dy.
    dz_dy, dz_dx = np.gradient(dem)

    # Slope is returned in degrees through simple calculation.
    # No windowing is done (e.g. Horn's method).
    slope = np.rad2deg(np.arctan(np.sqrt(dz_dx ** 2 + dz_dy ** 2)))
    if get_aspect:
        aspect = np.rad2deg(np.arctan2(-dz_dy, dz_dx))
        # Convert to clockwise about Y and restrict to (0, 360) from North
        aspect = (270 + aspect) % 360

        # TODO: do we need this line?
        # If no slope we don't need aspect
        aspect[slope == 0] = 0
        return slope, aspect
    return slope


def make_scale_factors(
    zsurf, rms_min=0, rms_max=50, nrms=10, write=True, fout=FZSURF_FACTORS
):
    """
    Determine scale factors to impose a specific RMS slope to synthetic surface.

    Returns
    -------
    10 element vector of multiplicative coefficients from 5-50 degrees RMS slope
    """
    rms_arr = np.linspace(rms_min, rms_max, nrms + 1)
    # Try to load scale factors and check they are correct length
    scale_factors = load_zsurf_scale_factors(fout)
    if scale_factors is not None and len(scale_factors) == len(rms_arr):
        print(f"Loaded scale_factors from {fout}")
        return scale_factors

    # Compute slope at range of factors, fit and infer slope at rms_arr
    print(f"Calculating new scale factors...")
    factors = np.arange(0, 105, 5, dtype="f8")  # TODO remove hardcoded values
    rms_derived = np.zeros_like(factors)

    # Calculate RMS slope for all factors
    for i, factor in enumerate(factors):
        rms_derived[i] = get_rms(
            get_slope_aspect(zsurf * factor, get_aspect=False)
        )

    # Fit with high order polynomial, infer factors at rms_arr
    rms_mult = np.polynomial.Polynomial.fit(rms_derived, factors, 9)
    scale_factors = rms_mult(rms_arr)
    if write:
        np.save(fout, scale_factors)
    return scale_factors


@lru_cache(1)
def load_zsurf_scale_factors(path=FZSURF_FACTORS):
    """
    Return scale factors at path.

    Parameters
    ----------
    path (optional: str): Path to zsurf numpy array

    Returns
    -------
    scale_factors (1D array): : Scale factors for zsurf.
    """
    try:
        zsurf_factors = np.load(path)
    except FileNotFoundError:
        zsurf_factors = None
    return zsurf_factors


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


def raytrace(dem, inc, azim):
    """
    Return line of sight boolean array of 2D dem viewed from (inc, azim).

    Wraps lineofsight raytrace module.

    Parameters
    ----------
    dem (2D arr): Input DEM to ray trace where values are z-height.
    inc (1D arr): Input vector incidence angle (0 is nadir).
    azim (1D arr): Input vector azimuth angle (0 is North).

    Returns
    -------
    out (2D bool arr, dem.shape): True - in los; False - not in los
    """
    # Make an input array where all values are illuminated
    los = np.ones_like(dem)
    # If inc == 0 everything is illuminated. No point in calling function.
    if inc != 0:
        # Passing the proper inputs to the fortran module.
        los = l.lineofsight(dem.flatten(), inc, azim, los)
    return los.T.astype(bool)


def bin_slope_azim(slope, azim, slope_bins, azim_bins):
    """
    Return count of facets in each slope, azim bin using 2d histogram.

    Parameters
    ----------
    slope (1D arr): Facet slope values (e.g. flattened from get_slope_aspect)
    azim (1D arr): Facet azim values (e.g. flattened from get_slope_aspect)
    slope_bins (1D arr): Bin edges of slopes.
    azim_bins (1D arr): Bin edges of azims.

    Return
    ------
    counts (2D arr nslope, naz): Counts in each slope/az bin; shape defined by bins.
    """
    hist = np.histogram2d(slope, azim, bins=(slope_bins, azim_bins))
    counts = hist[0].T  # Hist 2D transposes output
    return counts


def get_dem_los_buffer(dem, inc):
    """
    Return in_buffer, sections of dem too close to the edge to have shadows.

    Since dem is of finite size and illuminated from the West at some inc, some
    facets towards the left edge of dem cannot have shadows cast onto them. We
    compute the max possible shadow length and exclude pixels within that
    length from the western edge of dem.

    Parameters
    ----------
    dem (2D arr): Input DEM to ray trace where values are z-height.
    inc (1D arr): Input vector incidence angle (0 is nadir).

    Return
    ------
    in_buffer (2D arr): Bool array of whether dem is in buffer
    """
    # Find peak-to-peak of DEM
    elev_span = np.ceil(np.ptp(dem))

    # Calculate longest shadow for buffer.
    buffer = int(elev_span * np.tan(np.deg2rad(inc)))

    # Make in_buffer 2D arr: Pixels west of buffer == True, else False
    in_buffer = np.zeros(dem.shape, dtype=bool)
    in_buffer[:, : buffer + 1] = True
    return in_buffer


def make_los_table(
    nrms=10,
    ncinc=10,
    naz=36,
    ntheta=45,
    threshold=25,
    ftot_facets=FTOT_FACETS,
    flos_facets=FLOS_FACETS,
    flos_prob=FLOS_PROB,
    write=True,
):
    """
    Generate line of sight probability table with dims (rms, inc, slope, az).

    Tables are generated using raytracing on a synthetic Gaussian surface with
    a range of RMS slopes and observed from a range of solar inc angles. Final
    los_prob table is binned by facet slope/azimuth:

    - los_prob == 1: All facets in line of sight (i.e. visible / illuminated)
    - los_prob == 0: No facets in line of sight (i.e. not visible / shadowed)
    - los_prob == np.nan: Fewer than threshold facets observed in bin

    Parameters
    ----------
    nrms: Number of RMS slopes. Fixed value for now.
    ninc: Number of incidence Angles. Default=18
    naz: Number of azimuth bins. Default=36
    ntheta: Number of slope bins. Default=45
    threshold (int): Minimum number of valid facets to do binning, else nan

    Returns
    -------
    No values are returned. Three tables are saved in roughness data directory.
    total_facets_4D.npy: Number of facets in slope/azim bin.
    los_facets_4D.npy: Number of facets in line of sight in slope/azim bin.
    los_prob_4D.npy: Probability of facets in line of sight per slope/azim bin.
    """
    # Indicies for reporting progress
    niter = nrms * ncinc

    # Load in the synthetic DEM and scale factors
    zsurf = make_zsurf()
    factors = make_scale_factors(zsurf, nrms=nrms)

    # Init total and line of sight facet arrays binned by slope/azim
    tot_facets = np.full((nrms, ncinc, naz, ntheta), np.nan)
    los_facets = np.copy(tot_facets)

    # Setup our rms, inc and slope / azim bin arrays
    # TODO: remove hardcoded values
    rms_arr = np.linspace(0, 50, nrms)
    cincs = np.linspace(1, 0, ncinc, endpoint=False)  # (0, 90) degrees
    slopes = np.linspace(0, 90, ntheta + 1)
    azims = np.linspace(0, 360, naz + 1)
    incs = np.rad2deg(np.arccos(cincs))

    for r, rms in enumerate(rms_arr):
        # Flat surface => all facets in line of sight at any inc
        if rms == 0:
            # Cludge: at rms=0 most facets are undefined, but we expect all to
            # be visible. Set to threshold to ensure los_prob == 1
            tot_facets[r, :, :, :] = threshold
            los_facets[r, :, :, :] = threshold
            continue

        # Scale the dem to the prescirbed RMS slope and get slope, azim arrays
        dem = zsurf * factors[r]
        slope, azim = get_slope_aspect(dem)
        slope = slope.flatten()
        azim = azim.flatten()

        # Get binned counts of total facets in dem (doesn't change with inc)
        tot_facets[r, :, :, :] = bin_slope_azim(slope, azim, slopes, azims)
        for i, inc in enumerate(incs):
            print(f"{(r*ncinc+i)/niter:.2%}...........", end="\r", flush=True)
            # Nadir view => all facets in line of sight
            if inc == 0:
                los_facets[r, i, :, :] = tot_facets[r, i, :, :]
                continue

            # Run ray trace (always from West for now) and get azims in los
            los = raytrace(dem, inc, 270).flatten()

            # Get buffer beyond which raytrace breaks down due to finite dem
            in_buf = get_dem_los_buffer(dem, inc).flatten()

            # Get only slope and azim in los and not in buffer
            los_buf = los * ~in_buf
            azim_los_buf = azim[los_buf]
            slope_los_buf = slope[los_buf]

            # Get binned counts of facets in los (exclude those west of buffer)
            los_facets[r, i, :, :] = bin_slope_azim(
                slope_los_buf, azim_los_buf, slopes, azims
            )

    # If bins have fewer than threshold total facets -> set outputs to nan
    tot_facets[tot_facets < threshold] = np.nan
    los_facets[tot_facets < threshold] = np.nan

    # Get facet probabilities (fraction of facets in lineofsight / total)
    with np.errstate(divide="ignore"):  # Suppress div warning
        los_prob = los_facets / tot_facets

    if write:
        np.save(ftot_facets, tot_facets)
        np.save(flos_facets, los_facets)
        np.save(flos_prob, los_prob)
        print("\nWrote:")
        print("\n".join([ftot_facets, flos_facets, flos_prob]))
    return tot_facets, los_facets, los_prob


if __name__ == "__main__":
    _ = make_los_table()
