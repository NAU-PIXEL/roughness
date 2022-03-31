"""Generate line of sight lookup tables."""
from pathlib import Path
from functools import lru_cache
import numpy as np
from . import helpers as rh
from .config import (
    FZSURF,
    FZ_FACTORS,
    FLOOKUP,
    VERSION,
    MIN_FACETS,
    DEMSIZE,
    AZ0,
)

lineofsight = rh.import_lineofsight()  # Imports FORTRAN lineofsight module


def make_default_los_table():
    """Return default los table."""
    rmss, incs, azs, thetas = rh.get_lookup_bins()

    # Load in the synthetic DEM and scale factors
    zsurf = make_zsurf(DEMSIZE, default=True)
    factors = make_scale_factors(zsurf, rmss, default=True)
    return make_los_table(
        zsurf, factors, rmss, incs, azs, thetas, fout=FLOOKUP
    )


def make_los_table(
    zsurf, rmss, incs, naz=36, ntheta=45, thresh=MIN_FACETS, az0=AZ0, fout=None
):
    """
    Generate line of sight probability table with dims (rms, inc, slope, az).

    Tables are generated using raytracing on a synthetic Gaussian surface with
    a range of RMS slopes and observed from a range of solar inc angles. Final
    los_lookup table is binned by facet slope/azimuth:

    - los_lookup == 1: All facets in line of sight (i.e. visible / illuminated)
    - los_lookup == 0: No facets in line of sight (i.e. not visible / shadowed)
    - los_lookup == np.nan: Fewer than threshold facets observed in bin

    Parameters
    ----------
    zsurf (n x n arr): Artifical z surface with self-affine roughness.
    rmss (arr): RMS slope values
    incs (arr): Solar incidence angle values [degrees]
    naz: Number of facet azimuth bins. Default=36
    ntheta: Number of facet slope bins. Default=45
    thresh (int): Minimum number of facets to do binning, else nan
    az0 (float): Raytrace azimuth from north in degrees. Default=270
    fout (str): Path to write table with xarray compatible extension (e.g. .nc).

    Returns
    -------
    xarray.Dataset with 3 DataArrays:
        total: Total number of facets in bin
        los: Number of facets in line of sight in bin
        prob: Fraction of facets in line of sight in bin (los/total)
    """
    # Get line of sight lookup coordinate arrays
    azbins, slopebins = rh.get_facet_bins(naz, ntheta)

    # Load or compute scale factors
    # TODO: scale factors should be computed and associated with zsurf
    if fout == FLOOKUP:
        factors = make_scale_factors(zsurf, rmss, default=True)
    elif fout is not None:
        fout = Path(fout).with_suffix("_scale_factors.npy")
        factors = make_scale_factors(zsurf, rmss, fout=fout)
    else:
        factors = make_scale_factors(zsurf, rmss)

    # Init total and line of sight facet arrays binned by slope/azim
    tot_facets = np.full((len(rmss), len(incs), naz, ntheta), np.nan)
    los_facets = np.copy(tot_facets)

    for r, rms in enumerate(rmss):
        # Flat surface => all facets in line of sight at any inc
        if rms == 0:
            # Cludge: at rms=0 most facets are undefined, but we expect all to
            # be visible. Set to threshold to ensure los_lookup == 1
            tot_facets[r, :, :, :] = thresh
            los_facets[r, :, :, :] = thresh
            continue

        # Scale the dem to the prescirbed RMS slope and get slope, azim arrays
        dem = zsurf * factors[r]
        slope, azim = get_slope_aspect(dem)

        # Get binned counts of total facets in dem (doesn't change with inc)
        tot_facets[r, :, :, :] = bin_slope_azim(azim, slope, azbins, slopebins)
        for i, inc in enumerate(incs):
            # Nadir view => all facets in line of sight
            if inc == 0:
                los_facets[r, i, :, :] = tot_facets[r, i, :, :]
                continue

            # Run ray trace to compute dem facets in line of sight
            los = raytrace(dem, inc, az0)

            # Remove buffer before which we may have missed some shadows
            buf = get_dem_los_buffer(dem, inc)
            los = los[:, buf + 1 :]
            azim_buf = azim[:, buf + 1 :]
            slope_buf = slope[:, buf + 1 :]

            # # Get slope and azim in los and past buffer
            # los_buf = los * ~in_buf
            # azim_los_buf = azim[los_buf]
            # slope_los_buf = slope[los_buf]

            # Get binned counts of facets in los
            los_facets[r, i, :, :] = bin_slope_azim(
                azim_buf[los], slope_buf[los], azbins, slopebins
            )
            # If buffer, there will be fewer tot_facets, so we need to compute
            if buf > 0:
                tot_facets[r, i, :, :] = bin_slope_azim(
                    azim_buf, slope_buf, azbins, slopebins
                )
            # Report progress
            ppct = 100 * (r * len(incs) + i + 1) / (len(rmss) * len(incs))
            pbar = "=" * int(ppct // 5)
            print(f"Raytracing [{pbar:20s}] {ppct:.2f}%", end="\r", flush=True)

    print("\nGenerating lineofsight lookup table...")
    # If bins have fewer than threshold total facets -> set outputs to nan
    tot_facets[tot_facets < thresh] = np.nan
    los_facets[tot_facets < thresh] = np.nan

    # los_lookup is fraction of facets in lineofsight / total facets in bin
    with np.errstate(
        divide="ignore", invalid="ignore"
    ):  # Suppress div warning
        los_lookup = los_facets / tot_facets

    # Convert lookups to xarray
    lookups = (tot_facets, los_facets, los_lookup)
    ds = rh.lookup2xarray(lookups, rmss, incs)
    ds.attrs["description"] = "Roughness line of sight lookup DataSet."
    ds.attrs["version"] = VERSION
    ds.attrs["demsize"] = zsurf.shape
    ds.attrs["threshold"] = thresh
    ds.attrs["az0"] = az0
    if fout is not None:
        ds.to_netcdf(fout, mode="w")
        print(f"Wrote {fout}")
    return ds


# def make_los_table(
#     nrms=10,
#     ninc=10,
#     naz=36,
#     ntheta=45,
#     threshold=25,
#     demsize=10000,
#     az0=270,
#     fout="",
#     write=True,
#     default=False,
# ):
#     """
#     Generate line of sight probability table with dims (rms, inc, slope, az).

#     Tables are generated using raytracing on a synthetic Gaussian surface with
#     a range of RMS slopes and observed from a range of solar inc angles. Final
#     los_lookup table is binned by facet slope/azimuth:

#     - los_lookup == 1: All facets in line of sight (i.e. visible / illuminated)
#     - los_lookup == 0: No facets in line of sight (i.e. not visible / shadowed)
#     - los_lookup == np.nan: Fewer than threshold facets observed in bin

#     Parameters
#     ----------
#     nrms: Number of RMS slopes. Fixed value for now.
#     ninc: Number of incidence Angles. Default=18
#     naz: Number of azimuth bins. Default=36
#     ntheta: Number of slope bins. Default=45
#     threshold (int): Minimum number of valid facets to do binning, else nan

#     Returns
#     -------
#     No values are returned. Three tables are saved in roughness data directory.
#     total_facets_4D.npy: Number of facets in slope/azim bin.
#     los_facets_4D.npy: Number of facets in line of sight in slope/azim bin.
#     los_lookup_4D.npy: Probability of facets in line of sight per slope/azim bin.
#     """
#     if fout == "":
#         if default:
#             fout = FLOOKUP
#         else:
#             args = f"{nrms}_{ninc}_{naz}_{ntheta}_{threshold}_{demsize}"
#             fout = DATA_DIR / f"lookup_{args}.nc"
#     # Get line of sight lookup coordinate arrays
#     rms_arr, incs, _, _ = rh.get_lookup_coords(nrms, ninc)
#     azbins, slopebins = rh.get_facet_bins(naz, ntheta)

#     # Load in the synthetic DEM and scale factors
#     zsurf = make_zsurf(demsize, default=default, write=write)
#     factors = make_scale_factors(zsurf, rms_arr, default=default, write=write)

#     # Init total and line of sight facet arrays binned by slope/azim
#     tot_facets = np.full((nrms, ninc, naz, ntheta), np.nan)
#     los_facets = np.copy(tot_facets)

#     for r, rms in enumerate(rms_arr):
#         # Flat surface => all facets in line of sight at any inc
#         if rms == 0:
#             # Cludge: at rms=0 most facets are undefined, but we expect all to
#             # be visible. Set to threshold to ensure los_lookup == 1
#             tot_facets[r, :, :, :] = threshold
#             los_facets[r, :, :, :] = threshold
#             continue

#         # Scale the dem to the prescirbed RMS slope and get slope, azim arrays
#         dem = zsurf * factors[r]
#         slope, azim = get_slope_aspect(dem)
#         slope = slope.flatten()
#         azim = azim.flatten()

#         # Get binned counts of total facets in dem (doesn't change with inc)
#         tot_facets[r, :, :, :] = bin_slope_azim(azim, slope, azbins, slopebins)
#         for i, inc in enumerate(incs):
#             # Nadir view => all facets in line of sight
#             if inc == 0:
#                 los_facets[r, i, :, :] = tot_facets[r, i, :, :]
#                 continue

#             # Run ray trace to compute dem facets in line of sight
#             los = raytrace(dem, inc, az0).flatten()

#             # Get buffer beyond which raytrace breaks down due to finite dem
#             in_buf = get_dem_los_buffer(dem, inc).flatten()

#             # Get only slope and azim in los and not in buffer
#             los_buf = los * ~in_buf
#             azim_los_buf = azim[los_buf]
#             slope_los_buf = slope[los_buf]

#             # Get binned counts of remaining facets
#             los_facets[r, i, :, :] = bin_slope_azim(
#                 azim_los_buf, slope_los_buf, azbins, slopebins
#             )
#             # Report progress
#             ppct = (100 * (r * ninc + i) / (nrms * ninc)) + 1
#             pbar = "=" * int(ppct // 5)
#             print(f"Raytracing [{pbar:20s}] {ppct:.2f}%", end="\r", flush=True)

#     print("\nGenerating lineofsight lookup table...")
#     # If bins have fewer than threshold total facets -> set outputs to nan
#     tot_facets[tot_facets < threshold] = np.nan
#     los_facets[tot_facets < threshold] = np.nan

#     # los_lookup is fraction of facets in lineofsight / total facets in bin
#     with np.errstate(
#         divide="ignore", invalid="ignore"
#     ):  # Suppress div warning
#         los_lookup = los_facets / tot_facets

#     # Convert lookups to xarray
#     lookups = (tot_facets, los_facets, los_lookup)
#     ds = rh.lookup2xarray(lookups)
#     ds.attrs["description"] = "Roughness line of sight lookup DataSet."
#     ds.attrs["version"] = VERSION
#     ds.attrs["demsize"] = demsize
#     ds.attrs["threshold"] = threshold
#     ds.attrs["az0"] = az0
#     if write:
#         ds.to_netcdf(fout, mode="w")
#         print(f"Wrote {fout}")
#     return ds


def make_zsurf(n=10000, H=0.8, qr=100, fout=FZSURF, write=True, default=False):
    """
    Return a synthetic surface with self-affine roughness.
    Much of this is detailed in this paper:
    Tevis D B Jacobs et al 2017 Surf. Topogr.: Metrol. Prop. 5 013001
    https://doi.org/10.1088/2051-672X/aa51f8

    Parameters
    ----------
    n: Side length of synthetic surface. Default=10000
    H: Hurst exponent.
    qr: Rolloff wavelength in units of q (wavevector; 1/m).
    fout (str): Path to write z surface.

    Returns
    -------
    zsurf (n x n arr): Artifical z surface with self-affine roughness.
    """
    # Return zsurf from fout if it exists and is correct size
    if not default:
        fout = rh.fname_with_demsize(fout, n)
        zsurf = load_zsurf(fout)
        if zsurf is not None and zsurf.shape == (n, n):
            print(f"Loaded zsurf from {fout}")
            return zsurf

    # Make sure the dimensions are even
    print("Making new z surface...")
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
    rho, _ = rh.cart2pol(qxx, qyy)

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
    a, b = rh.pol2cart(Bq, Phi)

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
    if write:
        np.save(fout, norm_zsurf)
        print(f"Wrote zsurf to {fout}")
    return norm_zsurf


def make_scale_factors(
    zsurf, rms_arr, write=True, fout=FZ_FACTORS, default=False
):
    """
    Determine scale factors that produce rms_arr slopes from zsurf.

    Computes RMS slope of zsurf at several multiplicative scale factors, then
    interpolates to find scale_factors at desired rms_arr values.

    Parameters
    ----------
    zsurf (n x n arr): Artifical z surface with self-affine roughness.
    rms_arr (arr): Array of RMS values to scale zsurf to (max 80 degrees).
    write (bool): Write out scale factors to file.
    fout (str): Path to write scale factors.
    default (bool): Use default scale factors if they exist.

    Returns
    -------
    scale_factors (arr): Factors to scale zsurf by to get each rms_arr value.
    """
    if not default:
        fout = rh.fname_with_demsize(
            fout, f"{zsurf.shape[0]}_rmsmax_{rms_arr.max()}"
        )
        # Try to load scale factors and check they are correct length
        scale_factors = load_zsurf_scale_factors(fout)
        if scale_factors is not None and len(scale_factors) == len(rms_arr):
            print(f"Loaded scale_factors from {fout}")
            return scale_factors

    print("Calculating new scale factors...")
    # Scale factors from 0x to 10x in logspace (~0 to 80 degrees RMS)
    factors = np.array([0, *np.logspace(-1, 2, 20)])
    rms_derived = np.zeros_like(factors)
    i = 1  # rms=0 is factor=0 so we start at 1
    # Save RMS slope at each factor, end loop when we exceed max(rms_arr)
    while rms_derived[i - 1] < rms_arr.max() and i < len(factors):
        # Compute z surface * factor and get RMS
        zsurf_new = zsurf * factors[i]
        rms_derived[i] = get_rms(get_slope_aspect(zsurf_new, get_aspect=False))
        i += 1

    # Interpolate to get the factor at each rms in rms_arr (truncate at last i)
    scale_factors = np.interp(rms_arr, rms_derived[:i], factors[:i])

    if write:
        np.save(fout, scale_factors)
        print(f"Wrote scale_factors to {fout}")
    return scale_factors


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
        los = lineofsight.lineofsight(dem.flatten(), inc, azim, los)
    return los.T.astype(bool)


def bin_slope_azim(az, slope, az_bins, slope_bins):
    """
    Return count of facets in each slope, azim bin using 2d histogram.

    Parameters
    ----------
    slope (1D or 2D arr): Facet slope values.
    az (1D or 2D arr): Facet azim values.
    slope_bins (1D arr): Bin edges of slopes.
    az_bins (1D arr): Bin edges of azims.

    Return
    ------
    counts (2D arr nslope, naz): Counts in each slope/az bin; shape defined by bins.
    """
    h = np.histogram2d(slope.ravel(), az.ravel(), bins=(slope_bins, az_bins))
    counts = h[0].T  # Hist 2D transposes output
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
    buffer (int): Number of pixels to exclude from western edge of dem.
    """
    # Find peak-to-peak of DEM (max - min)
    elev_span = np.ptp(dem)

    # Calculate longest shadow for buffer. Round up for good measure
    buffer = int(elev_span * np.tan(np.deg2rad(inc))) + 1
    return buffer


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
        return slope, aspect
    return slope


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


# File I/O
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


@lru_cache(1)
def load_zsurf_scale_factors(path=FZ_FACTORS):
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
