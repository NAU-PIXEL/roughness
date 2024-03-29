"""Roughness helper functions"""
import os
import contextlib
from pathlib import Path
import numpy as np
import numpy.f2py
import xarray as xr
import jupytext
from . import config as cfg

# Line of sight helpers
def lookup2xarray(lookups, rms_coords=None, inc_coords=None, nodata=-999):
    """
    Convert list of default lookups to xarray.DataSet.

    Parameters:
        lookups (array): Lookup tables (e.g. from make_los_table).
        rms_coords (array): RMS slope coordinates (default: None).
        inc_coords (array): Incidence angle coordinates (default: None).
        nodata (float): Value to replace NaNs with (default: -999).

    Returns:
        xarray.DataArray: xarray.DataArray with labeled dims and coords.
    """
    names = cfg.LUT_NAMES
    longnames = cfg.LUT_LONGNAMES
    coords = get_lookup_coords(*lookups[0].shape)
    if rms_coords is not None:
        coords[0] = np.array(rms_coords)
    if inc_coords is not None:
        coords[1] = np.array(inc_coords)
    for i, lut in enumerate(lookups):
        # Make DataArray
        da = np2xr(lut, name=names[i], coords=coords)
        da.attrs["long_name"] = longnames[i]

        # Add DataArray to DataSet
        if i == 0:
            ds = da.to_dataset()
        else:
            ds[names[i]] = da
    return ds.where(ds != nodata)


def np2xr(arr, dims=None, coords=None, name=None, cnames=None, cunits=None):
    """
    Convert numpy array to xarray.DataArray.

    Parameters:
        array (np.array): Numpy array to convert to xarray.DataArray.
        dims (list of str): Dimension names (default: cfg.LUT_DIMS).
        coords (list of arr): Coordinate arrays of each dim.
        name (str): Name of xarray.DataArray (default: None)
        cnames (list of str): Coordinate names of each param.
        cunits (list of str): Coordinate units of each param (default: deg).

    Returns:
        xarray.DataArray
    """
    ndims = len(arr.shape)
    if dims is None:
        dims = cfg.LUT_DIMS[:ndims]
    if coords is None:
        coords = get_lookup_coords(*arr.shape)[:ndims]
    if cnames is None:
        cnames = cfg.LUT_DIMS_LONGNAMES[:ndims]
    if cunits is None:
        cunits = ["deg"] * ndims

    # Make xarray-readable (dim, coord) pairs
    coords_xr = list(zip(dims, coords))

    # Make DataArray
    da = xr.DataArray(arr, coords=coords_xr, name=name)
    for i, coord in enumerate(da.coords):
        da.coords[coord].attrs["long_name"] = cnames[i]
        da.coords[coord].attrs["units"] = cunits[i]
    return da


def wl2xr(arr, units="μm"):
    """
    Return wavelength numpy array as xarray.

    Parameters:
        arr (array): Wavelength array.
        units (str): Units of wavelengths (default: μm).
    
    Returns:
        xarray.DataArray
    """
    da = xr.DataArray(arr, coords=[("wavelength", arr)])
    da.coords["wavelength"].attrs["long_name"] = "Wavelength"
    da.coords["wavelength"].attrs["units"] = units
    return da


def wn2xr(arr, units="cm^-1"):
    """
    Return wavenumber numpy array as xarray.

    Parameters:
        arr (array): Wavenumber array.
        units (str): Units of wavenumbers (default: cm^-1).

    Returns:
        xarray.DataArray
    """
    da = xr.DataArray(arr, coords=[("wavenumber", arr)])
    da.coords["wavelength"].attrs["long_name"] = "Wavenumber"
    da.coords["wavelength"].attrs["units"] = units
    return da


def spec2xr(arr, wls, units="W/m²/sr/μm", wl_units="μm"):
    """
    Return spectral numpy array as xarray.

    Parameters:
        arr (array): Spectral array.
        wls (array): Wavelength array.
        units (str): Units of spectral array (default: W/m²/sr/μm).
        wl_units (str): Units of wavelength array (default: μm).

    Returns:
        xarray.DataArray
    """
    da = xr.DataArray(arr, coords=[wls], dims=["wavelength"], name="Radiance")
    da.attrs["units"] = units
    da.coords["wavelength"].attrs["long_name"] = "Wavelength"
    da.coords["wavelength"].attrs["units"] = wl_units
    return da


def get_lookup_coords(nrms=10, ninc=10, naz=36, ntheta=45):
    """
    Return lookup table coordinate arrays

    Parameters:
        nrms (int): Number of RMS slopes in [0, 50] degrees.
        ninc (int): Number of incidence angles in [0, 90] degrees.
        naz (int): Number of facet azimuth bins in [0, 360] degrees.
        ntheta (int): Number of facet slope bins in [0, 90] degrees.

    Returns:
        lookup_coords (list of array): Coord arrays (rms, cinc, az, theta)
    """
    rmss, incs, azs, thetas = get_lookup_bins(nrms, ninc, naz, ntheta)
    # Get slope and az bin centers
    azs = (azs[:-1] + azs[1:]) / 2
    thetas = (thetas[:-1] + thetas[1:]) / 2
    return [rmss, incs, azs, thetas]


def get_lookup_bins(nrms=10, ninc=10, naz=36, ntheta=45):
    """
    Return coordinate arrays corresponding to number of elements in each axis.

    Return lookup axes in the following order with ranges:
        rms: [0, 50] degrees
        inc: [0, 90] degrees
        az: [0, 360] degrees
        theta: [0, 90] degrees

    Parameters:
        nrms (int): Number of RMS slopes in [0, 50] degrees.
        ninc (int): Number of incidence angles in [0, 90] degrees.
        naz (int): Number of facet azimuth bins in [0, 360] degrees.
        ntheta (int): Number of facet slope bins in [0, 90] degrees.

    Return
    ------
    lookup_coords (tuple of array): Coordinate bins (rms, cinc, az, theta)
    """
    rms_coords = np.linspace(0, 45, nrms)
    cinc_coords = np.linspace(1, 0, ninc)  # cos([0, 90] degrees)
    azim_coords = np.linspace(0, 360, naz + 1)
    slope_coords = np.linspace(0, 90, ntheta + 1)
    inc_coords = np.rad2deg(np.arccos(cinc_coords))
    return (rms_coords, inc_coords, azim_coords, slope_coords)


def facet_grids(los_table, units="degrees"):
    """
    Return 2D grids of surface facet slope and azimuth angles of los_table.

    Assumes los_table axes are (az, theta) with ranges:
        az: [0, 360] degrees
        theta: [0, 90] degrees

    Parameters:
        los_table (arr): Line of sight table (dims: az, theta)
        units (str): Return grids in specified units ('degrees' or 'radians')

    Return
    ------
        thetas, azs (tuple of 2D array): Coord grids of facet slope and az
    """
    if isinstance(los_table, xr.DataArray):
        az_arr = los_table.az.values
        theta_arr = los_table.theta.values
    else:
        naz, ntheta = los_table.shape
        _, _, az_arr, theta_arr = get_lookup_coords(naz=naz, ntheta=ntheta)
    if units == "radians":
        az_arr = np.radians(az_arr)
        theta_arr = np.radians(theta_arr)

    # Make coordinate grids
    theta_grid, az_grid = np.meshgrid(theta_arr, az_arr)
    if isinstance(los_table, xr.DataArray):
        theta_grid = xr.DataArray(
            theta_grid,
            dims=["az", "theta"],
            coords={"az": az_arr, "theta": theta_arr},
        )
        theta_grid.name = f"theta [{units}]"
        az_grid = xr.DataArray(
            az_grid,
            dims=["az", "theta"],
            coords={"az": az_arr, "theta": theta_arr},
        )
        az_grid.name = f"azimuth [{units}]"
    return theta_grid, az_grid


def get_facet_bins(naz=36, ntheta=45):
    """
    Return az, theta bin arrays of los_table.

    Assumes los_table axes are (az, theta) with ranges:
        az: [0, 360] degrees
        theta: [0, 90] degrees

    Parameters:
        los_table (array): Line of sight table (dims: az, theta)
        units (str): Return grids in specified units ('degrees' or 'radians')

    Return
    ------
        thetas, azs (tuple of 2D array): Bin edges of facet slope and az
    """
    azim_coords = np.linspace(0, 360, naz + 1)
    slope_coords = np.linspace(0, 90, ntheta + 1)
    return (azim_coords, slope_coords)


# File I/O helpers
def rm_regex(dirpath, regex):
    """Remove all files matching regex in dirpath."""
    dirpath = Path(dirpath)
    for f in dirpath.rglob(regex):
        f.unlink()


def fname_with_demsize(filename, demsize):
    """
    Return filename with demsize appended to the end.

    Parameters:
        fname (str or Path): Filename to append to.
        demsize (int): Length of dem in pixels.

    Returns:
        fname_with_demsize (str): New filename with new demsize appended.
    """
    filename = Path(filename)
    return filename.with_name(f"{filename.stem}_s{demsize}{filename.suffix}")


def versions_match(version_a, version_b, precision=2):
    """
    Check if semantic versions match to precision (default 2).

    Examples
    --------
    >>> versions_match('1.0.0', '1.2.3', precision=1)
    True
    >>> versions_match('1.2.0', '1.2.3', precision=2)
    True
    >>> versions_match('1.2.3', '1.2.3', precision=3)
    True
    """
    va_split = version_a.split(".")
    vb_split = version_b.split(".")
    for i in range(precision):
        if va_split[i] != vb_split[i]:
            return False
    return True


def check_data_updated():
    """Check if data version is up to date."""
    version = get_data_version()
    if version is None or not versions_match(version, cfg.__version__):
        print("WARNING: The roughness/data folder is not up to date!")
        print("Update to the newest lookup tables with -d flag.")
        return False
    return True


def get_data_version(data_version_file=cfg.FDATA_VERSION):
    """Get data version in data/data_version.txt."""
    data_version = None
    try:
        with open(data_version_file, "r", encoding="utf8") as f:
            data_version = f.readline().strip()
    except FileNotFoundError:
        pass
    return data_version


def set_data_version(data_version_file=cfg.FDATA_VERSION):
    """Set data version in data/data_version.txt."""
    with open(data_version_file, "w", encoding="utf8") as f:
        f.write(cfg.__version__)


@contextlib.contextmanager
def change_working_directory(path):
    """Change working directory and revert to previous on exit."""
    prev_cwd = Path.cwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(prev_cwd)


# Fortran helpers
def compile_lineofsight(los_f90=cfg.FLOS_F90, verbose=False):
    """Compile lineofsight module with numpy f2py. Error if no compiler."""
    with open(los_f90, encoding="utf8") as f:
        src = f.read()
    failed = numpy.f2py.compile(
        src, "lineofsight", verbose=verbose, source_fn=los_f90
    )
    if failed:
        msg = "Cannot compile Fortran raytracing code. Please ensure you have \
               a F90 compatible Fortran compiler (e.g. gfortran) installed."
        raise ImportError(msg)


def import_lineofsight(fortran_dir=cfg.FORTRAN_DIR):
    """Import lineofsight module. Compile first if not found."""
    # pragma pylint: disable=import-outside-toplevel
    lineofsight = None
    with change_working_directory(fortran_dir):
        try:
            from .fortran import lineofsight
        except (ImportError, ModuleNotFoundError):
            try:
                compile_lineofsight()
                from .fortran import lineofsight
            except (ImportError, ModuleNotFoundError):
                msg = "Cannot compile lineofsight FORTRAN module. Please \
                       ensure you have a F90 compatible Fortran compiler (e.g.\
                       gfortran) installed."
                print(msg)
    return lineofsight


def build_jupyter_notebooks(nbpath=cfg.EXAMPLES_DIR):
    """Build Jupyter notebooks using Jupytext."""
    print("Setting up Jupyter notebooks")
    for nb_py in nbpath.rglob("*.py"):
        fout = Path(nb_py).with_suffix(".ipynb")
        if fout.exists():
            print(f"Skipping existing notebook {fout}")
        else:
            jupytext.write(jupytext.read(nb_py), fout)
            print(f"Wrote {fout}")


# Geometry helpers
def get_ieg(ground_zen, ground_az, sun_zen, sun_az, sc_zen, sc_az):
    """
    Return local solar incidence, spacecraft emergence and phase angles 
    (i, e, g) and azimuth given input viewing geometry.

    Input zeniths and azimuths in radians.

    Parameters:
        ground_zen (float): Ground zenith angle [rad].
        ground_az (float): Ground azimuth angle [rad].
        sun_zen (float): Sun zenith angle [rad].
        sun_az (float): Sun azimuth angle [rad].
        sc_zen (float): Spacecraft zenith angle [rad].
        sc_az (float): Spacecraft azimuth angle [rad].

    Returns:
        (i, e, g, az) (tuple of float): Geometry relative to ground.
    """
    ground = sph2cart(ground_zen, ground_az)
    sun = sph2cart(sun_zen, sun_az)
    sc = sph2cart(sc_zen, sc_az)
    inc = get_angle_between(ground, sun, True)
    em = get_angle_between(ground, sc, True)
    phase = get_angle_between(sun, sc, True)
    sun_sc_az = get_local_az(ground, sun, sc)
    return (inc, em, phase, sun_sc_az)


def get_ieg_xr(ground_zen, ground_az, sun_zen, sun_az, sc_zen, sc_az):
    """
    Return local solar incidence, spacecraft emergence and phase angles 
    (i, e, g) and azimuth given input viewing geometry as xarray.DataArray.

    Input zeniths and azimuths in radians.

    Parameters:
        ground_zen (DataArray): Ground zenith angle [rad].
        ground_az (DataArray): Ground azimuth angle [rad].
        sun_zen (DataArray): Sun zenith angle [rad].
        sun_az (DataArray): Sun azimuth angle [rad].
        sc_zen (DataArray): Spacecraft zenith angle [rad].
        sc_az (DataArray): Spacecraft azimuth angle [rad].

    Returns:
        (i, e, g, az) (tuple of DataArray): Geometry relative to ground.
    """
    ground = sph2cart_xr(ground_zen, ground_az)
    sun = sph2cart_xr(sun_zen, sun_az)
    sc = sph2cart_xr(sc_zen, sc_az)
    inc = get_angle_between_xr(ground, sun, True)
    em = get_angle_between_xr(ground, sc, True)
    phase = get_angle_between_xr(sun, sc, True)
    sun_sc_az = get_local_az_xr(ground, sun, sc)
    return inc, em, phase, sun_sc_az


def get_angle_between(vec1, vec2, safe_arccos=False):
    """
    Return angle between cartesian vec1 and vec2 using dot product.

    If vec1 or vec2 are NxMx3, compute element-wise.

    Parameters:
        vec1, vec2 (array): Cartesian vectors.
        safe_arccos (bool): Clip dot product to [-1, 1] before arccos.
    """
    if isinstance(vec1, xr.DataArray) or isinstance(vec2, xr.DataArray):
        return get_angle_between_xr(vec1, vec2, safe_arccos)
    dot = element_dot(vec1, vec2)
    if safe_arccos:
        # Restrict dot product to [-1, 1] to safely pass to arccos
        dot = np.clip(dot, -1, 1)
    return np.rad2deg(np.arccos(dot))


def get_angle_between_xr(vec1, vec2, safe_arccos=False):
    """
    Return angle between cartesian DataArrays using dot product.

    Parameters:
        vec1, vec2 (DataArray): Cartesian vectors.
        safe_arccos (bool): Clip dot product to [-1, 1] before arccos.
    """
    dot = vec1.dot(vec2, dims="xyz")
    if safe_arccos:
        # Restrict dot product to [-1, 1] to safely pass to arccos
        dot = dot.clip(-1, 1)
    return np.rad2deg(np.arccos(dot))


def get_azim(ground_sc_ground, ground_sun_ground):
    """
    Return the azimuth arccos(dot product of the spacecraft and sun vectors)

    Parameters:
        ground_sc_ground (array): Ground to spacecraft vector.
        ground_sun_ground (array): Ground to sun vector.
    """
    dot_azim = np.degrees(
        np.arccos(np.sum(ground_sc_ground * ground_sun_ground, axis=2))
    )
    oob = dot_azim[(dot_azim < 0) * (dot_azim > 180)]
    if oob.any():
        w = f"Azimuth {oob} outside (0, 180); Setting to 0"
        print(w)
        dot_azim[(dot_azim < 0) * (dot_azim > 180)] = 0
    return dot_azim


def get_local_az(ground, sun, sc):
    """
    Return azimuth angle of the spacecraft with respect to the sun and local
    slope. Assumes inputs are same size and shape and are in 3D
    Cartesian coords (ixjxk).

    Parameters:
        ground (array): Ground vector.
        sun (array): Sun vector.
        sc (array): Spacecraft vector.
    """
    sc_rel_ground = element_norm(element_triple_cross(ground, sc, ground))
    sun_rel_ground = element_norm(element_triple_cross(ground, sun, ground))
    az = get_angle_between(sc_rel_ground, sun_rel_ground, True)
    az[((az < 0) & (az > 180)) | np.isnan(az)] = 0
    return az


def get_local_az_xr(ground, sun, sc):
    """
    Return azimuth angle of the spacecraft with respect to the sun and local
    slope. Assumes inputs are same size and shape and are in 3D
    Cartesian coords (ixjxk).
    
    Parameters:
        ground (DataArray): Ground vector.
        sun (DataArray): Sun vector.
        sc (DataArray): Spacecraft vector.
    """
    sc_rel_ground = inc_rel_ground_xr(sc, ground)
    sun_rel_ground = inc_rel_ground_xr(sun, ground)
    az = get_angle_between_xr(sc_rel_ground, sun_rel_ground, True)
    az = az.where(((az >= 0) & (az <= 180)))
    return az


def inc_rel_ground_xr(vec, ground):
    """Return vec relative to ground vector."""
    cross = triple_cross_xr(ground, vec, ground)
    # Normalize vector by magnitude
    mag = np.sqrt((cross**2).sum(dim="xyz"))
    return cross / mag


def triple_cross_xr(a, b, c):
    """Return cross product of a, b, c in 3D Cartesian coordinates."""
    return b * a.dot(c, dims="xyz") - c * a.dot(b, dims="xyz")


def inc_to_tloc(inc, az, lat):
    """
    Convert solar incidence and az to decimal local time (in 6-18h).

    Parameters:
        inc (float): Solar incidence in degrees [0, 90)
        az (float): Solar azimuth in degrees [0, 360)
        lat (float): Latitude in degrees [-90, 90)
    """
    inc, lat = np.deg2rad(inc), np.deg2rad(lat)
    hr_angle = np.arccos(np.cos(inc) / np.cos(lat))
    if isinstance(az, xr.DataArray):
        hr_angle = hr_angle.where(az % 360 > 180, hr_angle * -1)
    elif isinstance(az, np.ndarray):
        hr_angle[az % 360 < 180] *= -1
    elif az % 360 < 180:
        hr_angle *= -1
    tloc = 12 + np.rad2deg(hr_angle) / 15
    return tloc


def tloc_to_inc(tloc, lat=0, az=False):
    """
    Return the solar incidence angle given the local time in hrs.

    Parameters:
        tloc (float) Local time (0, 24) [hrs]
        lat (float) Latitude (-90, 90) [deg]
        az (bool) Return azimuth angle (default: False)

    Returns:
        inc (float) Solar incidence angle [deg]
        az (float) Solar azimuth angle (if az=True) [deg]
    """
    latr = np.deg2rad(lat)
    hr_angle = np.deg2rad(15 * (12 - tloc))  # (morning is +; afternoon is -)
    inc = np.rad2deg(np.arccos(np.cos(hr_angle) * np.cos(latr)))
    if az:
        az = np.arctan2(np.sin(hr_angle), -np.sin(latr) * np.cos(-hr_angle))
        az = (np.rad2deg(az) + 360) % 360  # [0, 360) from N
        return inc, az
    return inc


def get_inc_az_from_subspacecraft(sslon, sslat, lon, lat):
    """
    Return inc and az angle at (lat, lon) given subspacecraft point
    (sslat, sslon). Return azimuth in degrees Clockwise from North (0 - 360).

    From calculator at https://the-moon.us/wiki/Sun_Angle
    
    Parameters:
        sslon (float): Subsolar longitude [deg]
        sslat (float): Subsolar latitude [deg]
        lon (float): Longitude of point of interest [deg]
        lat (float): Latitude of point of interest [deg]
        

    Returns:
        inc, az (float): Solar incidence and azimuth angles [deg]
    """
    lon, lat, sslon, sslat, dlon = (
        np.deg2rad(lon),
        np.deg2rad(lat),
        np.deg2rad(sslon),
        np.deg2rad(sslat),
        np.deg2rad(lon - sslon),
    )
    inc = np.arccos(
        np.sin(sslat) * np.sin(lat)
        + np.cos(sslat) * np.cos(lat) * np.cos(dlon)
    )
    az = np.arctan2(
        np.cos(lat) * np.sin(dlon),
        np.cos(sslat) * np.sin(lat)
        - np.sin(sslat) * np.cos(lat) * np.cos(dlon),
    )
    return np.rad2deg(inc), np.rad2deg(az) % 360


# Linear algebra
def as_cart3D(vecs):
    """Return list of vecs as shape (N, M, 3) if they are not already."""
    for i, vec in enumerate(vecs):
        if vec.ndim == 1:
            vecs[i] = vec[np.newaxis, np.newaxis, :]
    return vecs


def element_cross(A, B):
    """
    Return element-wise cross product of two 3D arrays in cartesian coords.

    Parameters:
        A, B (array): 3D arrays of vectors in cartesian coords.

    Returns:
        out (array): 3D array of vector cross products in cartesian coords.
    """
    A, B = as_cart3D([A, B])
    out = np.zeros_like(A)
    out[:, :, 0] = A[:, :, 1] * B[:, :, 2] - A[:, :, 2] * B[:, :, 1]
    out[:, :, 1] = A[:, :, 2] * B[:, :, 0] - A[:, :, 0] * B[:, :, 2]
    out[:, :, 2] = A[:, :, 0] * B[:, :, 1] - A[:, :, 1] * B[:, :, 0]
    return out


def element_dot(A, B):
    """
    Return element-wise dot product of two 3D arrays in Cartesian coords.

    Parameters:
        A, B (array): 3D arrays of vectors in cartesian coords.

    Returns:
        out (array): 2D array of vector dot products.
    """
    A, B = as_cart3D([A, B])
    return np.sum(A * B, axis=2)


def element_norm(A):
    """
    Return input array of vectors normalized to length 1.

    Parameters:
        A (array): 3D array of vectors in cartesian coords.

    Returns:
        out (array): 3D array of normalized vectors.
    """
    A = as_cart3D([A])[0]
    mag = np.sqrt(np.sum(A**2, axis=2))
    return A / mag[:, :, np.newaxis]


def element_triple_cross(A, B, C):
    """
    Return element-wise triple cross product of three 3D arr in Cartesian

    Parameters:
        A, B, C (array): 3D arrays of vectors in cartesian coords.

    Returns:
        out (array): 3D array of triple cross products.
    """
    A, B, C = as_cart3D([A, B, C])
    return (
        B * (element_dot(A, C))[:, :, np.newaxis]
        - C * (element_dot(A, B))[:, :, np.newaxis]
    )


# Coordinate transformations
def cart2pol(x, y):
    """
    Convert ordered coordinate pairs from Cartesian (X,Y) to polar (r,theta).

    Parameters:
        x,y (array): Cartesian coordinates

    Returns:
        r (array): Distance from origin.
        theta (array): Polar angle, [radians].
    """
    r = np.sqrt(x**2 + y**2)
    theta = np.arctan2(y, x)
    return (r, theta)


def pol2cart(rho, phi):
    """
    Convert ordered coordinate pairs from polar (r,theta) to Cartesian (X,Y).

    Parameters:
        rho (array): Distance from origin.
        phi (array): Polar angle, [radians].

    Returns:
        x,y (array): Cartesian coordinates
    """
    x = rho * np.cos(phi)
    y = rho * np.sin(phi)
    return (x, y)


def sph2cart(theta, phi, radius=1):
    """
    Convert from spherical (theta, phi, r) to cartesian (x, y, z) coordinates.

    Theta and phi must be specified in radians and returned units correspond to
    units of r, default is unitless (r=1 is unit vector). Returns vectors along
    z-axis (e.g., if theta and phi are int/float return 1x1x3 vector; if theta
    and phi are MxN arrays return 3D array of vectors MxNx3).

    Parameters:
        theta (num or array): Polar angle [rad].
        phi (num or array): Azimuthal angle [rad].
        radius (num or array): Radius (default=1).

    Returns:
        (array): Cartesian vector(s), same shape as theta and phi.
    """
    if isinstance(theta, xr.DataArray):
        theta = theta.values
    if isinstance(phi, xr.DataArray):
        phi = phi.values
    return np.dstack(
        [
            radius * np.sin(theta) * np.cos(phi),
            radius * np.sin(theta) * np.sin(phi),
            radius * np.cos(theta),
        ]
    ).squeeze()


def sph2cart_xr(theta, phi, radius=1):
    """
    Convert spherical to cartesian coordinates using xarray.
    
    Parameters:
        theta (DataArray): Polar angle [rad].
        phi (DataArray): Azimuthal angle [rad].
        radius (num or array): Radius (default=1).

    Returns:
        (DataArray): Cartesian vector(s), same shape as theta and phi.
    """
    return xr.concat(
        [
            radius * np.sin(theta) * np.cos(phi),
            radius * np.sin(theta) * np.sin(phi),
            radius * np.cos(theta),
        ],
        dim="xyz",
    )


def cart2sph(x, y, z):
    """
    Convert from cartesian (x, y, z) to spherical (theta, phi, r).

    Parameters:
        x, y, z (array): Cartesian coordinates.

    Returns:
        theta (array): Polar angle [rad].
        phi (array): Azimuthal angle [rad].
        radius (array): Radius.
    """
    r = np.sqrt(x**2 + y**2 + z**2)
    theta = np.arctan2(np.sqrt(x**2 + y**2), z)
    phi = np.arctan2(y, x)
    return (theta, phi, r)


def xy2lonlat_coords(x, y, extent):
    """
    Convert x,y coordinates to lat,lon coordinates.
    
    Parameters:
        x, y (array): x,y coordinates.
        extent (tuple): Lon/lat extent (lonmin, lonmax, latmin, latmax).

    Returns:
        lon, lat (array): Lon/lat coordinates.
    """
    lon = np.linspace(extent[0], extent[1], len(x))
    lat = np.linspace(extent[3], extent[2], len(y))
    return lon, lat


# Image I/O
def xarr2geotiff(xarr, savefile, crs=cfg.MOON2000_ESRI):
    """
    Write xarray image to geotiff with rasterio.

    xarr can must have dims (wavelength, lat, lon) or (lat, lon).
    
    Parameters:
        xarr (xarray.DataArray): Image to save.
        savefile (str): Filename to save to.
        crs (str): Coordinate reference system (default: MOON2000_ESRI).
    """
    xarr = xarr.rio.write_crs(crs)
    if "wavelength" in xarr.dims:
        xarr = xarr.transpose("wavelength", "lat", "lon")
    else:
        xarr = xarr.transpose("lat", "lon")
    xarr = xarr.rio.set_spatial_dims("lon", "lat")
    xarr.rio.to_raster(savefile)
    if Path(savefile).exists():
        print("Wrote to", savefile)
    else:
        print("Failed to write to", savefile)
