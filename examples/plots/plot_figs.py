# %%
import functools
import warnings
from datetime import datetime
from pathlib import Path
from multiprocessing import Process
import numpy as np
from scipy import stats
from scipy.optimize import curve_fit

# import pandas as pd
import xarray as xr
import matplotlib as mpl
import matplotlib.pyplot as plt

# import seaborn as sns
from roughness import (
    config as cfg,
    diviner as rd,
    emission as re,
    helpers as rh,
    m3_correction as m3,
    plotting as rp,
    roughness as rn,
)


# %%
# Constants
WARNINGS = 0
DATE = datetime.now().strftime("%y%m%d")
FIGDIR = Path(__name__).parent / DATE

# %%
# Helpers
def _reset_plot_style(mplstyle=True):
    """Reset matplotlib style defaults, use MoonPIES mplstyle if True."""
    mpl.rcParams.update(mpl.rcParamsDefault)
    if mplstyle:
        if mplstyle is True:
            mplstyle = next(Path(__name__).parent.glob("*.mplstyle"))
        try:
            mpl.style.use(mplstyle)
        except (OSError, FileNotFoundError):
            global WARNINGS
            if WARNINGS < 1:
                warnings.warn(f"Could not find mplstyle file {mplstyle}")
                WARNINGS += 1


# %%
def _reset_plot_style_dec(func):
    """Decorator for resetting matplotlib style defaults before and after."""

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        _reset_plot_style()
        value = func(*args, **kwargs)
        _reset_plot_style()
        return value

    return wrapper


# %%
def _save_or_show(fig, ax, fsave, figdir=FIGDIR, **kwargs):
    """Save figure or show fig and return ax."""
    if fsave:
        fsave = Path(figdir) / Path(fsave)  # Prepend path
        fsave.parent.mkdir(parents=True, exist_ok=True)  # Make dir if needed
        fig.savefig(fsave, **kwargs)
    else:
        plt.show()
    return ax


# %%
def _generate_all():
    """Produce all figures in figdir in parallel with default args."""
    # Get all functions in this module
    funcs = [
        obj
        for name, obj in globals().items()
        if callable(obj) and obj.__module__ == __name__ and name[0] != "_"
    ]

    # Run each with defaults in its own process
    print(f"Starting {len(funcs)} plots...")
    procs = [Process(target=func) for func in funcs]
    _ = [p.start() for p in procs]
    _ = [p.join() for p in procs]
    print(f"All plots written to {FIGDIR}")


# %%
# Plot functions
@_reset_plot_style_dec
def ref_emiss_3um(fsave="ref_emiss_3um.png", figdir=FIGDIR):
    """
    FIGURE 1
    Plot reflectance and emission overlap near 3 microns.
    """

    def exp(t, A, K):
        return A * np.exp(K * t)

    # Setup figure
    tifdir = "/work/team_moon/data/warped/210202/"
    basename = "M3G20090620T051312"
    m3ext = (-153, -152, 4, 14)
    rad = m3.readm3geotif_xr(tifdir + basename + "_rad.tif", m3ext)
    wls = rad.wavelength.values
    spec = rad.sel(lat=8.5, method="nearest").mean("lon").values

    wlmax = 10  # [microns] max radiance to fit
    tmin = 300  # [K]
    tmax = 400  # [K]
    tstep = 20
    abc = 3  # [um] absorption band center
    abdepth = 0.05  # 3um band depth (reflectance)
    abwidth = 0.2  # [microns] 3um band width
    yerr = 0.1  # [%]

    # Curve fit to capture tail of radiance
    fitmin = 2  # [microns]
    fitrange = wls > fitmin
    popt, _ = curve_fit(exp, wls[fitrange], spec[fitrange], p0=[1, -1])
    x = np.concatenate([wls, np.arange(3, wlmax, wls[1] - wls[0])])
    rad_tail = exp(x[x > fitmin], *popt)

    # Make radiance, emission and combined spectra
    sample_rad = np.concatenate([spec[wls <= fitmin], rad_tail])
    sample_em = re.cmrad2mrad(re.bbrw(x, tmax))
    refl3 = np.mean(
        sample_rad[np.where((abc - abwidth < x) & (x < abc + abwidth))]
    )
    absorb = stats.norm(loc=abc, scale=abwidth).pdf(x) * refl3 * abdepth
    sample_rad_total = sample_rad - absorb

    # Make plot
    fig, axs = plt.subplots(2, figsize=(3.6, 4))
    cmap = plt.get_cmap("cividis")
    cnorm = mpl.colors.Normalize(vmin=tmin, vmax=tmax)
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=cnorm)

    for i, ax in enumerate(axs):
        ax.plot(
            x, sample_rad_total, "-", lw=1, label="Reflected", c="k", zorder=1
        )
        if i == 1:  # Refl errorbars
            ax.errorbar(
                x,
                sample_rad_total,
                sample_rad_total * yerr,
                errorevery=8,
                elinewidth=1,
                capsize=2,
                capthick=1,
                c="k",
                zorder=1,
            )
        ax.plot(
            x,
            sample_em,
            "-",
            lw=1,
            label=f"Emitted ({tmax} K)",
            c="tab:red",
            zorder=0,
        )

        # Plot rad and emission at temps in tmin, tmax
        for temp in np.arange(tmax, tmin - tstep, -tstep):
            emiss = re.cmrad2mrad(re.bbrw(x, temp))
            total_rad = sample_rad + emiss
            ax.plot(x, total_rad, "-", lw=1, c=cmap(cnorm(temp)), zorder=-1)

            # Plot curve minus absorption
            total_rad -= absorb
            ax.plot(
                x, total_rad, "--", lw=0.75, c=cmap(cnorm(temp)), zorder=-1
            )

        ax.set_ylabel(r"Radiance $\left( \frac{W}{m^2 \ sr \ \mu m}\right)$")
        ax.set_xlim(0.5, wlmax)
        ax.set_ylim(-2, 65)

    axs[1].set_xlabel("Wavelength $(\mu m)$")
    axs[1].set_xlim(2, 3.5)
    axs[1].set_ylim(0, 13)
    axs[1].fill_betweenx(
        [0, 100], [3, 3], [100, 100], color="tab:gray", ec="none", alpha=0.3
    )
    fig.colorbar(sm, ax=axs, label="Temperature [K]")
    axs[0].legend(fontsize=8)
    axs[1].plot(
        [-1, -1],
        [-1, -1],
        "-",
        lw=1,
        c=cmap(cnorm(tmax)),
        label="Reflected+Emitted",
    )
    axs[1].plot(
        [-1, -1],
        [-1, -1],
        "--",
        lw=1,
        c=cmap(cnorm(tmax)),
        label="3$\mu$m absorption",
    )
    h_, l_ = axs[1].get_legend_handles_labels()
    axs[1].legend(h_[-2:], l_[-2:], loc="upper right", fontsize=8)
    return _save_or_show(fig, axs, fsave, figdir)


# %%
@_reset_plot_style_dec
def c4_resids(fsave="c4_resids.png", figdir=FIGDIR):
    """
    FIGURE 4
    Plot channel 4 residuals between model and Diviner for
    - Roughness model
    - M3 L2 (Clark et al., 2010)
    - Li et al. (2017)
    - Wohler et al. (2017)

    For the following regions:
    -
    """

    # Make plot
    fig, axs = plt.subplots(2, 2, sharex=True, sharey=True, figsize=(3.6, 3.6))

    for i, ax in enumerate(axs.flat):
        ax.set_xlim(-15, 15)
    axs[0, 0].set_ylabel("Count")
    axs[1, 0].set_ylabel("Count")
    axs[1, 0].set_xlabel("Residuals [K]")
    axs[1, 1].set_xlabel("Residuals [K]")

    return _save_or_show(fig, axs, fsave, figdir)


# %%
@_reset_plot_style_dec
def ohibd_map(fsave="ohibd_map.png", figdir=FIGDIR):
    """
    FIGURE 5
    Plot mapview OHIBD for roughness corrected and wohler corrected scene at
    early time and noontime. Spectra shown below.
    """
    # Load data
    labels = {"A": "Bandfield", "B": "Wohler", "C": "Bandfield", "D": "Wohler"}
    files = {
        "A": "/work/team_moon/data/comparisons/m3_bandfield_refl/220705/M3G20090209T212512.nc",
        "B": "/work/team_moon/data/comparisons/m3_bandfield_refl/220705/M3G20090209T212512.nc",
        "C": "/work/team_moon/data/comparisons/m3_bandfield_refl/220705/M3G20090612T230542.nc",
        "D": "/work/team_moon/data/comparisons/m3_bandfield_refl/220705/M3G20090612T230542.nc",
    }
    data = {k: xr.open_dataset(files[k]) for k in files}
    ext = [-56.35, -55.5, 8, 10]
    t1 = np.mean(m3.get_tloc_range_spice("M3G20090209T212512", ext[0], ext[1]))
    t2 = np.mean(m3.get_tloc_range_spice("M3G20090612T230542", ext[0], ext[1]))
    tlocs = {"A": t1, "B": t2, "C": t1, "D": t2}

    # Parse data, make ohibd maps
    ohibds = {}
    specs = {}
    vmin, vmax = (1, 0)
    for k, ds in data.items():
        ref = ds.ref
        spec = m3.contin_remove(ref.where((ref > 1e-3) & (ref < 0.5)))
        spec = spec.where((spec > 0) * (spec < 2))
        ohibd = m3.ohibd(spec)
        ohibd = ohibd.where((ohibd > 0) & (ohibd < 1))
        cmin, cmax = ohibd.min().values, ohibd.max().values
        vmin, vmax = min(vmin, cmin), max(vmax, cmax)
        specs[k] = spec
        ohibds[k] = ohibd

    aspect = (ext[3] - ext[2]) / (ext[1] - ext[0])
    # Make plot
    fig = plt.figure(figsize=(7.2, 7.2 * aspect))
    ax_dict = fig.subplot_mosaic(
        """AB\nCD\nEE""", gridspec_kw={"height_ratios": [1, 1, 0.3]}
    )
    for k, ohibd in ohibds.items():
        ax = ax_dict[k]
        ax.imshow(ohibd, cmap="cividis", vmin=vmin, vmax=vmax, extent=ext)
        tloc = f"{tlocs[k]:.2f} h"
        ax.set_title(f"{labels[k]} {tloc}")
        if k in ("A", "C"):
            ax.set_ylabel("Latitude")
        elif k in ("B", "D"):
            ax.set_xlabel("Longitude")
        specs[k].mean(["x", "y"]).plot(label=tloc, ax=ax_dict["E"])
    ax_dict["E"].legend()
    ax_dict["E"].set_title("")

    return _save_or_show(fig, ax_dict, fsave, figdir)


# %%
@_reset_plot_style_dec
def ohibd_lat_tloc(fsave="ohibd_lat_tloc.png", figdir=FIGDIR):
    """
    FIGURE 6
    Plot roughness OHIBD vs lat and tloc.
    Markers:
    - highlands
    - maria
    - swirl
    - pyroclastic
    """

    # Make plot
    fig, axs = plt.subplots(2, sharey=True, figsize=(3.6, 7.2))
    axs[0].set_ylim(0, 1)

    # OHIBD vs lat
    ax = axs[0]

    ax.set_xlim(-90, 90)

    ax.set_xlabel("Latitude")
    ax.set_ylabel("OHIBD")

    # OHIBD vs tloc
    ax = axs[1]

    ax.set_xlim(6, 18)
    ax.set_xlabel("Time [h]")
    ax.set_ylabel("OHIBD")

    return _save_or_show(fig, axs, fsave, figdir)


# %%
@_reset_plot_style_dec
def bt_iirs(fsave="bt_iirs.png", figdir=FIGDIR):
    """
    FIGURE 7
    Plot roughness predicted BT in 4.5—4.85 µm and delta vs local time

    """
    rms = 15
    albedo = 0.12
    emissivity = 0.96
    lat = 0
    wls = np.array([3, 4.5, 4.85])  # [um]
    tlocs = np.arange(7, 18)
    colors = ["tab:blue", "tab:orange", "tab:gray"]
    markers = ["D", "s", "o"]

    bts = np.zeros((len(wls), len(tlocs)))
    for i, tloc in enumerate(tlocs):
        inc, az = rh.tloc_to_inc(tloc, lat, az=True)
        geom = (inc, az, 0, 0)
        tparams = {"tloc": tloc, "albedo": albedo, "lat": lat}
        emission = re.rough_emission_lookup(
            geom, wls, rms, tparams, emissivity=emissivity
        )
        bt = re.btempw(wls, re.mrad2cmrad(emission))
        # wnrad = re.wlrad2wnrad(wls, emission)
        # div_bt = rd.divrad2bt(rd.divfilt_rad(wnrad))
        bts[:, i] = re.btempw(wls, re.mrad2cmrad(emission))

    # Make plot
    fig, ax = plt.subplots(1, figsize=(3.6, 3.6))
    for j, (wl, bt) in enumerate(zip(wls, bts)):
        ax.plot(tlocs, bt, markers[j], lw=1, c=colors[j], label=f"{wl} $\mu$m")

    ax.set_xlabel("Local time [hr]")
    ax.set_ylabel("Brightness temperature [K]")
    ax.set_xlim(6, 18)
    ax.set_ylim(280, 440)
    ax.legend(loc="upper left", title="BT", fontsize=8)

    # Plot delta vs local time
    ax2 = ax.twinx()
    ax2.plot(tlocs, bts[2] - bts[1], "-", c="k", label="$4.85 - 4.5 \mu$m")
    ax2.plot(
        tlocs, bts[1] - bts[0], "--", c="tab:gray", label="$4.5 - 3 \mu$m"
    )
    ax2.set_ylabel("$\Delta$ Brightness Temperature [K]")
    ax2.set_ylim(-10, 10)
    ax2.legend(loc="upper right", title="$\Delta$ BT", fontsize=8)

    # Label incidence on top axis
    incs = np.array([rh.tloc_to_inc(tloc, lat) for tloc in tlocs])
    incs[tlocs < 12] = -incs[tlocs < 12]
    iax = ax.twiny()
    iax.set_xlim(-90, 90)
    iax.set_xticks(np.arange(-90, 91, 30))
    iax.set_xticklabels(np.abs(np.arange(-90, 91, 30)))
    iax.set_xlabel("Incidence [deg]")

    return _save_or_show(fig, ax, fsave, figdir)
