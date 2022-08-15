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


# Constants
MPLSTYLE = cfg.FIGURES_DIR / "roughness.mplstyle"
DATE = datetime.now().strftime("%y%m%d")
FIGDIR = cfg.FIGURES_DIR / "tmp" / DATE
FPRIO = cfg.DATA_DIR_M3 / "210301_rois.json"
MPLWARN = True

# Helpers
def _reset_plot_style(mplstyle=MPLSTYLE):
    """Reset matplotlib style defaults, use custom mplstyle path."""
    mpl.rcParams.update(mpl.rcParamsDefault)
    try:
        mpl.style.use(mplstyle)
    except (OSError, FileNotFoundError):
        mpl_read_warn = f"Could not read mplstyle file {mplstyle}"
        global MPLWARN
        if MPLWARN:
            warnings.warn(mpl_read_warn)
            MPLWARN = False


def _reset_plot_style_dec(func):
    """Decorator for resetting matplotlib style defaults before and after."""

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        _reset_plot_style()
        value = func(*args, **kwargs)
        _reset_plot_style()
        return value

    return wrapper


def _save_or_show(fig, ax, fsave, figdir=FIGDIR, **kwargs):
    """Save figure or show fig and return ax."""
    if fsave:
        fsave = Path(figdir) / Path(fsave)  # Prepend path
        fsave.parent.mkdir(parents=True, exist_ok=True)  # Make dir if needed
        fig.savefig(fsave, **kwargs)
        print("Saved to", fsave)
    else:
        plt.show()
    return ax


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


@_reset_plot_style_dec
def temp_compare(
    fsave="temp_compare.png",
    figdir=FIGDIR,
    basename="M3G20090612T230542",
    ext=None,
    ddirs=None,
):
    """Four panel mapview comparison of temperatures"""
    # Setup
    if ext is None:
        ext = _ext_dict()[basename]
    img_dict = _find_imgs(ddirs)  # dict[keys] -> {basename: img_path}

    # Load data
    imgs = {}
    for src, d in img_dict.items():
        if basename not in d:
            continue
        img = _get_tempxr(d[basename], basename, ext, src)
        imgs[src] = img
        if src == "diviner":
            mean = img.median()
            sd = max(img.std(), 10)  # at least +/- 10 K
            vmin, vmax = (mean - 2 * sd, mean + 2 * sd)
            aspect = img.aspect

    # Make plot
    fig, axs = plt.subplots(
        2, 2, sharex=True, sharey=True, figsize=(4 + 1.5, 4 * aspect)
    )
    for src, ax in zip(img_dict.keys(), axs.flat):
        if src not in imgs:
            continue
        p = ax.imshow(
            imgs[src], cmap="magma", extent=ext, vmin=vmin, vmax=vmax
        )
        ax.set_title(src.capitalize())
    fig.colorbar(p, ax=axs.flatten(), label="Temperature [K]")
    return _save_or_show(fig, axs, fsave, figdir)


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
    # Setup
    ddir = Path(cfg.DATA_DIR_M3) / "comparisons"
    ddirs = {
        "diviner": ddir / "diviner_T" / "210422",
        "li": ddir / "m3_li_T",
        "wohler": ddir / "m3_wohler_T",
        "rough": ddir / "m3_bandfield_refl" / "220705" / "no_rerad",
    }

    # Load data
    d = _find_imgs(
        ddirs
    )  # dict[('li', 'wohler', 'rough')] -> {basename: img_path}
    ext_d = _ext_dict()
    ext_d = {
        k: v for k, v in ext_d.items() if v[1] - v[0] < 2
    }  # remove big lon rois
    basenames = {
        k: v for k, v in ext_d.items() if k in d["diviner"]
    }  # need diviner data for residuals
    basenames = [
        b for b in basenames if all(b in v for v in d.values())
    ]  # keep only if exists for Li and Wohler too
    labels = {
        "rough": "This work",
        "li": "Li et al. (2017)",
        "wohler": "Wohler et al. (2017)",
        "clark": "Clark et al. (2010)",
    }
    # basenames = ['M3G20090210T033052']
    resids = {b: {} for b in basenames}
    colors = {
        "rough": "tab:green",
        "wohler": "tab:brown",
        "li": "tab:pink",
        "clark": "tab:blue",
    }
    for basename in basenames:
        ext = ext_d[basename]
        div = _get_tempxr(d["diviner"][basename], basename, ext, "diviner")
        for src in ("rough", "li", "wohler", "clark"):
            if src != "clark" and basename not in d[src]:
                continue
            if src == "clark":
                da = _get_tempxr(None, basename, ext, src)
            else:
                da = _get_tempxr(d[src][basename], basename, ext, src)
            # Compute residuals from diviner
            if src == "wohler":
                res = (div.lat[0] - div.lat[1]).values
                da = da.reindex_like(div, "nearest", tolerance=res)
                resid = da - div
            else:
                da = da.where(da > 50)
                if len(div.lon) == 1:
                    resid = da - div.isel(lon=0).interp(lat=da.lat)
                else:
                    resid = da - div.interp(lat=da.lat, lon=m3.lon180(da.lon))
            resids[basename][src] = resid

    # Make plot
    vmin, vmax = (-30, 30)
    fig, axs = plt.subplots(len(basenames), figsize=(3.8, 2 * len(basenames)))
    for basename, ax in zip(basenames, axs):
        for i, src in enumerate(("clark", "wohler", "li", "rough")):
            if src not in resids[basename]:
                continue
            # Clip to min/max (retains outlier values in last bin)
            resid = resids[basename][src].clip(vmin, vmax).values
            resid = resid[np.isfinite(resid)]
            freq, bins = np.histogram(
                resid, range=(vmin, vmax), bins=21, density=True
            )

            # Plot as offset bar hist
            if i == 0:
                width = (bins[1] - bins[0]) / 4
            x = bins[:-1] + width * (i - 2)  # (-2, -1, 0, 1)
            ax.bar(
                x, freq, width=width, facecolor=colors[src], label=labels[src]
            )
            ax.set_ylabel(basename, fontsize=8)
            ext = ext_d[basename]
            lat = f"Lat {ext[2]:3.1f}-{ext[3]:3.1f}$^o$"
            tloc = f"t {np.mean(m3.get_tloc_range_spice(basename, ext[0], ext[1])):2.2f} h"
            ax.annotate(
                lat,
                (0.02, 0.96),
                ha="left",
                va="top",
                xycoords="axes fraction",
            )
            ax.annotate(
                tloc,
                (0.02, 0.81),
                ha="left",
                va="top",
                xycoords="axes fraction",
            )
            ax.tick_params(direction="out", right=False, top=False)
            ax.yaxis.set_major_formatter(mpl.ticker.FormatStrFormatter("%.2f"))
            ax.set_xlim(vmin, vmax)
    axs[0].legend(
        fontsize=8,
        ncol=2,
        bbox_to_anchor=(0, 1.02, 1, 0.2),
        loc="lower left",
        mode="expand",
        borderaxespad=0,
    )
    axs[-1].set_xlabel("Diviner C4 (8.25 $\mu m$) residuals [K]", fontsize=10)

    return _save_or_show(fig, axs, fsave, figdir)


@_reset_plot_style_dec
def ibd3um_map(fsave="ibd3um_map.png", figdir=FIGDIR):
    """
    FIGURE 5
    Plot mapview IBD3um for roughness corrected and Li corrected scene at
    early time and noontime. Spectra shown below.

    Plots:
        | Rough | Li  |
    t1  |   A   |  B  |
    t2  |   C   |  D  |
    spec|       E     |
    """

    def get_tloc(bn, ext):
        return np.mean(m3.get_tloc_range_spice(bn, ext[0], ext[1]))

    # Load data
    fprio = cfg.DATA_DIR_M3 / "210301_rois.json"
    rois = m3.read_m3_rois(fprio, True)
    ddir = Path(cfg.DATA_DIR_M3) / "comparisons"
    dwork = {"A": "rough", "B": "li", "C": "rough", "D": "li"}
    dtime = {"A": "t1", "B": "t1", "C": "t2", "D": "t2"}
    dlabels = {"rough": "This work", "li": "Li et al., 2017"}
    dtlocs = {}
    fmt = {"rough": "--", "li": "-"}
    basenames = {"t1": "M3G20090209T212512", "t2": "M3G20090612T230542"}
    # basenames = {"t1": "M3G20090516T040653", "t2": "M3G20090612T143522"}
    ddirs = {
        "rough": "m3_bandfield_refl/220705/no_rerad",
        # "rough": "m3_bandfield_refl/220705/",
        "li": "m3_li_refl",
    }

    # Get extent, search for and load files, and get tloc for labels
    title = [k for k, (_, *b) in rois.items() if basenames["t1"] in b][0]
    ext = [ext for _, (ext, *b) in rois.items() if basenames["t1"] in b][0]
    data, labels = {}, {}
    for k, work in dwork.items():
        basename = basenames[dtime[k]]
        dpath = ddir / ddirs[work]
        dtlocs[k] = f"{get_tloc(basename, ext):.2f} h"
        labels[k] = f"{get_tloc(basename, ext):.2f} h ({dlabels[work]})"
        fimg = dpath.glob(f"{basename}*")
        fimg = [f for f in fimg if f.suffix != ".hdr"][0]
        if work == "rough":
            da = xr.open_dataset(fimg).ref
        elif work == "li":
            da = xr.open_dataarray(fimg, engine="rasterio")
            da = m3.assign_loc(m3.bands2wls(da, basename), basename)
            da = m3.subset_m3xr_ext(da, ext)
        data[k] = da

    # Parse data, make ibd3um maps
    ibd3ums = {}
    specs = {}
    vmin, vmax = (1, 0)
    for k, da in data.items():
        ref = da  # .where((da > 1e-3) & (da < 0.5))
        spec = m3.contin_remove(ref)
        spec = spec  # .where((spec > -2) * (spec < 2))

        ibd3um = m3.ibd3um(spec)
        ibd3um = ibd3um  # .where((ibd3um > 0) & (ibd3um < 30))
        cmin, cmax = ibd3um.min().values, ibd3um.max().values
        vmin, vmax = min(vmin, cmin), max(vmax, cmax)
        specs[k] = ref.mean(["x", "y"]).isel(wavelength=slice(0, -1))
        ibd3ums[k] = ibd3um
    vmin, vmax = (-4, 20)

    # Make plot axes dict
    aspect = (ext[3] - ext[2]) / (ext[1] - ext[0])
    fig = plt.figure(figsize=(5, 5 * aspect + 4))
    outer_gs = fig.add_gridspec(
        2, 1, top=0.93, hspace=0.1, height_ratios=[5 * aspect, 4]
    )
    img_gs = outer_gs[0].subgridspec(2, 2, hspace=0.1 / aspect, wspace=0.15)
    ax_dict = {
        k: ax
        for k, ax in zip(
            dwork.keys(), img_gs.subplots(sharex=True, sharey=True).flatten()
        )
    }
    ax_dict["E"] = fig.add_subplot(outer_gs[1])

    for k in ("D", "C", "B", "A"):
        ibd3um = ibd3ums[k]
        ax = ax_dict[k]
        p = ax.imshow(ibd3um, cmap="YlGnBu", vmin=vmin, vmax=vmax, extent=ext)
        # tloc = f"{tlocs[tdict[k]]:.2f} h"
        # label = f"{labels[k]} {tloc}"
        # ax.set_title(labels[k])
        ax.tick_params(direction="out", right=False, top=False)
        if k in ("A", "B"):
            ax.tick_params(axis="x", bottom=False)
            ax.set_title(dlabels[dwork[k]])
        if k in ("A", "C"):
            ax.set_ylabel(f"$t=${dtlocs[k]}\nLatitude [$^o$]")
        if k in ("C", "D"):
            ax.set_xlabel("Longitude [$^o$]")
            # if k in ("B", "D"):
            ax.tick_params(axis="y", left=False)
        # Spectra plot
        specs[k].plot(label=labels[k], ls=fmt[dwork[k]], ax=ax_dict["E"])
    fig.suptitle(title)
    cb = fig.colorbar(p, pad=0.12, ax=[ax_dict["E"]], location="top")
    cb.set_label(label="IBD$_{3 \mu m}$")
    cb.ax.xaxis.set_ticks_position("bottom")
    ax_dict["E"].legend(loc="upper left")
    ax_dict["E"].set_ylabel("Reflectance")
    ax_dict["E"].set_xlabel("Wavelength [$\mu m$]")
    ax_dict["E"].set_xlim((0.5, 3))
    ax_dict["E"].set_title("")

    return _save_or_show(fig, ax_dict, fsave, figdir)


@_reset_plot_style_dec
def ibd3um_lat_tloc(fsave="ibd3um_lat_tloc.png", figdir=FIGDIR):
    """
    FIGURE 6
    Plot roughness IBD3um vs lat and tloc.
    Markers:
    - highlands
    - maria
    - swirl
    - pyroclastic
    """
    fprio = cfg.DATA_DIR_M3 / "210301_rois.json"
    rois = m3.read_m3_rois(fprio, True)
    outdir = Path("/work/team_moon/data/comparisons/m3_bandfield_refl/220705/")
    # outdir = Path('/work/team_moon/data/comparisons/m3_bandfield_refl/220705/no_rerad/')

    tlocs = []
    tlocs_low = []
    tlocs_upp = []
    ibd3um_tloc_mean = []
    ibd3um_tloc_std = []
    lats = []
    ibd3um_lat_mean = []
    ibd3um_lat_std = []
    for roi, (ext, *basenames) in rois.items():
        if not (outdir / f"{basenames[0]}.nc").exists():
            continue
        for basename in basenames:
            if not (outdir / f"{basename}.nc").exists():
                continue

            fin = outdir / f"{basename}.nc"
            ds = xr.open_dataset(fin)
            # Get tlocs
            tloc_low, tloc_upp = m3.get_tloc_range_spice(
                basename, ext[0], ext[1]
            )
            tloc = np.mean([tloc_low, tloc_upp])

            # get ibd3um img
            ref = ds.ref
            spec = m3.contin_remove(ref.where((ref > 1e-3) & (ref < 0.5)))
            spec = spec.where((spec > 0) * (spec < 2))
            ibd3um = m3.ibd3um(spec)
            ibd3um = ibd3um.where((ibd3um > 0) & (ibd3um < 30))

            tlocs.append(tloc)
            tlocs_low.append(tloc_low)
            tlocs_upp.append(tloc_upp)
            ibd3um_tloc_mean.append(ibd3um.mean().values)
            ibd3um_tloc_std.append(ibd3um.std().values)

            # get lat
            lat_res = 3  # degrees
            ibd_meanx = ibd3um.mean("x")
            lat_arr = ibd3um.lat.mean("x")
            if ext[3] - ext[2] < lat_res:
                lats.append(lat_arr.mean().values)
                ibd3um_lat_mean.append(ibd_meanx.mean().values)
                ibd3um_lat_std.append(ibd_meanx.std().values)
            else:
                yarr = np.arange(ext[2], ext[3], lat_res)
                for lat in yarr:
                    ibd_y = ibd_meanx.where(
                        (lat - lat_res / 2 < lat_arr)
                        & (lat_arr < lat + lat_res / 2)
                    )
                    lats.append(lat)
                    ibd3um_lat_mean.append(ibd_y.mean().values)
                    ibd3um_lat_std.append(ibd_y.std().values)

    # Make plot
    fig, axs = plt.subplots(2, sharey=True, figsize=(3.6, 7.2))

    # IBD3um vs lat
    ax = axs[0]
    ax.errorbar(lats, ibd3um_lat_mean, yerr=ibd3um_lat_std, fmt="x")

    ax.set_ylim(0, 30)
    ax.set_xlim(-70, 70)
    ax.set_xlabel("Latitude")
    ax.set_ylabel("IBD$_{3\mu m}$")

    # IBD3um vs tloc
    ax = axs[1]
    ax.errorbar(tlocs, ibd3um_tloc_mean, yerr=ibd3um_tloc_std, fmt="x")

    ax.set_xlim(6, 18)
    ax.set_xlabel("Time [h]")
    ax.set_ylabel("IBD$_{3\mu m}$")

    return _save_or_show(fig, axs, fsave, figdir)


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
    ls = 180
    wls = np.array([3, 4.5, 4.85])  # [um]
    tlocs = np.arange(7, 18)
    colors = ["tab:blue", "tab:orange", "tab:gray"]
    markers = ["D", "s", "o"]

    bts = np.zeros((len(wls), len(tlocs)))
    for i, tloc in enumerate(tlocs):
        inc, az = rh.tloc_to_inc(tloc, lat, az=True)
        geom = (inc, az, 0, 0)
        tparams = {"tloc": tloc, "albedo": albedo, "lat": lat, "ls": ls}
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


# Helpers
def _read_li(fimg, basename, ext=None):
    """"""
    fimg = Path(fimg).with_suffix("")  # No extension on Li imgs (trim .hdr)
    da = xr.open_dataarray(fimg, engine="rasterio")
    da = m3.assign_loc(m3.bands2wls(da, basename), basename)
    if ext is not None:
        da = m3.subset_m3xr_ext(da, ext)
    return da


def _get_tempxr(fimg, basename, ext, src="rough"):
    """
    Return temperature DataArray from file from one of the following:
    srcs: ['rough', 'diviner', 'li', 'wohler].
    """
    if src == "clark":  # Get temperature from M3 L2 data (band 2 of sup file)
        da = m3.get_m3xr(basename, "sup")
        da = m3.subset_m3xr_ext(da, ext).isel(band=1)
    elif src == "rough":
        da = xr.open_dataset(fimg).bt
    elif src == "li":
        fimg = Path(fimg).with_suffix("")  # No ext on Li imgs (trim .hdr)
        da = xr.open_dataarray(fimg, engine="rasterio").isel(band=0)
        da = m3.assign_loc(da, basename)
        da = m3.subset_m3xr_ext(da, ext)
    elif src == "diviner":
        # Diviner data channels (C3-C9, Tbol)
        da = xr.open_dataarray(fimg, engine="rasterio")
        da = da.assign_coords(x=m3.lon180(da.x))
        div_channels = [f"c{i}" for i in range(3, 10)] + ["tbol"]
        da = da.assign_coords(band=div_channels).sel(band="c4")
    elif src == "wohler":
        da = xr.open_dataarray(fimg, engine="rasterio").isel(band=0)
        da = da.assign_coords(x=m3.lon180(da.x))
    da.attrs["aspect"] = (ext[3] - ext[2]) / (ext[1] - ext[0])
    if src in ("diviner", "wohler"):
        da = da.rename({"x": "lon", "y": "lat"})
        da = da.sel(lon=slice(ext[0], ext[1]), lat=slice(ext[3], ext[2]))
    return da


def _ext_dict(fprio=FPRIO):
    """"""
    prio = m3.read_m3_rois(fprio, True)
    ext_dict = {}
    for _, (ext, *basenames) in prio.items():
        for basename in basenames:
            ext_dict[basename] = ext
    return ext_dict


def _find_imgs(dir_dict=None, img_exts=(".tif", ".hdr", ".nc")):
    """Return list of common images in dirs"""

    def _get_basename(fpath):
        i0 = max(fpath.stem.find("M3G20"), fpath.stem.find("M3T20"))
        return fpath.stem[i0 : i0 + 18]

    if dir_dict is None:
        ddir = Path(cfg.DATA_DIR_M3) / "comparisons"
        dir_dict = {
            "diviner": ddir / "diviner_T" / "210422",
            "li": ddir / "m3_li_T",
            "wohler": ddir / "m3_wohler_T",
            "rough": ddir / "m3_bandfield_refl" / "220705",  # / 'no_rerad
        }
    # Find imgs
    img_dict = {}
    for key, dpath in dir_dict.items():
        dpath = Path(dpath)
        imgs = []
        for img_ext in img_exts:
            imgs.extend(dpath.glob("*" + img_ext))
        if len(imgs) == 0:
            print(f"No images found in {dpath.as_posix()}")
            continue
        imgs = {_get_basename(img): img for img in imgs}
        img_dict[key] = imgs
    return img_dict
