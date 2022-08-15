import datetime
import json
from pathlib import Path
import numpy as np
from roughness import m3_correction as m3
from roughness import config as cfg
from roughness import diviner as rd
from roughness import emission as re

# Set params
kwin = {"dask": False, "chunks": {"band": None, "y": 10, "x": None}}
kwre = {
    "rms": 15,
    "emissivity": 0.96,
    "rerad": True,
    "rerad_albedo": 0.12,
    "flookup": cfg.FLOOKUP,
    "tlookup": cfg.TLOOKUP,
    "cast_shadow_time": 0.05,
}
dwls = rd.load_div_filters().wavelength.values

fprio = "/home/ctaiudovicic/projects/enthalpy/data/210301_priority_list.json"
outdir = "/work/team_moon/data/comparisons/m3_bandfield_refl/220811/"

# Load data from prio
with open(fprio, encoding="utf8") as f:
    prio = json.load(f)
size = []
for k, v in prio.items():
    ext, *basenames = v
    size.append([m3.m3_size_est(basenames[0], ext)[1], k])
sprio = {}
for _, k in sorted(size):
    sprio[k] = prio[k]

# Setup outdir
outdir = Path(outdir)
outdir.mkdir(parents=True, exist_ok=True)

# Loop through priority dict
for i, (k, v) in enumerate(sprio.items()):
    if i % 2 == 0:
        continue
    ext, *basenames = v
    for basename in basenames:
        roi_size = m3.m3_size_est(basename, ext)[1]
        now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        print(f"Starting {k} {basename} {ext} {roi_size} MB at {now}")
        wls = np.concatenate([m3.get_m3_wls(basename), dwls])
        ref, em = m3.m3_refl(basename, ext, wls=wls, kwin=kwin, kwre=kwre)
        wl3um = ref.wavelength[-1]  # ~3 microns (last wl in m3)
        bt = re.btempw(wl3um, re.mrad2cmrad(em.sel(wavelength=wl3um))).drop(
            "wavelength"
        )
        # tbol = rd.emission2tbol_xr(em, wls)

        # Save images as dataset netcdf
        out = ref.to_dataset(name="ref", promote_attrs=True)
        out = out.assign(
            {"em": em, "bt": bt}
        )  # , 'tbol': tbol}) tbol does slow interp
        out.to_netcdf(outdir / f"{basename}.nc")
        print(f"Saved to {outdir.as_posix()}/{basename}.nc")
