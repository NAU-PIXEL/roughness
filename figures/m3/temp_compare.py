from pathlib import Path
import plot_figs as pf
from roughness import config as cfg

# Setup
figdirs = {"rerad": pf.FIGDIR / "rerad", "no_rerad": pf.FIGDIR / "no_rerad"}
ddir = Path(cfg.DATA_DIR_M3) / "comparisons"
ddirs = {
    "diviner": ddir / "diviner_T" / "210422",
    "li": ddir / "m3_li_T",
    "wohler": ddir / "m3_wohler_T",
}
dddirs = {
    "rerad": {**ddirs},
    "no_rerad": {**ddirs},
}
dddirs["rerad"]["rough"] = ddir / "m3_bandfield_refl" / "220705"
dddirs["no_rerad"]["rough"] = (
    ddir / "m3_bandfield_refl" / "220705" / "no_rerad"
)

d = pf._find_imgs(ddirs)  # dict[ddirs keys] -> {basename: img_path}
ext_d = pf._ext_dict()
ext_d = {
    k: v for k, v in ext_d.items() if v[1] - v[0] < 2
}  # remove big lon rois
basenames = set(d["diviner"].keys()).intersection(
    ext_d.keys()
)  # Get keys in both

for rerad, ddirs in dddirs.items():
    figdir = figdirs[rerad]
    for basename in basenames:
        fsave = f"{basename}_{rerad}.png"
        _ = pf.temp_compare(fsave, figdir, basename, ddirs=ddirs)
