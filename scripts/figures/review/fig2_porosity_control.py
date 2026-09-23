#!/usr/bin/env python
"""Figure 2 — Porosity control.

Left:  a row of generated cross-sections at increasing target porosity.
Right: requested porosity (x) vs measured porosity (y), dashed 1:1 line,
       R² and MAE annotated.

Data format
-----------
--csv     CSV with columns  requested_porosity, measured_porosity
          (fractions, e.g. 0.031; use --percent if your file is in %).
--slices  folder of 2-D greyscale images (.png/.tif/.npy), one per porosity
          level. Files are sorted by name, so name them e.g.
          01_p0.01.png, 02_p0.02.png ... The target porosity shown under
          each panel comes from --slice-porosity (comma list, same order).
          If --slices is omitted, placeholder synthetic data is generated.
--pixel-um  voxel size in µm (default 25) for the scale bar.

Example
-------
python fig2_porosity_control.py --csv porosity.csv --slices slices/ \
    --slice-porosity 0.005,0.01,0.02,0.04,0.08 --out fig2_porosity_control
"""
import argparse
from pathlib import Path

import numpy as np
import matplotlib
import matplotlib.patheffects
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ============================ SHARED STYLE BLOCK ============================
# Identical in fig2/fig3/fig4. Edit here to change every figure at once.
FONT_FAMILY = "DejaVu Sans"       # sans-serif present in every matplotlib install
FONT_SIZE = 11                    # body text / tick labels (pt)
LABEL_SIZE = 12                   # axis labels
TITLE_SIZE = 13                   # panel titles
FIG_WIDTH_IN = 13.33              # 16:9 slide width at 1 in = 1 in of slide
FIG_HEIGHT_IN = 5.2               # ~40 % of slide height; two figures stack on one slide
DPI = 300
C_REAL = "#0072B2"                # blue  (dark in greyscale)
C_SYNTH = "#E69F00"               # orange (light in greyscale) — CVD-safe pair
C_ACCENT = "#0072B2"
C_GREY = "#666666"
C_LIGHT = "#bbbbbb"
IMG_CMAP = "gray"                 # all tomography images
SCALEBAR_UM = 1000                # scale bar length in µm
PIXEL_UM = 25.0                   # default voxel size

plt.rcParams.update({
    "font.family": FONT_FAMILY, "font.size": FONT_SIZE,
    "axes.labelsize": LABEL_SIZE, "axes.titlesize": TITLE_SIZE,
    "xtick.labelsize": FONT_SIZE, "ytick.labelsize": FONT_SIZE,
    "legend.fontsize": FONT_SIZE, "legend.frameon": False,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.linewidth": 0.8, "xtick.major.width": 0.8, "ytick.major.width": 0.8,
    "savefig.dpi": DPI, "pdf.fonttype": 42, "ps.fonttype": 42,
})


def add_scalebar(ax, shape, pixel_um=PIXEL_UM, length_um=SCALEBAR_UM):
    """White bar with black outline in the lower-left corner of an image axis."""
    h, w = shape
    px = length_um / pixel_um
    x0, y0 = 0.05 * w, 0.93 * h
    ax.plot([x0, x0 + px], [y0, y0], color="black", lw=4, solid_capstyle="butt")
    ax.plot([x0, x0 + px], [y0, y0], color="white", lw=2.5, solid_capstyle="butt")
    ax.text(x0 + px / 2, y0 - 0.03 * h, f"{length_um:g} µm", color="white",
            ha="center", va="bottom", fontsize=FONT_SIZE - 1,
            path_effects=[matplotlib.patheffects.withStroke(linewidth=2, foreground="black")])


def show_image(ax, img, title=None, pixel_um=PIXEL_UM):
    ax.imshow(img, cmap=IMG_CMAP, vmin=0, vmax=255, interpolation="nearest")
    ax.set_axis_off()
    if title:
        ax.set_title(title, fontsize=FONT_SIZE, pad=4)
    add_scalebar(ax, img.shape[:2], pixel_um)


def load_image(path):
    path = Path(path)
    if path.suffix == ".npy":
        img = np.load(path)
    else:
        if path.suffix in (".tif", ".tiff"):
            import tifffile
            img = tifffile.imread(path)
        else:
            img = plt.imread(path)
            if img.ndim == 3:
                img = img[..., :3].mean(-1)
            if img.max() <= 1.0:
                img = img * 255
    img = np.asarray(img, dtype=np.float32)
    if img.max() > 255:                       # 16-bit → 8-bit range
        img = img / img.max() * 255
    return img
# ========================== END SHARED STYLE BLOCK ==========================


# ----------------------------- placeholder data -----------------------------
def synthetic_slice(porosity, size=192, rng=None):
    """A fake laminate cross-section: bright material, dark elongated pores."""
    rng = rng or np.random.default_rng(0)
    img = np.full((size, size), 170.0)
    img += rng.normal(0, 8, img.shape)
    yy = np.arange(size)[:, None]
    img += 12 * np.sin(yy / 6.0)                       # ply texture
    n_pores = int(porosity * size * size / 40)
    for _ in range(n_pores):
        cy, cx = rng.integers(0, size, 2)
        ry, rx = rng.integers(2, 4), rng.integers(4, 14)
        y, x = np.ogrid[:size, :size]
        img[((y - cy) / ry) ** 2 + ((x - cx) / rx) ** 2 <= 1] = 40
    return np.clip(img, 0, 255)


def placeholder_csv(n=60, rng=None):
    rng = rng or np.random.default_rng(1)
    req = rng.uniform(0.002, 0.10, n)
    meas = req + rng.normal(0, 0.003, n)
    return req, np.clip(meas, 0, None)


# --------------------------------- figure -----------------------------------
def make_figure(req, meas, slices, slice_por, pixel_um, out):
    n = len(slices)
    fig = plt.figure(figsize=(FIG_WIDTH_IN, FIG_HEIGHT_IN))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.9, 1.0], wspace=0.15,
                          left=0.03, right=0.98, top=0.90, bottom=0.14)
    gs_left = gs[0].subgridspec(1, n, wspace=0.06)
    for i, (img, p) in enumerate(zip(slices, slice_por)):
        ax = fig.add_subplot(gs_left[i])
        show_image(ax, img, f"target {100 * p:.1f} %", pixel_um)
    fig.text(0.03 + 0.62 / 2, 0.95, "Generated cross-sections at increasing target porosity",
             ha="center", fontsize=TITLE_SIZE)

    ax = fig.add_subplot(gs[1])
    lim = (0, 1.05 * max(req.max(), meas.max()) * 100)
    ax.plot(lim, lim, ls="--", color=C_LIGHT, lw=1, label="1:1")
    ax.scatter(req * 100, meas * 100, s=28, color=C_ACCENT, edgecolor="white",
               linewidth=0.5, zorder=3, label="generated volume")
    ss_res = np.sum((meas - req) ** 2)
    ss_tot = np.sum((meas - meas.mean()) ** 2)
    r2 = 1 - ss_res / ss_tot
    mae = np.mean(np.abs(meas - req)) * 100
    ax.text(0.04, 0.96, f"$R^2$ = {r2:.3f}\nMAE = {mae:.2f} pp",
            transform=ax.transAxes, va="top", ha="left", fontsize=FONT_SIZE)
    ax.set_xlabel("Requested porosity (%)")
    ax.set_ylabel("Measured porosity (%)")
    ax.set_xlim(lim); ax.set_ylim(lim); ax.set_aspect("equal")
    ax.legend(loc="lower right")
    ax.set_title("Requested vs measured porosity")

    fig.savefig(f"{out}.png"); fig.savefig(f"{out}.pdf")
    print(f"wrote {out}.png / .pdf   R²={r2:.3f}  MAE={mae:.2f} pp  (n={len(req)})")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--csv"); ap.add_argument("--slices")
    ap.add_argument("--slice-porosity", help="comma list of target porosities, same order as sorted slice files")
    ap.add_argument("--percent", action="store_true", help="CSV values are in percent, not fractions")
    ap.add_argument("--pixel-um", type=float, default=PIXEL_UM)
    ap.add_argument("--out", default="fig2_porosity_control")
    a = ap.parse_args()

    if a.csv:
        import pandas as pd
        df = pd.read_csv(a.csv)
        req, meas = df["requested_porosity"].to_numpy(float), df["measured_porosity"].to_numpy(float)
        if a.percent:
            req, meas = req / 100, meas / 100
    else:
        print("no --csv: using placeholder scatter data"); req, meas = placeholder_csv()

    if a.slices:
        files = sorted(p for p in Path(a.slices).iterdir() if p.suffix in (".png", ".tif", ".tiff", ".npy"))
        slices = [load_image(f) for f in files]
        slice_por = [float(x) for x in a.slice_porosity.split(",")] if a.slice_porosity else [np.nan] * len(files)
    else:
        print("no --slices: using placeholder synthetic slices")
        slice_por = [0.005, 0.01, 0.02, 0.04, 0.08]
        slices = [synthetic_slice(p, rng=np.random.default_rng(i)) for i, p in enumerate(slice_por)]

    make_figure(req, meas, slices, slice_por, a.pixel_um, a.out)


if __name__ == "__main__":
    main()
