#!/usr/bin/env python
"""Figure 4 — Real vs synthetic.

Left:  grid of real and synthetic slices side by side (real column, synthetic
       column by default). --blind shuffles all panels, hides the labels, and
       writes <out>_key.csv so you can reveal the answer afterwards.
Right: two stacked histograms, real vs synthetic: pore size, and greyscale
       intensity.

Data format
-----------
--real / --synth   folders of 2-D greyscale slices (.png/.tif/.npy). The
                   first N of each (sorted by name, N = --n-pairs) are shown.
                   Greyscale histograms are computed from these images
                   (material+pore voxels; pixels below --air-threshold are
                   dropped so background air does not dominate).
--pores-real / --pores-synth
                   CSV with one column  pore_size_um  (one row per pore, e.g.
                   equivalent spherical diameter in µm). Any extra columns are
                   ignored. If omitted, placeholder pore sizes are generated.
--pixel-um         voxel size in µm (default 25).

If --real/--synth are omitted, placeholder synthetic data is generated.

Example
-------
python fig4_real_vs_synthetic.py --real slices_real/ --synth slices_gen/ \
    --pores-real pores_real.csv --pores-synth pores_gen.csv --blind --seed 3
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
def synthetic_slice(porosity, size=192, rng=None, contrast=1.0):
    rng = rng or np.random.default_rng(0)
    img = np.full((size, size), 170.0) + rng.normal(0, 8 * contrast, (size, size))
    yy = np.arange(size)[:, None]
    img += 12 * np.sin(yy / 6.0)
    y, x = np.ogrid[:size, :size]
    for _ in range(int(porosity * size * size / 40)):
        cy, cx = rng.integers(0, size, 2)
        img[((y - cy) / rng.integers(2, 4)) ** 2 + ((x - cx) / rng.integers(4, 14)) ** 2 <= 1] = 40
    return np.clip(img, 0, 255)


def placeholder_pores(n, rng, scale):
    return rng.lognormal(mean=np.log(scale), sigma=0.5, size=n)


# --------------------------------- figure -----------------------------------
def make_figure(real, synth, pores_real, pores_synth, pixel_um, out, blind, seed, air_thr):
    n = min(len(real), len(synth))
    # grid: 2 rows × n columns. Row 0 real, row 1 synthetic — or shuffled if blind.
    panels = [("real", i, real[i]) for i in range(n)] + [("synthetic", i, synth[i]) for i in range(n)]
    if blind:
        rng = np.random.default_rng(seed)
        panels = [panels[k] for k in rng.permutation(len(panels))]

    fig = plt.figure(figsize=(FIG_WIDTH_IN, FIG_HEIGHT_IN * 1.15))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.45, 1.0], wspace=0.10,
                          left=0.05, right=0.98, top=0.90, bottom=0.10)
    gl = gs[0].subgridspec(2, n, wspace=0.05, hspace=0.12)
    key_rows = []
    for k, (kind, idx, img) in enumerate(panels):
        r, c = divmod(k, n)
        ax = fig.add_subplot(gl[r, c])
        show_image(ax, img, f"panel {k + 1}" if blind else None, pixel_um)
        if not blind and c == 0:
            ax.text(-0.04, 0.5, kind, transform=ax.transAxes, rotation=90,
                    ha="right", va="center", fontsize=LABEL_SIZE)
        key_rows.append((k + 1, kind, idx))
    fig.text(0.05 + 0.55 / 2, 0.95,
             "Which are real?  (blind)" if blind else "Real (top) vs synthetic (bottom) slices",
             ha="center", fontsize=TITLE_SIZE)

    # ---- right: two stacked histograms ----
    gr = gs[1].subgridspec(2, 1, hspace=0.55)
    ax1 = fig.add_subplot(gr[0]); ax2 = fig.add_subplot(gr[1])

    bins = np.geomspace(max(min(pores_real.min(), pores_synth.min()), 1), max(pores_real.max(), pores_synth.max()), 30)
    ax1.hist(pores_real, bins=bins, density=True, histtype="stepfilled", color=C_REAL, alpha=0.35, lw=0)
    ax1.hist(pores_real, bins=bins, density=True, histtype="step", color=C_REAL, lw=1.8, label="real")
    ax1.hist(pores_synth, bins=bins, density=True, histtype="step", color=C_SYNTH, lw=1.8, ls="--", label="synthetic")
    ax1.set_xscale("log"); ax1.set_xlabel("Pore equivalent diameter (µm)"); ax1.set_ylabel("Density")
    ax1.set_title("Pore size distribution"); ax1.legend()

    gr_real = np.concatenate([im[im > air_thr].ravel() for im in real])
    gr_synth = np.concatenate([im[im > air_thr].ravel() for im in synth])
    gbins = np.linspace(0, 255, 64)
    ax2.hist(gr_real, bins=gbins, density=True, histtype="stepfilled", color=C_REAL, alpha=0.35, lw=0)
    ax2.hist(gr_real, bins=gbins, density=True, histtype="step", color=C_REAL, lw=1.8, label="real")
    ax2.hist(gr_synth, bins=gbins, density=True, histtype="step", color=C_SYNTH, lw=1.8, ls="--", label="synthetic")
    ax2.set_xlabel("Greyscale intensity (8-bit)"); ax2.set_ylabel("Density")
    ax2.set_title("Greyscale distribution (sample voxels)"); ax2.legend()
    for ax in (ax1, ax2):
        ax.set_yticks([])          # densities: shape matters, not the number
        ax.spines["left"].set_visible(False)

    fig.savefig(f"{out}.png"); fig.savefig(f"{out}.pdf")
    print(f"wrote {out}.png / .pdf   ({n} real + {n} synthetic panels, blind={blind})")
    if blind:
        import csv
        with open(f"{out}_key.csv", "w", newline="") as f:
            w = csv.writer(f); w.writerow(["panel", "kind", "index"]); w.writerows(key_rows)
        print(f"answer key: {out}_key.csv")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--real"); ap.add_argument("--synth")
    ap.add_argument("--pores-real"); ap.add_argument("--pores-synth")
    ap.add_argument("--n-pairs", type=int, default=3, help="rows of the image grid")
    ap.add_argument("--blind", action="store_true", help="shuffle panels and hide labels")
    ap.add_argument("--seed", type=int, default=0, help="shuffle seed for --blind")
    ap.add_argument("--air-threshold", type=float, default=60, help="pixels ≤ this are background air, excluded from the grey histogram")
    ap.add_argument("--pixel-um", type=float, default=PIXEL_UM)
    ap.add_argument("--out", default="fig4_real_vs_synthetic")
    a = ap.parse_args()

    def load_folder(d):
        files = sorted(p for p in Path(d).iterdir() if p.suffix in (".png", ".tif", ".tiff", ".npy"))
        return [load_image(f) for f in files[: a.n_pairs]]

    if a.real and a.synth:
        real, synth = load_folder(a.real), load_folder(a.synth)
    else:
        print("no --real/--synth: using placeholder slices")
        real = [synthetic_slice(0.02, rng=np.random.default_rng(10 + i)) for i in range(a.n_pairs)]
        synth = [synthetic_slice(0.02, rng=np.random.default_rng(20 + i), contrast=0.9) for i in range(a.n_pairs)]
    if a.pores_real and a.pores_synth:
        import pandas as pd
        pr = pd.read_csv(a.pores_real)["pore_size_um"].to_numpy(float)
        ps = pd.read_csv(a.pores_synth)["pore_size_um"].to_numpy(float)
    else:
        print("no --pores-*: using placeholder pore sizes")
        rng = np.random.default_rng(1)
        pr, ps = placeholder_pores(3000, rng, 120), placeholder_pores(3000, rng, 130)
    make_figure(real, synth, pr, ps, a.pixel_um, a.out, a.blind, a.seed, a.air_threshold)


if __name__ == "__main__":
    main()
