#!/usr/bin/env python
"""Figure 3 — Stacking sequence and shape control.

Top row:    through-thickness slices of generated samples with different
            stacking sequences, each labelled with its sequence.
Bottom row: generated samples with different shapes / thicknesses.

Data format
-----------
--images   folder of 2-D greyscale images (.png/.tif/.npy).
--manifest CSV with columns   file, row, label
             file  : filename inside --images
             row   : "sequence" (top row) or "shape" (bottom row)
             label : text under the panel, e.g. [0/90]s  or  notch, 4 mm
           Panels appear left-to-right in manifest order.
--pixel-um voxel size in µm (default 25) for the scale bar.

If --images is omitted, placeholder synthetic data is generated.

Example
-------
python fig3_sequence_shape.py --images panels/ --manifest panels/manifest.csv
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
def synthetic_laminate(angles, size=192, rng=None, shape="box", thickness=1.0):
    """Fake through-thickness slice: one texture band per ply, dark pores,
    black background outside the sample for non-box shapes."""
    rng = rng or np.random.default_rng(0)
    img = np.full((size, size), 170.0) + rng.normal(0, 8, (size, size))
    n_ply = len(angles)
    h = int(size * thickness)
    ply_h = max(h // n_ply, 1)
    y, x = np.ogrid[:size, :size]
    for i, a in enumerate(angles):
        y0, y1 = i * ply_h, min((i + 1) * ply_h, h)
        k = 2 * np.pi / 12
        tex = 10 * np.sin(k * (x * np.cos(np.deg2rad(a)) + y * np.sin(np.deg2rad(a))))
        img[y0:y1] += tex[y0:y1]
        img[y1 - 1:y1] -= 25            # ply interface
    for _ in range(int(size * size * 0.02 / 40)):
        cy, cx = rng.integers(0, h), rng.integers(0, size)
        img[((y - cy) / 2.5) ** 2 + ((x - cx) / 9) ** 2 <= 1] = 40
    img[h:] = 20                        # air below a thinner sample
    if shape == "notch":
        img[(y < h) & (np.abs(x - size // 2) < 14) & (y < h // 3)] = 20
    if shape == "hole":
        img[((y - h // 2) ** 2 + (x - size // 2) ** 2) < 22 ** 2] = 20
    return np.clip(img, 0, 255)


PLACEHOLDER = [
    ("sequence", "[0/90]$_{2s}$", dict(angles=[0, 90, 0, 90, 90, 0, 90, 0])),
    ("sequence", "[0/45/90/-45]$_s$", dict(angles=[0, 45, 90, -45, -45, 90, 45, 0])),
    ("sequence", "[45/-45]$_{2s}$", dict(angles=[45, -45, 45, -45, -45, 45, -45, 45])),
    ("sequence", "[0]$_8$", dict(angles=[0] * 8)),
    ("shape", "flat, 8 plies", dict(angles=[0, 90] * 4)),
    ("shape", "flat, 16 plies", dict(angles=[0, 90] * 8)),
    ("shape", "thin, 4 plies", dict(angles=[0, 90, 90, 0], thickness=0.5)),
    ("shape", "notch", dict(angles=[0, 90] * 4, shape="notch")),
    ("shape", "hole", dict(angles=[0, 90] * 4, shape="hole")),
]


# --------------------------------- figure -----------------------------------
def make_figure(panels, pixel_um, out):
    """panels: list of (row, label, image)."""
    rows = {"sequence": [p for p in panels if p[0] == "sequence"],
            "shape": [p for p in panels if p[0] == "shape"]}
    ncol = max(len(rows["sequence"]), len(rows["shape"]))
    fig, axes = plt.subplots(2, ncol, figsize=(FIG_WIDTH_IN, FIG_HEIGHT_IN * 1.15),
                             gridspec_kw=dict(left=0.03, right=0.99, top=0.90, bottom=0.04,
                                              wspace=0.06, hspace=0.30))
    axes = np.atleast_2d(axes)
    for r, key in enumerate(["sequence", "shape"]):
        for c in range(ncol):
            ax = axes[r, c]
            if c < len(rows[key]):
                _, label, img = rows[key][c]
                show_image(ax, img, label, pixel_um)
            else:
                ax.set_axis_off()
    fig.text(0.5, 0.95, "Stacking-sequence control (through-thickness slices)",
             ha="center", fontsize=TITLE_SIZE)
    fig.text(0.5, 0.47, "Shape and thickness control", ha="center", fontsize=TITLE_SIZE)
    fig.savefig(f"{out}.png"); fig.savefig(f"{out}.pdf")
    print(f"wrote {out}.png / .pdf   ({len(rows['sequence'])} sequence + {len(rows['shape'])} shape panels)")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--images"); ap.add_argument("--manifest")
    ap.add_argument("--pixel-um", type=float, default=PIXEL_UM)
    ap.add_argument("--out", default="fig3_sequence_shape")
    a = ap.parse_args()
    if a.images and a.manifest:
        import pandas as pd
        df = pd.read_csv(a.manifest)
        panels = [(r.row.strip(), r.label, load_image(Path(a.images) / r.file)) for r in df.itertuples()]
    else:
        print("no --images/--manifest: using placeholder synthetic panels")
        panels = [(row, label, synthetic_laminate(rng=np.random.default_rng(i), **kw))
                  for i, (row, label, kw) in enumerate(PLACEHOLDER)]
    make_figure(panels, a.pixel_um, a.out)


if __name__ == "__main__":
    main()
