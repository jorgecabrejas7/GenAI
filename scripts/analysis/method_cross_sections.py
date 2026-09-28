#!/usr/bin/env python
"""Cross-section comparison of the four generators at three porosity levels, beside a real crop. (CPU.)

Rows: real reference crop (campaign 18 real floor), ldm06 (campaign 18), the porosity-only LDM
(campaign 25), SliceGAN (campaign 22) and the pixel-space DDPM (campaign 23). Columns: phi 0.01,
0.03, 0.06. The conditional models are shown at the requested level; the unconditional baselines
are shown as the volume whose DELIVERED phi is nearest the level, with the delivered value printed
under every panel so the mismatch is visible. All volumes are centre-cropped to 128^3 (the real
crop's size) and the through-thickness slice holding the most pore voxels is shown, grey only.

Usage:
    python scripts/analysis/method_cross_sections.py --out runs/campaigns/18-eval-v4-final/figures_side_by_side/methods_cross_sections
"""
from __future__ import annotations
import argparse, json
from pathlib import Path

import numpy as np, tifffile
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patheffects

REPO = Path(__file__).resolve().parents[2]
C = REPO / "runs/campaigns"
LEVELS = [0.01, 0.03, 0.06]
UM, CROP = 25.0, 128


def load(d: Path):
    return tifffile.imread(d / "volume.tif"), tifffile.imread(d / "label.tif")


def crop_center(a, n=CROP):
    z, y, x = [(s - n) // 2 for s in a.shape[:3]]
    return a[z:z + n, y:y + n, x:x + n]


def phi_of(l):
    m, p = (l == 0).sum(), (l == 1).sum()
    return p / max(m + p, 1)


def richest_y(l):
    return int(np.argmax((l == 1).sum(axis=(0, 2))))


def nearest_unconditional(volumes_dir: Path, level: float, used: set):
    best = None
    for d in sorted(volumes_dir.iterdir()):
        if not (d / "label.tif").exists() or d.name in used:
            continue
        l = tifffile.imread(d / "label.tif")
        if l.shape[0] < CROP:
            continue
        p = phi_of(crop_center(l))
        if best is None or abs(p - level) < abs(best[1] - level):
            best = (d, p)
    used.add(best[0].name)
    return best[0]


def panel(ax, g, l, title):
    y = richest_y(l)
    ax.imshow(g[:, y, :], cmap="gray", vmin=0, vmax=255, interpolation="nearest")
    ax.set_title(title, fontsize=9, pad=3); ax.set_axis_off()
    h, w = g.shape[0], g.shape[2]; px = 1000 / UM
    ax.plot([0.04 * w, 0.04 * w + px], [0.95 * h] * 2, color="black", lw=3, solid_capstyle="butt")
    ax.plot([0.04 * w, 0.04 * w + px], [0.95 * h] * 2, color="white", lw=1.8, solid_capstyle="butt")
    ax.text(0.04 * w, 0.92 * h, "1 mm", color="white", fontsize=7, va="bottom",
            path_effects=[matplotlib.patheffects.withStroke(linewidth=1.5, foreground="black")])


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", required=True); ap.add_argument("--real-panel", default="Na_05")
    ap.add_argument("--seed", default="seed101")
    a = ap.parse_args()
    rows = [("real CT", None), ("ldm06 (ours)", C / "18-eval-v4-final/microstructure/volumes"),
            ("porosity-only LDM", C / "25-ldm-phi-only/microstructure/volumes"),
            ("SliceGAN (unconditional)", C / "22-slicegan-baseline/slicegan/volumes"),
            ("pixel DDPM (unconditional)", C / "23-ddpm3d-baseline/ddpm3d/volumes")]
    fig, axes = plt.subplots(len(rows), len(LEVELS), figsize=(3.2 * len(LEVELS), 3.3 * len(rows)))
    used_sg, used_dd = set(), set()
    for r, (name, vdir) in enumerate(rows):
        for c, lv in enumerate(LEVELS):
            if vdir is None:
                d = C / f"18-eval-v4-final/real_floor/volumes/micro_phi{lv}__{a.real_panel}__a"
                g, l = load(d)
            elif "unconditional" in name:
                d = nearest_unconditional(vdir, lv, used_sg if "SliceGAN" in name else used_dd)
                g, l = (crop_center(v) for v in load(d))
            else:
                g, l = (crop_center(v) for v in load(vdir / f"phi{lv}_{a.seed}"))
            req = f"requested {100 * lv:g} %  " if vdir is not None and "unconditional" not in name else ("target level  " if vdir is None else "nearest to  " + f"{100 * lv:g} %  ")
            panel(axes[r, c], g, l, f"{req}delivered {100 * phi_of(l):.1f} %")
            if c == 0:
                axes[r, c].text(-0.06, 0.5, name, transform=axes[r, c].transAxes, rotation=90, ha="right", va="center", fontsize=11)
    fig.suptitle("Through-thickness cross-sections, 128³ crops, 25 µm/voxel — same slice rule for every panel (most pore voxels)", fontsize=11)
    fig.tight_layout(rect=(0.02, 0, 1, 0.97))
    out = Path(a.out); out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out.with_suffix(".png"), dpi=200); fig.savefig(out.with_suffix(".pdf"))
    print("wrote", out.with_suffix(".png"))


if __name__ == "__main__":
    main()
