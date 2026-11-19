#!/usr/bin/env python
"""One figure per air mask: its upper (red) and lower (green) surface on grey slices (campaign 40).

Surfaces = first and last voxel of the longest non-air run in each thickness
column (``score_air.surfaces``, as ``paint_material_limits.py``).  Panels:
the T x B planes at the requested A indices (full width, or windows of
``--window`` along B), then the T x A planes at the requested B indices side
by side, all magnified by ``--zoom``; below, the upper and lower surface
height maps over A x B.  One PNG per ``--mask`` (``label=path``).

    python scripts/analysis/fig_surface_lines.py --grey <grey.tif> --thickness-axis 1 \
        --mask "U-Net air ft=<air.tif>" --a 40 112 185 --b 200 785 1400 --out <dir>
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import tifffile  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from score_air import surfaces  # noqa: E402


def paint(g2: np.ndarray, top: np.ndarray, bot: np.ndarray, zoom: int) -> np.ndarray:
    """g2 (T, W) grey, top/bot (W,) -> zoomed RGB with 1-voxel lines."""
    rgb = np.repeat(g2[..., None], 3, -1).copy()
    w = np.arange(g2.shape[1])
    rgb[top, w] = (255, 0, 0)
    rgb[bot, w] = (0, 255, 0)
    return np.kron(rgb, np.ones((zoom, zoom, 1), np.uint8))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--grey", type=Path, required=True)
    ap.add_argument("--thickness-axis", type=int, default=0)
    ap.add_argument("--mask", action="append", required=True, help="label=path")
    ap.add_argument("--a", type=int, nargs="+", default=[40, 112, 185])
    ap.add_argument("--b", type=int, nargs="+", default=[200, 785, 1400])
    ap.add_argument("--window", type=int, default=100000, help="split A planes along B (default: full width)")
    ap.add_argument("--zoom", type=int, default=2)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    g = np.moveaxis(tifffile.imread(a.grey), a.thickness_axis, 0)
    T, A, B = g.shape
    for spec in a.mask:
        name, path = spec.split("=", 1)
        m = np.moveaxis(tifffile.imread(path) > 0, a.thickness_axis, 0)
        top, bot = surfaces(m)  # (A, B)
        panels = []
        for i in a.a:
            for w0 in range(0, B, a.window):
                w1 = min(B, w0 + a.window)
                panels.append((f"A={i}, B {w0}:{w1}", paint(g[:, i, w0:w1], top[i, w0:w1], bot[i, w0:w1], a.zoom)))
        side = [paint(g[:, :, j], top[:, j], bot[:, j], a.zoom) for j in a.b]
        n = len(panels) + 2
        fig = plt.figure(figsize=(20, 2.9 * len(panels) + 6.5))
        gs_h = [1] * len(panels) + [2.2, 1.6]
        gs = fig.add_gridspec(n, len(a.b), height_ratios=gs_h)
        for k, (t, im) in enumerate(panels):
            ax = fig.add_subplot(gs[k, :]); ax.imshow(im); ax.set_title(t, fontsize=9); ax.axis("off")
        for k, (j, im) in enumerate(zip(a.b, side)):
            ax = fig.add_subplot(gs[len(panels), k]); ax.imshow(im); ax.set_title(f"B={j} (T x A)", fontsize=9)
            ax.axis("off")
        sub = gs[len(panels) + 1, :].subgridspec(1, 2)
        for k, (h, lab) in enumerate(((top, "upper"), (bot, "lower"))):
            ax = fig.add_subplot(sub[0, k])
            im = ax.imshow(h, aspect="auto", cmap="viridis", interpolation="nearest")
            ax.set_title(f"{lab} surface height (T index); A down, B across", fontsize=9); ax.set_xticks([]); ax.set_yticks([])
            fig.colorbar(im, ax=ax, fraction=0.02)
        fig.suptitle(f"{name}   (red = upper surface, green = lower surface; air {m.mean() * 100:.1f} %)", fontsize=12)
        fig.tight_layout()
        fn = a.out / (re.sub(r"[^A-Za-z0-9+]+", "_", name).strip("_") + ".png")
        fig.savefig(fn, dpi=100)
        plt.close(fig)
        print("wrote", fn)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
