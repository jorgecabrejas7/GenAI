#!/usr/bin/env python
"""Side-by-side comparison of a few air masks on one volume (campaign 40).

One column per ``--mask label=path[@thickness_axis]`` (default axis =
``--thickness-axis``).  Rows: the T x B plane at ``--a`` cut into
``--windows`` pieces along B, the T x A planes at ``--b``, each with the
upper (red) / lower (green) surface of that mask (longest non-air run per
column, ``score_air.surfaces``), magnified by ``--zoom``; then the upper and
lower surface height of the first mask and, in the other columns, the
difference to the first mask (voxels, + = deeper in T).

    python scripts/analysis/fig_side_by_side.py --grey <grey.tif> --thickness-axis 1 \
        --mask "U-Net=<air.tif>" --mask "SAM2=<air.tif>@0" --out cmp.png
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import tifffile  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from fig_surface_lines import paint  # noqa: E402
from score_air import surfaces  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--grey", type=Path, required=True)
    ap.add_argument("--thickness-axis", type=int, default=0)
    ap.add_argument("--mask", action="append", required=True, help="label=path[@thickness_axis]")
    ap.add_argument("--a", type=int, default=112)
    ap.add_argument("--windows", type=int, default=3)
    ap.add_argument("--b", type=int, nargs="+", default=[200, 785])
    ap.add_argument("--zoom", type=int, default=2)
    ap.add_argument("--diff-range", type=float, default=5)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    g = np.moveaxis(tifffile.imread(a.grey), a.thickness_axis, 0)
    T, A, B = g.shape
    names, surf, air = [], [], []
    for spec in a.mask:
        name, path = spec.split("=", 1)
        ax = a.thickness_axis
        if "@" in path:
            path, ax = path.rsplit("@", 1)
            ax = int(ax)
        m = np.moveaxis(tifffile.imread(path) > 0, ax, 0)
        assert m.shape == g.shape, (name, m.shape, g.shape)
        names.append(name); surf.append(surfaces(m)); air.append(m.mean() * 100)
    edges = np.linspace(0, B, a.windows + 1).astype(int)
    rows = [("A", a.a, w0, w1) for w0, w1 in zip(edges[:-1], edges[1:])] + [("B", j, 0, A) for j in a.b]
    n = len(names)
    hr = [1] * a.windows + [1.6] * len(a.b) + [1.1, 1.1]
    fig, axs = plt.subplots(len(hr), n, figsize=(7.5 * n, sum(hr) * 2.4), gridspec_kw={"height_ratios": hr},
                            squeeze=False)
    for c, (name, (top, bot)) in enumerate(zip(names, surf)):
        for r, (kind, i, w0, w1) in enumerate(rows):
            if kind == "A":
                im = paint(g[:, i, w0:w1], top[i, w0:w1], bot[i, w0:w1], a.zoom)
                t = f"A={i}, B {w0}:{w1}"
            else:
                im = paint(g[:, :, i], top[:, i], bot[:, i], a.zoom)
                t = f"B={i} (T x A)"
            axs[r, c].imshow(im); axs[r, c].axis("off")
            axs[r, c].set_title((f"{name}  (air {air[c]:.1f} %; red = upper, green = lower)\n" if r == 0 else "") + t, fontsize=10)
        for k, (h, lab) in enumerate(((top, "upper"), (bot, "lower"))):
            axx = axs[len(rows) + k, c]
            if c == 0:
                im = axx.imshow(h, aspect="auto", cmap="viridis", interpolation="nearest")
                axx.set_title(f"{lab} surface height (T index); A down, B across", fontsize=9)
            else:
                d = h.astype(float) - surf[0][k]
                im = axx.imshow(d, aspect="auto", cmap="RdBu_r", vmin=-a.diff_range, vmax=a.diff_range,
                                interpolation="nearest")
                p95 = np.percentile(np.abs(d), 95)
                axx.set_title(f"{lab}: minus {names[0]} (voxels); median {np.median(d):+.0f}, |d| p95 {p95:.0f}",
                              fontsize=9)
            axx.set_xticks([]); axx.set_yticks([])
            fig.colorbar(im, ax=axx, fraction=0.025)
    fig.tight_layout()
    a.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(a.out, dpi=90)
    print("wrote", a.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
