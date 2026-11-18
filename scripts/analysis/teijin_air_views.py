#!/usr/bin/env python
"""Orthogonal slices of a grey volume with one or more air masks outlined (campaign 40).

Shows where air masks agree and disagree on a scan with no reference label.
Each mask is drawn as a 1-voxel outline in its own colour on the raw grey,
plus a local-std panel (the texture cue the hand envelope thresholds).
Slices: through-thickness at several positions along the long axis and the
other in-plane axis, and one in-plane slice per requested thickness index.

    python scripts/analysis/teijin_air_views.py --tif <grey.tif> \
        --mask unet_s32000=<dir>/air.tif --mask otsu=<file> --out <dir>/views
Masks are uint8/bool TIFFs of the same shape as the grey (255 or 1 = air).
"""
from __future__ import annotations

import argparse
from pathlib import Path

import imageio.v2 as iio
import numpy as np
import tifffile
from scipy import ndimage

COLOURS = [(240, 60, 40), (40, 200, 255), (250, 210, 0), (60, 220, 90), (230, 80, 230), (255, 140, 0)]


def outline(m: np.ndarray) -> np.ndarray:
    return m & ~ndimage.binary_erosion(m)


def local_std(v: np.ndarray, w: int = 7) -> np.ndarray:
    f = v.astype(np.float32)
    m = ndimage.uniform_filter(f, w)
    return np.sqrt(np.clip(ndimage.uniform_filter(f * f, w) - m * m, 0, None))


def panel(grey: np.ndarray, masks: list[np.ndarray]) -> np.ndarray:
    rgb = np.repeat(grey[..., None], 3, -1).astype(np.uint8)
    for k, m in enumerate(masks):
        rgb[outline(m)] = COLOURS[k % len(COLOURS)]
    return rgb


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tif", type=Path, required=True)
    ap.add_argument("--thickness-axis", type=int, default=1)
    ap.add_argument("--mask", action="append", default=[], help="name=path (repeatable)")
    ap.add_argument("--fractions", type=float, nargs="+", default=[0.25, 0.5, 0.75],
                    help="positions (fraction of the axis) of the through-thickness slices")
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)

    v = np.moveaxis(tifffile.imread(a.tif), a.thickness_axis, 0)  # (T, A, B)
    names, masks = [], []
    for s in a.mask:
        n, p = s.split("=", 1)
        names.append(n)
        masks.append(np.moveaxis(tifffile.imread(p) > 0, a.thickness_axis, 0))
    std = np.clip(local_std(v) * 4, 0, 255).astype(np.uint8)
    T, A, B = v.shape
    legend = ", ".join(f"{n}={COLOURS[k % len(COLOURS)]}" for k, n in enumerate(names))
    (a.out / "legend.txt").write_text(legend + "\n" + "\n".join(
        f"{n}: air fraction {m.mean():.4f}" for n, m in zip(names, masks)) + "\n")

    for f in a.fractions:
        i, j = int(f * (A - 1)), int(f * (B - 1))
        for tag, sl in ((f"A{i}", np.s_[:, i, :]), (f"B{j}", np.s_[:, :, j])):
            g = v[sl]
            img = np.concatenate([panel(g, [m[sl] for m in masks]),
                                  np.full((4, g.shape[1], 3), 255, np.uint8),
                                  np.repeat(std[sl][..., None], 3, -1)], 0)
            iio.imwrite(a.out / f"thick_{tag}.png", img)
    for t in (T // 2, int(0.1 * T), int(0.9 * T)):
        sl = np.s_[t]
        iio.imwrite(a.out / f"plane_T{t}.png", panel(v[sl], [m[sl] for m in masks]))
    print(legend)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
