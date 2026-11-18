#!/usr/bin/env python
"""Split an RGB surface overlay (grey + red upper line + green lower line) into grey and an air label.

Campaign 40.  The author's overlay of 01_Teijin_CuartoPulg has the upper
material surface drawn in pure red (255,0,0) and the lower surface in pure
green (0,255,0) on the grey; every other voxel has R = G = B.

Per column along the thickness axis: top = first red voxel, bottom = last
green voxel; the specimen is [top, bottom] (line voxels count as specimen),
air is everything above top and below bottom.  A column with no red (or no
green) voxel takes the nearest column's value along the long in-plane axis.
Grey under the lines is the mean of the nearest grey voxels above and below
in the same column.

Outputs (<out>/): grey.tif (uint8), air.tif (0/255), surfaces.npz (top,
bottom, n_red, n_green per column, missing masks), report.json.  All in the
input axis order without the colour axis.

    python scripts/analysis/overlay_to_air_label.py --tif <overlay.tif> --thickness-axis 1 --out <dir>
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import tifffile


def fill_missing(h: np.ndarray, miss: np.ndarray) -> np.ndarray:
    """Nearest valid value along the last axis (columns are (A, B))."""
    out = h.copy()
    idx = np.arange(h.shape[1])
    for i in range(h.shape[0]):
        ok = ~miss[i]
        if ok.any() and miss[i].any():
            j = np.searchsorted(idx[ok], idx[miss[i]]).clip(0, ok.sum() - 1)
            jl = (j - 1).clip(0)
            cand = np.stack([idx[ok][j], idx[ok][jl]])
            best = cand[np.argmin(np.abs(cand - idx[miss[i]]), 0), np.arange(miss[i].sum())]
            out[i, miss[i]] = h[i, best]
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tif", type=Path, required=True)
    ap.add_argument("--thickness-axis", type=int, default=1)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)

    rgb = np.moveaxis(tifffile.imread(a.tif), a.thickness_axis, 0)  # (T, A, B, 3)
    r, g, b = rgb[..., 0], rgb[..., 1], rgb[..., 2]
    red = (r == 255) & (g == 0) & (b == 0)
    green = (g == 255) & (r == 0) & (b == 0)
    grey_ok = (r == g) & (g == b)
    T = rgb.shape[0]

    n_red, n_green = red.sum(0), green.sum(0)
    top = np.argmax(red, 0)
    bottom = T - 1 - np.argmax(green[::-1], 0)
    miss_t, miss_b = n_red == 0, n_green == 0
    top, bottom = fill_missing(top, miss_t), fill_missing(bottom, miss_b)

    t = np.arange(T)[:, None, None]
    air = (t < top[None]) | (t > bottom[None])

    grey = r.copy()
    del rgb, g, b
    bad = ~grey_ok
    for i in np.unique(np.nonzero(bad)[2]):  # per B index, fill along T
        sl = grey[:, :, i].astype(np.float32)
        ok = grey_ok[:, :, i]
        for j in np.flatnonzero((~ok).any(0)):
            col, okc = sl[:, j], ok[:, j]
            if okc.any():
                col[~okc] = np.interp(np.flatnonzero(~okc), np.flatnonzero(okc), col[okc])
        grey[:, :, i] = sl.round().astype(np.uint8)

    back = lambda x: np.moveaxis(x, 0, a.thickness_axis)  # noqa: E731
    tifffile.imwrite(a.out / "grey.tif", back(grey), imagej=True)
    tifffile.imwrite(a.out / "air.tif", (back(air) * 255).astype(np.uint8), imagej=True)
    np.savez_compressed(a.out / "surfaces.npz", top=top, bottom=bottom, n_red=n_red, n_green=n_green,
                        missing_top=miss_t, missing_bottom=miss_b)
    th = bottom - top + 1
    rep = {"input": str(a.tif), "thickness_axis": a.thickness_axis, "shape_TAB": list(grey.shape),
           "columns": int(top.size), "missing_top_columns": int(miss_t.sum()),
           "missing_bottom_columns": int(miss_b.sum()),
           "columns_red_gt2": int((n_red > 2).sum()), "columns_green_gt2": int((n_green > 2).sum()),
           "line_voxels_refilled": int(bad.sum()), "air_fraction": float(air.mean()),
           "thickness_min_mean_max": [int(th.min()), float(th.mean()), int(th.max())]}
    (a.out / "report.json").write_text(json.dumps(rep, indent=1))
    print(rep)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
