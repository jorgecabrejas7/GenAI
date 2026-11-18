#!/usr/bin/env python
"""Score predicted air masks against a reference air mask (campaign 40).

Metrics: air Dice, air fraction, and the surface error per column along the
thickness axis: top and bottom of the longest non-air run of the prediction
against the reference, in voxels (mean
absolute, median, 95th percentile, fraction of columns within 2 voxels).
Columns where the prediction has no specimen count as error = thickness.
``--ignore-edge`` drops that many voxels at both ends of each in-plane axis
(edge columns of the overlay carry no line).

    python scripts/analysis/score_air.py --ref <air.tif> --thickness-axis 1 \
        --pred name=<air.tif> [--pred ...] --out <metrics.json>
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import tifffile


def surfaces(air: np.ndarray):
    """Top and bottom of the longest non-air run per column (as paint_material_limits.py)."""
    T = air.shape[0]
    cur = np.zeros(air.shape[1:], np.int32)
    best = np.zeros_like(cur)
    best_end = np.full(cur.shape, -1, np.int32)
    for t in range(T):
        cur = np.where(air[t], 0, cur + 1)
        better = cur > best
        best = np.where(better, cur, best)
        best_end = np.where(better, t, best_end)
    top = np.where(best > 0, best_end - best + 1, T)
    return top, best_end


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ref", type=Path, required=True)
    ap.add_argument("--thickness-axis", type=int, default=1)
    ap.add_argument("--pred", action="append", required=True, help="name=path")
    ap.add_argument("--ignore-edge", type=int, default=2)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    e = a.ignore_edge
    crop = np.s_[:, e:-e or None, e:-e or None]
    ref = np.moveaxis(tifffile.imread(a.ref) > 0, a.thickness_axis, 0)[crop]
    rt, rb = surfaces(ref)
    res = {"ref": str(a.ref), "ignore_edge": e, "ref_air_fraction": float(ref.mean()), "pred": {}}
    for spec in a.pred:
        n, p = spec.split("=", 1)
        pr = np.moveaxis(tifffile.imread(p) > 0, a.thickness_axis, 0)[crop]
        pt, pb = surfaces(pr)
        err = np.concatenate([np.abs(pt - rt).ravel(), np.abs(pb - rb).ravel()]).astype(np.float64)
        r = {"path": p, "air_fraction": float(pr.mean()),
             "air_dice": float(2 * (pr & ref).sum() / max(pr.sum() + ref.sum(), 1)),
             "surface_mae": float(err.mean()), "surface_median": float(np.median(err)),
             "surface_p95": float(np.percentile(err, 95)), "surface_within2": float((err <= 2).mean()),
             "top_mean_signed": float((pt - rt).mean()), "bottom_mean_signed": float((pb - rb).mean()),
             "top_median_signed": float(np.median(pt - rt)), "bottom_median_signed": float(np.median(pb - rb))}
        res["pred"][n] = r
        print(n, {k: round(v, 4) for k, v in r.items() if k != "path"}, flush=True)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(res, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
