#!/usr/bin/env python
"""Campaign 41 (paper review W1): per-axis grey AND pore seam ratios, with a
real floor at the same periods. (CPU.)

Reviewer W1 / Q2 (2026-11-24): the assembly table reports grey seams only,
grey seams barely move when the pore band is present, and the combined pore
ratio (0.743) sits below both per-axis values. This reports, per volume and
per axis, the seam ratio of the grey image and of the binary pore label at the
chunk planes (period 192) and at the window planes (period 64).

Definition (same convention as ``poregen.eval_v4.metrics``): for one axis, the
plane discontinuity is the mean |v[i] - v[i-1]| over the plane at boundary i.
The seam ratio of a family of planes is their mean discontinuity divided by
the mean over INTERIOR planes, i.e. boundaries that are not multiples of 64.
Chunk planes are boundaries at multiples of 192; window planes are multiples
of 64 that are not multiples of 192. A ratio near 1 means the plane is no
different from ordinary slice-to-slice change. The pore field is the binary
label == 1, so its |diff| is the fraction of voxels that change class.

Planes are taken on the volume's own grid from index 0, which is the grid the
sampler used (twins: the air padding is cut from the high end only, see
c36_compare_folders.py, so plane positions are unchanged).

A combined ratio over the axes that HAVE chunk planes is also given (weighted
by the plane area), so it can be set beside the per-axis values.

Sets (positional ``name=glob`` pairs; each glob matches label files, the grey
volume is found beside it):
    python scripts/analysis/c41_seams.py --out runs/campaigns/41-paper-review-checks/seams \
        "assembly=runs/campaigns/18-eval-v4-final-v4-run0005-a2s/assembly_modes/volumes/*/label.tif" \
        "real_coupon=runs/campaigns/36-pore-morphology/compare/*/real_label.tif" \
        "twin_a2s=runs/campaigns/36-pore-morphology/compare/*/twin_a2s_label.tif" \
        "twin_production=runs/campaigns/36-pore-morphology/compare/*/twin_production_label.tif"
"""
from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path

import numpy as np
import pandas as pd
import tifffile

TILE, CHUNK = 64, 192
AXES = ("z", "y", "x")


def grey_path(label: Path) -> Path:
    n = label.name
    if n == "label.tif":
        return label.with_name("volume.tif")
    return label.with_name(n.replace("_label.tif", "_volume.tif"))


def plane_disc(v: np.ndarray, axis: int, binary: bool) -> np.ndarray:
    """Mean |v[i] - v[i-1]| for every boundary i = 1..n-1 along ``axis``."""
    n = v.shape[axis]
    out = np.empty(n - 1, np.float64)
    prev = np.take(v, 0, axis=axis)
    for i in range(1, n):
        cur = np.take(v, i, axis=axis)
        if binary:
            out[i - 1] = np.count_nonzero(cur != prev) / cur.size
        else:
            out[i - 1] = np.abs(cur.astype(np.int16) - prev.astype(np.int16)).mean()
        prev = cur
    return out


def ratios(disc: np.ndarray) -> dict:
    b = np.arange(1, len(disc) + 1)
    chunk = (b % CHUNK) == 0
    tile = ((b % TILE) == 0) & ~chunk
    interior = (b % TILE) != 0
    base = disc[interior].mean() if interior.any() else np.nan
    r = {"n_chunk_planes": int(chunk.sum()), "n_tile_planes": int(tile.sum()),
         "interior_mean": float(base)}
    r["chunk_ratio"] = float(disc[chunk].mean() / base) if chunk.any() and base > 0 else None
    r["tile_ratio"] = float(disc[tile].mean() / base) if tile.any() and base > 0 else None
    r["chunk_mean"] = float(disc[chunk].mean()) if chunk.any() else None
    return r


def measure(label_f: Path) -> list[dict]:
    lab = tifffile.imread(label_f)
    grey = tifffile.imread(grey_path(label_f))
    pore = (lab == 1)
    rows = []
    for field, arr, binary in (("pore", pore, True), ("grey", grey, False)):
        comb_num = comb_den = 0.0
        for a, name in enumerate(AXES):
            r = ratios(plane_disc(arr, a, binary))
            area = arr.size / arr.shape[a]
            if r["chunk_mean"] is not None and r["interior_mean"] > 0:
                comb_num += r["chunk_mean"] * r["n_chunk_planes"] * area
                comb_den += r["interior_mean"] * r["n_chunk_planes"] * area
            rows.append({"field": field, "ax": name, **r})
        rows.append({"field": field, "ax": "combined",
                     "chunk_ratio": comb_num / comb_den if comb_den else None})
    phi = float(pore.mean())
    for r in rows:
        r.update(volume=str(label_f), shape=list(lab.shape), pore_fraction=phi)
    return rows


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("sets", nargs="+", help="name=glob of label files")
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    rows = []
    for spec in a.sets:
        name, pattern = spec.split("=", 1)
        files = sorted(glob.glob(pattern))
        print(f"{name}: {len(files)} volumes", flush=True)
        for f in files:
            for r in measure(Path(f)):
                rows.append({"set": name, **r})
            print("  done", f, flush=True)
            pd.DataFrame(rows).to_csv(a.out / "seams_per_volume.csv", index=False)
    d = pd.DataFrame(rows)
    d["case"] = d.volume.map(lambda p: Path(p).parent.name)
    d["arm"] = np.where(d.set == "assembly", d.case.str.replace(r"_seed\d+$", "", regex=True), d.set)
    summ = (d.groupby(["arm", "field", "ax"])[["chunk_ratio", "tile_ratio"]]
             .agg(["mean", "std", "count"]).round(4))
    summ.to_csv(a.out / "seams_summary.csv")
    (a.out / "seams_summary.txt").write_text(
        "Campaign 41 - grey and pore seam ratios per axis (chunk period 192, window period 64)\n\n"
        + summ.to_string() + "\n")
    print(summ.to_string())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
