#!/usr/bin/env python
"""Per-pore morphology tables for real coupons and generated volumes. (CPU.)

Campaign 36. One parquet per volume (one row per pore, every column of
``poregen.metrics.pore_morphology``) plus one JSON line per volume in
``summary.jsonl`` (porosity, pore count, split, bin, provenance). The
segmentation is used as is; filter on ``n_voxels`` when reading.

Sources
  --real [VOLUME_ID ...]   real coupons from data/<split>/volumes.zarr, label =
                           build_label(mask, sample_mask) (0 material, 1 pore,
                           2 air); no ids = all of them. ``--crop-from`` takes
                           the crop of each coupon from a generated campaign's
                           manifests (the twins), so real and twin cover the
                           same voxels.
  --generated DIR          every ``volumes/<case>/label.tif`` under an eval_v4
                           assessment directory.

Porosity bin (whole volume, pore / non-air): 0-0.5, 0.5-1, 1-2, 2-5, 5-12 %
(author 2026-10-26).
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import tifffile
import zarr

from poregen.dataset.loader import build_label
from poregen.metrics.pore_morphology import measure

REPO = Path(__file__).resolve().parents[2]
VOXEL_UM = 25.0
BINS = ((0.0, 0.005), (0.005, 0.01), (0.01, 0.02), (0.02, 0.05), (0.05, 0.12))
OUT = REPO / "runs/campaigns/36-pore-morphology"


def bin_of(phi: float) -> str:
    for lo, hi in BINS:
        if lo <= phi < hi:
            return f"{lo * 100:g}-{hi * 100:g}%"
    return "out"


def wait_for_memory(gb: float = 45.0) -> None:
    """Block until the host has ``gb`` GB available. Measurement runs beside a
    whole-coupon generation; together they took the host to 1 GB free on
    2026-10-28, and an OOM kill would take the generation with it."""
    while True:
        with open("/proc/meminfo") as fh:
            avail = next(int(l.split()[1]) for l in fh if l.startswith("MemAvailable")) / 2 ** 20
        if avail >= gb:
            return
        time.sleep(60)


def write(dest: Path, name: str, df, summary: dict) -> None:
    dest.mkdir(parents=True, exist_ok=True)
    df.to_parquet(dest / f"{name}.parquet", index=False)
    summary["porosity_bin"] = bin_of(summary["porosity"])
    with open(dest / "summary.jsonl", "a") as fh:
        fh.write(json.dumps({"name": name, **summary}) + "\n")
    print(name, f"pores {summary['n_pores']} phi {summary['porosity']:.4f}",
          f"{summary['seconds']:.0f}s", flush=True)


def done(dest: Path) -> set[str]:
    f = dest / "summary.jsonl"
    return {json.loads(l)["name"] for l in open(f)} if f.exists() else set()


def real(ids, split: str, out: Path, crop_from: Path | None) -> None:
    root = REPO / "data" / split
    z = zarr.open(str(root / "volumes.zarr"), mode="r")
    splits = json.load(open(root / "splits.json"))
    crops = {}
    if crop_from is not None:
        for m in crop_from.glob("volumes/*/manifest.json"):
            notes = json.load(open(m)).get("notes", {})
            if "real_volume_id" in notes:
                crops[notes["real_volume_id"]] = (m.parent.name, notes["real_crop_zyx"])
        ids = ids or sorted(crops)
    ids = ids or sorted(z.keys())
    have = done(out)
    for v in ids:
        name = crops[v][0] if v in crops else v
        if name in have:
            continue
        t0 = time.time()
        g = z[v]
        if v in crops:
            (z0, z1), (y0, y1), (x0, x1) = crops[v][1]
            sl = np.s_[z0:z1, y0:y1, x0:x1]
        else:
            sl = np.s_[:, :, :]
        label = build_label(np.asarray(g["mask"][sl]), np.asarray(g["sample_mask"][sl]))
        df, s = measure(label, VOXEL_UM)
        del label
        s.update(source="real", volume_id=v, split=splits["volumes"].get(v),
                 panel=splits["panel_id"].get(v), crop=None if v not in crops else crops[v][1],
                 seconds=time.time() - t0)
        write(out, name, df, s)


def generated(root: Path, out: Path) -> None:
    have = done(out)
    for d in sorted(root.glob("volumes/*/")):
        if d.name in have or not (d / "label.tif").exists():
            continue
        wait_for_memory()
        t0 = time.time()
        m = json.load(open(d / "manifest.json"))
        notes = m.get("notes", {})
        label = tifffile.imread(d / "label.tif")
        if "real_crop_zyx" in notes:  # twins: cut the air padding off
            label = label[tuple(slice(0, hi - lo) for lo, hi in notes["real_crop_zyx"])]
        df, s = measure(label, VOXEL_UM)
        s.update(source="generated", case=d.name, seed=m.get("seed"),
                 requested_phi=m.get("requested_global_phi"),
                 chunk_overlap=m.get("chunk_overlap", notes.get("chunk_overlap")),
                 real_volume_id=notes.get("real_volume_id"), request_bin=notes.get("bin"),
                 geometry_from=notes.get("geometry_from"), seconds=time.time() - t0)
        write(out, d.name, df, s)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--real", nargs="*")
    g.add_argument("--generated", type=Path)
    ap.add_argument("--crop-from", type=Path, help="generated assessment dir whose twins give the crops")
    ap.add_argument("--split", default="split_v4")
    ap.add_argument("--out", type=Path, required=True, help="sub-directory of campaign 36")
    a = ap.parse_args()
    out = a.out if a.out.is_absolute() else OUT / a.out
    if a.generated is not None:
        generated(a.generated if a.generated.is_absolute() else REPO / a.generated, out)
    else:
        cf = None if a.crop_from is None else (a.crop_from if a.crop_from.is_absolute() else REPO / a.crop_from)
        real(a.real, a.split, out, cf)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
