#!/usr/bin/env python
"""Can split_v3's labels be recreated from the raw volumes? Check every dataset volume. (CPU, ~1 h in two shards.)

split_v3's labels came from PoreGen's old ``compute_mask`` (GenAI cce9000, ``src/poregen/dataset/io.py:90``):
``onlypores(xct)`` at the function defaults — sauvola_radius 30, sauvola_k 0.125, min_size_filtering -1, no wall
detection (frontwall = backwall = 0). That local copy is gone; the installed, commit-pinned reference
``preprocess_tools.onlypores.onlypores`` is the same function. This script recomputes the labels of each volume
with that call, in memory, and compares them voxel by voxel with ``data/split_v3/volumes.zarr/<id>/{mask,
sample_mask}``. It also records a SHA-256 of the stored arrays so a future rebuild can be checked without this run.
Nothing is written into raw_data or data/.

    python scripts/analysis/split_v3_reproducibility.py --shard 0/2 --out runs/campaigns/30-reference-onlypores/audit
"""
from __future__ import annotations

import argparse
import csv
import gc
import hashlib
import time
from pathlib import Path

import numpy as np
import zarr
from preprocess_tools import io as ref_io, onlypores as ref_onlypores

REPO = Path(__file__).resolve().parents[2]
RAW = REPO / "raw_data" / "MedidasDB"
V3 = REPO / "data" / "split_v3" / "volumes.zarr"
DEFAULTS = dict(sauvola_radius=30, sauvola_k=0.125, min_size_filtering=-1)


def sha(a: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--shard", default="0/1")
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    i, n = map(int, a.shard.split("/"))
    z = zarr.open(store=str(V3), mode="r")
    ids = sorted(z.keys())[i::n]
    a.out.mkdir(parents=True, exist_ok=True)
    csv_path = a.out / f"split_v3_reproducibility_shard{i}of{n}.csv"
    done = set()
    if csv_path.exists():
        with open(csv_path) as f:
            done = {r["volume"] for r in csv.DictReader(f)}
    fields = ["volume", "shape", "pore_voxels_stored", "pore_voxels_recomputed", "pore_diff_voxels",
              "sample_voxels_stored", "sample_voxels_recomputed", "sample_diff_voxels",
              "sha256_mask_stored", "sha256_sample_mask_stored", "seconds"]
    new = not csv_path.exists()
    with open(csv_path, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        if new:
            w.writeheader()
        for vid in ids:
            if vid in done:
                continue
            t0 = time.time()
            raw = RAW / (vid.split("__", 1)[1] + ".tif")
            xct = ref_io.load_tif(str(raw))
            pores, sample, _binary = ref_onlypores.onlypores(xct, **DEFAULTS)
            del _binary, xct
            pores = pores.astype(bool)
            sample = sample.astype(bool)
            g = z[vid]
            m_stored = g["mask"][:]
            s_stored = g["sample_mask"][:]
            row = {
                "volume": vid, "shape": "x".join(map(str, pores.shape)),
                "pore_voxels_stored": int((m_stored != 0).sum()), "pore_voxels_recomputed": int(pores.sum()),
                "pore_diff_voxels": int(((m_stored != 0) != pores).sum()),
                "sample_voxels_stored": int((s_stored != 0).sum()), "sample_voxels_recomputed": int(sample.sum()),
                "sample_diff_voxels": int(((s_stored != 0) != sample).sum()),
                "sha256_mask_stored": sha(m_stored), "sha256_sample_mask_stored": sha(s_stored),
                "seconds": f"{time.time() - t0:.0f}",
            }
            w.writerow(row)
            f.flush()
            print(f"{vid}: pore diff {row['pore_diff_voxels']}  sample diff {row['sample_diff_voxels']}  "
                  f"({row['seconds']} s)", flush=True)
            del pores, sample, m_stored, s_stored
            gc.collect()
    print("ALLDONE", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
