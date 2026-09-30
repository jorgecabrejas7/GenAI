#!/usr/bin/env python
"""Are the split_v3 labels exactly the reference `v3` outputs in raw_data?  (CPU, read-only.)

For every split_v3 volume, compares the stored ``volumes.zarr`` ``mask`` and
``sample_mask`` voxel for voxel with the reference outputs of the ``v3``
parameter set (``<stem>_onlypores_r30_k0.125_min-1_nowalls.tif`` and
``_samplemask_...``, 255 = True) that ``scripts/build_reference_onlypores.py
--config v3`` writes.  If every volume is identical, split_v3 can be rebuilt
from raw_data (``docs/REPRODUCE_SPLIT_V3.md``).

Writes <out>/verify_split_v3.csv (one row per volume) and verify_split_v3.json
(totals).  Exit code 1 if any voxel differs or any file is missing.

    python scripts/analysis/verify_split_v3_reference.py \
        --out runs/campaigns/31-dataset-v4/split_v3_reproduction
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path

import numpy as np
import zarr
from preprocess_tools.io import load_tif

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))
from poregen.dataset.io import read_reference_report, reference_outputs  # noqa: E402

V3 = REPO / "data" / "split_v3"
RAW = REPO / "raw_data" / "MedidasDB"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", required=True, type=Path)
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)

    splits = json.loads((V3 / "splits.json").read_text())
    excluded = set(splits["excluded_volume_ids"])
    vols = sorted(v for v in splits["volumes"] if v not in excluded)
    g = zarr.open_group(str(V3 / "volumes.zarr"), mode="r")

    rows, t0 = [], time.time()
    for i, vid in enumerate(vols, 1):
        ref = reference_outputs(RAW / f"{vid.split('__', 1)[1]}.tif", "v3")
        row = {"volume_id": vid}
        if not all(p.exists() for p in ref.values()):
            row["error"] = "reference outputs missing"
            rows.append(row)
            print(f"[{i:2d}/{len(vols)}] {vid}: MISSING", flush=True)
            continue
        rep = read_reference_report(ref["report"])
        pore_ref = load_tif(ref["onlypores"]) == 255
        pore_v3 = np.asarray(g[vid]["mask"]) > 0
        row.update(shape_equal=pore_ref.shape == pore_v3.shape,
                   pore_voxels_v3=int(np.count_nonzero(pore_v3)),
                   pore_voxels_ref=int(np.count_nonzero(pore_ref)),
                   pore_voxels_differing=int(np.count_nonzero(pore_ref != pore_v3)) if pore_ref.shape == pore_v3.shape else -1)
        del pore_ref, pore_v3
        samp_ref = load_tif(ref["samplemask"]) == 255
        samp_v3 = np.asarray(g[vid]["sample_mask"]) > 0
        row.update(sample_voxels_v3=int(np.count_nonzero(samp_v3)),
                   sample_voxels_ref=int(np.count_nonzero(samp_ref)),
                   sample_voxels_differing=int(np.count_nonzero(samp_ref != samp_v3)) if samp_ref.shape == samp_v3.shape else -1,
                   walls=f"{rep['frontwall']}/{rep['backwall']}",
                   reference_commit=rep["reference_commit"], error="")
        del samp_ref, samp_v3
        rows.append(row)
        print(f"[{i:2d}/{len(vols)}] {vid}: pore diff {row['pore_voxels_differing']}, "
              f"sample diff {row['sample_voxels_differing']}", flush=True)

    fields = ["volume_id", "shape_equal", "pore_voxels_v3", "pore_voxels_ref", "pore_voxels_differing",
              "sample_voxels_v3", "sample_voxels_ref", "sample_voxels_differing", "walls",
              "reference_commit", "error"]
    with open(a.out / "verify_split_v3.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)
    ok_rows = [r for r in rows if not r.get("error")]
    identical = [r for r in ok_rows if r["shape_equal"] and r["pore_voxels_differing"] == 0
                 and r["sample_voxels_differing"] == 0]
    summary = {
        "n_volumes": len(vols),
        "n_missing": len(rows) - len(ok_rows),
        "n_identical": len(identical),
        "pore_voxels_total": sum(r["pore_voxels_v3"] for r in ok_rows),
        "sample_voxels_total": sum(r["sample_voxels_v3"] for r in ok_rows),
        "pore_voxels_differing_total": sum(max(r["pore_voxels_differing"], 0) for r in ok_rows),
        "sample_voxels_differing_total": sum(max(r["sample_voxels_differing"], 0) for r in ok_rows),
        "reference_commits": sorted({r["reference_commit"] for r in ok_rows}),
        "wall_s": round(time.time() - t0, 1),
    }
    summary["all_identical"] = summary["n_identical"] == len(vols)
    (a.out / "verify_split_v3.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
    return 0 if summary["all_identical"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
