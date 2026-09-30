#!/usr/bin/env python
"""Per-volume VVF of a split_v4 root beside split_v3's, from its labels.json. (CPU, seconds.)

``scripts/build_split_v4.py --stage labels`` counts, per volume, the pore and
sample voxels of the new labels and of the split_v3 labels in the same pass.
This script only tabulates those counts: VVF = pore voxels / sample voxels over
the whole volume, the ratio, the pore voxels each label set has alone, the
walls, and the sample-mask voxels that differ.

Writes <out>/vvf_per_volume.csv and <out>/vvf_per_volume.md, and prints the
summary lines.

    python scripts/analysis/dataset_v4_vvf_table.py --root data/split_v4 \
        --out runs/campaigns/31-dataset-v4/ipynb
"""
from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path

import numpy as np


def specimen(volume_id: str) -> str:
    m = re.search(r"(Na_\d+_\d+)_volume", volume_id)
    if m:
        return m.group(1)
    m = re.search(r"Pegaso_probetas_(\d+_\d+)_volu", volume_id)
    if m:
        return f"Pegaso_{m.group(1)}"
    m = re.search(r"Juan_Ignacio_probetas_(\d+)_", volume_id)
    return f"JI_{m.group(1)}" if m else volume_id


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    a = ap.parse_args()
    labels = json.loads((a.root / "labels.json").read_text())
    splits = json.loads((a.root / "splits.json").read_text())["volumes"]
    a.out.mkdir(parents=True, exist_ok=True)

    rows = []
    for vid, r in sorted(labels["volumes"].items(), key=lambda kv: specimen(kv[0])):
        c = r["counts"]
        rows.append({
            "specimen": specimen(vid), "panel": r["panel_id"], "split": splits.get(vid, ""),
            "frontwall": r["frontwall"], "backwall": r["backwall"], "depth": r["shape_zyx"][0],
            "vvf_pct": 100 * r["vvf"], "vvf_split_v3_pct": 100 * r["vvf_split_v3"],
            "ratio_new_over_v3": r["vvf"] / r["vvf_split_v3"] if r["vvf_split_v3"] else float("nan"),
            "pore_voxels": c["pore"], "pore_voxels_split_v3": c["v3_pore"],
            "pore_new_only": c["pore_new_only"], "pore_split_v3_only": c["pore_v3_only"],
            "sample_voxels": c["sample"], "sample_voxels_differing": c["sample_diff"],
            "volume_id": vid,
        })
    with open(a.out / "vvf_per_volume.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

    L = [f"# Per-volume VVF, {a.root.name} vs split_v3", "",
         f"Settings: {labels['segmentation']} {labels['params']}. VVF = pore voxels / "
         "sample-mask voxels over the whole volume.", "",
         "| specimen | panel | split | walls | VVF % | split_v3 VVF % | new / v3 | pore voxels new only | pore voxels v3 only | sample voxels differing |",
         "|---|---|---|---|---|---|---|---|---|---|"]
    for r in rows:
        L.append(f"| {r['specimen']} | {r['panel']} | {r['split']} | {r['frontwall']}/{r['backwall']} of {r['depth']} "
                 f"| {r['vvf_pct']:.4f} | {r['vvf_split_v3_pct']:.4f} | {r['ratio_new_over_v3']:.3f} "
                 f"| {r['pore_new_only']} | {r['pore_split_v3_only']} | {r['sample_voxels_differing']} |")
    ratio = np.array([r["ratio_new_over_v3"] for r in rows])
    summary = [
        f"volumes: {len(rows)}",
        f"new/v3 VVF ratio: median {np.median(ratio):.3f}, min {ratio.min():.3f}, max {ratio.max():.3f}",
        f"volumes with any pore voxel the new labels add: {sum(r['pore_new_only'] > 0 for r in rows)}",
        f"volumes whose sample mask differs from split_v3: {sum(r['sample_voxels_differing'] > 0 for r in rows)}",
        f"pooled VVF: {100 * sum(r['pore_voxels'] for r in rows) / sum(r['sample_voxels'] for r in rows):.4f} % "
        f"(split_v3 {100 * sum(r['pore_voxels_split_v3'] for r in rows) / sum(r['sample_voxels'] for r in rows):.4f} %)",
    ]
    L += ["", *[f"- {s}" for s in summary], ""]
    (a.out / "vvf_per_volume.md").write_text("\n".join(L))
    print("\n".join(summary))


if __name__ == "__main__":
    main()
