#!/usr/bin/env python
"""Whole-sample void volume fraction of the real specimens from the stored labels, margins and holes included. (CPU.)

Campaign 21's table comes from the stride-64 patch index, which drops the margins past the last full
patch (14 % of the box on average) and every patch near a drilled hole. This reads the full label
volumes instead: VVF = pore voxels / sample voxels, air excluded, nothing dropped.

Why the stored labels and not a fresh run of the repo-root onlypores.py: verified on Na_01_2, the
root program's Sauvola binary is identical to the dataset's, and its pores INSIDE the sample agree
with the stored labels to 12 voxels out of 14.4 million. But its material mask leaks: on these
aligned volumes it marks 8.8 % of the box that is air as sample, and 85 million dark air voxels
(mean grey 34) then count as pores, which reads as 8.6 % where the sample holds 1.38 %. The stored
sample mask (the reference preprocess_tools.onlypores.material_mask) does not leak, so the label volumes are the
whole-sample answer at the same segmentation.

    python scripts/analysis/real_vvf_whole_sample.py --pattern Na_ --out runs/campaigns/21-real-porosity-by-region
"""
from __future__ import annotations
import argparse, csv, re, time
from pathlib import Path

import numpy as np, zarr

REPO = Path(__file__).resolve().parents[2]


def specimen(key: str) -> str:
    m = re.search(r"((?:Na|JI|Pegaso)_\d+(?:_\d+)?)_volume", key)
    return m.group(1) if m else key


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--store", default=str(REPO / "data/split_v3/volumes.zarr"))
    ap.add_argument("--pattern", default="Na_")
    ap.add_argument("--out", default=str(REPO / "runs/campaigns/21-real-porosity-by-region"))
    a = ap.parse_args()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    c21 = {}
    tbl = out / "per_volume_table.md"
    for line in tbl.read_text().splitlines()[1:] if tbl.exists() else []:
        f = line.split("\t")
        if len(f) > 3:
            c21[f[0]] = float(f[3])
    g = zarr.open(a.store, mode="r")
    keys = sorted(k for k in g.keys() if a.pattern in k)
    rows = []
    for k in keys:
        t0 = time.time(); grp = g[k]; m, s = grp["mask"], grp["sample_mask"]
        n_pore = n_sample = 0
        for z0 in range(0, m.shape[0], 16):
            n_pore += int((m[z0:z0 + 16] == 1).sum()); n_sample += int((s[z0:z0 + 16] > 0).sum())
        name = specimen(k); vvf = 100 * n_pore / max(n_sample, 1); ref = c21.get(name, float("nan"))
        rows.append({"specimen": name, "shape": "x".join(map(str, m.shape)), "sample_voxels": n_sample, "pore_voxels": n_pore,
                     "vvf_whole_sample_percent": f"{vvf:.3f}", "vvf_campaign21_partition_percent": f"{ref:.2f}", "diff_pp": f"{vvf - ref:+.3f}"})
        print(f"{name}: vvf {vvf:.3f} %  (campaign 21: {ref:.2f})  {time.time() - t0:.0f} s", flush=True)
    with open(out / "vvf_whole_sample.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
    md = ["| specimen | VVF whole sample (%) | campaign 21, patch partition (%) | difference (pp) |", "|---|---|---|---|"]
    md += [f"| {r['specimen']} | {float(r['vvf_whole_sample_percent']):.2f} | {r['vvf_campaign21_partition_percent']} | {r['diff_pp'][:-1]} |" for r in rows]
    (out / "vvf_whole_sample.md").write_text("\n".join(md) + "\n")
    print("wrote", out / "vvf_whole_sample.csv")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
