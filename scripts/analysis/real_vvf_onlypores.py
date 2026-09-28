#!/usr/bin/env python
"""Whole-sample void volume fraction of the real Na specimens from the RAW volumes with onlypores. (CPU.)

Campaign 21's per-volume table comes from the stride-64 patch index: it drops the margins past the
last full 64-voxel patch (14 % of the box on average) and every patch near a drilled hole, so it is
not the whole sample. This script segments each raw TIFF in raw_data/MedidasDB with the repo's
onlypores.py at its defaults and counts pores over the filled sample mask, margins and holes
included. Output: <out>/vvf_onlypores_whole_sample.{csv,md} with campaign 21's number beside it.

    python scripts/analysis/real_vvf_onlypores.py --pattern "Na_" --out runs/campaigns/21-real-porosity-by-region
"""
from __future__ import annotations
import argparse, csv, re, sys, time
from pathlib import Path

import numpy as np, tifffile

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
from onlypores import onlypores  # the repo-root program, at its defaults


def specimen_name(path: Path) -> str:
    m = re.search(r"(Na_\d+_\d+|JI_\d+|Pegaso_\d+_\d+|\w+_\d+_\d+)_volume", path.name)
    return m.group(1) if m else path.stem


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--raw", default=str(REPO / "raw_data/MedidasDB"))
    ap.add_argument("--pattern", default="Na_")
    ap.add_argument("--out", default=str(REPO / "runs/campaigns/21-real-porosity-by-region"))
    a = ap.parse_args()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    c21 = {}
    tbl = out / "per_volume_table.md"
    if tbl.exists():
        for line in tbl.read_text().splitlines()[1:]:
            f = line.split("\t")
            if len(f) > 3:
                c21[f[0]] = float(f[3])
    files = sorted(p for p in Path(a.raw).glob("*.tif") if a.pattern in p.name)
    rows, csv_path = [], out / "vvf_onlypores_whole_sample.csv"
    done = set()
    if csv_path.exists():                      # resumable: keep finished rows
        for r in csv.DictReader(open(csv_path)):
            rows.append(r); done.add(r["specimen"])
    for p in files:
        name = specimen_name(p)
        if name in done:
            continue
        t0 = time.time()
        xct = tifffile.imread(p)
        pores, sample, _ = onlypores(xct)
        if pores is None:
            print(f"{name}: onlypores returned None", flush=True); continue
        n_pore, n_sample = int(pores.sum()), int(sample.sum())
        vvf = 100 * n_pore / max(n_sample, 1)
        rows.append({"specimen": name, "shape": "x".join(map(str, xct.shape)), "sample_voxels": n_sample,
                     "pore_voxels": n_pore, "vvf_onlypores_percent": f"{vvf:.3f}",
                     "vvf_campaign21_percent": f"{c21.get(name, float('nan')):.2f}",
                     "diff_pp": f"{vvf - c21.get(name, float('nan')):.3f}"})
        with open(csv_path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
        print(f"{name}: vvf {vvf:.3f} %  (campaign 21: {c21.get(name, float('nan')):.2f})  {time.time() - t0:.0f} s", flush=True)
        del xct, pores, sample
    md = ["| specimen | VVF onlypores, whole sample (%) | campaign 21, patch partition (%) | difference (pp) |", "|---|---|---|---|"]
    md += [f"| {r['specimen']} | {float(r['vvf_onlypores_percent']):.2f} | {r['vvf_campaign21_percent']} | {float(r['diff_pp']):+.2f} |" for r in rows]
    (out / "vvf_onlypores_whole_sample.md").write_text("\n".join(md) + "\n")
    print("wrote", csv_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
