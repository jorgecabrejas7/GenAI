#!/usr/bin/env python
"""Campaign 41 (paper review W5): the twins table split into training and
held-out coupons. (CPU, reads campaign 36's reports only.)

Reviewer W5 (2026-11-24): "11 of 25 twins are training coupons, and the >5 %
bin is training-only, so Table X partly measures in-distribution
reproduction. Separate training-coupon and held-out-coupon twins."

Inputs (campaign 36, written by pore_morphology_report.py):
    report_twins/min8_train-val-test/{twins.csv, per_volume.csv}   pores >= 8 voxels
    report_twins/min1_train-val-test/per_volume.csv                 every pore (sub-8 share)
A coupon's split is its split in data/split_v4/patch_index.parquet; val and
test are "held-out".

Per twin: porosity relative error, pore-density ratio (twin / its real
coupon), sub-8-voxel pore share, median-sphericity difference, and the MEDIAN
normalised W1 over every pore property against its own coupon. Then the mean
per sampler x bin x group.

    python scripts/analysis/c41_twins_split.py --out runs/campaigns/41-paper-review-checks/twins_split
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parents[2]
R = REPO / "runs/campaigns/36-pore-morphology/report_twins"
ID_COLS = {"bin", "sampler", "twin", "real_porosity", "twin_porosity"}


def coupon(name: str) -> str:
    """Short coupon id from a twin name or a volume id. Volume ids spell the
    suffix three ways: _volume_eq_aligned, _volume_eq_cleaned_aligned and (four
    Pegaso coupons) _volumen_eq_aligned."""
    s = str(name).removeprefix("twin_")
    m = re.search(r"probetas_(.+?)_volumen?_", s)
    return m.group(1) if m else s


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    idx = pd.read_parquet(REPO / "data/split_v4/patch_index.parquet", columns=["volume_id", "split"]).drop_duplicates()
    split = {coupon(v): s for v, s in zip(idx.volume_id, idx.split)}

    tw = pd.read_csv(R / "min8_train-val-test/twins.csv")
    props = [c for c in tw.columns if c not in ID_COLS]
    tw["coupon"] = tw.twin.map(coupon)
    tw["median_norm_w1"] = tw[props].median(axis=1)  # median: a few properties have a near-zero IQR
    tw["porosity_rel_err"] = (tw.twin_porosity - tw.real_porosity) / tw.real_porosity

    pv8 = pd.read_csv(R / "min8_train-val-test/per_volume.csv")
    pv1 = pd.read_csv(R / "min1_train-val-test/per_volume.csv")
    for pv in (pv8, pv1):
        pv["coupon"] = pv.name.map(coupon)
    real8 = pv8[pv8.source == "real"].set_index("coupon")
    rows = []
    for _, t in tw.iterrows():
        g8 = pv8[(pv8.source == t.sampler) & (pv8.coupon == t.coupon)]
        g1 = pv1[(pv1.source == t.sampler) & (pv1.coupon == t.coupon)]
        r1 = pv1[(pv1.source == "real") & (pv1.coupon == t.coupon)]
        if g8.empty or t.coupon not in real8.index:
            continue
        rr = real8.loc[t.coupon]
        rows.append({"sampler": t.sampler, "bin": t.bin, "coupon": t.coupon, "split": split.get(t.coupon, "?"),
                     "porosity_rel_err": t.porosity_rel_err,
                     "density_ratio": float(g8.pores_per_mm3.iloc[0] / rr.pores_per_mm3),
                     "sub8_share_twin": float(g1.small_pore_share.iloc[0]) if not g1.empty else None,
                     "sub8_share_real": float(r1.small_pore_share.iloc[0]) if not r1.empty else None,
                     "sphericity_corr_diff": float(g8.median_sphericity_corr.iloc[0] - rr.median_sphericity_corr),
                     "median_norm_w1": t.median_norm_w1})
    d = pd.DataFrame(rows)
    d["group"] = d.split.map({"train": "train", "val": "held-out", "test": "held-out"}).fillna("unknown")
    d.to_csv(a.out / "twins_split_per_twin.csv", index=False)
    cols = ["porosity_rel_err", "density_ratio", "sub8_share_twin", "sub8_share_real", "sphericity_corr_diff", "median_norm_w1"]
    by_group = d.groupby(["sampler", "group"])[cols].agg(["mean", "std"]).round(3)
    by_group.insert(0, "n_twins", d.groupby(["sampler", "group"]).size())
    by_bin = d.groupby(["sampler", "bin", "group"])[cols].mean().round(3)
    by_bin.insert(0, "n_twins", d.groupby(["sampler", "bin", "group"]).size())
    txt = ("Campaign 41 - campaign-36 twins split by coupon split (held-out = val + test)\n"
           "pores >= 8 voxels except the sub-8 shares (every pore)\n\n"
           + by_group.to_string() + "\n\nper bin\n" + by_bin.to_string() + "\n")
    (a.out / "twins_split.txt").write_text(txt)
    print(txt)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
