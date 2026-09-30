#!/usr/bin/env python
"""Per-specimen porosity from the repo-root onlypores.py run AS IT COMES on the raw TIFFs, beside the stored labels. (CPU.)

For every raw volume matching --pattern: onlypores(xct) at its defaults -> (pore_mask, sample_mask);
VVF = pores / sample; porosity per 64^3 cell of pores / sample within the cell (cells with < 10 % sample
dropped). The same three quantities from the stored labels of the same volume are written beside them,
plus the voxel-level agreement: pores that both find, pores only onlypores finds, and where those extra
pores sit (inside or outside the stored sample mask, and their mean grey level).

    python scripts/analysis/real_porosity_onlypores_raw.py --pattern Na_
"""
from __future__ import annotations
import argparse, csv, re, sys, time
from pathlib import Path

import numpy as np, tifffile, zarr

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
from onlypores import onlypores
CELL = 64


def specimen(name):
    m = re.search(r"((?:Na|JI|Pegaso)_\d+(?:_\d+)?)_volume", name); return m.group(1) if m else name


def cell_phis(pore, sample):
    Z, Y, X = pore.shape; out = []
    for z0 in range(0, Z, CELL):
        p_ = pore[z0:z0 + CELL]; s_ = sample[z0:z0 + CELL]
        for y0 in range(0, Y, CELL):
            for x0 in range(0, X, CELL):
                n = int(s_[:, y0:y0 + CELL, x0:x0 + CELL].sum())
                if n >= 0.1 * CELL ** 3 * (p_.shape[0] / CELL):
                    out.append(int(p_[:, y0:y0 + CELL, x0:x0 + CELL].sum()) / n)
    return np.asarray(out)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--raw", default=str(REPO / "raw_data/MedidasDB")); ap.add_argument("--store", default=str(REPO / "data/split_v3/volumes.zarr"))
    ap.add_argument("--pattern", default="Na_"); ap.add_argument("--out", default=str(REPO / "runs/campaigns/21-real-porosity-by-region/porosity_onlypores_raw"))
    a = ap.parse_args(); out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    g = zarr.open(a.store, mode="r"); zkeys = {specimen(k): k for k in g.keys()}
    files = sorted(p for p in Path(a.raw).glob("*.tif") if a.pattern in p.name)
    rows = []; csv_path = out / "onlypores_vs_labels.csv"
    done = {r["specimen"] for r in csv.DictReader(open(csv_path))} if csv_path.exists() else set()
    if done: rows = list(csv.DictReader(open(csv_path)))
    for p in files:
        name = specimen(p.name)
        if name in done: continue
        t0 = time.time(); x = tifffile.imread(p)
        pore_o, samp_o, _ = onlypores(x); pore_o = pore_o.astype(bool); samp_o = samp_o.astype(bool)
        grp = g[zkeys[name]]; pore_l = grp["mask"][:] == 1; samp_l = grp["sample_mask"][:] > 0
        v_o = cell_phis(pore_o, samp_o); v_l = cell_phis(pore_l, samp_l)
        np.save(out / f"cells_onlypores_{name}.npy", v_o)
        both = int((pore_o & pore_l).sum()); only_o = pore_o & ~pore_l; only_l = int((pore_l & ~pore_o).sum())
        only_o_in = int((only_o & samp_l).sum()); only_o_out = int((only_o & ~samp_l).sum())
        r = {"specimen": name,
             "vvf_onlypores_percent": 100 * pore_o.sum() / samp_o.sum(), "vvf_labels_percent": 100 * pore_l.sum() / samp_l.sum(),
             "sample_onlypores_frac_of_box": samp_o.mean(), "sample_labels_frac_of_box": samp_l.mean(),
             "cell_median_onlypores": 100 * np.median(v_o), "cell_median_labels": 100 * np.median(v_l),
             "cell_p95_onlypores": 100 * np.percentile(v_o, 95), "cell_p95_labels": 100 * np.percentile(v_l, 95),
             "pores_both": both, "pores_only_onlypores_inside_label_sample": only_o_in,
             "pores_only_onlypores_outside_label_sample": only_o_out, "pores_only_labels": only_l,
             "grey_mean_of_onlypores_only_pores": float(x[only_o].mean()) if only_o.any() else float("nan"),
             "grey_mean_of_shared_pores": float(x[pore_o & pore_l].mean()) if both else float("nan")}
        rows.append({k: (f"{v:.4f}" if isinstance(v, float) else v) for k, v in r.items()})
        with open(csv_path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
        print(f"{name}: onlypores VVF {r['vvf_onlypores_percent']:.2f} % (sample {r['sample_onlypores_frac_of_box']:.3f} of box) | labels {r['vvf_labels_percent']:.2f} % (sample {r['sample_labels_frac_of_box']:.3f}) | shared pores {both}, extra onlypores inside/outside label sample {only_o_in}/{only_o_out} (grey {r['grey_mean_of_onlypores_only_pores']:.0f}), labels-only {only_l}  {time.time()-t0:.0f}s", flush=True)
        del x, pore_o, samp_o, pore_l, samp_l
    print("wrote", csv_path)


if __name__ == "__main__":
    main()
