#!/usr/bin/env python
"""Local-porosity histograms and statistics per real specimen, from the full stored labels. (CPU.)

For every volume matching --pattern: phi = pore / (pore + material) in every 64^3 cell of the full
label volume (partial edge cells included, cells with < 10 % material dropped), the whole-sample
VVF, and per-specimen statistics of the cell distribution. Writes <out>/porosity_histograms/
  cell_stats.csv          one row per specimen: whole-sample VVF, n cells, mean/median/sd/p5/p95/max of cell phi
  panel_stats.csv         one row per panel (Na_01 ... Na_10), specimens pooled
  cells_<specimen>.npy    the cell phi values, for re-plotting
  histograms_by_specimen.png / .pdf   10 x 5 grid, one histogram per specimen, whole-sample VVF marked
  histograms_by_panel.png / .pdf      one histogram per panel, five specimens pooled

    python scripts/analysis/real_porosity_histograms.py --pattern Na_
"""
from __future__ import annotations
import argparse, csv, re
from pathlib import Path

import numpy as np, zarr
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from poregen.paths import data_root

REPO = Path(__file__).resolve().parents[2]
CELL = 64


def specimen(key):
    m = re.search(r"((?:Na|JI|Pegaso)_\d+(?:_\d+)?)_volume", key); return m.group(1) if m else key


def cell_phis(mask, sample):
    """phi per 64^3 cell over the whole volume; cells with < 10 % material voxels are dropped."""
    Z, Y, X = mask.shape; out = []
    for z0 in range(0, Z, CELL):
        m = mask[z0:z0 + CELL]; s = sample[z0:z0 + CELL] > 0
        pore = (m == 1); mat = s & ~pore
        for y0 in range(0, Y, CELL):
            for x0 in range(0, X, CELL):
                p = pore[:, y0:y0 + CELL, x0:x0 + CELL].sum(); q = mat[:, y0:y0 + CELL, x0:x0 + CELL].sum()
                n = p + q
                if n >= 0.1 * CELL ** 3 * (m.shape[0] / CELL):
                    out.append(p / n)
    return np.asarray(out, dtype=np.float64)


def stats(v):
    return dict(n_cells=len(v), mean=v.mean(), median=np.median(v), sd=v.std(ddof=1) if len(v) > 1 else 0.0,
                p5=np.percentile(v, 5), p95=np.percentile(v, 95), max=v.max())


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--store", default=str(data_root() / "volumes.zarr")); ap.add_argument("--pattern", default="Na_")
    ap.add_argument("--out", default=str(REPO / "runs/campaigns/21-real-porosity-by-region/porosity_histograms"))
    ap.add_argument("--replot", action="store_true", help="re-draw from the saved cells_*.npy and cell_stats.csv without touching the store")
    a = ap.parse_args(); out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    rows, cells = [], {}
    if a.replot:
        for r in csv.DictReader(open(out / "cell_stats.csv")):
            rows.append({k: (float(v) if k not in ("specimen", "panel") else v) for k, v in r.items()})
            cells[r["specimen"]] = np.load(out / f"cells_{r['specimen']}.npy")
        keys = []
    else:
        g = zarr.open(a.store, mode="r"); keys = sorted(k for k in g.keys() if a.pattern in k)
    for k in keys:
        grp = g[k]; mask = grp["mask"][:]; sample = grp["sample_mask"][:]
        name = specimen(k); v = cell_phis(mask, sample); cells[name] = v; np.save(out / f"cells_{name}.npy", v)
        pore = int((mask == 1).sum()); samp = int((sample > 0).sum())
        rows.append({"specimen": name, "panel": name.rsplit("_", 1)[0], "vvf_whole_percent": 100 * pore / samp,
                     **{kk: (vv if kk == "n_cells" else 100 * vv) for kk, vv in stats(v).items()}})
        print(f"{name}: VVF {rows[-1]['vvf_whole_percent']:.2f} %  cells {len(v)}  cell mean {100*v.mean():.2f}  p95 {100*np.percentile(v,95):.2f}  max {100*v.max():.2f}", flush=True)
        del mask, sample
    if not a.replot:
        with open(out / "cell_stats.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader()
            for r in rows: w.writerow({k: (f"{v:.3f}" if isinstance(v, float) else v) for k, v in r.items()})
    panels = sorted({r["panel"] for r in rows}); prow = []
    for p in panels:
        v = np.concatenate([cells[r["specimen"]] for r in rows if r["panel"] == p])
        tot_p = sum(1 for r in rows if r["panel"] == p)
        prow.append({"panel": p, "n_specimens": tot_p, "vvf_mean_of_specimens_percent": np.mean([r["vvf_whole_percent"] for r in rows if r["panel"] == p]),
                     **{kk: (vv if kk == "n_cells" else 100 * vv) for kk, vv in stats(v).items()}})
    with open(out / "panel_stats.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(prow[0])); w.writeheader()
        for r in prow: w.writerow({k: (f"{v:.3f}" if isinstance(v, float) else v) for k, v in r.items()})

    # figures: shared log-spaced bins so panels are comparable; x in percent
    bins = np.concatenate([[0], np.geomspace(0.02, 90, 44)])   # first bin = cells with (almost) no pores
    fig, axes = plt.subplots(len(panels), 5, figsize=(16, 2.2 * len(panels)), sharex=True)
    for i, p in enumerate(panels):
        specs = [r for r in rows if r["panel"] == p]
        for j in range(5):
            ax = axes[i, j]
            if j < len(specs):
                r = specs[j]; v = 100 * cells[r["specimen"]]
                ax.hist(v, bins=bins, color="#0072B2", alpha=0.8)
                ax.axvline(r["vvf_whole_percent"], color="#b7791f", lw=1.2)
                ax.set_title(f"{r['specimen']}   VVF {r['vvf_whole_percent']:.2f} %", fontsize=8)
                ax.set_xscale("symlog", linthresh=0.02); ax.set_yticks([])
                for s in ("top", "right"): ax.spines[s].set_visible(False)
            else:
                ax.set_axis_off()
    for ax in axes[-1]: ax.set_xlabel("porosity of a 64³ cell (%)", fontsize=8)
    fig.suptitle("Local porosity per 64³ cell, all Na specimens (orange line = whole-sample VVF)", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.98)); fig.savefig(out / "histograms_by_specimen.png", dpi=150); fig.savefig(out / "histograms_by_specimen.pdf"); plt.close(fig)

    fig, axes = plt.subplots(2, 5, figsize=(16, 6), sharex=True)
    for ax, p, r in zip(axes.ravel(), panels, prow):
        v = 100 * np.concatenate([cells[s["specimen"]] for s in rows if s["panel"] == p])
        ax.hist(v, bins=bins, color="#0072B2", alpha=0.8); ax.set_xscale("symlog", linthresh=0.02); ax.set_yticks([])
        ax.axvline(r["vvf_mean_of_specimens_percent"], color="#b7791f", lw=1.2)
        ax.set_title(f"{p}   VVF {r['vvf_mean_of_specimens_percent']:.2f} %\ncell median {r['median']:.2f} %, p95 {r['p95']:.2f} %", fontsize=8)
        for s in ("top", "right"): ax.spines[s].set_visible(False)
    for ax in axes[-1]: ax.set_xlabel("porosity of a 64³ cell (%)", fontsize=8)
    fig.suptitle("Local porosity per 64³ cell, pooled by panel", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.96)); fig.savefig(out / "histograms_by_panel.png", dpi=150); fig.savefig(out / "histograms_by_panel.pdf")
    print("wrote", out)


if __name__ == "__main__":
    main()
