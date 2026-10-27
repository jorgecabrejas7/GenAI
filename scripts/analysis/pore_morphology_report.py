#!/usr/bin/env python
"""Campaign 36 report: generated pore populations against real coupons, per
porosity bin, per split, per sampler. (CPU.)

Reads the per-pore parquets written by pore_morphology.py. Every comparison is
made on pores with at least ``--min-voxels`` voxels (default 1 = the
segmentation as is) and can be restricted to real coupons of given splits.

The yardstick is real against real. For each volume v of a bin (a real coupon
or a generated volume) and each property, D(v) = W1(v, pooled real coupons of
the same bin, without v) / IQR(pooled real). The real coupons' D values are the
floor: how far one real coupon sits from the others. A generated volume reads
"as real as a real coupon" when its D falls inside the real D range.

Population-level numbers per volume: porosity (pore / non-air), pores per mm^3
of material, and the median of every property.

Twins: each twin against the real crop of its own coupon (same voxels), with
the same normalised W1, next to the real coupon's own D as the floor.

Writes <campaign>/report/{tables.md, per_volume.csv, distances.csv} and one
histogram figure per bin.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy.stats import wasserstein_distance  # noqa: E402

REPO = Path(__file__).resolve().parents[2]
C36 = REPO / "runs/campaigns/36-pore-morphology"
PROPS = ("r_eq_um", "sphericity_corr", "sphericity", "cov_elongation", "cov_flatness",
         "cov_anisotropy", "cov_phi_deg", "inertia_phi_deg", "dist_surface_um")
BINS = ("0-0.5%", "0.5-1%", "1-2%", "2-5%", "5-12%")
SAMPLE = 200_000
VOXEL_MM3 = 0.025 ** 3


def md_table(t: pd.DataFrame) -> str:
    t = t.copy()
    t.columns = [" ".join(map(str, c)) if isinstance(c, tuple) else str(c) for c in t.columns]
    t = t.reset_index()
    head = "| " + " | ".join(map(str, t.columns)) + " |"
    sep = "|" + "---|" * len(t.columns)
    rows = ["| " + " | ".join(map(str, r)) + " |" for r in t.itertuples(index=False)]
    return "\n".join([head, sep, *rows])


def load(sub: str) -> pd.DataFrame:
    rows = []
    for d in sorted(C36.glob(sub)):
        f = d / "summary.jsonl"
        if f.exists():
            for line in open(f):
                r = json.loads(line)
                r["dir"] = str(d)
                rows.append(r)
    return pd.DataFrame(rows)


def pores(row, min_vox: int) -> pd.DataFrame:
    df = pd.read_parquet(Path(row["dir"]) / f"{row['name']}.parquet", columns=["n_voxels", *PROPS])
    return df[df.n_voxels >= min_vox]


def sample(a: np.ndarray, rng) -> np.ndarray:
    a = a[np.isfinite(a)]
    return a if len(a) <= SAMPLE else rng.choice(a, SAMPLE, replace=False)


def volume_stats(row, df) -> dict:
    out = {"name": row["name"], "porosity": row["porosity"], "n_pores_kept": len(df),
           "pores_per_mm3": len(df) / (row["nonair_voxels"] * VOXEL_MM3)}
    for p in PROPS:
        out[f"median_{p}"] = float(np.nanmedian(df[p])) if len(df) else np.nan
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--min-voxels", type=int, default=1)
    ap.add_argument("--real-splits", nargs="+", default=["train", "val", "test"])
    ap.add_argument("--out", type=Path, default=C36 / "report")
    a = ap.parse_args()
    rng = np.random.default_rng(0)
    out = a.out / f"min{a.min_voxels}_{'-'.join(a.real_splits)}"
    out.mkdir(parents=True, exist_ok=True)

    real = load("real_[ab]")
    real = real[real.split.isin(a.real_splits)]
    gens = {s: load(f"gen_{s}_pore_population") for s in ("production", "a2s")}

    per_volume, dist_rows, md = [], [], [f"# Campaign 36 - pores with >= {a.min_voxels} voxels, "
                                          f"real splits {a.real_splits}\n"]
    for b in BINS:
        rb = real[real.porosity_bin == b]
        if rb.empty:
            continue
        rpores = {r["name"]: pores(r, a.min_voxels) for _, r in rb.iterrows()}
        pooled = pd.concat(rpores.values())
        iqr = {p: float(np.subtract(*np.nanpercentile(pooled[p], [75, 25]))) or 1.0 for p in PROPS}
        ref = {p: sample(pooled[p].to_numpy(), rng) for p in PROPS}

        def dist(df, exclude=None):
            res = {}
            for p in PROPS:
                if exclude is not None and len(rpores) > 1:
                    rest = pd.concat([v for k, v in rpores.items() if k != exclude])[p].to_numpy()
                    r_ = sample(rest, rng)
                else:
                    r_ = ref[p]
                x = sample(df[p].to_numpy(), rng)
                res[p] = wasserstein_distance(x, r_) / iqr[p] if len(x) and len(r_) else np.nan
            return res

        for _, r in rb.iterrows():
            df = rpores[r["name"]]
            per_volume.append({"bin": b, "source": "real", "split": r["split"], **volume_stats(r, df)})
            if len(rpores) > 1:
                dist_rows.append({"bin": b, "source": "real", "name": r["name"], **dist(df, r["name"])})
        gen_hist = {}
        for s, g in gens.items():
            if g.empty:
                continue
            gb = g[g.request_bin == b]
            for _, r in gb.iterrows():
                df = pores(r, a.min_voxels)
                per_volume.append({"bin": b, "source": s, "split": "", **volume_stats(r, df)})
                dist_rows.append({"bin": b, "source": s, "name": r["name"], **dist(df)})
                gen_hist.setdefault(s, []).append(df)

        # figure: pooled histograms per property
        fig, axs = plt.subplots(3, 3, figsize=(13, 10))
        for ax, p in zip(axs.ravel(), PROPS):
            lo, hi = np.nanpercentile(pooled[p], [0.5, 99.5])
            bins_ = np.linspace(lo, hi, 50)
            ax.hist(pooled[p].clip(lo, hi), bins_, density=True, histtype="step", lw=2, color="k",
                    label=f"real ({len(rb)} coupons)")
            for s, c in (("production", "tab:blue"), ("a2s", "tab:orange")):
                if s in gen_hist:
                    ax.hist(pd.concat(gen_hist[s])[p].clip(lo, hi), bins_, density=True,
                            histtype="step", lw=1.5, color=c, label=f"{s} ({len(gen_hist[s])} vols)")
            ax.set_title(p, fontsize=10)
        axs[0, 0].legend(fontsize=8)
        fig.suptitle(f"bin {b}, pores >= {a.min_voxels} voxels")
        fig.tight_layout()
        fig.savefig(out / f"hist_{b.replace('%', 'pct')}.png", dpi=110)
        plt.close(fig)

    pv, dd = pd.DataFrame(per_volume), pd.DataFrame(dist_rows)
    pv.to_csv(out / "per_volume.csv", index=False)
    dd.to_csv(out / "distances.csv", index=False)

    md.append("## Normalised W1 to the pooled real bin (median over volumes; real = leave-one-coupon-out floor)\n")
    if not dd.empty:
        t = dd.groupby(["bin", "source"])[list(PROPS)].median().round(3)
        md.append(t.pipe(md_table) + "\n")
        md.append("## Real floor range (min-max over real coupons)\n")
        rr = dd[dd.source == "real"].groupby("bin")[list(PROPS)].agg(["min", "max"]).round(3)
        md.append(rr.pipe(md_table) + "\n")
    md.append("## Population per volume (median over volumes)\n")
    cols = ["porosity", "pores_per_mm3", "n_pores_kept", *[f"median_{p}" for p in PROPS]]
    md.append(pv.groupby(["bin", "source"])[cols].median().round(4).pipe(md_table) + "\n")
    (out / "tables.md").write_text("\n".join(md))
    print("wrote", out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
