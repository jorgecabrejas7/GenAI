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
#: Every per-pore property of pore_morphology.py except the identifiers, the
#: in-plane barycentre (a position in the crop, not a property of the pore)
#: and the two flags (reported as fractions). Eigenvector components are
#: compared as |v| because their sign is arbitrary.
PROPS = (
    "n_voxels", "volume_um3", "r_eq_um", "area_um2", "sphericity", "area_corr_um2",
    "sphericity_corr", "bary_z_um", "dist_surface_um",
    "inertia_Ixx_um2", "inertia_Iyy_um2", "inertia_Izz_um2",
    "inertia_Ixy_um2", "inertia_Ixz_um2", "inertia_Iyz_um2",
    *[f"{src}_eig{k}_um2" for src in ("inertia", "cov") for k in (1, 2, 3)],
    *[f"{src}_vec{k}_{c}" for src in ("inertia", "cov") for k in (1, 2, 3) for c in "xyz"],
    *[f"{src}_{d}" for src in ("cov", "inertia")
      for d in ("anisotropy", "elongation", "flatness", "elongation_avg", "flatness_avg",
                "phi_deg", "theta_deg")],
)
FLAGS = ("touches_air", "touches_edge")
#: The split_v4 segmentation's min_size_filtering is 8 (data/split_v4/labels.json;
#: the smallest real 26-connected pore over the 81 coupons is 9 voxels). A
#: generated pore under 8 voxels is outside the training distribution
#: (author 2026-10-27: quantify it as a model error; the threshold is 8).
MIN_REAL = 8
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
    df = pd.read_parquet(Path(row["dir"]) / f"{row['name']}.parquet", columns=[*PROPS, *FLAGS])
    for c in df.columns:
        if "_vec" in c:
            df[c] = df[c].abs()
    return df[df.n_voxels >= min_vox]


def sample(a: np.ndarray, rng, n: int = SAMPLE) -> np.ndarray:
    a = a[np.isfinite(a)].astype(np.float64)
    return a if len(a) <= n else rng.choice(a, n, replace=False)


def small_pore_error(row) -> dict:
    """Pores below the real segmentation's minimum: count, share of pores and
    share of pore volume. 0 for every real coupon by construction."""
    n = pd.read_parquet(Path(row["dir"]) / f"{row['name']}.parquet", columns=["n_voxels"]).n_voxels
    small = n < MIN_REAL
    return {"small_pores": int(small.sum()),
            "small_pore_share": float(small.mean()) if len(n) else np.nan,
            "small_pore_volume_share": float(n[small].sum() / n.sum()) if n.sum() else np.nan}


def volume_stats(row, df) -> dict:
    out = {"name": row["name"], "porosity": row["porosity"], "n_pores_kept": len(df),
           "pores_per_mm3": len(df) / (row["nonair_voxels"] * VOXEL_MM3),
           **{f"share_{f}": float(df[f].mean()) if len(df) else np.nan for f in FLAGS},
           **small_pore_error(row)}
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
        # Each coupon contributes an equal-size subsample, drawn once; the
        # pooled reference and every leave-one-coupon-out reference are
        # concatenations of these, so no coupon's pore count dominates the
        # pool and the floor costs no re-read.
        per = max(SAMPLE // len(rpores), 2000)
        csamp = {k: {p: sample(v[p].to_numpy(), rng, per) for p in PROPS} for k, v in rpores.items()}
        ref = {p: np.concatenate([c[p] for c in csamp.values()]) for p in PROPS}

        def dist(df, exclude=None):
            res = {}
            for p in PROPS:
                if exclude is not None and len(rpores) > 1:
                    r_ = np.concatenate([c[p] for k, c in csamp.items() if k != exclude])
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
        nc = 8
        nr = -(-len(PROPS) // nc)
        fig, axs = plt.subplots(nr, nc, figsize=(3 * nc, 2.4 * nr))
        for ax, p in zip(axs.ravel(), PROPS):
            lo, hi = np.nanpercentile(pooled[p], [0.5, 99.5])
            bins_ = np.linspace(lo, hi, 50)
            ax.hist(pooled[p].clip(lo, hi), bins_, density=True, histtype="step", lw=2, color="k",
                    label=f"real ({len(rb)} coupons)")
            for s, c in (("production", "tab:blue"), ("a2s", "tab:orange")):
                if s in gen_hist:
                    ax.hist(pd.concat(gen_hist[s])[p].clip(lo, hi), bins_, density=True,
                            histtype="step", lw=1.5, color=c, label=f"{s} ({len(gen_hist[s])} vols)")
            ax.set_title(p, fontsize=8)
            ax.tick_params(labelsize=6)
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
    md.append(f"## Model error: pores under {MIN_REAL} voxels (all pores counted, median over volumes)\n")
    md.append(pv.groupby(["bin", "source"])[["small_pores", "small_pore_share", "small_pore_volume_share"]]
              .median().round(4).pipe(md_table) + "\n")
    md.append("## Population per volume (median over volumes)\n")
    cols = ["porosity", "pores_per_mm3", "n_pores_kept", *[f"share_{f}" for f in FLAGS],
            *[f"median_{p}" for p in PROPS]]
    md.append(pv.groupby(["bin", "source"])[cols].median().round(4).pipe(md_table) + "\n")
    (out / "tables.md").write_text("\n".join(md))
    print("wrote", out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
