"""T-E — p(local patch porosity | global volume porosity).

Answers three questions for the ldm05 conditioning design:

* How wide is the spread of patch porosity inside one specimen?
* How far off-distribution is "uniform painting" at inference, i.e. giving every
  patch a local porosity equal to the volume target?
* What should inference draw instead?  The script emits a reusable sampler
  artifact: a per-bin quantile table of the ratio ``local / global``, which an
  inference script can interpolate for any global target.

The global porosity of a volume is the exact foreground-normalised value taken
from the T-A voxel pass when available, otherwise the mean over the volume's
patches.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import (  # noqa: E402
    OUT_ROOT, PATCH_INDEX, savefig, set_style, write_json, plt,
)

TEST_ID = "T-E"
OUT_DIR = OUT_ROOT / TEST_ID
QUANTILES = np.round(np.arange(0.0, 1.001, 0.01), 3)
BIN_EDGES = np.array([0.0, 0.0025, 0.005, 0.0075, 0.01, 0.015, 0.02, 0.03,
                      0.05, 0.08, 1.0])


def global_porosity_table(df: pd.DataFrame) -> tuple[pd.Series, str]:
    cache = OUT_ROOT / "T-A" / "fine_profiles.npz"
    if cache.exists():
        profiles = np.load(cache, allow_pickle=True)["profiles"].item()
        vals = {}
        for vid, p in profiles.items():
            fg = np.asarray(p["z"]["fg"], float)
            phi = np.asarray(p["z"]["phi"], float)
            ok = np.isfinite(phi) & (fg > 0)
            if ok.any():
                vals[vid] = float(np.sum(phi[ok] * fg[ok]) / np.sum(fg[ok]))
        s = pd.Series(vals)
        missing = set(df["volume_id"].unique()) - set(s.index)
        if not missing:
            return s, "exact foreground-normalised volume porosity (voxel pass, T-A)"
    return (df.groupby("volume_id", observed=True)["porosity"].mean(),
            "mean over the volume's patches (voxel pass unavailable)")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="all", choices=["all", "train"])
    args = ap.parse_args()
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    df = pd.read_parquet(PATCH_INDEX)
    df = df[df["porosity"] <= 1.0]
    if args.split == "train":
        df = df[df["split"] == "train"]
    gphi, gsource = global_porosity_table(df)
    df = df.assign(global_phi=df["volume_id"].map(gphi).astype(float))
    df = df[np.isfinite(df["global_phi"])]
    local = df["porosity"].to_numpy(dtype=np.float64)
    glob = df["global_phi"].to_numpy(dtype=np.float64)
    print(f"[T-E] {len(df):,} patches, {df['volume_id'].nunique()} volumes; "
          f"global phi from: {gsource}", flush=True)

    results = {
        "test_id": TEST_ID,
        "n_patches": int(len(df)),
        "n_volumes": int(df["volume_id"].nunique()),
        "global_phi_source": gsource,
        "global_phi_range": [float(gphi.min()), float(gphi.max())],
        "quantile_levels": QUANTILES.tolist(),
        "bin_edges": BIN_EDGES.tolist(),
        "bins": [],
        "overall": {},
    }

    # ---------- overall marginals ----------
    results["overall"] = {
        "local_phi_mean": float(local.mean()),
        "local_phi_median": float(np.median(local)),
        "local_phi_p90": float(np.percentile(local, 90)),
        "local_phi_p99": float(np.percentile(local, 99)),
        "local_phi_max": float(local.max()),
        "frac_local_zero": float(np.mean(local <= 0)),
        "frac_abs_diff_lt_0.001": float(np.mean(np.abs(local - glob) < 0.001)),
        "frac_abs_diff_lt_0.005": float(np.mean(np.abs(local - glob) < 0.005)),
        "ratio_local_over_global_median": float(np.median(local / glob)),
        "ratio_local_over_global_p10": float(np.percentile(local / glob, 10)),
        "ratio_local_over_global_p90": float(np.percentile(local / glob, 90)),
        "within_volume_cv_median": float(
            df.groupby("volume_id", observed=True)["porosity"]
              .apply(lambda s: s.std() / max(s.mean(), 1e-12)).median()
        ),
    }

    # ---------- per-bin conditionals ----------
    bin_idx = np.digitize(glob, BIN_EDGES) - 1
    sampler_table = {"bin_centres_global_phi": [], "ratio_quantiles": [],
                     "local_phi_quantiles": [], "n_patches": [], "n_volumes": []}
    for b in range(len(BIN_EDGES) - 1):
        m = bin_idx == b
        if m.sum() < 500:
            continue
        lo, hi = float(BIN_EDGES[b]), float(BIN_EDGES[b + 1])
        lv, gv = local[m], glob[m]
        vols = df.loc[m, "volume_id"].nunique()
        ratio = lv / gv
        entry = {
            "bin_index": b,
            "global_phi_range": [lo, hi],
            "global_phi_mean": float(gv.mean()),
            "n_patches": int(m.sum()),
            "n_volumes": int(vols),
            "local_p10": float(np.percentile(lv, 10)),
            "local_p25": float(np.percentile(lv, 25)),
            "local_p50": float(np.median(lv)),
            "local_p75": float(np.percentile(lv, 75)),
            "local_p90": float(np.percentile(lv, 90)),
            "local_p99": float(np.percentile(lv, 99)),
            "local_max": float(lv.max()),
            "local_mean": float(lv.mean()),
            "local_std": float(lv.std()),
            "frac_local_zero": float(np.mean(lv <= 0)),
            "skewness": float(((lv - lv.mean()) ** 3).mean() / max(lv.std(), 1e-12) ** 3),
            "frac_abs_diff_lt_0.001": float(np.mean(np.abs(lv - gv) < 0.001)),
            "frac_abs_diff_lt_0.005": float(np.mean(np.abs(lv - gv) < 0.005)),
            "frac_local_above_2x_global": float(np.mean(lv > 2 * gv)),
            "frac_local_below_half_global": float(np.mean(lv < 0.5 * gv)),
            "ratio_quantiles": np.quantile(ratio, QUANTILES).tolist(),
            "local_phi_quantiles": np.quantile(lv, QUANTILES).tolist(),
        }
        results["bins"].append(entry)
        sampler_table["bin_centres_global_phi"].append(entry["global_phi_mean"])
        sampler_table["ratio_quantiles"].append(entry["ratio_quantiles"])
        sampler_table["local_phi_quantiles"].append(entry["local_phi_quantiles"])
        sampler_table["n_patches"].append(entry["n_patches"])
        sampler_table["n_volumes"].append(entry["n_volumes"])
        print(f"  bin {lo:.4f}-{hi:.4f}: n={m.sum():7d} vols={vols:2d} "
              f"local p10/p50/p90 = {entry['local_p10']:.4f}/{entry['local_p50']:.4f}/"
              f"{entry['local_p90']:.4f}  |L-G|<0.005: {entry['frac_abs_diff_lt_0.005']*100:5.1f}%",
              flush=True)

    results["sampler"] = {
        "usage": (
            "Given a global target G: pick the two nearest bin_centres_global_phi, "
            "linearly interpolate ratio_quantiles between them, draw u ~ U(0,1), "
            "read the interpolated quantile at u, and set local phi = G * ratio. "
            "This reproduces the marginal p(local | global). It does NOT reproduce "
            "the spatial correlation measured in T-D: for a coherent volume, "
            "smooth the drawn field over the correlation length before use."
        ),
        "quantile_levels": QUANTILES.tolist(),
        **sampler_table,
    }
    p = write_json(results, OUT_DIR)
    print(f"[T-E] wrote {p}", flush=True)

    # ---------------- figures ----------------
    set_style()
    fig, axs = plt.subplots(1, 3, figsize=(12.5, 3.9), constrained_layout=True)

    ax = axs[0]
    sample = np.random.default_rng(0).choice(len(local), size=min(60000, len(local)),
                                             replace=False)
    ax.scatter(glob[sample], local[sample], s=1.2, alpha=0.06, color="#1b6ca8",
               edgecolors="none", rasterized=True)
    ax.plot([0, 0.12], [0, 0.12], "k--", lw=1.0, label="local = global\n(uniform painting)")
    centres = np.array(sampler_table["bin_centres_global_phi"])
    for q, style, lab in ((10, ":", "p10"), (50, "-", "p50"), (90, "--", "p90")):
        vals = [b[f"local_p{q}"] for b in results["bins"]]
        ax.plot(centres, vals, style, color="#c2571a", lw=1.6, label=lab)
    ax.set_xlabel("volume (global) porosity $\\varphi_G$")
    ax.set_ylabel("patch (local) porosity $\\varphi_L$")
    ax.set_title("Local vs global porosity")
    ax.set_xlim(0, 0.115); ax.set_ylim(0, 0.16)
    ax.legend(frameon=False, loc="upper left")

    ax = axs[1]
    for b in results["bins"]:
        lo, hi = b["global_phi_range"]
        m = (glob >= lo) & (glob < hi)
        if m.sum() < 500:
            continue
        h, edges = np.histogram(local[m], bins=np.linspace(0, 0.12, 121), density=True)
        ax.plot(0.5 * (edges[1:] + edges[:-1]), h, lw=1.2,
                label=f"$\\varphi_G\\in$[{lo:.3f},{hi:.3f}) n={b['n_patches']//1000}k")
    ax.set_xlabel("patch porosity $\\varphi_L$")
    ax.set_ylabel("probability density (1/$\\varphi$ units)")
    ax.set_yscale("log")
    ax.set_title("Conditional distributions $p(\\varphi_L\\,|\\,\\varphi_G)$")
    ax.legend(frameon=False, fontsize=7)

    ax = axs[2]
    labels, f1, f5 = [], [], []
    for b in results["bins"]:
        labels.append(f"{b['global_phi_range'][0]:.3f}")
        f1.append(100 * b["frac_abs_diff_lt_0.001"])
        f5.append(100 * b["frac_abs_diff_lt_0.005"])
    xpos = np.arange(len(labels))
    ax.bar(xpos - 0.2, f1, width=0.4, color="#8c2d04", label="$|\\varphi_L-\\varphi_G|<0.001$")
    ax.bar(xpos + 0.2, f5, width=0.4, color="#fdae6b", label="$|\\varphi_L-\\varphi_G|<0.005$")
    ax.set_xticks(xpos); ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_xlabel("lower edge of $\\varphi_G$ bin")
    ax.set_ylabel("share of real patches (%)")
    ax.set_title("How often is uniform painting realistic?")
    ax.legend(frameon=False)
    fig.suptitle("T-E  Conditional distribution of local porosity given global porosity",
                 fontsize=12)
    figs = savefig(fig, OUT_DIR, "TE_fig1_conditional")

    fig, ax = plt.subplots(figsize=(6.2, 4.0), constrained_layout=True)
    R = np.array(sampler_table["ratio_quantiles"])
    for i, c in enumerate(centres):
        ax.plot(QUANTILES, R[i], lw=1.2, label=f"$\\varphi_G$={c:.4f}")
    ax.axhline(1.0, color="k", ls="--", lw=1.0)
    ax.set_xlabel("quantile level $u$")
    ax.set_ylabel("ratio $\\varphi_L / \\varphi_G$ (dimensionless)")
    ax.set_yscale("log")
    ax.set_title("T-E  Sampler artifact: quantiles of $\\varphi_L/\\varphi_G$")
    ax.legend(frameon=False, fontsize=7, ncol=2)
    figs += savefig(fig, OUT_DIR, "TE_fig2_sampler_quantiles")
    print("[T-E] figures:", *figs, sep="\n  ", flush=True)


if __name__ == "__main__":
    main()
