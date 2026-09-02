"""T-F — Which transform of porosity should feed the conditioning embedding?

Compares three candidate transforms, for patch (local) porosity and for volume
(global) porosity separately:

* ``raw``        - phi as-is
* ``log_eps``    - log(phi + eps), several eps values
* ``sqrt``       - sqrt(phi), a milder compressive transform
* ``quantile``   - the empirical CDF of the training patches (rank -> [0, 1])
* ``quantile_gauss`` - the same CDF pushed through the normal inverse CDF

Every transform is min-max mapped onto [0, 1] using its 0.1 / 99.9 percentiles,
which is what an embedding layer sees.  On that common scale we report:

* ``central90_span``  - width of the p5-p95 interval as a share of the range.
  A value near 0.9 means the data uses the input range evenly; a small value
  means most of the range is empty and the embedding wastes capacity.
* ``effective_bins``  - exp(Shannon entropy) over 256 equal-width bins, i.e. the
  number of equally-populated levels the transform actually resolves (max 256).
* ``gini``            - inequality of the bin occupancy (0 = perfectly even).
* ``median_position`` - where the median sits inside the range (0.5 is ideal).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import norm

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import OUT_ROOT, PATCH_INDEX, savefig, set_style, write_json, plt  # noqa: E402

TEST_ID = "T-F"
OUT_DIR = OUT_ROOT / TEST_ID
N_BINS = 256


def transforms(train_local: np.ndarray) -> dict:
    """Build the candidate transforms. The CDF is fitted on training patches."""
    grid = np.quantile(train_local, np.linspace(0, 1, 4001))
    grid = np.maximum.accumulate(grid)
    u_grid = np.linspace(0, 1, 4001)

    def q(v):
        return np.interp(v, grid, u_grid)

    return {
        "raw": (lambda v: v.copy(), "$\\varphi$"),
        "sqrt": (lambda v: np.sqrt(np.maximum(v, 0)), "$\\sqrt{\\varphi}$"),
        "log_eps1e-4": (lambda v: np.log(np.maximum(v, 0) + 1e-4), "$\\log(\\varphi+10^{-4})$"),
        "log_eps1e-3": (lambda v: np.log(np.maximum(v, 0) + 1e-3), "$\\log(\\varphi+10^{-3})$"),
        "log_eps1e-2": (lambda v: np.log(np.maximum(v, 0) + 1e-2), "$\\log(\\varphi+10^{-2})$"),
        "quantile": (q, "CDF$(\\varphi)$"),
        "quantile_gauss": (lambda v: norm.ppf(np.clip(q(v), 1e-4, 1 - 1e-4)),
                           "$\\Phi^{-1}$(CDF$(\\varphi)$)"),
    }


def score(vals: np.ndarray) -> dict:
    v = vals[np.isfinite(vals)]
    lo, hi = np.percentile(v, [0.1, 99.9])
    if hi <= lo:
        return {"degenerate": True}
    u = np.clip((v - lo) / (hi - lo), 0.0, 1.0)
    p5, p50, p95 = np.percentile(u, [5, 50, 95])
    hist, _ = np.histogram(u, bins=N_BINS, range=(0.0, 1.0))
    p = hist / hist.sum()
    nz = p[p > 0]
    H = float(-(nz * np.log(nz)).sum())
    srt = np.sort(p)
    n = len(srt)
    gini = float((2 * np.arange(1, n + 1) - n - 1).dot(srt) / (n * srt.sum()))
    return {
        "p0.1": float(lo), "p99.9": float(hi),
        "central90_span": float(p95 - p5),
        "median_position": float(p50),
        "effective_bins": float(np.exp(H)),
        "effective_bins_frac_of_256": float(np.exp(H) / N_BINS),
        "occupied_bins": int((hist > 0).sum()),
        "gini_bin_occupancy": gini,
        "skewness": float(((u - u.mean()) ** 3).mean() / max(u.std(), 1e-12) ** 3),
        "excess_kurtosis": float(((u - u.mean()) ** 4).mean() / max(u.std(), 1e-12) ** 4 - 3.0),
        "frac_in_lowest_decile_of_range": float(np.mean(u < 0.1)),
        "frac_in_highest_decile_of_range": float(np.mean(u > 0.9)),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.parse_args()
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    df = pd.read_parquet(PATCH_INDEX)
    df = df[df["porosity"] <= 1.0]
    train_local = df.loc[df["split"] == "train", "porosity"].to_numpy(dtype=np.float64)
    local = df["porosity"].to_numpy(dtype=np.float64)

    cache = OUT_ROOT / "T-A" / "fine_profiles.npz"
    if cache.exists():
        profiles = np.load(cache, allow_pickle=True)["profiles"].item()
        gvals = []
        for p in profiles.values():
            fg = np.asarray(p["z"]["fg"], float); phi = np.asarray(p["z"]["phi"], float)
            ok = np.isfinite(phi) & (fg > 0)
            gvals.append(float(np.sum(phi[ok] * fg[ok]) / np.sum(fg[ok])))
        glob = np.array(gvals)
        gsrc = "exact foreground-normalised volume porosity (T-A voxel pass)"
    else:
        glob = df.groupby("volume_id", observed=True)["porosity"].mean().to_numpy()
        gsrc = "mean over each volume's patches"

    tf = transforms(train_local)
    results = {
        "test_id": TEST_ID,
        "n_patches": int(len(local)),
        "n_volumes": int(len(glob)),
        "global_phi_source": gsrc,
        "epsilon_note": "log transforms use log(phi + eps) with eps in {1e-4, 1e-3, 1e-2}.",
        "raw_stats": {
            "local": {"mean": float(local.mean()), "median": float(np.median(local)),
                      "std": float(local.std()), "min": float(local.min()),
                      "max": float(local.max()),
                      "frac_zero": float(np.mean(local <= 0)),
                      "frac_below_1e-3": float(np.mean(local < 1e-3)),
                      "p1": float(np.percentile(local, 1)),
                      "p99": float(np.percentile(local, 99))},
            "global": {"mean": float(glob.mean()), "median": float(np.median(glob)),
                       "std": float(glob.std()), "min": float(glob.min()),
                       "max": float(glob.max())},
        },
        "local": {}, "global": {},
    }
    for name, (fn, _) in tf.items():
        results["local"][name] = score(fn(local))
        results["global"][name] = score(fn(glob))
        sl, sg = results["local"][name], results["global"][name]
        print(f"  {name:16s} local: span90={sl['central90_span']:.3f} "
              f"eff_bins={sl['effective_bins']:6.1f} med_pos={sl['median_position']:.3f} | "
              f"global: span90={sg['central90_span']:.3f} "
              f"eff_bins={sg['effective_bins']:6.1f} med_pos={sg['median_position']:.3f}",
              flush=True)

    def best(scope):
        cand = {k: v for k, v in results[scope].items() if not v.get("degenerate")}
        return max(cand, key=lambda k: (cand[k]["effective_bins"], cand[k]["central90_span"]))

    results["recommendation"] = {
        "local_best_by_effective_bins": best("local"),
        "global_best_by_effective_bins": best("global"),
    }
    p = write_json(results, OUT_DIR)
    print(f"[T-F] wrote {p}", flush=True)

    # ---------------- figures ----------------
    set_style()
    names = list(tf.keys())
    fig, axs = plt.subplots(2, len(names), figsize=(2.05 * len(names), 5.2),
                            constrained_layout=True)
    for j, name in enumerate(names):
        fn, lab = tf[name]
        for i, (vals, scope, colour) in enumerate(
                ((local, "local", "#1b6ca8"), (glob, "global", "#c2571a"))):
            ax = axs[i][j]
            t = fn(vals)
            s = results[scope][name]
            u = np.clip((t - s["p0.1"]) / (s["p99.9"] - s["p0.1"]), 0, 1)
            ax.hist(u, bins=60, range=(0, 1), color=colour, alpha=0.85)
            ax.set_yscale("log")
            ax.set_title(lab if i == 0 else "", fontsize=8)
            ax.set_xlabel("normalised embedding input" if i == 1 else "")
            ax.set_ylabel(f"{scope} count" if j == 0 else "")
            ax.text(0.5, 0.95, f"eff. bins {s['effective_bins']:.0f}\nspan90 "
                                f"{s['central90_span']:.2f}",
                    transform=ax.transAxes, ha="center", va="top", fontsize=7)
            ax.set_xticks([0, 0.5, 1])
    fig.suptitle("T-F  Porosity transforms mapped onto the embedding input range [0, 1]\n"
                 "top: patch (local) porosity, 2.27M patches — bottom: volume (global) "
                 "porosity, 80 volumes", fontsize=11)
    figs = savefig(fig, OUT_DIR, "TF_fig1_transform_histograms")

    fig, ax = plt.subplots(figsize=(7.2, 3.8), constrained_layout=True)
    xpos = np.arange(len(names))
    ax.bar(xpos - 0.2, [results["local"][n]["effective_bins"] for n in names],
           width=0.4, color="#1b6ca8", label="patch (local) $\\varphi$")
    ax.bar(xpos + 0.2, [results["global"][n]["effective_bins"] for n in names],
           width=0.4, color="#c2571a", label="volume (global) $\\varphi$")
    ax.axhline(N_BINS, color="k", ls=":", lw=0.9)
    ax.text(len(names) - 0.5, N_BINS * 1.02, "256 = ideal", fontsize=8, ha="right")
    ax.set_xticks(xpos); ax.set_xticklabels(names, rotation=30, ha="right")
    ax.set_ylabel("effective distinguishable levels  $e^{H}$")
    ax.set_title("T-F  Range use per transform (higher is better, 256 bins)")
    ax.legend(frameon=False)
    figs += savefig(fig, OUT_DIR, "TF_fig2_effective_bins")
    print("[T-F] figures:", *figs, sep="\n  ", flush=True)


if __name__ == "__main__":
    main()
