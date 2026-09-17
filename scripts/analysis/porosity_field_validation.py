"""Does the coherent porosity field deliver what it was asked for? (CPU.)

WHY THIS EXISTS. `build_porosity_field` draws from a fitted marginal, smooths
with a Gaussian, then rescales and clips. Two of those three steps change the
thing the step before it established, and nothing measured the result:

  * the smoothing sigma set the 1/e correlation length to TWICE the requested
    one, because smoothed white noise has rho(r) = exp(-r^2/4s^2) and so falls
    to 1/e at 2s, not at s. Every coherent field ever generated was twice as
    smooth as the T-D length it was built from;
  * smoothing REDUCES VARIANCE, so the marginal after smoothing is not the
    marginal that was fitted, and the rescale-and-clip does not restore it.

The first is a defect and is fixed. The second is not a defect — it is what
smoothing does — but it is a property of the delivered field that has to be
stated rather than discovered later.

Usage:
    python scripts/analysis/porosity_field_validation.py
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
#: TWO grids, because they answer different questions.
#:
#: `diagnostic` is big enough that even the 901-voxel x length is many grid
#: steps, so it measures what the GENERATOR delivers. `production` is the grid
#: a 192x1024x1024 canvas actually uses, where a 901-voxel requested length is
#: most of the box — what it delivers there is limited by the canvas and not by
#: the code, and reporting only the diagnostic number would overstate what a
#: real request receives.
GRIDS = {"diagnostic": (24, 96, 160), "production": (3, 16, 16)}
SEEDS = (101, 202, 303)
TARGETS = (0.01, 0.03, 0.06)


def one_over_e_length(field: np.ndarray, axis: int, stride: int) -> float:
    """1/e correlation length along one axis, in VOXELS.

    Paired-Pearson per lag on the demeaned field, interpolated to the crossing
    — the same definition `metrics.correlation_length_1_over_e` uses, so the
    number is comparable with everything else in this project.
    """
    f = field - field.mean()
    n = field.shape[axis]
    lags = np.arange(0, min(n - 1, 24))
    rho = []
    for lag in lags:
        a = np.take(f, np.arange(0, n - lag), axis=axis)
        b = np.take(f, np.arange(lag, n), axis=axis)
        sa, sb = a.std(), b.std()
        rho.append(float((a * b).mean() / (sa * sb)) if sa > 0 and sb > 0 else np.nan)
    rho = np.asarray(rho)
    below = np.flatnonzero(rho < np.exp(-1.0))
    if not below.size:
        return float("nan")
    i = below[0]
    if i == 0:
        return 0.0
    # linear interpolation between the lag above and the lag below 1/e
    t = (rho[i - 1] - np.exp(-1.0)) / (rho[i - 1] - rho[i])
    return float((lags[i - 1] + t) * stride)


def main() -> int:
    from poregen.diffusion.porosity_field import (
        DEFAULT_TD_RESULTS,
        DEFAULT_TE_RESULTS,
        build_porosity_field,
        load_corr_lengths_voxels,
        load_sampler,
    )

    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path,
                    default=REPO / "runs" / "campaigns" / "26-field-validation")
    ap.add_argument("--stride", type=int, default=64)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    sampler = load_sampler(DEFAULT_TE_RESULTS)
    corr = load_corr_lengths_voxels(DEFAULT_TD_RESULTS)
    print(f"requested 1/e lengths (voxels), z/y/x: {tuple(round(c, 1) for c in corr)}")

    rows = []
    for grid_name, grid in GRIDS.items():
      for target in TARGETS:
        for seed in SEEDS:
            f = build_porosity_field(grid, target, sampler, corr,
                                     stride_voxels=args.stride, seed=seed)
            rows.append({
                "grid": grid_name, "target": target, "seed": seed,
                "delivered_mean": float(f.mean()),
                "delivered_sd": float(f.std()),
                "delivered_min": float(f.min()), "delivered_max": float(f.max()),
                "corr_len_vox": {ax: one_over_e_length(f, a, args.stride)
                                 for a, ax in enumerate(("z", "y", "x"))},
            })

    # The marginal the sampler was fitted to, drawn WITHOUT smoothing, so the
    # two can be compared on the same target.
    from poregen.diffusion.porosity_field import _draw_marginal
    raw = {}
    for target in TARGETS:
        u = np.random.default_rng(0).uniform(0, 1, size=100000)
        d = _draw_marginal(float(target), sampler, u)
        raw[f"{target:g}"] = {"mean": float(d.mean()), "sd": float(d.std())}

    def agg(sel, key):
        v = [r[key] for r in sel]
        return {"mean": float(np.mean(v)), "sd": float(np.std(v))}

    summary = {}
    for grid_name in GRIDS:
      for target in TARGETS:
        sel = [r for r in rows if r["target"] == target and r["grid"] == grid_name]
        summary[f"{grid_name}/{target:g}"] = {
            "grid": grid_name,
            "requested_mean": target,
            "delivered_mean": agg(sel, "delivered_mean"),
            "unsmoothed_marginal_sd": raw[f"{target:g}"]["sd"],
            "delivered_sd": agg(sel, "delivered_sd"),
            "sd_ratio_delivered_over_fitted": (
                agg(sel, "delivered_sd")["mean"] / raw[f"{target:g}"]["sd"]
                if raw[f"{target:g}"]["sd"] else None),
            "corr_len_vox": {
                ax: {"requested": float(corr[a]),
                     "delivered": float(np.mean([r["corr_len_vox"][ax] for r in sel])),
                     "ratio": float(np.mean([r["corr_len_vox"][ax] for r in sel]) / corr[a])}
                for a, ax in enumerate(("z", "y", "x"))},
        }

    out = {
        "question": "Does build_porosity_field deliver the correlation length "
                    "and the marginal it was asked for?",
        "grids": {k: list(v) for k, v in GRIDS.items()}, "stride_voxels": args.stride,
        "seeds": list(SEEDS), "targets": list(TARGETS),
        "td_results": str(DEFAULT_TD_RESULTS), "te_results": str(DEFAULT_TE_RESULTS),
        "requested_corr_len_vox": [float(c) for c in corr],
        "note": (
            "THE MARGINAL IS NOT PRESERVED, AND THAT IS NOT A BUG. Smoothing "
            "reduces variance, so the delivered field is narrower than the "
            "fitted T-E marginal whatever the correlation length is; the "
            "rescale-and-clip restores the MEAN and does not restore the "
            "spread. sd_ratio_delivered_over_fitted is how much narrower. A "
            "field asked for a correlation length must give up marginal "
            "fidelity to get it, and this campaign reports which was "
            "delivered rather than implying both were."
        ),
        "per_field": rows,
        "summary": summary,
    }
    (args.out / "results.json").write_text(json.dumps(out, indent=2) + "\n")

    print(f"\n{'cell':>20} {'req mean':>9} {'got mean':>9} "
          f"{'fitted sd':>10} {'got sd':>8} {'sd ratio':>9}")
    for k, v in summary.items():
        print(f"{k:>20} {v['requested_mean']:9.4f} {v['delivered_mean']['mean']:9.4f} "
              f"{v['unsmoothed_marginal_sd']:10.4f} {v['delivered_sd']['mean']:8.4f} "
              f"{v['sd_ratio_delivered_over_fitted']:9.3f}")
    for grid_name, grid in GRIDS.items():
        print(f"\n{grid_name} grid {grid} = "
              f"{tuple(g * args.stride for g in grid)} voxels")
        print(f"{'axis':>5} {'requested':>10} {'delivered':>10} {'ratio':>7}")
        for ax in ("z", "y", "x"):
            c = summary[f"{grid_name}/{TARGETS[1]:g}"]["corr_len_vox"][ax]
            print(f"{ax:>5} {c['requested']:10.1f} {c['delivered']:10.1f} "
                  f"{c['ratio']:7.3f}")
    print(f"\n-> {args.out}/results.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
