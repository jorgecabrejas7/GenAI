#!/usr/bin/env python
"""Campaign 41 (paper review W5): bootstrap 95 % intervals for the
microstructure ratios of campaign 18 B and the comparison models.

Reviewer W5 (2026-11-24): "Bootstrap over panels for the reference
distributions and report 95% confidence intervals for the main ratios."

Units resampled, with replacement, independently:
    reference : the real PANELS of the level (each panel gives one real_a and
                one real_b crop; the pair stays together)
    model     : the generated volumes (one per seed)
For each draw the ratio is recomputed exactly as the campaign does it:
    ratio = d(generated, real_a + real_b) / d(real_a, real_b)
with ``poregen.eval_v4.microstructure.compare_sets`` for S2 W1, PSD W1 and
the Ripley log ratio (per-volume profiles are computed once with
``profile_volume``). With ``--fid`` the FID ratio is bootstrapped too, from
Inception features computed once per volume and axis (the campaign's crops,
``collect_fid_crops`` per single case), pooled per draw.

The point estimate is the campaign's own number (all units, no resampling).
With 3 panels and 3 seeds per level the intervals are coarse, and a draw
that repeats one panel cannot measure between-panel spread: they bound what
these data can say, no more.

    python scripts/analysis/c41_bootstrap.py --out runs/campaigns/41-paper-review-checks/bootstrap \
        "PoreGen 130k=runs/campaigns/18-eval-v4-final-v4-run0005-a2s:microstructure" \
        "PoreGen 40k=runs/campaigns/27-budget-matched-40k-v4-run0005-a2s:microstructure" \
        "phi-only LDM=runs/campaigns/25-ldm-phi-only-v4-run0002-a2s:microstructure" [--fid --device cuda]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from poregen.eval_v4 import microstructure as MS
from poregen.eval_v4.io import load_cases
from poregen.eval_v4.measure import _micro_reference

REPO = Path(__file__).resolve().parents[2]
C18B = REPO / "runs/campaigns/18-eval-v4-final-v4-run0005-a2s"
METRICS = ("s2_w1", "psd_w1", "ripley_log_ratio")


def panel_of(case) -> str:
    return case.manifest.case.split("__")[1]


def fid_feats(case, device, n_total: int, seed: int) -> dict:
    per = n_total // len(MS.FID_AXES)
    return {ax: MS.inception_features(MS.collect_fid_crops([case], ax, per, seed + 10_000 * i), device)
            for i, ax in enumerate(MS.FID_AXES)}


def fid_of(fa: list[dict], fb: list[dict]) -> float:
    vals = []
    for ax in MS.FID_AXES:
        a = np.concatenate([f[ax] for f in fa]); b = np.concatenate([f[ax] for f in fb])
        vals.append(MS.frechet_distance(a, b))
    return float(np.mean(vals))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("models", nargs="+", help="name=root:assessment")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--fid", action="store_true")
    ap.add_argument("--fid-crops", type=int, default=3000, help="crops per volume (split over the 3 axes)")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(a.seed)
    ref = _micro_reference(C18B)
    import torch
    dev = torch.device(a.device)
    rows = []
    for level, pair in sorted(ref.items()):
        panels = sorted({panel_of(c) for c in pair["a"]})
        A = {panel_of(c): c for c in pair["a"]}; B = {panel_of(c): c for c in pair["b"]}
        pa = {p: MS.profile_volume(A[p], group="real_a") for p in panels}
        pb = {p: MS.profile_volume(B[p], group="real_b") for p in panels}
        if a.fid:
            fa = {p: fid_feats(A[p], dev, a.fid_crops, 1) for p in panels}
            fb = {p: fid_feats(B[p], dev, a.fid_crops, 2) for p in panels}
        for spec in a.models:
            name, rest = spec.split("=", 1)
            root, assess = rest.rsplit(":", 1)
            gen = [c for c in load_cases(REPO / root, assess)
                   if abs(float(c.manifest.requested_global_phi or -1) - level) < 1e-6]
            if not gen:
                print(f"{name} {level}: no cases"); continue
            pg = [MS.profile_volume(c, group="generated") for c in gen]
            fg = [fid_feats(c, dev, a.fid_crops, 3) for c in gen] if a.fid else None

            def ratios(gi, pi):
                ra = [pa[panels[i]] for i in pi]; rb = [pb[panels[i]] for i in pi]
                g = [pg[i] for i in gi]
                num = MS.compare_sets(g, ra + rb); den = MS.compare_sets(ra, rb)
                out = {m: (num[m] / den[m] if den[m] else np.nan) for m in METRICS}
                if a.fid:
                    out["fid"] = fid_of([fg[i] for i in gi], [fa[panels[i]] for i in pi] + [fb[panels[i]] for i in pi]) \
                        / fid_of([fa[panels[i]] for i in pi], [fb[panels[i]] for i in pi])
                return out

            point = ratios(list(range(len(pg))), list(range(len(panels))))
            # Leave-one-panel-out: the reference with one panel removed, all
            # generated volumes kept. With 3 panels the percentile bootstrap is
            # biased (a draw that repeats a panel shrinks the real-vs-real
            # spread it divides by); these 3 values show the panel dependence
            # directly.
            lopo = [ratios(list(range(len(pg))), [i for i in range(len(panels)) if i != k])
                    for k in range(len(panels))] if len(panels) > 2 else []
            draws = []
            for _ in range(a.n_boot):
                gi = rng.integers(0, len(pg), len(pg)); pi = rng.integers(0, len(panels), len(panels))
                if len(set(pi)) < 2:   # a reference drawn from one panel has no real-vs-real spread
                    continue
                draws.append(ratios(list(gi), list(pi)))
            d = pd.DataFrame(draws)
            for m in point:
                lo, hi = np.nanpercentile(d[m], [2.5, 97.5])
                lv = [x[m] for x in lopo]
                rows.append({"model": name, "level": level, "metric": m, "ratio": point[m],
                             "ci_lo": lo, "ci_hi": hi,
                             "lopo_min": min(lv) if lv else np.nan, "lopo_max": max(lv) if lv else np.nan, "n_draws": int(d[m].notna().sum()),
                             "n_generated": len(pg), "n_panels": len(panels), "panels": ",".join(panels)})
            print(name, level, {m: round(point[m], 3) for m in point}, flush=True)
            pd.DataFrame(rows).to_csv(a.out / "bootstrap_ratios.csv", index=False)
    t = pd.DataFrame(rows)
    t["ci95"] = t.apply(lambda r: f"{r.ratio:.2f} [{r.ci_lo:.2f}, {r.ci_hi:.2f}] lopo {r.lopo_min:.2f}-{r.lopo_max:.2f}", axis=1)
    piv = t.pivot_table(index=["model", "level"], columns="metric", values="ci95", aggfunc="first")
    (a.out / "bootstrap_ratios.txt").write_text(
        f"Campaign 41 - microstructure ratios, point [95 % bootstrap interval], {a.n_boot} draws, "
        "panels and seeds resampled; draws with one distinct panel dropped; lopo = range over leave-one-panel-out references\n\n" + piv.to_string() + "\n")
    (a.out / "bootstrap_meta.json").write_text(json.dumps(vars(a), default=str, indent=1))
    print(piv.to_string())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
