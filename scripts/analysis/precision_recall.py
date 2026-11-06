#!/usr/bin/env python
"""Improved precision and recall (Kynkaanniemi et al. 2019) beside the FID, as
Rombach et al. 2022 report them. Campaign 37. (CPU by default.)

The feature space is EXACTLY the one our FID uses: Inception pool3 (2048-d) on
64x64 material-only slices along the three axes, the crops and preprocessing of
``poregen.eval_v4.microstructure`` (``collect_fid_crops``, ``inception_features``).
So the three numbers of one row come from the same crops.

Precision = share of generated features inside the real k-NN manifold (fidelity).
Recall    = share of real features inside the generated k-NN manifold (coverage).
A manifold is the union of balls around each point, radius = distance to its
k-th nearest neighbour in its own set (k = 3, the paper's value). FID is the
per-axis mean, as in campaign 18; precision/recall use the pooled crops.

Sets (author 2026-11-06, "yes to everything"):
  per porosity level 0.01 / 0.03 / 0.06 of campaign 18's microstructure test:
    real floor  : the matched real crops real_a vs real_b (two disjoint crops of
                  one panel) -> the value real material scores against itself
    ours        : campaign 18 microstructure volumes (ldm06-run-0005, 130k)
    ours 40k    : campaign 27 (same model at the phi-only baseline's budget)
    phi-only LDM: campaign 25
    SliceGAN, pixel DDPM: their 192^3 volumes, scored against every level (they
                  cannot be asked for a porosity; the nearest level is named)
  per porosity bin of campaign 36:
    the a2s twins of the bin against their own real coupons (crops cut from the
    coupon arrays, padding excluded); floor = real coupons of the bin, odd
    against even index.

``--crops`` per set (default 10 000), split evenly over the three axes and the
cases of the set. Output: <out>/results.json, table.md, fig_pr.png.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import tifffile  # noqa: E402
import torch  # noqa: E402

from poregen.eval_v4 import microstructure as MS  # noqa: E402
from poregen.eval_v4.io import Case, iter_case_dirs, load_cases  # noqa: E402
from poregen.eval_v4.manifest import Manifest  # noqa: E402
from poregen.eval_v4.measure import _micro_reference  # noqa: E402

REPO = Path(__file__).resolve().parents[2]
C = REPO / "runs/campaigns"
C18 = C / "18-eval-v4-final-v4-run0005"
C36 = C / "36-pore-morphology"
K = 3
LEVELS = (0.01, 0.03, 0.06)
MODELS = {
    "ours (130k)": (C18, "microstructure"),
    "ours (40k)": (C / "27-budget-matched-40k-v4-run0005", "microstructure"),
    "phi-only LDM": (C / "25-ldm-phi-only-v4-run0002", "microstructure"),
}
BASELINES = {
    "SliceGAN": (C / "22-slicegan-baseline-v4", "slicegan"),
    "pixel DDPM": (C / "23-ddpm3d-baseline-v4", "ddpm3d"),
}


def load_cases_tolerant(root, assessment):
    """The baselines' manifests (campaigns 22/23, 2026-10-06/07) carry
    ``porosity_clamped``, a field the Manifest class has since dropped
    (a53e250); it meant nothing for an unconditional model. Dropped on read."""
    import dataclasses  # noqa: PLC0415
    known = {f.name for f in dataclasses.fields(Manifest)}
    out = []
    for d in iter_case_dirs(root, assessment):
        m = json.loads((d / "manifest.json").read_text())
        out.append(Case(path=d, manifest=Manifest.from_dict({k: v for k, v in m.items() if k in known})))
    return out


# ── the metric ──────────────────────────────────────────────────────────────

def knn_radii(x: torch.Tensor, k: int) -> torch.Tensor:
    """Distance from each point to its k-th nearest OTHER point of the same set."""
    r = torch.empty(len(x))
    for s in range(0, len(x), 1024):
        d = torch.cdist(x[s:s + 1024], x)
        r[s:s + 1024] = d.kthvalue(k + 1, dim=1).values  # +1: the point itself at 0
    return r


def inside(query: torch.Tensor, ref: torch.Tensor, ref_r: torch.Tensor) -> float:
    """Share of query points that fall inside at least one ball of ref."""
    hit = torch.zeros(len(query), dtype=torch.bool)
    for s in range(0, len(query), 1024):
        d = torch.cdist(query[s:s + 1024], ref)
        hit[s:s + 1024] = (d <= ref_r[None]).any(dim=1)
    return float(hit.float().mean())


def precision_recall(real: np.ndarray, gen: np.ndarray, k: int = K) -> dict:
    r, g = torch.from_numpy(real).float(), torch.from_numpy(gen).float()
    rr, gr = knn_radii(r, k), knn_radii(g, k)
    return {"precision": inside(g, r, rr), "recall": inside(r, g, gr), "k": k,
            "n_real": len(real), "n_generated": len(gen)}


# ── features ────────────────────────────────────────────────────────────────

def features_of_cases(cases, n_total: int, seed: int, device) -> dict[str, np.ndarray]:
    per = n_total // len(MS.FID_AXES)
    return {ax: MS.inception_features(MS.collect_fid_crops(cases, ax, per, seed + 10_000 * i), device)
            for i, ax in enumerate(MS.FID_AXES)}


class ArrayCase:
    """A real coupon as the crop collector expects it: xct, material, manifest."""

    def __init__(self, xct, label, manifest):
        self.xct, self._mat, self.manifest = xct, label != 2, manifest

    def material_voxels(self):
        return self._mat


def coupon_case(coupon_dir: Path, twin_case) -> ArrayCase:
    xct = tifffile.imread(coupon_dir / "real_volume.tif")
    label = tifffile.imread(coupon_dir / "real_label.tif")
    return ArrayCase(xct, label, twin_case.manifest)


def score(fr: dict, fg: dict) -> dict:
    fid_axes = {ax: MS.frechet_distance(fr[ax], fg[ax]) for ax in fr}
    pr = precision_recall(np.concatenate(list(fr.values())), np.concatenate(list(fg.values())))
    return {"fid": float(np.mean(list(fid_axes.values()))), "fid_per_axis": fid_axes, **pr}


# ── main ────────────────────────────────────────────────────────────────────

def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--crops", type=int, default=10_000)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--out", type=Path, default=C / "37-precision-recall")
    ap.add_argument("--skip-twins", action="store_true")
    a = ap.parse_args()
    device = torch.device(a.device)
    a.out.mkdir(parents=True, exist_ok=True)
    rows: list[dict] = []
    done_file = a.out / "results.json"
    rows = json.load(open(done_file))["rows"] if done_file.exists() else []
    have = {(r["group"], r["model"], r["level"]) for r in rows}

    def add(group, model, level, fr, fg, extra=None):
        if (group, model, level) in have:
            return
        s = score(fr, fg)
        rows.append({"group": group, "model": model, "level": level, **s, **(extra or {})})
        print(f"{group:8s} {model:14s} {level!s:8s} FID {s['fid']:.2f}  P {s['precision']:.3f}  R {s['recall']:.3f}", flush=True)
        json.dump({"k": K, "crops_per_set": a.crops, "extractor": MS.FID_EXTRACTOR, "rows": rows},
                  open(done_file, "w"), indent=1)

    # ── campaign-18 levels ───────────────────────────────────────────────
    ref = _micro_reference(C18)
    base_feats = {name: features_of_cases([c for c in load_cases_tolerant(root, ass)
                                           if tuple(c.manifest.volume_shape) == (192, 192, 192)],
                                          a.crops, 7, device)
                  for name, (root, ass) in BASELINES.items()}
    for lv in LEVELS:
        ra, rb = ref[lv]["a"], ref[lv]["b"]
        fa = features_of_cases(ra, a.crops, 1, device)
        fb = features_of_cases(rb, a.crops, 2, device)
        add("level", "real vs real (floor)", lv, fa, fb)
        freal = {ax: np.concatenate([fa[ax], fb[ax]]) for ax in fa}
        for name, (root, ass) in MODELS.items():
            cases = [c for c in load_cases(root, ass)
                     if abs(float(c.manifest.requested_global_phi or -1) - lv) < 1e-6]
            if not cases:
                print("no cases:", name, lv); continue
            add("level", name, lv, freal, features_of_cases(cases, a.crops, 3, device),
                {"n_cases": len(cases)})
        for name, fg in base_feats.items():
            add("level", name, lv, freal, fg)

    # ── campaign-36 twins per bin ────────────────────────────────────────
    if not a.skip_twins:
        from poregen.eval_v4.cases import TWIN_COUPONS, twin_name  # noqa: PLC0415
        twins = {c.manifest.case: c for c in load_cases(C36 / "gen_a2s", "twins")}
        for b, vids in TWIN_COUPONS.items():
            names = [twin_name(v) for v in vids if twin_name(v) in twins]
            if not names:
                continue
            gen = [twins[n] for n in names]
            real = []
            for n in names:
                d = C36 / "compare" / n.removeprefix("twin_")
                if (d / "real_volume.tif").exists():
                    real.append(coupon_case(d, twins[n]))
            if len(real) < 2:
                print("twins: fewer than two real coupons on disk for bin", b); continue
            fr = features_of_cases(real, a.crops, 11, device)
            add("bin", "real vs real (floor)", b,
                features_of_cases(real[0::2], a.crops, 12, device),
                features_of_cases(real[1::2], a.crops, 13, device))
            add("bin", "twins (a2s)", b, fr, features_of_cases(gen, a.crops, 14, device),
                {"n_cases": len(gen), "n_real": len(real)})
            del fr

    # ── table and figure ─────────────────────────────────────────────────
    md = [f"# Campaign 37 - improved precision / recall (k = {K}) beside the FID\n",
          f"{a.crops} crops per set, Inception pool3 on 64x64 material slices, three axes.\n",
          "| group | level/bin | model | FID | precision | recall |", "|---|---|---|---|---|---|"]
    for r in rows:
        md.append(f"| {r['group']} | {r['level']} | {r['model']} | {r['fid']:.2f} | {r['precision']:.3f} | {r['recall']:.3f} |")
    (a.out / "table.md").write_text("\n".join(md) + "\n")
    fig, ax = plt.subplots(figsize=(6, 5))
    for r in rows:
        m = "*" if "floor" in r["model"] else "o"
        ax.scatter(r["recall"], r["precision"], marker=m, s=60)
        ax.annotate(f"{r['model']} {r['level']}", (r["recall"], r["precision"]), fontsize=6)
    ax.set_xlabel("recall (coverage of real)"); ax.set_ylabel("precision (fidelity)")
    ax.set_xlim(0, 1.02); ax.set_ylim(0, 1.02); ax.grid(alpha=.3)
    fig.tight_layout(); fig.savefig(a.out / "fig_pr.png", dpi=130)
    print("wrote", a.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
