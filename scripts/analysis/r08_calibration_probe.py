"""Is r08's high-porosity over-prediction calibration, or capacity?

r08-run-0002 over-predicts pore by ~2.2x wherever porosity is appreciable, on
training panels as well as held-out ones. Two explanations have different
consequences:

* **Calibration.** The class-weighted cross-entropy (pore weight 16.1) makes a
  false positive cheap relative to a false negative, so the decision boundary
  sits too low. The learned probabilities would then be fine and a single fixed
  correction — dividing out the weights, or raising the pore threshold — should
  remove the bias across every panel and every bin at once.
* **Capacity.** z=4 cannot represent dense pore structure. No fixed
  post-hoc rule fixes that: the error stays, and the sweep over z is the
  experiment that matters.

This probe decides between them without training anything. For a stratified
sample (~``--per-bin`` patches per ground-truth porosity bin per panel) it runs
one forward pass, then scores the SAME probabilities three ways:

1. ``argmax``   — what the model delivers today.
2. ``deweight`` — ``p_c / w_c`` renormalised, then argmax. Undoes the CE
   reweighting exactly if the network learned the reweighted posterior.
3. ``tau``      — pore where ``p_pore > tau``, else argmax of the other two,
   for a sweep of thresholds.

The verdict rule: if one fixed rule brings porosity MAE under the 0.005 gate in
every bin of every panel, the cause is calibration. If the MAE stays large
where porosity is high, it is capacity.

Usage
-----
    python scripts/analysis/r08_calibration_probe.py --run runs/vae/r08-run-0002-...
"""

from __future__ import annotations

import argparse
import glob
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import REPO, write_findings, write_json  # noqa: E402

sys.path.insert(0, str(REPO / "src"))

from poregen.dataset.loader import build_label  # noqa: E402
from poregen.experiments.train_vae import build_model  # noqa: E402
from poregen.models.vae.base import CLASS_AIR, CLASS_PORE  # noqa: E402
from poregen.training.checkpoint import load_checkpoint  # noqa: E402

OUT_ROOT = REPO / "runs/campaigns/09-r08-latent-sweep"
INDEX = REPO / "data/split_v3/patch_index.parquet"
ZARR = REPO / "data/split_v3/volumes.zarr"

BINS = [0.0, 0.01, 0.03, 0.06, float("inf")]
BIN_LABELS = ["<1%", "1-3%", "3-6%", ">=6%"]
TAUS = (0.5, 0.6, 0.7, 0.8, 0.9, 0.95)
GATE = 0.005
PATCH = 64

# The three panels holding the dense microstructure. The calibration probe
# located the capacity residual here and nowhere else: after de-weighting
# r08-run-0002 they were the only panels still failing a bin, and their pore
# Dice is 0.69-0.77 against 0.95-0.96 for the JI panels. Porosity TOTALS are
# fixable by calibration; the Dice gap is not, so this is the number a latent
# rung has to move. Na_10 and Pegaso_1 are TRAIN panels, which is why this
# lives in the probe (all 17 panels) and not in the rung report (val/test).
DENSE_PANELS = ("Na_10", "Na_09", "Pegaso_1")
_T0 = time.time()


def log(msg: str) -> None:
    print(f"[{time.time() - _T0:7.1f}s] {msg}", flush=True)


def stratified(df: pd.DataFrame, per_bin: int, seed: int = 0) -> pd.DataFrame:
    """Up to ``per_bin`` patches from each porosity bin of each panel."""
    d = df.assign(bin=pd.cut(df.porosity, bins=BINS, right=False,
                             labels=BIN_LABELS))
    out = []
    for (_, _), grp in d.groupby(["panel_id", "bin"], observed=True):
        out.append(grp.sample(min(len(grp), per_bin), random_state=seed))
    return pd.concat(out).reset_index(drop=True)


@torch.no_grad()
def probe(model, rows: pd.DataFrame, g, device, weights: np.ndarray,
          batch: int = 32) -> dict:
    """One forward pass per patch; score the same probabilities every way."""
    w = torch.tensor(weights, dtype=torch.float32, device=device).view(1, 3, 1, 1, 1)
    variants = ["argmax", "deweight"] + [f"tau{t:g}" for t in TAUS]
    acc = {v: {"pf": [], "inter": [], "card": []} for v in variants}
    true_pf = []

    for i in range(0, len(rows), batch):
        ch = rows.iloc[i:i + batch]
        xs, ts = [], []
        for r in ch.itertuples():
            sl = np.s_[r.z0:r.z0 + PATCH, r.y0:r.y0 + PATCH, r.x0:r.x0 + PATCH]
            grp = g[r.volume_id]
            xs.append(np.asarray(grp["xct"][sl]).astype(np.float32) / 255.0)
            ts.append(build_label(np.asarray(grp["mask"][sl]),
                                  np.asarray(grp["sample_mask"][sl])))
        X = torch.from_numpy(np.stack(xs)).unsqueeze(1).to(device)
        T = torch.from_numpy(np.stack(ts).astype(np.int64)).to(device)
        if device.type == "cuda":
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                logits = model(X, T).class_logits
        else:
            logits = model(X, T).class_logits
        p = torch.softmax(logits.float(), dim=1)

        gt = T == CLASS_PORE
        true_pf.append(gt.flatten(1).float().mean(1).cpu().numpy())

        preds = {"argmax": p.argmax(1),
                 "deweight": (p / w).argmax(1)}
        other = torch.stack([p[:, 0], p[:, CLASS_AIR]], dim=1)
        other_cls = torch.where(other.argmax(1) == 0,
                                torch.zeros_like(gt, dtype=torch.long),
                                torch.full_like(gt, CLASS_AIR, dtype=torch.long))
        for t in TAUS:
            preds[f"tau{t:g}"] = torch.where(p[:, CLASS_PORE] > t,
                                             torch.full_like(other_cls, CLASS_PORE),
                                             other_cls)
        for v, pred in preds.items():
            pp = pred == CLASS_PORE
            acc[v]["pf"].append(pp.flatten(1).float().mean(1).cpu().numpy())
            acc[v]["inter"].append((pp & gt).flatten(1).sum(1).float().cpu().numpy())
            acc[v]["card"].append((pp.flatten(1).sum(1) + gt.flatten(1).sum(1))
                                  .float().cpu().numpy())
        if (i // batch) % 20 == 0:
            log(f"  {i + len(ch)}/{len(rows)} patches")

    t = np.concatenate(true_pf)
    out = {"true_pf": t, "variants": {}}
    for v in variants:
        pf = np.concatenate(acc[v]["pf"])
        inter = np.concatenate(acc[v]["inter"])
        card = np.concatenate(acc[v]["card"])
        dice = np.where(card > 0, 2.0 * inter / np.maximum(card, 1), 1.0)
        out["variants"][v] = {"pf": pf, "dice": dice}
    return out


def summarise(res: dict, rows: pd.DataFrame) -> dict:
    t = res["true_pf"]
    b = pd.cut(t, bins=BINS, right=False, labels=BIN_LABELS)
    panels = rows["panel_id"].to_numpy()
    splits = rows["split"].to_numpy()
    out = {}
    for v, d in res["variants"].items():
        err = np.abs(d["pf"] - t)
        per_bin = {}
        for lab in BIN_LABELS:
            sel = np.asarray(b == lab)
            if sel.sum():
                per_bin[lab] = {"n": int(sel.sum()),
                                "mae": float(err[sel].mean()),
                                "bias": float((d["pf"] - t)[sel].mean()),
                                "dice": float(d["dice"][sel].mean())}
        per_panel = {}
        for p in sorted(set(panels)):
            ps = panels == p
            per_panel[p] = {
                "split": splits[ps][0],
                "dice": float(d["dice"][ps].mean()),
                "bins": {lab: float(err[ps & np.asarray(b == lab)].mean())
                         for lab in BIN_LABELS
                         if (ps & np.asarray(b == lab)).sum()},
            }
        dense = np.isin(panels, DENSE_PANELS)
        groups = {}
        for name, sel in (("dense", dense), ("rest", ~dense)):
            if sel.sum():
                groups[name] = {
                    "panels": sorted(set(panels[sel].tolist())),
                    "n": int(sel.sum()),
                    "dice_pore": float(d["dice"][sel].mean()),
                    "porosity_mae": float(err[sel].mean()),
                }
        worst = max((c["mae"] for c in per_bin.values()), default=float("nan"))
        out[v] = {"overall_mae": float(err.mean()),
                  "overall_dice": float(d["dice"].mean()),
                  "worst_bin_mae": worst,
                  "passes_every_bin": bool(worst < GATE),
                  "dense_vs_rest": groups,
                  "per_bin": per_bin, "per_panel": per_panel}
    return out


def build_findings(meta: dict, summ: dict) -> str:
    L = [f"# r08 calibration probe — {meta['experiment']} step {meta['step']}", "",
         "Is the high-porosity over-prediction CALIBRATION (the class-weighted "
         "cross-entropy put the decision boundary too low) or CAPACITY (z=4 "
         "cannot represent dense pore structure)? One forward pass, three "
         "scorings of the same probabilities.", "",
         f"Stratified sample: {meta['n_patches']} patches, up to "
         f"{meta['per_bin']} per porosity bin per panel, across "
         f"{meta['n_panels']} panels of all three splits. Training weights "
         f"{meta['class_weights']}. Gate {GATE} per bin.", "",
         "## Verdict table", "",
         "| scoring | overall MAE | worst-bin MAE | pore Dice | every bin < gate |",
         "|---|---|---|---|---|"]
    for v, s in summ.items():
        L.append(f"| {v} | {s['overall_mae']:.5f} | {s['worst_bin_mae']:.5f} "
                 f"| {s['overall_dice']:.4f} | "
                 f"{'**yes**' if s['passes_every_bin'] else 'no'} |")
    L += ["", "## Per bin", "",
          "| scoring | " + " | ".join(f"{l} (n)" for l in BIN_LABELS) + " |",
          "|---" * (len(BIN_LABELS) + 1) + "|"]
    for v, s in summ.items():
        cells = []
        for lab in BIN_LABELS:
            c = s["per_bin"].get(lab)
            cells.append("—" if c is None
                         else f"{c['mae']:.5f} ({c['n']})")
        L.append(f"| {v} | " + " | ".join(cells) + " |")

    L += ["", "## Dense panels vs the rest", "",
          "`" + "`, `".join(DENSE_PANELS) + "` hold the dense microstructure. "
          "Porosity totals there are fixable by calibration; the pore Dice gap "
          "is not, so this is the number a latent rung has to move.", "",
          "| scoring | dense pore Dice | rest pore Dice | gap | dense φ MAE | rest φ MAE |",
          "|---|---|---|---|---|---|"]
    for v, s_ in summ.items():
        g = s_.get("dense_vs_rest", {})
        if "dense" in g and "rest" in g:
            L.append(f"| {v} | {g['dense']['dice_pore']:.4f} "
                     f"| {g['rest']['dice_pore']:.4f} "
                     f"| {g['rest']['dice_pore'] - g['dense']['dice_pore']:+.4f} "
                     f"| {g['dense']['porosity_mae']:.5f} "
                     f"| {g['rest']['porosity_mae']:.5f} |")

    best = min(summ, key=lambda v: summ[v]["worst_bin_mae"])
    L += ["", f"## Per panel — best scoring (`{best}`)", "",
          "| panel | split | " + " | ".join(BIN_LABELS) + " | pore Dice |",
          "|---" * (len(BIN_LABELS) + 3) + "|"]
    for p, c in summ[best]["per_panel"].items():
        cells = [f"{c['bins'][l]:.5f}" if l in c["bins"] else "—"
                 for l in BIN_LABELS]
        L.append(f"| {p} | {c['split']} | " + " | ".join(cells)
                 + f" | {c['dice']:.4f} |")

    verdict = ("CALIBRATION — one fixed rule removes the bias in every bin"
               if summ[best]["passes_every_bin"] else
               "CAPACITY (or not fixable post hoc) — no fixed rule brings "
               "every bin under the gate")
    L += ["", "## Verdict", "", f"**{verdict}.** Best scoring `{best}`: "
          f"worst-bin MAE {summ[best]['worst_bin_mae']:.5f} against a "
          f"{GATE} gate, pore Dice {summ[best]['overall_dice']:.4f} "
          f"(argmax baseline {summ['argmax']['overall_dice']:.4f}).", "",
          "## Caveats", "",
          "- A stratified sample is not the split distribution: it "
          "deliberately over-weights the rare high-porosity bins so they can "
          "be judged at all. Read the per-bin numbers, not the overall MAE.",
          "- `deweight` assumes the network learned the reweighted posterior "
          "exactly. Cross-entropy with class weights does converge to that in "
          "the infinite-data limit; with a Dice term added and finite data it "
          "is an approximation, so a partial improvement is still consistent "
          "with a calibration cause.",
          "- Raising the pore threshold trades recall for precision by "
          "construction, so a threshold that fixes the porosity bias can still "
          "cost Dice. Both are reported.", ""]
    return "\n".join(L)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--checkpoint", default=None,
                    help="checkpoint inside the run dir (default best.ckpt, "
                         "which only exists once the final full eval has run; "
                         "use latest.ckpt to check a gate mid-training)")
    ap.add_argument("--per-bin", type=int, default=96)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--device", default="cuda",
                    help="cuda, or cpu to check a gate while a training run "
                         "holds the GPU — slower, but it cannot OOM the run")
    ap.add_argument("--out-name", default="calibration_probe",
                    help="subdirectory under the campaign root")
    args = ap.parse_args()

    import zarr
    run_dir = Path(args.run)
    cfg = yaml.safe_load((run_dir / "resolved_config.yaml").read_text())
    ckpt = run_dir / (args.checkpoint or "best.ckpt")
    if not ckpt.exists():
        raise SystemExit(f"{ckpt} not found — best.ckpt is only written by the "
                         "final full eval; pass --checkpoint latest.ckpt to "
                         "probe a run that is still training.")
    device = torch.device(args.device)
    model = build_model(cfg, device)
    step, _ = load_checkpoint(str(ckpt), model=model, map_location=device,
                              restore_rng=False)
    model.eval()
    weights = np.asarray(cfg["loss"]["class_weights"], dtype=np.float64)
    log(f"loaded step {step}; training class weights {weights.tolist()}")

    df = pd.read_parquet(INDEX, columns=["split", "volume_id", "panel_id",
                                         "porosity", "z0", "y0", "x0"])
    rows = stratified(df, args.per_bin)
    log(f"stratified sample: {len(rows)} patches over "
        f"{rows.panel_id.nunique()} panels")

    g = zarr.open_group(str(ZARR), mode="r")
    res = probe(model, rows, g, device, weights, args.batch)
    summ = summarise(res, rows)

    meta = {
        "experiment": f"{cfg['experiment']['name']}/{cfg['experiment']['variant']}",
        "run_dir": str(run_dir), "step": step,
        "z_channels": cfg["model"]["z_channels"],
        "class_weights": weights.tolist(),
        "per_bin": args.per_bin, "n_patches": int(len(rows)),
        "n_panels": int(rows.panel_id.nunique()),
        "taus": list(TAUS), "gate": GATE, "device": args.device,
    }
    out_dir = OUT_ROOT / args.out_name
    out_dir.mkdir(parents=True, exist_ok=True)
    write_json({**meta, "summary": summ}, out_dir)
    write_findings(build_findings(meta, summ), out_dir)
    log(f"wrote {out_dir}")
    for v, s in summ.items():
        log(f"  {v:10s} overall {s['overall_mae']:.5f}  worst-bin "
            f"{s['worst_bin_mae']:.5f}  dice {s['overall_dice']:.4f}  "
            f"{'PASS' if s['passes_every_bin'] else 'fail'}")


if __name__ == "__main__":
    main()
