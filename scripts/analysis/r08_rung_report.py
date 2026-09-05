"""Full-split report for one r08 rung — gates, porosity bins, per-volume, per-panel.

The periodic validation during training sees ``val_batches`` batches, which is
a few thousand patches out of 140 594. That is enough to steer training and not
enough to judge a gate: the high-porosity bins are sparse, and split_v3's val
and test panels (Na_08, Na_05) are low-porosity, so a headline
``porosity_mae`` can pass while a bin nobody looked at does not.

This runs the WHOLE val and test splits through a finished checkpoint and
reports:

* the three training gates — ``porosity_mae`` < 0.005, pore Dice >= 0.88,
  air Dice > 0.98;
* ``porosity_mae`` per ground-truth porosity bin WITH the patch count, so a bin
  is only judged when it holds more than ``--min-bin-n`` patches;
* per-volume porosity MAE, which separates the Juan_Ignacio test volume
  (layup B, 10-voxel ply pitch) from the Nacho panel;
* per-panel rows, since a whole panel is one split in split_v3 and a panel is
  the unit that can be atypical.

Outputs -> ``runs/campaigns/09-r08-latent-sweep/<experiment>/``: results.json
and findings.md. Run it once per rung; ``--compare`` then assembles the
cross-rung table.

Usage
-----
    python scripts/analysis/r08_rung_report.py --run runs/vae/r08-run-0002-...
    python scripts/analysis/r08_rung_report.py --compare
"""

from __future__ import annotations

import argparse
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

from poregen.experiments.train_vae import build_model, resolve_data_root  # noqa: E402
from poregen.metrics.seg import porosity_binned_mae  # noqa: E402
from poregen.models.vae.base import CLASS_AIR, CLASS_PORE  # noqa: E402
from poregen.training.checkpoint import load_checkpoint  # noqa: E402
from poregen.training.data import build_patch_dataloaders  # noqa: E402
from poregen.training.engine import to_device_inputs  # noqa: E402

OUT_ROOT = REPO / "runs/campaigns/09-r08-latent-sweep"
SPLITS_JSON = REPO / "data/split_v3/splits.json"

GATES = {
    "porosity_mae": ("<", 0.005),
    "dice_pore": (">=", 0.88),
    "dice_air": (">", 0.98),
}
BINS = (0.0, 0.01, 0.03, 0.06, float("inf"))
# A single pore-probability threshold, CALIBRATED ON VAL and applied unchanged
# to test. argmax is tau = 0.5 by construction, so the grid includes it and a
# rung is free to come out uncalibrated. The point of fitting on val and never
# refitting on test is that the test number then measures the model plus a
# fixed decision rule, not a rule tuned to the thing being measured.
TAU_GRID = (0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9)
BIN_LABELS = ("phi < 1%", "1-3%", "3-6%", ">= 6%")
MIN_BIN_N = 500
_T0 = time.time()


def log(msg: str) -> None:
    print(f"[{time.time() - _T0:7.1f}s] {msg}", flush=True)


# ---------------------------------------------------------------------------
# Full-split evaluation
# ---------------------------------------------------------------------------

@torch.no_grad()
def evaluate_split(model, loader, device, autocast_dtype=torch.bfloat16) -> dict:
    """Per-patch predicted and true pore/air fractions over a whole split.

    Everything is read off the ARGMAX — the label a generated volume will
    actually carry — so these numbers are what the sampler delivers, not what
    the soft probabilities promise.
    """
    model.eval()
    pred_por, true_por, pred_air, true_air = [], [], [], []
    tau_por: dict[float, list] = {t: [] for t in TAU_GRID}
    dice_acc = {c: [] for c in ("material", "pore", "air")}
    vol_ids: list[str] = []

    n_batches = len(loader)
    for i, batch in enumerate(loader):
        _, model_args = to_device_inputs(model, batch, device)
        label = model_args[-1]
        with torch.autocast(device_type=device.type, dtype=autocast_dtype):
            out = model(*model_args)
        pred = out.class_logits.argmax(dim=1)

        for c, name in enumerate(("material", "pore", "air")):
            p, t = (pred == c), (label == c)
            inter = (p & t).flatten(1).sum(1).float()
            card = p.flatten(1).sum(1).float() + t.flatten(1).sum(1).float()
            d = torch.where(card > 0, 2.0 * inter / card, torch.ones_like(card))
            dice_acc[name].append(d.cpu())

        # One softmax, every threshold read off it — the alternative is a
        # forward pass per tau over 556k patches.
        probs = torch.softmax(out.class_logits.float(), dim=1)
        p_pore = probs[:, CLASS_PORE]
        for t in TAU_GRID:
            tau_por[t].append((p_pore > t).flatten(1).float().mean(1).cpu())
        del probs, p_pore

        pred_por.append((pred == CLASS_PORE).flatten(1).float().mean(1).cpu())
        true_por.append((label == CLASS_PORE).flatten(1).float().mean(1).cpu())
        pred_air.append((pred == CLASS_AIR).flatten(1).float().mean(1).cpu())
        true_air.append((label == CLASS_AIR).flatten(1).float().mean(1).cpu())
        vol_ids.extend(batch["volume_id"])
        if (i + 1) % 200 == 0 or i + 1 == n_batches:
            log(f"    batch {i + 1}/{n_batches}")

    cat = lambda xs: torch.cat(xs).numpy()  # noqa: E731
    return {
        "tau_por": {t: cat(v) for t, v in tau_por.items()},
        "pred_por": cat(pred_por), "true_por": cat(true_por),
        "pred_air": cat(pred_air), "true_air": cat(true_air),
        "dice": {k: cat(v) for k, v in dice_acc.items()},
        "volume_id": np.array(vol_ids),
    }


def summarise(ev: dict, panel_of: dict[str, str], min_bin_n: int) -> dict:
    por_err = ev["pred_por"] - ev["true_por"]
    air_err = ev["pred_air"] - ev["true_air"]
    out = {
        "n_patches": int(por_err.size),
        "porosity_mae": float(np.abs(por_err).mean()),
        "porosity_bias": float(por_err.mean()),
        "air_mae": float(np.abs(air_err).mean()),
        "air_bias": float(air_err.mean()),
        "true_porosity_mean": float(ev["true_por"].mean()),
        "true_air_mean": float(ev["true_air"].mean()),
        **{f"dice_{k}": float(v.mean()) for k, v in ev["dice"].items()},
    }

    binned = porosity_binned_mae(torch.from_numpy(ev["pred_por"]),
                                 torch.from_numpy(ev["true_por"]), bins=BINS)
    out["bins"] = []
    for i, label in enumerate(BIN_LABELS):
        n = int(binned[f"porosity_n_bin_{i}"])
        mae = binned[f"porosity_mae_bin_{i}"]
        out["bins"].append({
            "bin": label, "n": n, "porosity_mae": mae,
            "judged": n > min_bin_n,
            "passes": (bool(mae < GATES["porosity_mae"][1]) if n > min_bin_n
                       else None),
        })

    def _group(keys: np.ndarray) -> list[dict]:
        rows = []
        for g in sorted(set(keys.tolist())):
            sel = keys == g
            rows.append({
                "name": g, "n": int(sel.sum()),
                "porosity_mae": float(np.abs(por_err[sel]).mean()),
                "porosity_bias": float(por_err[sel].mean()),
                "air_mae": float(np.abs(air_err[sel]).mean()),
                "dice_pore": float(ev["dice"]["pore"][sel].mean()),
                "dice_air": float(ev["dice"]["air"][sel].mean()),
                "true_porosity_mean": float(ev["true_por"][sel].mean()),
            })
        return rows

    out["per_volume"] = _group(ev["volume_id"])
    out["per_panel"] = _group(np.array([panel_of.get(v, "?")
                                        for v in ev["volume_id"]]))
    return out


def binned_worst(pred_por, true_por, min_bin_n: int) -> tuple[float, dict]:
    """Worst JUDGED-bin porosity MAE, and the per-bin detail.

    "Judged" means the bin holds more than ``min_bin_n`` patches. The gate is
    per bin, so the worst judged bin is the quantity a threshold should be
    chosen to minimise — not the overall MAE, which the sparse dense bins
    barely move.
    """
    err = np.abs(pred_por - true_por)
    b = pd.cut(true_por, bins=BINS, right=False, labels=BIN_LABELS)
    detail, worst = {}, 0.0
    for lab in BIN_LABELS:
        sel = np.asarray(b == lab)
        n = int(sel.sum())
        m = float(err[sel].mean()) if n else float("nan")
        detail[lab] = {"n": n, "porosity_mae": m, "judged": n > min_bin_n}
        if n > min_bin_n and m > worst:
            worst = m
    return worst, detail


def calibrate_tau(ev_val: dict, min_bin_n: int) -> dict:
    """Pick one tau on VAL by minimising the worst judged-bin porosity MAE.

    Fitted on val and never refitted, so the test number measures the model
    plus a FIXED decision rule rather than a rule tuned to the thing being
    measured. tau = 0.5 is argmax, so a rung that needs no calibration says so.
    """
    rows = []
    for t, pred in ev_val["tau_por"].items():
        worst, _ = binned_worst(pred, ev_val["true_por"], min_bin_n)
        rows.append({"tau": float(t), "worst_bin_mae": worst,
                     "porosity_mae": float(np.abs(pred - ev_val["true_por"]).mean())})
    rows.sort(key=lambda r: r["worst_bin_mae"])
    best = rows[0]
    return {"tau": best["tau"], "selected_on": "val worst judged-bin porosity MAE",
            "worst_bin_mae_val": best["worst_bin_mae"],
            "argmax_worst_bin_mae_val": next(
                r["worst_bin_mae"] for r in rows if r["tau"] == 0.5),
            "grid": sorted(rows, key=lambda r: r["tau"])}


def apply_tau(ev: dict, tau: float, min_bin_n: int) -> dict:
    """Score a split at a FIXED tau."""
    pred = ev["tau_por"][tau]
    worst, detail = binned_worst(pred, ev["true_por"], min_bin_n)
    err = pred - ev["true_por"]
    return {"tau": tau, "porosity_mae": float(np.abs(err).mean()),
            "porosity_bias": float(err.mean()),
            "worst_bin_mae": worst, "bins": detail}


def gate_status(s: dict) -> dict:
    ok = {}
    for k, (op, thr) in GATES.items():
        v = s[k]
        ok[k] = {"value": v, "op": op, "threshold": thr,
                 "passes": bool(v < thr if op == "<" else
                                v >= thr if op == ">=" else v > thr)}
    return ok


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def build_findings(meta: dict, per_split: dict, min_bin_n: int) -> str:
    L = [f"# r08 rung report — {meta['experiment']} (z={meta['z_channels']})", "",
         f"Checkpoint `{meta['checkpoint']}` at step {meta['step']}, "
         f"{meta['n_params']} params. WHOLE val and test splits of "
         f"`{meta['dataset_root']}`, read off the argmax.", "",
         "## Gates", "",
         "| gate | target | val | test | verdict |", "|---|---|---|---|---|"]
    for k, (op, thr) in GATES.items():
        v, t = per_split["val"]["gates"][k], per_split["test"]["gates"][k]
        verdict = "PASS" if v["passes"] else "**FAIL**"
        L.append(f"| {k} | {op} {thr} | {v['value']:.5f} | {t['value']:.5f} "
                 f"| {verdict} (val) |")
    L += ["", "The gate is defined on val; the test column is reported beside "
          "it because split_v3 splits by panel, so val and test are different "
          "panels rather than different samples of one.", ""]

    for split in ("val", "test"):
        s = per_split[split]
        L += [f"## {split} — porosity MAE per ground-truth bin", "",
              f"Mean true porosity {s['true_porosity_mean']:.5f}, "
              f"{s['n_patches']} patches. A bin is judged only when it holds "
              f"more than {min_bin_n} patches.", "",
              "| bin | patches | porosity MAE | judged | passes |",
              "|---|---|---|---|---|"]
        for b in s["bins"]:
            mae = "—" if not np.isfinite(b["porosity_mae"]) else f"{b['porosity_mae']:.5f}"
            p = "—" if b["passes"] is None else ("yes" if b["passes"] else "**no**")
            L.append(f"| {b['bin']} | {b['n']} | {mae} "
                     f"| {'yes' if b['judged'] else 'no'} | {p} |")
        L.append("")

    for split in ("val", "test"):
        L += [f"## {split} — per volume", "",
              "| volume | panel | patches | true φ | porosity MAE | bias "
              "| air MAE | pore Dice | air Dice |", "|---|---|---|---|---|---|---|---|---|"]
        vol_panel = per_split[split]["_volume_panel"]
        for r in per_split[split]["per_volume"]:
            panel = next((p for p, vols in vol_panel.items()
                          if r["name"] in vols), "?")
            L.append(f"| {r['name'].split('probetas')[-1].strip('_')} | {panel} "
                     f"| {r['n']} | {r['true_porosity_mean']:.5f} "
                     f"| {r['porosity_mae']:.5f} | {r['porosity_bias']:+.5f} "
                     f"| {r['air_mae']:.5f} | {r['dice_pore']:.4f} "
                     f"| {r['dice_air']:.4f} |")
        L += ["", f"### {split} — per panel", "",
              "| panel | patches | true φ | porosity MAE | pore Dice | air Dice |",
              "|---|---|---|---|---|---|"]
        for r in per_split[split]["per_panel"]:
            L.append(f"| {r['name']} | {r['n']} | {r['true_porosity_mean']:.5f} "
                     f"| {r['porosity_mae']:.5f} | {r['dice_pore']:.4f} "
                     f"| {r['dice_air']:.4f} |")
        L.append("")

    L += ["## Caveats", "",
          "- Every number is read off `argmax(class_logits)`, not the soft "
          "probabilities, so it is what a decoded volume carries.",
          "- Dice is averaged per patch, and a patch with neither prediction "
          "nor truth for a class scores 1 for it. For the sparse pore class "
          "that makes the mean optimistic on empty patches; the per-bin table "
          "is the honest read.",
          f"- Bins holding {min_bin_n} patches or fewer are reported but not "
          "judged — split_v3's val and test panels are low-porosity, so the "
          "high bins are thin by construction.", ""]
    return "\n".join(L)


def run_one(run_dir: Path, batch_size: int, min_bin_n: int) -> dict:
    cfg = yaml.safe_load((run_dir / "resolved_config.yaml").read_text())
    ckpt = run_dir / "checkpoints" / "best.ckpt"
    if not ckpt.exists():
        ckpt = run_dir / "best.ckpt"
    if not ckpt.exists():
        raise SystemExit(f"no best.ckpt under {run_dir}")

    device = torch.device("cuda")
    model = build_model(cfg, device)
    step, _ = load_checkpoint(str(ckpt), model=model, map_location=device,
                              restore_rng=False)
    model.eval()
    log(f"loaded {ckpt.name} at step {step}")

    cfg["data"].update(num_workers=4, persistent_workers=False, timeout=180,
                       batch_size=batch_size)
    _, val_loader, test_loader = build_patch_dataloaders(
        cfg, resolve_data_root(cfg, REPO))

    splits_meta = json.loads(SPLITS_JSON.read_text())
    panel_of = splits_meta["panel_id"]

    per_split, evs = {}, {}
    for name, loader in (("val", val_loader), ("test", test_loader)):
        log(f"evaluating {name} ({len(loader.dataset)} patches)")
        ev = evaluate_split(model, loader, device)
        evs[name] = ev
        s = summarise(ev, panel_of, min_bin_n)
        s["gates"] = gate_status(s)
        s["_volume_panel"] = {}
        for v in set(ev["volume_id"].tolist()):
            s["_volume_panel"].setdefault(panel_of.get(v, "?"), []).append(v)
        per_split[name] = s
        log(f"  {name}: porosity_mae {s['porosity_mae']:.5f} "
            f"dice_pore {s['dice_pore']:.4f} dice_air {s['dice_air']:.4f}")

    # One tau, fitted on val, applied unchanged to test.
    cal = calibrate_tau(evs["val"], min_bin_n)
    for name in ("val", "test"):
        per_split[name]["tau_calibrated"] = apply_tau(evs[name], cal["tau"], min_bin_n)
    log(f"tau calibrated on val = {cal['tau']:.2f} "
        f"(worst judged-bin MAE {cal['argmax_worst_bin_mae_val']:.5f} at argmax "
        f"-> {cal['worst_bin_mae_val']:.5f})")

    meta = {
        "experiment": f"{cfg['experiment']['name']}/{cfg['experiment']['variant']}",
        "run_dir": str(run_dir), "checkpoint": str(ckpt), "step": step,
        "z_channels": cfg["model"]["z_channels"],
        "model": cfg["model"]["name"],
        "n_params": sum(p.numel() for p in model.parameters()),
        "dataset_root": cfg["data"]["dataset_root"],
        "min_bin_n": min_bin_n,
    }
    out_dir = OUT_ROOT / meta["experiment"].replace("/", "_")
    out_dir.mkdir(parents=True, exist_ok=True)
    write_json({**meta, "calibration": cal, "splits": per_split}, out_dir)
    write_findings(build_findings(meta, per_split, min_bin_n), out_dir)
    log(f"wrote {out_dir}")
    return {**meta, "splits": per_split}


def _last(rows, split):
    r = [x for x in rows if x.get("split") == split]
    return r[-1] if r else None


def gather_rung(run_dir: Path) -> dict | None:
    """Everything known about one rung, from artefacts that already exist.

    The full-split rung report is NOT required. Running it for four rungs costs
    ~4.7 h of GPU and duplicates what the training run already computed: the
    engine's own final val_full/test_full are whole-split evaluations with the
    per-bin table and counts. This reads those, plus the CPU calibration probe
    (dense/rest Dice and tau), the tile-seam and the latent-sanity summary.
    """
    cfg_p = run_dir / "resolved_config.yaml"
    met_p = run_dir / "metrics.jsonl"
    if not cfg_p.exists() or not met_p.exists():
        return None
    cfg = yaml.safe_load(cfg_p.read_text())
    variant = cfg["experiment"]["variant"]
    exp = f"{cfg['experiment']['name']}/{variant}"
    rows = [json.loads(l) for l in open(met_p)]
    vf, tf = _last(rows, "val_full"), _last(rows, "test_full")
    if vf is None:
        return None

    ev = [x for x in rows if x.get("split") == "event"]
    wall_h = (ev[-1].get("elapsed", 0) / 3600.0) if ev else float("nan")
    stopped = ev[-1].get("event") if ev else None

    # active latent channels, from the last periodic val
    lv = _last(rows, "val") or {}
    active = lv.get("mu_active_fraction")
    n_active = lv.get("mu_n_active")

    key = exp.replace("/", "_")
    out = {"experiment": exp, "variant": variant, "run_dir": str(run_dir),
           "z_channels": cfg["model"]["z_channels"], "step": vf.get("step"),
           "wall_h": wall_h, "stopped": stopped,
           "mu_active_fraction": active, "mu_n_active": n_active,
           "val_full": vf, "test_full": tf}

    probe_p = OUT_ROOT / f"calibration_probe_{key}" / "results.json"
    if probe_p.exists():
        summ = json.loads(probe_p.read_text())["summary"]
        # Same rule the full report uses, on the probe's stratified sample:
        # the tau whose WORST judged bin is smallest. tau0.5 is argmax.
        best = min((k for k in summ if k.startswith("tau") or k == "argmax"),
                   key=lambda k: summ[k]["worst_bin_mae"])
        out["probe"] = {
            "tau": 0.5 if best == "argmax" else float(best[3:]),
            "tau_worst_bin": summ[best]["worst_bin_mae"],
            "argmax_worst_bin": summ["argmax"]["worst_bin_mae"],
            "dense_dice": summ["argmax"]["dense_vs_rest"]["dense"]["dice_pore"],
            "rest_dice": summ["argmax"]["dense_vs_rest"]["rest"]["dice_pore"],
            "dense_mae": summ["argmax"]["dense_vs_rest"]["dense"]["porosity_mae"],
        }

    seam_p = OUT_ROOT / key / "tile_seam" / "results.json"
    if seam_p.exists():
        recs = json.loads(seam_p.read_text())["records"]
        ov = [a for r in recs for a in r["assemblies"] if a["assembly"] == "B_overlapped"]
        if ov:
            out["seam"] = {
                "xct": float(np.mean([a["seam_xct_ratio"] for a in ov])),
                "pore_logit": float(np.mean([a["seam_mask_ratio"] for a in ov])),
                "overlapped_pore_dice": float(np.mean([a["dice_pore"] for a in ov])),
            }

    san = sorted((REPO / "runs/diagnostics/mask_sanity").glob(f"{run_dir.name}/summary.json"))
    if san:
        sets = json.loads(san[0].read_text())["sets"]
        out["drift_2x"] = sets.get("noise_2.0x_post_std", {}).get("porosity_drift_vs_real_mu")
    return out


def compare() -> None:
    rungs = []
    for d in sorted((REPO / "runs" / "vae").glob("r08-run-*")):
        g = gather_rung(d)
        if g:
            rungs.append(g)
    if not rungs:
        raise SystemExit("no finished r08 rungs found under runs/vae/")
    # Keep the newest run per variant, then order by descending z.
    best: dict[str, dict] = {}
    for g in rungs:
        best[g["variant"]] = g
    rungs = sorted(best.values(), key=lambda r: -r["z_channels"])

    def f(x, n=5):
        return "—" if x is None or (isinstance(x, float) and not np.isfinite(x)) else f"{x:.{n}f}"

    L = ["# r08 latent-compression sweep — decision table", "",
         f"{len(rungs)} rung(s). Built from artefacts that already exist: the "
         "training run's own final `val_full` / `test_full` (whole-split, with "
         "the per-bin table and counts), the CPU calibration probe, the "
         "tile-seam and the latent-sanity check. The separate full-split rung "
         "report is deferred — it costs ~70 min of GPU per rung and recomputes "
         "what the engine already produced.", "",
         "## Gates", "",
         "| rung | z | red. | step | wall h | val φ MAE | pore Dice val/test "
         "| air Dice val/test | active z | drift 2σ |",
         "|---|---|---|---|---|---|---|---|---|---|"]
    for r in rungs:
        v, t = r["val_full"], (r["test_full"] or {})
        red = 64 ** 3 / (r["z_channels"] * 16 ** 3)
        L.append(
            f"| {r['experiment']} | {r['z_channels']} | {red:.0f}x | {r['step']} "
            f"| {r['wall_h']:.1f} | {f(v.get('porosity_mae'))} "
            f"| {f(v.get('dice_pore'),4)} / {f(t.get('dice_pore'),4)} "
            f"| {f(v.get('dice_air'),4)} / {f(t.get('dice_air'),4)} "
            f"| {r.get('mu_n_active','—')} | {f(r.get('drift_2x'),3)} |")

    L += ["", "## Per-bin porosity MAE (whole split, counts in the run's own log)", "",
          "| rung | val <1% | val 1-3% | val 3-6% | val >=6% | test <1% | test 1-3% "
          "| test 3-6% | test >=6% |", "|---|---|---|---|---|---|---|---|---|"]
    for r in rungs:
        v, t = r["val_full"], (r["test_full"] or {})
        cells = [f(v.get(f"porosity_mae_bin_{i}")) for i in range(4)] + \
                [f(t.get(f"porosity_mae_bin_{i}")) for i in range(4)]
        L.append(f"| {r['experiment']} | " + " | ".join(cells) + " |")

    L += ["", "## Dense panels, tau, and assembly", "",
          "`Na_10`, `Na_09`, `Pegaso_1` hold the dense microstructure — the gap "
          "no decision rule closes. tau is chosen on the probe's stratified "
          "sample by minimising the worst judged bin; tau 0.50 means argmax "
          "needed no correction.", "",
          "| rung | DENSE pore Dice | rest | gap | tau | worst bin @tau | @argmax "
          "| seam grey | seam pore-logit | overlapped pore Dice |",
          "|---|---|---|---|---|---|---|---|---|---|"]
    for r in rungs:
        p_, s_ = r.get("probe"), r.get("seam")
        gap = f(p_["rest_dice"] - p_["dense_dice"], 4) if p_ else "—"
        L.append(
            f"| {r['experiment']} "
            f"| {f(p_['dense_dice'],4) if p_ else '—'} | {f(p_['rest_dice'],4) if p_ else '—'} | {gap} "
            f"| {f(p_['tau'],2) if p_ else '—'} | {f(p_['tau_worst_bin']) if p_ else '—'} "
            f"| {f(p_['argmax_worst_bin']) if p_ else '—'} "
            f"| {f(s_['xct'],3) if s_ else '—'} | {f(s_['pore_logit'],3) if s_ else '—'} "
            f"| {f(s_['overlapped_pore_dice'],4) if s_ else '—'} |")

    L += ["", "## Reading it", "",
          "- Gates: val φ MAE < 0.005, pore Dice >= 0.88, air Dice > 0.98, "
          "overlapped seams <= 1.1, drift < 0.30.",
          "- The per-bin gate applies after tau. A rung whose dense bins only "
          "pass at a high tau is buying accuracy with recall — read the "
          "overlapped pore Dice beside it.",
          "- Dense-panel pore Dice is the number capacity is supposed to move; "
          "no threshold moves it.",
          "- A blank cell means that artefact has not been produced for that "
          "rung, NOT that the value is zero.", ""]
    write_findings("\n".join(L), OUT_ROOT)
    write_json({"rungs": rungs}, OUT_ROOT, name="decision_table.json")
    print("\n".join(L))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", help="run directory under runs/vae/")
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--min-bin-n", type=int, default=MIN_BIN_N)
    ap.add_argument("--compare", action="store_true")
    args = ap.parse_args()
    if args.compare:
        compare()
        return
    if not args.run:
        raise SystemExit("--run or --compare required")
    run_one(Path(args.run), args.batch_size, args.min_bin_n)


if __name__ == "__main__":
    main()
