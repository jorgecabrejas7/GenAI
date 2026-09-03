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

        pred_por.append((pred == CLASS_PORE).flatten(1).float().mean(1).cpu())
        true_por.append((label == CLASS_PORE).flatten(1).float().mean(1).cpu())
        pred_air.append((pred == CLASS_AIR).flatten(1).float().mean(1).cpu())
        true_air.append((label == CLASS_AIR).flatten(1).float().mean(1).cpu())
        vol_ids.extend(batch["volume_id"])
        if (i + 1) % 200 == 0 or i + 1 == n_batches:
            log(f"    batch {i + 1}/{n_batches}")

    cat = lambda xs: torch.cat(xs).numpy()  # noqa: E731
    return {
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

    per_split = {}
    for name, loader in (("val", val_loader), ("test", test_loader)):
        log(f"evaluating {name} ({len(loader.dataset)} patches)")
        ev = evaluate_split(model, loader, device)
        s = summarise(ev, panel_of, min_bin_n)
        s["gates"] = gate_status(s)
        s["_volume_panel"] = {}
        for v in set(ev["volume_id"].tolist()):
            s["_volume_panel"].setdefault(panel_of.get(v, "?"), []).append(v)
        per_split[name] = s
        log(f"  {name}: porosity_mae {s['porosity_mae']:.5f} "
            f"dice_pore {s['dice_pore']:.4f} dice_air {s['dice_air']:.4f}")

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
    write_json({**meta, "splits": per_split}, out_dir)
    write_findings(build_findings(meta, per_split, min_bin_n), out_dir)
    log(f"wrote {out_dir}")
    return {**meta, "splits": per_split}


def compare() -> None:
    rows = []
    for d in sorted(OUT_ROOT.glob("r08_*")):
        f = d / "results.json"
        if f.exists():
            rows.append(json.loads(f.read_text()))
    if not rows:
        raise SystemExit(f"no rung reports under {OUT_ROOT}")
    rows.sort(key=lambda r: -r["z_channels"])
    def dense_cols(r):
        """Pore Dice on the dense panels, from that rung's calibration probe.

        Na_10 and Pegaso_1 are TRAIN panels, so the val/test report above
        cannot see them; the probe samples all 17. Blank when the probe has
        not been run for a rung.
        """
        f = OUT_ROOT / f"calibration_probe_{r['experiment'].replace('/', '_')}" / "results.json"
        if not f.exists():
            f = OUT_ROOT / "calibration_probe" / "results.json"
        if not f.exists():
            return "—", "—", "—"
        try:
            g = json.loads(f.read_text())["summary"]["argmax"]["dense_vs_rest"]
            return (f"{g['dense']['dice_pore']:.4f}", f"{g['rest']['dice_pore']:.4f}",
                    f"{g['rest']['dice_pore'] - g['dense']['dice_pore']:+.4f}")
        except (KeyError, ValueError):
            return "—", "—", "—"

    L = ["# r08 latent-compression sweep — comparison", "",
         f"{len(rows)} rung(s), whole val split. Reduction = 64³ / (z · 16³).", "",
         "The last three columns are the point of the sweep. `Na_10`, `Na_09` "
         "and `Pegaso_1` hold the dense microstructure, and after the class "
         "weights were tempered they are the only place a residual survives: "
         "porosity totals there are fine, pore Dice is not. Calibration cannot "
         "close that gap — more latent capacity is the only lever left, so the "
         "dense-panel Dice is what a rung has to move. They come from each "
         "rung's calibration probe, which samples all 17 panels; the val/test "
         "columns cannot see Na_10 or Pegaso_1 because those are train.", "",
         "| rung | z | reduction | step | porosity_mae | pore Dice | air Dice "
         "| material Dice | air_mae | params | DENSE pore Dice | rest | gap |",
         "|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in rows:
        v = r["splits"]["val"]
        red = 64 ** 3 / (r["z_channels"] * 16 ** 3)
        dense, rest, gap = dense_cols(r)
        L.append(f"| {r['experiment']} | {r['z_channels']} | {red:.0f}x "
                 f"| {r['step']} | {v['porosity_mae']:.5f} | {v['dice_pore']:.4f} "
                 f"| {v['dice_air']:.4f} | {v['dice_material']:.4f} "
                 f"| {v['air_mae']:.5f} | {r['n_params']} "
                 f"| {dense} | {rest} | {gap} |")
    L += ["", "Selection rule (D31/D33): among rungs with val porosity_mae < "
          "0.005 and air Dice > 0.98, take the SMALLEST z whose pore Dice is "
          "within 0.01 of the best and whose overlapped-decode seam ratio is "
          "<= 1.1; ties go to z=4.", ""]
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    write_findings("\n".join(L), OUT_ROOT)
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
