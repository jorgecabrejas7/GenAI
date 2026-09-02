"""Recompute the VAE validation metrics that the XCT-sigmoid bug corrupted.

Background
----------
The VAE's XCT head regresses ``xct / 255`` directly
(:func:`poregen.losses.total.compute_total_loss` calls
``recon_fn(output.xct_out, batch["xct"])``), so the decoder output already IS
the grey level in [0, 1].  Until 2026-09-01 ``poregen.training.engine._run_eval``
applied ``torch.sigmoid()`` to it before computing the reconstruction eval
metrics.  That squashed [0, 1] into [0.5, 0.731], which

* put a ~0.135 artefact floor under ``mae`` (the mean |sigmoid(x) - x| gap), and
* scaled ``sharpness_recon_over_gt`` by the sigmoid slope (~0.21).

The training loss, the mask metrics, ``porosity_mae`` and the KL/latent metrics
were never affected — the loss always consumed the raw decoder output.

What this script does
---------------------
For every requested run it loads ``best.ckpt``, rebuilds the model and the
validation loader from that run's own ``resolved_config.yaml``, and calls the
real :func:`poregen.training.engine._run_eval` — the same function that wrote
the historical numbers — over a deterministic slice of the validation set.

A probe (:class:`_DualDecodeProbe`) rides along inside that single pass and
computes the **old, buggy** ``mae`` and ``sharpness_recon_over_gt`` from the
same forward pass, using ``torch.sigmoid(output.xct_out)`` exactly as the
pre-fix line ``xct_sigmoid = torch.sigmoid(output.xct_logits)`` did.  Because
both numbers come from one forward pass on one batch, the buggy→fixed delta is
exact — it carries no eval-noise or reparameterisation-noise term.  The buggy
value is also the bridge back to history: it must land on the value recorded in
the run's own ``metrics.jsonl``.

Nothing under ``runs/vae/`` is written.  All output goes to
``runs/campaigns/07-vae-metric-recompute/vae_metric_recompute/``.

Usage
-----
::

    python scripts/analysis/recompute_vae_metrics.py --n-batches 200
    python scripts/analysis/recompute_vae_metrics.py --runs r07-run-0006 --n-batches 400
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import sys
import time
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))

from poregen.experiments.train_vae import build_model  # noqa: E402
from poregen.losses import compute_total_loss  # noqa: E402
from poregen.metrics.recon import sharpness_proxy  # noqa: E402
from poregen.models.vae.base import decode_xct  # noqa: E402
from poregen.training import engine as engine_mod  # noqa: E402
from poregen.training.checkpoint import load_checkpoint  # noqa: E402
from poregen.training.data import build_patch_dataloaders  # noqa: E402
from poregen.training.device import get_autocast_dtype, select_device  # noqa: E402

logger = logging.getLogger("recompute_vae_metrics")

RUNS_DIR = REPO_ROOT / "runs" / "vae"
OUT_DIR = REPO_ROOT / "runs" / "campaigns" / "07-vae-metric-recompute" / "vae_metric_recompute"

# Metrics carried through to the report.  The first two are the ones the bug
# corrupted; the rest are controls that must reproduce their logged values.
AFFECTED_METRICS = ["mae", "sharpness_recon_over_gt"]
CONTROL_METRICS = [
    "xct_loss",
    "porosity_mae",
    "total",
    "kl",
    "porosity_bias",
    "dice_pos_only",
    "mask_bce",
    "mu_std",
    "kl_collapsed_fraction",
    "mu_active_fraction",
]
REPORT_METRICS = AFFECTED_METRICS + CONTROL_METRICS


def parse_run_name(name: str) -> dict[str, Any]:
    """Pull experiment id, run index and latent width out of a run dir name."""
    exp = name.split("-run-")[0]
    idx = re.search(r"-run-(\d+)-", name)
    z = re.search(r"-z(\d+)-", name)
    arch = re.search(r"archv2-([a-z_]+?)(?:-z\d+)?-c\d+-", name)
    return {
        "experiment": exp,
        "run_index": int(idx.group(1)) if idx else -1,
        "z_channels": int(z.group(1)) if z else None,
        "arch": arch.group(1) if arch else None,
        "short": f"{exp}-{idx.group(1)}" if idx else exp,
    }


def logged_val_full(run_dir: Path, best_step: int) -> tuple[dict[str, Any] | None, int | None]:
    """Return the ``val_full`` metrics record matching *best_step*.

    ``train_loop`` writes the full-eval record at step ``S`` and the checkpoint
    saved from it at step ``S`` or ``S + 1`` (see the ``+ (0 if full_eval)``
    offset in ``engine._replay_metrics``), so accept both.
    """
    path = run_dir / "metrics.jsonl"
    if not path.exists():
        return None, None
    records: dict[int, dict[str, Any]] = {}
    for line in path.open():
        line = line.strip()
        if not line:
            continue
        try:
            rec = json.loads(line)
        except json.JSONDecodeError:
            continue  # truncated tail — skip
        if rec.get("split") == "val_full":
            records[int(rec["step"])] = rec
    for candidate in (best_step, best_step - 1, best_step + 1):
        if candidate in records:
            return records[candidate], candidate
    return None, None


class _DualDecodeProbe:
    """Compute the pre-fix reconstruction metrics inside one ``_run_eval`` pass.

    ``_run_eval`` does, per batch and in this order::

        xct_recon = decode_xct(output.xct_out)
        ...
        mae_acc.append(F.l1_loss(xct_recon, xct_dev))
        sharp_recon_acc.append(sharpness_proxy(xct_recon))
        sharp_gt_acc.append(sharpness_proxy(xct_dev))

    So swapping ``engine.decode_xct`` for :meth:`decode` (which stashes the raw
    decoder output and still returns the correct value) and ``engine.F`` for
    this object (which intercepts only ``l1_loss``, where the ground truth
    finally becomes visible) is enough to accumulate the sigmoid-path numbers
    alongside the clamp-path ones.  ``_run_eval`` uses no other ``F.*``
    attribute, and ``compute_total_loss`` holds its own ``F`` import, so the
    loss and every other metric run untouched.

    The buggy ratio is rebuilt with ``_run_eval``'s own formula:
    ``mean(sharp_recon) / mean(sharp_gt)`` — a ratio of means, not a mean of
    ratios.
    """

    def __init__(self) -> None:
        self.mae: list[torch.Tensor] = []
        self.sharp_recon: list[torch.Tensor] = []
        self.sharp_gt: list[torch.Tensor] = []
        self._raw: torch.Tensor | None = None

    def decode(self, xct_out: torch.Tensor) -> torch.Tensor:
        self._raw = xct_out
        return decode_xct(xct_out)

    def l1_loss(self, recon: torch.Tensor, target: torch.Tensor):
        buggy = torch.sigmoid(self._raw)
        self.mae.append(F.l1_loss(buggy, target))
        self.sharp_recon.append(sharpness_proxy(buggy))
        self.sharp_gt.append(sharpness_proxy(target))
        self._raw = None
        return F.l1_loss(recon, target)

    def __getattr__(self, name: str):
        # Everything except l1_loss falls through to torch.nn.functional.
        return getattr(F, name)

    def results(self) -> dict[str, float]:
        sr = float(torch.stack(self.sharp_recon).mean().item())
        sg = float(torch.stack(self.sharp_gt).mean().item())
        return {
            "mae": float(torch.stack(self.mae).mean().item()),
            "sharpness_recon_over_gt": sr / sg if sg > 0.0 else float("nan"),
        }


def eval_pass(
    model: torch.nn.Module,
    val_loader,
    loss_fn,
    *,
    n_batches: int,
    step: int,
    device: torch.device,
    autocast_dtype: torch.dtype,
    seed: int,
) -> tuple[dict[str, float], dict[str, float]]:
    """One ``_run_eval`` pass; returns (fixed metrics, buggy recon metrics).

    Reseeding ``val_loader.generator`` before ``iter()`` fixes the batch order:
    the DataLoader draws its worker base seed and then the sampler permutation
    from that generator, in that order.  Seeding the global RNG fixes the
    reparameterisation noise, so a rerun of this script reproduces itself.
    """
    val_loader.generator.manual_seed(seed + 1)
    torch.manual_seed(seed)
    data_iter = iter(val_loader)

    probe = _DualDecodeProbe()
    orig_decode, orig_F = engine_mod.decode_xct, engine_mod.F
    engine_mod.decode_xct, engine_mod.F = probe.decode, probe
    try:
        agg, _ = engine_mod._run_eval(
            model, data_iter, loss_fn, n_batches, step, device, autocast_dtype,
            desc="eval",
        )
    finally:
        engine_mod.decode_xct, engine_mod.F = orig_decode, orig_F
        del data_iter

    fixed = {k: v for k, v in agg.items() if isinstance(v, (int, float))}
    return fixed, probe.results()


def recompute_run(
    run_dir: Path,
    *,
    n_batches: int,
    device: torch.device,
) -> dict[str, Any]:
    """Load ``best.ckpt`` and re-run validation on both code paths."""
    cfg = yaml.safe_load((run_dir / "resolved_config.yaml").read_text())
    # Worker lifetime only — does not affect any metric.  Turning it off keeps
    # the generator-reseed replay free of persistent-worker state.
    cfg["data"]["persistent_workers"] = False

    ckpt = run_dir / "best.ckpt"
    model = build_model(cfg, device)
    best_step, ckpt_meta = load_checkpoint(
        str(ckpt), model=model, map_location=device, restore_rng=False
    )
    model.eval()

    data_root = REPO_ROOT / "data" / cfg["data"]["dataset_root"]
    _, val_loader, _ = build_patch_dataloaders(cfg, data_root)
    n_batches = min(n_batches, len(val_loader))

    loss_fn = lambda output, batch, step: compute_total_loss(output, batch, step, cfg)  # noqa: E731
    autocast_dtype = get_autocast_dtype(device)
    seed = int(cfg["training"]["seed"])

    t0 = time.time()
    fixed, buggy_recon = eval_pass(
        model, val_loader, loss_fn, n_batches=n_batches, step=best_step,
        device=device, autocast_dtype=autocast_dtype, seed=seed,
    )
    # The bug touched only these two metrics; every other value is identical on
    # both code paths by construction (same forward pass, same loss).
    buggy = {**fixed, **buggy_recon}
    elapsed = time.time() - t0

    logged, logged_step = logged_val_full(run_dir, best_step)

    del model, val_loader
    torch.cuda.empty_cache()

    meta = parse_run_name(run_dir.name)
    return {
        **meta,
        "run": run_dir.name,
        "run_dir": str(run_dir),
        "model_name": cfg["model"]["name"],
        "xct_loss_type": cfg["loss"]["xct_loss_type"],
        "has_mask_head": "porosity_mae" in fixed,
        "best_step": best_step,
        "best_metric": ckpt_meta.get("best_metric"),
        "best_value": ckpt_meta.get("best_value"),
        "n_batches": n_batches,
        "batch_size": int(cfg["data"]["batch_size"]),
        "n_patches": n_batches * int(cfg["data"]["batch_size"]),
        "val_batches_total": None,
        "logged_step": logged_step,
        "logged_n_batches": (logged or {}).get("n_batches"),
        "logged": {k: logged.get(k) for k in REPORT_METRICS} if logged else None,
        "recomputed_fixed": {k: fixed.get(k) for k in REPORT_METRICS},
        "recomputed_buggy": {k: buggy.get(k) for k in REPORT_METRICS},
        "eval_seconds": round(elapsed, 1),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--n-batches", type=int, default=200,
                    help="validation batches per pass (deterministic seeded subset)")
    ap.add_argument("--runs", nargs="*", default=None,
                    help="substring filters; default = every run dir holding a best.ckpt")
    ap.add_argument("--out", type=Path, default=OUT_DIR)
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    candidates = sorted(d for d in RUNS_DIR.iterdir()
                        if d.is_dir() and (d / "best.ckpt").exists()
                        and (d / "resolved_config.yaml").exists())
    if args.runs:
        candidates = [d for d in candidates if any(f in d.name for f in args.runs)]

    skipped = []
    for d in sorted(RUNS_DIR.iterdir()):
        if not d.is_dir() or d in candidates:
            continue
        reason = []
        if not (d / "best.ckpt").exists():
            reason.append("no best.ckpt")
        if not (d / "resolved_config.yaml").exists():
            reason.append("no resolved_config.yaml")
        if reason and (not args.runs or any(f in d.name for f in args.runs)):
            skipped.append({"run": d.name, "reason": ", ".join(reason)})

    device = select_device()
    logger.info("device=%s | %d runs to evaluate | %d batches/pass",
                device, len(candidates), args.n_batches)

    args.out.mkdir(parents=True, exist_ok=True)
    results: list[dict[str, Any]] = []
    failures: list[dict[str, str]] = []

    for i, run_dir in enumerate(candidates, 1):
        logger.info("[%d/%d] %s", i, len(candidates), run_dir.name)
        try:
            results.append(recompute_run(run_dir, n_batches=args.n_batches, device=device))
        except Exception as exc:  # noqa: BLE001 — one bad run must not stop the sweep
            logger.exception("FAILED %s", run_dir.name)
            failures.append({"run": run_dir.name, "error": f"{type(exc).__name__}: {exc}"})
            torch.cuda.empty_cache()

    payload = {
        "generated": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "n_batches_per_pass": args.n_batches,
        "device": str(device),
        "affected_metrics": AFFECTED_METRICS,
        "control_metrics": CONTROL_METRICS,
        "runs": results,
        "skipped": skipped,
        "failures": failures,
    }
    out_json = args.out / "results.json"
    out_json.write_text(json.dumps(payload, indent=2))
    logger.info("wrote %s (%d runs, %d skipped, %d failed)",
                out_json, len(results), len(skipped), len(failures))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
