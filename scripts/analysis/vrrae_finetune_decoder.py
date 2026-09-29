#!/usr/bin/env python
"""Mounayer 2024 §4.2's last stage: a decoder-only fine-tune on the fixed basis.

WHY THIS EXISTS. An RR model trains with a PER-BATCH SVD and is deployed with a
single frozen U_f. Those are not the same model, and this family has been
measured without the step the method's own paper puts between them: freeze the
encoder and the bottleneck, project on U_f, and train the decoder for a few
hundred steps against that projection.

The cost of skipping it is not hypothetical. conv_k8 at beta 1e-5 scored, on
the SAME held-out patches with the fixed basis, 0.0184 L1 at its last
checkpoint and 0.0393 at the one its own early stopping chose — and on its own
selection data, 80.9 % of the baseline error removed against 12.8 %. The basis
suits the late weights and not the earlier ones. If the fine-tune is what
closes that, the whole family's numbers are a stage short.

The mechanism is ``finetune_decoder_on_fixed_basis``, which already freezes the
encoder and the bottleneck, builds the optimizer over decoder parameters only,
and keeps BatchNorm frozen everywhere but the decoder. This is its driver: it
loads a checkpoint that already carries U_f, runs the stage, validates before
and after, and writes a NEW run directory rather than touching the original.

Usage
-----
    python scripts/analysis/vrrae_finetune_decoder.py \\
        --run runs/vae/vrrae-run-0008-... --ckpt best.ckpt --steps 500
"""
from __future__ import annotations

import argparse
import json
import shutil
import time
from pathlib import Path

import torch
import yaml

REPO = Path(__file__).resolve().parents[2]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", type=Path, required=True)
    ap.add_argument("--ckpt", default="best.ckpt")
    ap.add_argument("--steps", type=int, default=500)
    ap.add_argument("--lr", type=float, default=None,
                    help="default: the run's own learning rate")
    ap.add_argument("--out", type=Path, default=None,
                    help="default: <run>_ft_<ckpt stem>")
    a = ap.parse_args()

    from poregen.experiments.train_vae import build_model
    from poregen.losses.total import compute_total_loss
    from poregen.models.vae.v2.vrrae_finetune import (
        finetune_decoder_on_fixed_basis, load_vrrae_state_dict,
    )
    from poregen.training import build_patch_dataloaders

    run = a.run if a.run.is_absolute() else REPO / a.run
    cfg = yaml.safe_load((run / "resolved_config.yaml").read_text())
    lr = a.lr if a.lr is not None else float(cfg["training"]["lr"])
    split = cfg["data"]["dataset_root"]
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    payload = torch.load(run / a.ckpt, map_location="cpu", weights_only=False)
    state = payload["model"]
    if not any(k.endswith("inference_basis") for k in state):
        raise RuntimeError(
            f"{run.name}/{a.ckpt} carries no inference_basis. The fine-tune "
            "projects on U_f, so a checkpoint without one cannot be its input; "
            "run scripts/analysis/vrrae_stamp_best.py first.")

    model = build_model(cfg, dev)
    load_vrrae_state_dict(model, state, strict=False)
    model.to(dev)

    train, val, _ = build_patch_dataloaders(cfg, REPO / "data" / split)

    def loss_fn(out, batch, step):
        return compute_total_loss(out, batch, step, cfg)

    @torch.no_grad()
    def validate(n_batches: int = 20) -> float:
        model.eval()
        tot, n = 0.0, 0
        it = iter(val)
        for _ in range(n_batches):
            try:
                b = next(it)
            except StopIteration:
                break
            b = {k: (v.to(dev) if torch.is_tensor(v) else v) for k, v in b.items()}
            out = model(b["xct"], b.get("mask"))
            tot += float(compute_total_loss(out, b, 0, cfg)["xct_loss"])
            n += 1
        return tot / max(n, 1)

    torch.manual_seed(0)
    before = validate()
    print(f"run          {run.name}")
    print(f"checkpoint   {a.ckpt}  step {payload.get('step')}")
    print(f"val xct BEFORE the fine-tune: {before:.4f}", flush=True)

    t0 = time.time()
    losses = finetune_decoder_on_fixed_basis(
        model, iter(train), loss_fn, steps=a.steps, lr=lr, device=dev)
    dt = time.time() - t0

    torch.manual_seed(0)
    after = validate()
    print(f"val xct AFTER  the fine-tune: {after:.4f}   "
          f"({(before - after) / before:+.1%})")
    print(f"train loss {losses[0]:.4f} -> {losses[-1]:.4f} over {a.steps} steps "
          f"in {dt:.0f}s at lr {lr:g}")

    # A NEW RUN DIRECTORY. The fine-tuned decoder is a different model from the
    # one the checkpoint holds, and overwriting it would make the before/after
    # comparison impossible to repeat.
    out = a.out or run.parent / f"{run.name}__ft_{Path(a.ckpt).stem}"
    out.mkdir(parents=True, exist_ok=True)
    shutil.copy(run / "resolved_config.yaml", out / "resolved_config.yaml")
    sd = model.state_dict()
    torch.save({"model": sd, "step": payload.get("step"),
                "finetune": {"source": str(run / a.ckpt), "steps": a.steps,
                             "lr": lr, "val_xct_before": before,
                             "val_xct_after": after}},
               out / "best.ckpt")
    (out / "finetune.json").write_text(json.dumps(
        {"source_run": run.name, "source_ckpt": a.ckpt,
         "source_step": payload.get("step"), "steps": a.steps, "lr": lr,
         "seconds": dt, "val_xct_before": before, "val_xct_after": after,
         "train_loss_first": losses[0], "train_loss_last": losses[-1]},
        indent=2) + "\n")
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
