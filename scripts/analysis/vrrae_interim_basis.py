#!/usr/bin/env python
"""Give a mid-training VRRAE checkpoint a usable U_f, from a small sample. (CPU.)

WHY THIS EXISTS. Training derives U_f in a full training-set pass AFTER the
last step, so a checkpoint taken mid-run carries no ``inference_basis`` and
both ``vae_val_l1.py`` and ``vae_recon_figure.py`` refuse it outright: in eval
mode the RR layer PROJECTS onto that basis, so a model without one is a
different model. There is no way to look at a run in progress without building
one.

WHAT THIS IS NOT. The basis here comes from a few dozen training patches, not
from the training set. It is enough to LOOK at reconstructions and nowhere near
enough to put a number in a table — the filename and the figure title say so,
and nothing produced this way belongs in campaign 28.

Usage:
    python scripts/analysis/vrrae_interim_basis.py \\
        --run runs/vae/vrrae-run-0007-... --ckpt <name>.ckpt --n-patches 64 \\
        --out /tmp/conv_k8_step10k
"""
from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import torch
import yaml
from torch.utils.data import DataLoader, Subset

REPO = Path(__file__).resolve().parents[2]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", type=Path, required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--n-patches", type=int, default=64)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()

    from poregen.experiments.train_vae import build_model
    from poregen.training import build_patch_dataloaders

    run = a.run if a.run.is_absolute() else REPO / a.run
    cfg = yaml.safe_load((run / "resolved_config.yaml").read_text())
    split = cfg["data"]["dataset_root"]
    cfg["data"].update(batch_size=a.batch, num_workers=0, timeout=0,
                       persistent_workers=False, prefetch_factor=None)

    dev = torch.device("cpu")
    model = build_model(cfg, dev)
    state = torch.load(run / a.ckpt, map_location="cpu", weights_only=False)["model"]
    step = torch.load(run / a.ckpt, map_location="cpu", weights_only=False).get("step")
    state.pop("bottleneck.rr.inference_basis", None)
    missing, unexpected = model.load_state_dict(state, strict=False)
    if unexpected:
        raise RuntimeError(f"unexpected keys {list(unexpected)}")

    # TRAIN patches, deliberately: U_f is a property of the training
    # distribution, and taking it from the held-out patches the figure is drawn
    # on would let the model see them before it reconstructs them.
    train, _, _ = build_patch_dataloaders(cfg, REPO / "data" / split)
    ds = train.dataset
    stride = max(1, len(ds) // a.n_patches)
    idx = list(range(0, len(ds), stride))[: a.n_patches]
    loader = DataLoader(Subset(ds, idx), batch_size=a.batch, num_workers=0)

    info = model.finalize_inference_basis(loader, device=dev,
                                          autocast_dtype=torch.float32)
    basis = model.bottleneck.rr.inference_basis
    print(f"U_f {tuple(basis.shape)} from {info['n_samples']} samples "
          f"in {info['n_batches']} batches of {len(idx)} {split} train patches")

    # A self-contained run directory, so the figure and L1 harnesses can read it
    # exactly as they read a finished run — and so nothing is written into the
    # live run directory of a training job.
    out = a.out if a.out.is_absolute() else REPO / a.out
    out.mkdir(parents=True, exist_ok=True)
    shutil.copy(run / "resolved_config.yaml", out / "resolved_config.yaml")
    sd = model.state_dict()
    sd["bottleneck.rr.inference_basis"] = basis
    torch.save({"model": sd, "step": step,
                "interim_basis_n_patches": len(idx)}, out / "best.ckpt")
    print(f"wrote {out}/best.ckpt at step {step}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
