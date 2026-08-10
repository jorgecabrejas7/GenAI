"""Standalone GPU-memory profiler for the ConvVAE3DNoAttnDualBranchV2 training step.

Phase-2 (VRRAE batch/latent sizing) memory profiling.  Drives the *real*
``poregen.training.engine.train_step`` (identical forward / adversarial /
backward / optimizer path used in production) with synthetic batches of the
correct shape.  GPU memory depends only on tensor shapes & dtypes, not values,
so synthetic data gives a faithful peak-memory measurement without needing the
Zarr dataset or DataLoader workers.

Each invocation profiles a SINGLE (batch_size, discriminator) config and prints
one JSON line, so the caller can run each config in a clean subprocess (no
cross-config allocator fragmentation).

Usage
-----
    python scripts/profile_vae_memory.py --batch-size 128 --disc on
    python scripts/profile_vae_memory.py --batch-size 256 --disc off --compile
"""

from __future__ import annotations

import argparse
import json
import sys

import torch

from poregen.models.vae import build_vae
from poregen.models.discriminator import PatchDiscriminator2D
from poregen.losses import compute_total_loss
from poregen.training import get_autocast_dtype, make_scaler, select_device
from poregen.training.engine import train_step

# Model configs = resolved production experiments.
MODEL_KWS = {
    "dualbranch": dict(
        name="v2.conv_noattn_dualbranch",
        in_channels=1, z_channels=16, base_channels=32, n_blocks=2, patch_size=64,
    ),
    "vrrae": dict(
        name="v2.vrrae",
        in_channels=1, z_channels=16, base_channels=32, n_blocks=5, patch_size=64,
        vrrae_dim=2048, vrrae_rank=300, vrrae_basis_history_size=20,
    ),
}
LOSS_CFGS = {
    "dualbranch": {
        "loss": {
            "xct_loss_type": "charbonnier",
            "xct_weight": 1.0,
            "mask_bce_weight": 1.0,
            "mask_bce_pos_weight": 1.0,
            "mask_dice_weight": 1.0,
            "use_tversky": True,
            "tversky_alpha": 0.5,
            "tversky_beta": 0.5,
            "kl_free_bits": 0.1,
            "kl_warmup_steps": 0,
            "kl_max_beta": 0.05,
            "use_focal": True,
            "focal_gamma": 2.0,
            "focal_alpha": 0.25,
        }
    },
    "vrrae": {
        "loss": {
            "xct_loss_type": "charbonnier",
            "xct_weight": 1.0,
            "kl_free_bits": 0.1,
            "kl_warmup_steps": 0,
            "kl_max_beta": 0.05,
        }
    },
}


def make_batch(bs: int, device: torch.device) -> dict[str, torch.Tensor]:
    """Synthetic training batch: 64^3 XCT in [0,1] and a {0,1} pore mask."""
    xct = torch.rand(bs, 1, 64, 64, 64, device=device)
    # ~5 % porosity, matching CFRP microstructure sparsity.
    mask = (torch.rand(bs, 1, 64, 64, 64, device=device) < 0.05).float()
    return {"xct": xct, "mask": mask}


def profile(model_key: str, batch_size: int, disc_on: bool, use_compile: bool,
            warmup: int, measure: int) -> dict:
    device = select_device(None)
    autocast_dtype = get_autocast_dtype(device)
    scaler = make_scaler(device)

    model = build_vae(**MODEL_KWS[model_key]).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=2.0e-4, weight_decay=0.01)

    discriminator = disc_optimizer = None
    disc_weight = 0.0
    if disc_on:
        discriminator = PatchDiscriminator2D(in_channels=1, base_channels=64).to(device)
        disc_optimizer = torch.optim.AdamW(
            discriminator.parameters(), lr=2.0e-4, weight_decay=0.01, betas=(0.5, 0.999)
        )
        disc_weight = 0.05

    loss_cfg = LOSS_CFGS[model_key]
    loss_fn = lambda output, batch, step: compute_total_loss(output, batch, step, loss_cfg)

    if use_compile:
        model = torch.compile(model, mode="max-autotune", dynamic=False)
        if discriminator is not None:
            discriminator = torch.compile(discriminator, mode="max-autotune", dynamic=False)
        loss_fn = torch.compile(loss_fn, mode="max-autotune")

    n_params = sum(p.numel() for p in model.parameters())
    n_disc = sum(p.numel() for p in discriminator.parameters()) if discriminator else 0

    batch = make_batch(batch_size, device)

    def one_step(step: int) -> None:
        train_step(
            model, batch, optimizer, scaler, loss_fn,
            step=step, device=device, autocast_dtype=autocast_dtype,
            max_grad_norm=None, scheduler=None,
            discriminator=discriminator, disc_optimizer=disc_optimizer,
            disc_weight=disc_weight,
        )

    # Warmup — triggers compile, lazy Adam-state allocation, allocator growth.
    for s in range(warmup):
        one_step(s)
    torch.cuda.synchronize()

    # Steady-state peak: reset stats after warmup, then measure.
    torch.cuda.reset_peak_memory_stats()
    for s in range(warmup, warmup + measure):
        one_step(s)
    torch.cuda.synchronize()

    return {
        "model": model_key,
        "batch_size": batch_size,
        "disc": "on" if disc_on else "off",
        "compile": use_compile,
        "autocast_dtype": str(autocast_dtype),
        "vae_params": n_params,
        "disc_params": n_disc,
        "peak_alloc_gb": round(torch.cuda.max_memory_allocated() / 1e9, 3),
        "peak_reserved_gb": round(torch.cuda.max_memory_reserved() / 1e9, 3),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=list(MODEL_KWS), default="dualbranch")
    ap.add_argument("--batch-size", type=int, required=True)
    ap.add_argument("--disc", choices=["on", "off"], default="off")
    ap.add_argument("--compile", action="store_true")
    ap.add_argument("--warmup", type=int, default=6)
    ap.add_argument("--measure", type=int, default=15)
    args = ap.parse_args()

    try:
        result = profile(
            args.model, args.batch_size, args.disc == "on", args.compile,
            args.warmup, args.measure,
        )
        result["status"] = "ok"
    except torch.cuda.OutOfMemoryError as exc:  # type: ignore[attr-defined]
        result = {
            "model": args.model, "batch_size": args.batch_size, "disc": args.disc,
            "compile": args.compile, "status": "OOM", "error": str(exc)[:200],
        }
    print("RESULT_JSON " + json.dumps(result))
    sys.stdout.flush()


if __name__ == "__main__":
    main()
