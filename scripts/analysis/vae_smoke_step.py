"""One training step of a VAE experiment, for memory. (GPU.)

A queue slot is hours. Spending one to discover at step 0 that the batch does
not fit is avoidable, and `vrrae/b` at batch 1536 with a 4096-wide SVD is the
configuration most likely not to: it doubles vrrae04's batch and widens the RR
layer more than four-fold.

Runs forward, loss and backward — not just a forward pass, because the backward
is where the SVD's gradient and the optimiser state land, and that is the peak.

EXIT CODES, and they matter: 0 it fits, **1 a real CUDA OOM**, 2 anything else.
The chain uses a config's fallback ONLY on 1. The previous version returned 1
for every failure, so a KeyError in this script itself was read as "does not
fit" and sent a config to its fallback that had never been tested — and that
fallback took the whole host out of memory.

Usage:
    python scripts/analysis/vae_smoke_step.py --experiment vrrae/b
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parents[2]


def synthetic_batch(bs: int, patch: int, device) -> dict:
    """Every key a VAE's forward and its loss can read, at the real shapes.

    Not just the grey. The 3-class model's encoder takes the label, and the
    loss reads the pore mask and the label for its mask and class terms; a
    batch without them measures a step the trainer never takes, and — as it
    turned out — raises before it measures anything at all. They also cost
    memory in the real step, so leaving them out under-states the peak.
    """
    shape = (bs, 1, patch, patch, patch)
    return {
        "xct": torch.rand(*shape, device=device),
        "mask": (torch.rand(*shape, device=device) > 0.97).float(),
        "label": torch.randint(0, 3, (bs, patch, patch, patch), device=device),
    }


def main() -> int:
    from poregen.configuration import resolve_experiment
    from poregen.experiments.train_vae import (
        adversarial_schedule, build_discriminator, build_model, build_optimizer)
    from poregen.losses import build_vae_loss_fn
    from poregen.training.device import get_autocast_dtype, make_scaler
    from poregen.training.engine import train_step

    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--experiment", required=True)
    ap.add_argument("--batch-size", type=int, default=None,
                    help="override; default is the config's own, which is the point")
    args = ap.parse_args()

    cfg = resolve_experiment(args.experiment).cfg
    bs = args.batch_size or cfg["data"]["batch_size"]
    from poregen.training.device import select_device
    dev = select_device()  # applies the POREGEN_CUDA_MEM_FRACTION cap, so an oversize batch raises instead of draining the host
    if dev.type != "cuda":
        print("no CUDA device; a memory smoke test on CPU would mean nothing")
        return 2

    torch.cuda.reset_peak_memory_stats(dev)
    model = build_model(cfg, dev)
    # The trainer's own loss builder: a config with loss.perceptual_weight > 0
    # carries the frozen LPIPS network, and its activations are part of the
    # step's peak.
    loss_fn = build_vae_loss_fn(cfg, dev)
    opt = build_optimizer(cfg, model)
    # The discriminator too, and the step is taken AT disc_start, so the
    # adversarial path (generator term, adaptive weight, discriminator update)
    # is inside the measured peak. Before this, a run learned its GAN-path
    # memory only when it reached disc_start hours later.
    disc, disc_opt, disc_weight = build_discriminator(cfg, dev)
    sched = adversarial_schedule(cfg)
    patch = int(cfg["model"]["patch_size"])
    # Random input at the real shape and the real batch: the allocator does not
    # care what the numbers are, only how many.
    batch = synthetic_batch(bs, patch, dev)

    rank = cfg["model"].get("vrrae_rank")
    if rank is not None and rank > bs:
        print(f"REFUSING: vrrae_rank {rank} exceeds batch {bs}. RRLayer CLIPS the "
              f"effective rank to the batch size, so this would train a narrower "
              f"bottleneck than the config claims and nothing would say so.")
        return 2

    try:
        model.train()
        # The trainer's own step: its input contract (each model declares the
        # batch keys its forward() consumes), precision, gradient clipping and
        # the adversarial path, so the peak is the training peak.
        train_step(
            model, batch, opt, make_scaler(dev), loss_fn,
            step=sched["disc_start"], device=dev,
            autocast_dtype=get_autocast_dtype(dev),
            max_grad_norm=cfg["training"]["max_grad_norm"],
            discriminator=disc, disc_optimizer=disc_opt, disc_weight=disc_weight,
            **sched,
        )
    except torch.cuda.OutOfMemoryError as exc:
        peak = torch.cuda.max_memory_allocated(dev) / 2**30
        print(f"OOM at batch {bs}: peak {peak:.1f} GiB before failing\n  {exc}")
        return 1                      # a REAL out-of-memory: the fallback applies
    except Exception as exc:          # noqa: BLE001
        # Anything else is a fault in this script or the config, NOT evidence
        # about memory, and must not be read as one.
        import traceback

        traceback.print_exc()
        print(f"\nNOT A MEMORY RESULT: {type(exc).__name__}: {exc}")
        return 2

    peak = torch.cuda.max_memory_allocated(dev) / 2**30
    total = torch.cuda.get_device_properties(dev).total_memory / 2**30
    n = sum(p.numel() for p in model.parameters())
    print(f"{args.experiment}: one step at batch {bs} FITS"
          + (f" (with {type(disc).__name__}, at step {sched['disc_start']})" if disc else ""))
    print(f"  parameters      {n/1e6:.2f} M")
    print(f"  peak allocated  {peak:.1f} GiB of {total:.0f} GiB")
    print(f"  headroom        {total - peak:.1f} GiB")
    # This machine shares its memory between host and device, so "it fits" is
    # not the same as "it fits beside everything else". Said, not assumed.
    print("  NOTE: host and device share one pool here; this is the step's own "
          "peak with nothing else running.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
