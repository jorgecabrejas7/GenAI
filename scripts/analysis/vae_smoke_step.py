"""One training step of a VAE experiment, for memory. (GPU.)

A queue slot is hours. Spending one to discover at step 0 that the batch does
not fit is avoidable, and `vrrae/b` at batch 1536 with a 4096-wide SVD is the
configuration most likely not to: it doubles vrrae04's batch and widens the RR
layer more than four-fold.

Runs forward, loss and backward — not just a forward pass, because the backward
is where the SVD's gradient and the optimiser state land, and that is the peak.

Exit 0 if it fits, 1 if it does not. The chain reads that to decide between a
config and its fallback.

Usage:
    python scripts/analysis/vae_smoke_step.py --experiment vrrae/b
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parents[2]


def main() -> int:
    from poregen.configuration import resolve_experiment
    from poregen.experiments.train_vae import build_model
    from poregen.losses import compute_total_loss

    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--experiment", required=True)
    ap.add_argument("--batch-size", type=int, default=None,
                    help="override; default is the config's own, which is the point")
    args = ap.parse_args()

    cfg = resolve_experiment(args.experiment).cfg
    bs = args.batch_size or cfg["data"]["batch_size"]
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if dev.type != "cuda":
        print("no CUDA device; a memory smoke test on CPU would mean nothing")
        return 1

    torch.cuda.reset_peak_memory_stats(dev)
    model = build_model(cfg, dev)
    opt = torch.optim.AdamW(model.parameters(), lr=float(cfg["training"]["lr"]))
    patch = int(cfg["model"]["patch_size"])
    # Random input at the real shape and the real batch: the allocator does not
    # care what the numbers are, only how many.
    batch = {"xct": torch.rand(bs, 1, patch, patch, patch, device=dev)}

    rank = cfg["model"].get("vrrae_rank")
    if rank is not None and rank > bs:
        print(f"REFUSING: vrrae_rank {rank} exceeds batch {bs}. RRLayer CLIPS the "
              f"effective rank to the batch size, so this would train a narrower "
              f"bottleneck than the config claims and nothing would say so.")
        return 1

    try:
        model.train()
        out = model(batch["xct"])
        losses = compute_total_loss(out, batch, 0, cfg["loss"])
        losses["total"].backward()
        opt.step()
        opt.zero_grad(set_to_none=True)
    except torch.cuda.OutOfMemoryError as exc:
        peak = torch.cuda.max_memory_allocated(dev) / 2**30
        print(f"OOM at batch {bs}: peak {peak:.1f} GiB before failing\n  {exc}")
        return 1

    peak = torch.cuda.max_memory_allocated(dev) / 2**30
    total = torch.cuda.get_device_properties(dev).total_memory / 2**30
    n = sum(p.numel() for p in model.parameters())
    print(f"{args.experiment}: one step at batch {bs} FITS")
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
