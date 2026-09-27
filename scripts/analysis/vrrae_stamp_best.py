#!/usr/bin/env python
"""Give a finished RR run's ``best.ckpt`` an inference basis. (GPU, one pass.)

Training derives U_f after the last step, so only the final checkpoint carries
one and every eval harness refuses best.ckpt. Runs trained after e637495 stamp
it themselves; this is the repair for the ones that finished before that, and
it does exactly what the trainer now does — a REFIT on the best weights, never
a copy of the final basis, because eval-mode projection through another
encoder's basis reports a model that existed at no step of training.

It matters when best and final differ. conv_k8 at beta 1e-5 early-stopped at
17 538 with its best validation at 14 239, so its table row currently comes
from a model 3 299 steps past the one that was chosen.

Usage:
    POREGEN_CUDA_MEM_FRACTION=0.8 choom -n 1000 -- \\
        python scripts/analysis/vrrae_stamp_best.py --run runs/vae/vrrae-run-0008-...
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import torch
import yaml

REPO = Path(__file__).resolve().parents[2]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", type=Path, required=True)
    ap.add_argument("--batch-size", type=int, default=None,
                    help="override the run's own batch size for the pass")
    # THE PASS IS DATALOADER-BOUND, NOT GPU-BOUND. The encoder forward for a
    # 0.5 M-parameter model is trivial beside loading 128 64-cubed patches, so
    # at the run's own 2 workers the card sits at 1 % and 14 239 batches take
    # about four hours. The accumulation is order-independent — a sum of
    # per-batch projectors — so widening the loader changes nothing but the
    # wall clock.
    ap.add_argument("--num-workers", type=int, default=8,
                    help="upper bound; the shm budget may lower it")
    ap.add_argument("--shm-budget-gib", type=float, default=24.0,
                    help="shared memory the loader may hold in flight")
    a = ap.parse_args()

    from poregen.experiments.train_vae import build_model
    from poregen.models.vae.v2.vrrae_finetune import finalize_best_checkpoint
    from poregen.training import build_patch_dataloaders

    run = a.run if a.run.is_absolute() else REPO / a.run
    cfg = yaml.safe_load((run / "resolved_config.yaml").read_text())
    if a.batch_size:
        cfg["data"]["batch_size"] = a.batch_size
    # THE LOADER IS SIZED IN BYTES, NOT IN WORKERS. Worker batches travel
    # through /dev/shm, which is 61 GiB here, and a batch of 1024 carries
    # 4 GiB of xct + label + mask — so 8 workers at prefetch 4 asks for 128 GiB
    # and dies with "No space left on device" after the pass has already
    # started. conv_k8's batch of 128 is 0.5 GiB and the same settings are
    # fine, which is exactly why a fixed worker count is the wrong knob.
    bytes_per_batch = cfg["data"]["batch_size"] * 64 ** 3 * (4 + 8 + 4)
    cap = a.shm_budget_gib * 2 ** 30
    workers = max(1, min(a.num_workers, int(cap // bytes_per_batch)))
    prefetch = max(1, int(cap // (workers * bytes_per_batch)))
    cfg["data"]["num_workers"] = workers
    cfg["data"]["prefetch_factor"] = prefetch
    cfg["data"]["persistent_workers"] = False
    split = cfg["data"]["dataset_root"]
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    latest = torch.load(run / "latest.ckpt", map_location="cpu", weights_only=False)
    final_step = latest.get("step")
    model = build_model(cfg, dev)
    # The FINAL weights, so the same-step shortcut inside the helper is correct
    # if it is ever taken. load_state_dict cannot take inference_basis through
    # the strict path — the buffer is registered as None — so it is assigned.
    state = dict(latest["model"])
    basis = {k: state.pop(k) for k in list(state) if k.endswith("inference_basis")}
    model.load_state_dict(state, strict=False)
    for key, value in basis.items():
        mod = model
        for part in key.split(".")[:-1]:
            mod = getattr(mod, part)
        setattr(mod, "inference_basis", value.to(dev))

    train, _, _ = build_patch_dataloaders(cfg, REPO / "data" / split)
    n_batches = len(train)
    print(f"run          {run.name}")
    print(f"final step   {final_step}")
    print(f"pass         {n_batches} batches of {cfg['data']['batch_size']} on {dev}")
    print(f"loader       {workers} workers x prefetch {prefetch} = "
          f"{workers * prefetch * bytes_per_batch / 2 ** 30:.0f} GiB in flight")

    t0 = time.time()
    info = finalize_best_checkpoint(
        run / "best.ckpt", model, train,
        final_step=final_step, device=dev, autocast_dtype=torch.bfloat16)
    dt = time.time() - t0

    if info is None:
        print("NOTHING TO DO: no best.ckpt, no RR bottleneck, or already stamped.")
        return 0
    print(f"best step    {info['best_step']}  refitted={info['refitted']}")
    print(f"keys         {info['keys']}")
    print(f"took         {dt:.0f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
