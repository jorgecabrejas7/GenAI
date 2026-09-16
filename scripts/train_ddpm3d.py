"""Train the 3-D pixel-space DDPM baseline. (GPU, streams the patch store.)

    python scripts/train_ddpm3d.py --out runs/campaigns/23-ddpm3d-baseline/train

NOTHING ELSE MAY READ THE STORE WHILE THIS RUNS — see docs/DEVELOPMENT.md. It
streams `patches_xct.bin` continuously, and concurrent store reads deadlocked a
training run in this project once already.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parents[1]


def main() -> int:
    from poregen.baselines.ddpm3d.data import PatchVolumes
    from poregen.baselines.ddpm3d.train import DDPMConfig, train

    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path,
                    default=REPO / "runs" / "campaigns" / "23-ddpm3d-baseline" / "train")
    ap.add_argument("--data-root", type=Path, default=REPO / "data" / "split_v3")
    ap.add_argument("--steps", type=int, default=120_000)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--max-hours", type=float, default=24.0)
    ap.add_argument("--num-workers", type=int, default=4)
    ap.add_argument("--seed", type=int, default=101)
    ap.add_argument("--resume", type=Path, default=None)
    ap.add_argument("--device", default=None)
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(message)s")

    device = torch.device(args.device) if args.device else (
        torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"))
    cfg = DDPMConfig(steps=args.steps, batch_size=args.batch_size,
                     max_hours=args.max_hours, num_workers=args.num_workers,
                     seed=args.seed)
    ds = PatchVolumes(args.data_root, split="train")
    logging.info("device %s | %d train patches | %s", device, len(ds), cfg)
    ck = train(ds, cfg, args.out, device, resume=args.resume)
    logging.info("done -> %s", ck)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
