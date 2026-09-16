"""Train the SliceGAN baseline on split_v3 TRAIN patches. (GPU.)

    python scripts/train_slicegan.py --out runs/campaigns/22-slicegan-baseline/train
    python scripts/train_slicegan.py --resume runs/campaigns/22-slicegan-baseline/train/latest.ckpt

The slice bank is built once and cached beside the run, because extracting it
walks the patch store and there is no reason to do that twice.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parents[1]


def main() -> int:
    from poregen.baselines.slicegan.data import build_slice_bank, load_bank, save_bank
    from poregen.baselines.slicegan.train import TrainConfig, train

    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path,
                    default=REPO / "runs" / "campaigns" / "22-slicegan-baseline" / "train")
    ap.add_argument("--bank", type=Path, default=None,
                    help="cached slice bank .npz (default: <out>/slice_bank.npz)")
    ap.add_argument("--n-slices", type=int, default=200_000)
    ap.add_argument("--steps", type=int, default=60_000)
    ap.add_argument("--max-hours", type=float, default=24.0)
    ap.add_argument("--g-batch", type=int, default=4)
    ap.add_argument("--slices-per-volume", type=int, default=16)
    ap.add_argument("--seed", type=int, default=101)
    ap.add_argument("--resume", type=Path, default=None)
    ap.add_argument("--device", default=None)
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s  %(message)s")

    args.out.mkdir(parents=True, exist_ok=True)
    bank_path = args.bank or (args.out / "slice_bank.npz")
    if bank_path.exists():
        logging.info("loading cached slice bank %s", bank_path)
        bank = load_bank(bank_path)
    else:
        bank = build_slice_bank(n_slices=args.n_slices, seed=args.seed, repo_root=REPO)
        save_bank(bank, bank_path)
        logging.info("cached slice bank -> %s", bank_path)
    logging.info("slice bank: %d slices", len(bank))

    device = torch.device(args.device) if args.device else (
        torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"))
    cfg = TrainConfig(steps=args.steps, max_hours=args.max_hours,
                      g_batch=args.g_batch, slices_per_volume=args.slices_per_volume,
                      seed=args.seed)
    logging.info("device %s | %s", device, cfg)
    ck = train(bank, cfg, args.out, device, resume=args.resume)
    logging.info("done -> %s", ck)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
