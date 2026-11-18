#!/usr/bin/env python
"""Train the standalone 3-class 3D U-Net segmenter on the split_v4 patches.

Input: the grey patch (``xct / 255``).  Target: the reference 3-class label
(0 material, 1 pore, 2 air) of ``patches_label.bin``.  Loss: class-weighted
cross-entropy (``data/split_v4/class_weights.json``, sqrt-inverse rule) plus
the soft Dice of pore and air (``poregen.losses.mask.multiclass_dice_loss``),
the same label losses as the VAE's class head.  Augmentation: random flips on
the three axes and a random 90-degree turn in the ply plane (axes 1, 2; the
stacking axis 0 is never swapped).

Validation every ``--val-every`` steps on a fixed val subset:
``poregen.metrics.seg.multiclass_metrics`` (per-class Dice, porosity and air
MAE/bias).  ``best.ckpt`` = highest val pore Dice.  Run directory
``runs/unet_seg/<exp>-run-NNNN-<date>/`` with metrics.jsonl, tb/, config.json.

    python -m poregen.experiments.train_unet_seg --exp us01 --steps 40000
"""
from __future__ import annotations

import argparse
import json
import math
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from poregen.losses.mask import multiclass_dice_loss
from poregen.metrics.seg import multiclass_metrics
from poregen.models.unet3d import UNet3D

REPO = Path(__file__).resolve().parents[3]


class GreyLabelPatches(Dataset):
    def __init__(self, data_root: Path, split: str, augment: bool) -> None:
        meta = json.loads((data_root / "patches_meta.json").read_text())
        shape = tuple(meta["shape"])
        self.xct = np.memmap(data_root / "patches_xct.bin", dtype=np.uint8, mode="r", shape=shape)
        self.label = np.memmap(data_root / "patches_label.bin", dtype=np.uint8, mode="r", shape=shape)
        df = pd.read_parquet(data_root / "patch_index.parquet", columns=["split", "porosity"])
        ok = (df["split"] == split) & (df["porosity"] <= 1.0)
        self.rows = np.flatnonzero(ok.to_numpy())
        self.augment = augment

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, i: int):
        r = int(self.rows[i])
        x = np.asarray(self.xct[r]); y = np.asarray(self.label[r])
        if self.augment:
            rng = np.random.default_rng()
            for ax in range(3):
                if rng.random() < 0.5:
                    x = np.flip(x, ax); y = np.flip(y, ax)
            k = int(rng.integers(4))
            x = np.rot90(x, k, axes=(1, 2)); y = np.rot90(y, k, axes=(1, 2))
        x = torch.from_numpy(np.ascontiguousarray(x, dtype=np.float32)).mul_(1 / 255.0).unsqueeze(0)
        return x, torch.from_numpy(np.ascontiguousarray(y).astype(np.int64))


@torch.no_grad()
def validate(model, loader, device) -> dict:
    acc: dict[str, list[float]] = {}
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            logits = model(x)
        for k, v in multiclass_metrics(logits.float(), y).items():
            acc.setdefault(k, []).append(v)
    return {k: float(np.mean(v)) for k, v in acc.items()}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--exp", default="us01")
    ap.add_argument("--data-root", type=Path, default=REPO / "data/split_v4")
    ap.add_argument("--base", type=int, default=24)
    ap.add_argument("--levels", type=int, default=3)
    ap.add_argument("--steps", type=int, default=40_000)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--warmup", type=int, default=1000)
    ap.add_argument("--dice-weight", type=float, default=1.0)
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--log-every", type=int, default=100)
    ap.add_argument("--val-every", type=int, default=2000)
    ap.add_argument("--val-patches", type=int, default=4096)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out-root", type=Path, default=REPO / "runs/unet_seg")
    a = ap.parse_args()

    torch.manual_seed(a.seed)
    device = torch.device("cuda")
    arch = dict(in_channels=1, n_classes=3, base=a.base, levels=a.levels)
    model = UNet3D(**arch).to(device)
    cw = json.loads((a.data_root / "class_weights.json").read_text())["class_weights"]
    weights = torch.tensor(cw, dtype=torch.float32, device=device)

    n_prev = len(list(a.out_root.glob(f"{a.exp}-run-*")))
    out = a.out_root / f"{a.exp}-run-{n_prev + 1:04d}-{datetime.now():%Y%m%d-%H%M%S}"
    out.mkdir(parents=True)
    (out / "config.json").write_text(json.dumps({**vars(a), "arch": arch, "class_weights": cw,
                                                 "params": sum(p.numel() for p in model.parameters())},
                                                indent=1, default=str))
    from torch.utils.tensorboard import SummaryWriter  # noqa: PLC0415
    tb = SummaryWriter(str(out / "tb"))

    train = GreyLabelPatches(a.data_root, "train", augment=True)
    val = GreyLabelPatches(a.data_root, "val", augment=False)
    g = torch.Generator().manual_seed(a.seed)
    tl = DataLoader(train, batch_size=a.batch, shuffle=True, num_workers=a.workers, pin_memory=True,
                    drop_last=True, persistent_workers=True, generator=g)
    vsub = torch.utils.data.Subset(val, torch.randperm(len(val), generator=g)[: a.val_patches].tolist())
    vl = DataLoader(vsub, batch_size=a.batch, shuffle=False, num_workers=4, pin_memory=True)

    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1.0, (s + 1) / a.warmup) * 0.5 * (1 + math.cos(math.pi * min(s, a.steps) / a.steps)))
    mfile = open(out / "metrics.jsonl", "a")

    def log(row: dict):
        mfile.write(json.dumps(row) + "\n"); mfile.flush()
        for k, v in row.items():
            if isinstance(v, (int, float)) and k != "step":
                tb.add_scalar(f"{row['split']}/{k}", v, row["step"])

    step, best, t0 = 0, -1.0, time.time()
    run = {"loss": 0.0, "ce": 0.0, "dice": 0.0}
    model.train()
    while step < a.steps:
        for x, y in tl:
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                logits = model(x)
            logits = logits.float()
            ce = F.cross_entropy(logits, y, weight=weights)
            dice = multiclass_dice_loss(logits, y, classes=(1, 2))["class_dice"]
            loss = ce + a.dice_weight * dice
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step(); sched.step(); step += 1
            run["loss"] += loss.item(); run["ce"] += ce.item(); run["dice"] += dice.item()
            if step % a.log_every == 0:
                log({"split": "train", "step": step, **{k: v / a.log_every for k, v in run.items()},
                     "lr": sched.get_last_lr()[0], "elapsed_s": time.time() - t0})
                run = {k: 0.0 for k in run}
            if step % a.val_every == 0 or step == a.steps:
                model.eval(); v = validate(model, vl, device); model.train()
                log({"split": "val", "step": step, **v, "elapsed_s": time.time() - t0})
                print(f"step {step} val pore Dice {v['dice_pore']:.3f} air Dice {v['dice_air']:.3f} "
                      f"material Dice {v['dice_material']:.4f} porosity MAE {v['porosity_mae']:.5f} "
                      f"bias {v['porosity_bias']:+.5f} air MAE {v['air_mae']:.5f}", flush=True)
                state = {"model": model.state_dict(), "arch": arch, "step": step, "val": v}
                torch.save(state, out / "last.ckpt")
                if v["dice_pore"] > best:
                    best = v["dice_pore"]; torch.save(state, out / "best.ckpt")
            if step >= a.steps:
                break
    print("done", out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
