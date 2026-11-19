#!/usr/bin/env python
"""Fine-tune the 3-class U-Net for OUTSIDE AIR only, on a volume with surface lines (campaign 40).

Author 2026-11-19: "I do not care about the pores in this fine tuned version,
only about the air ... we have the original checkpoint for pores."  So every
weight is trained and the loss sees only air vs specimen: the 3-class head is
kept (same checkpoint format, same inference code), and the specimen
probability is p(material) + p(pore), i.e. the 2-class logits are
[logsumexp(l_material, l_pore), l_air].  Pore output of the result is NOT valid.

Data: a thickness-first grey volume and its air label (0/255), e.g. the
CuartoPulg overlay split by ``overlay_to_air_label.py``.  Columns where the
author's line is missing or doubled (``surfaces.npz``: n_red or n_green not 1
or 2) get zero loss weight.  Input = grey / 255 (no envelope, no quantile map),
so intensity augmentation (gain, offset, gamma, noise) stands in for scanner
differences.  Half of the patches are centred on a surface (random column,
upper or lower surface, +-patch/4 jitter), half anywhere.  Augmentation: flips
on all three axes, 90-degree turns in the ply plane.

Loss: 2-class cross-entropy + soft Dice of air, masked.  All CuartoPulg is
training data (the test is MediaPulg, which has no label), so there is no
validation split: ``last.ckpt`` is the result; ``ckpt_<step>.ckpt`` every
``--save-every`` steps.

    python -m poregen.experiments.train_unet_air_ft --init runs/unet_seg/us01-.../best.ckpt \
        --grey <grey.tif> --air <air.tif> --surfaces <surfaces.npz> --thickness-axis 1 --exp uaf01
"""
from __future__ import annotations

import argparse
import json
import math
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import tifffile
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, IterableDataset

from poregen.models.unet3d import load_unet3d

REPO = Path(__file__).resolve().parents[3]


def air_logits(logits: torch.Tensor) -> torch.Tensor:
    """(N, 3, ...) material/pore/air logits -> (N, 2, ...) specimen/air logits."""
    return torch.stack([torch.logsumexp(logits[:, :2], 1), logits[:, 2]], 1)


class SurfacePatches(IterableDataset):
    def __init__(self, grey: np.ndarray, air: np.ndarray, valid_col: np.ndarray, top: np.ndarray,
                 bottom: np.ndarray, patch: int, seed: int) -> None:
        self.g, self.y, self.v, self.top, self.bot = grey, air, valid_col, top, bottom
        self.p, self.seed = patch, seed

    def _sample(self, rng):
        T, A, B = self.g.shape
        p = self.p
        if rng.random() < 0.5:
            a, b = int(rng.integers(A)), int(rng.integers(B))
            t = (self.top if rng.random() < 0.5 else self.bot)[a, b] + int(rng.integers(-p // 4, p // 4 + 1))
            t0, a0, b0 = t - p // 2, a - p // 2, b - p // 2
        else:
            t0, a0, b0 = (int(rng.integers(s - p + 1)) for s in (T, A, B))
        t0, a0, b0 = (int(np.clip(o, 0, s - p)) for o, s in zip((t0, a0, b0), (T, A, B)))
        sl = np.s_[t0:t0 + p, a0:a0 + p, b0:b0 + p]
        x = self.g[sl].astype(np.float32) / 255.0
        y = (self.y[sl] > 0).astype(np.int64)
        m = np.broadcast_to(self.v[a0:a0 + p, b0:b0 + p][None], y.shape).astype(np.float32)
        for ax in range(3):
            if rng.random() < 0.5:
                x, y, m = np.flip(x, ax), np.flip(y, ax), np.flip(m, ax)
        k = int(rng.integers(4))
        x, y, m = (np.rot90(z, k, axes=(1, 2)) for z in (x, y, m))
        x = np.clip(x, 0, 1) ** float(np.exp(rng.uniform(-0.4, 0.4)))
        mu = x.mean()
        x = (x - mu) * rng.uniform(0.7, 1.3) + mu + rng.uniform(-0.12, 0.12)
        x = x + rng.normal(0, rng.uniform(0, 0.04), x.shape)
        return (torch.from_numpy(np.ascontiguousarray(x, np.float32))[None],
                torch.from_numpy(np.ascontiguousarray(y)), torch.from_numpy(np.ascontiguousarray(m)))

    def __iter__(self):
        info = torch.utils.data.get_worker_info()
        rng = np.random.default_rng([self.seed, info.id if info else 0, int(time.time())])
        while True:
            yield self._sample(rng)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--exp", default="uaf01")
    ap.add_argument("--init", type=Path, required=True)
    ap.add_argument("--grey", type=Path, required=True)
    ap.add_argument("--air", type=Path, required=True)
    ap.add_argument("--surfaces", type=Path, required=True)
    ap.add_argument("--thickness-axis", type=int, default=1)
    ap.add_argument("--patch", type=int, default=96)
    ap.add_argument("--steps", type=int, default=8000)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--warmup", type=int, default=300)
    ap.add_argument("--dice-weight", type=float, default=1.0)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--log-every", type=int, default=50)
    ap.add_argument("--save-every", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out-root", type=Path, default=REPO / "runs/unet_seg")
    a = ap.parse_args()

    torch.manual_seed(a.seed)
    device = torch.device("cuda")
    grey = np.ascontiguousarray(np.moveaxis(tifffile.imread(a.grey), a.thickness_axis, 0))
    air = np.ascontiguousarray(np.moveaxis(tifffile.imread(a.air), a.thickness_axis, 0))
    assert grey.shape == air.shape, (grey.shape, air.shape)
    s = np.load(a.surfaces)
    assert s["top"].shape == grey.shape[1:], (s["top"].shape, grey.shape)
    valid = ((s["n_red"] >= 1) & (s["n_red"] <= 2) & (s["n_green"] >= 1) & (s["n_green"] <= 2)).astype(np.uint8)

    model = load_unet3d(str(a.init), device).train()
    arch = torch.load(a.init, map_location="cpu", weights_only=False)["arch"]
    n_prev = len(list(a.out_root.glob(f"{a.exp}-run-*")))
    out = a.out_root / f"{a.exp}-run-{n_prev + 1:04d}-{datetime.now():%Y%m%d-%H%M%S}"
    out.mkdir(parents=True)
    (out / "config.json").write_text(json.dumps(
        {**vars(a), "arch": arch, "volume_shape_TAB": list(grey.shape), "air_fraction": float((air > 0).mean()),
         "valid_column_fraction": float(valid.mean()), "loss": "2-class CE + air soft Dice, all weights trained"},
        indent=1, default=str))

    ds = SurfacePatches(grey, air, valid, s["top"], s["bottom"], a.patch, a.seed)
    dl = DataLoader(ds, batch_size=a.batch, num_workers=a.workers, pin_memory=True, persistent_workers=True)
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda k: min(1.0, (k + 1) / a.warmup) * 0.5 * (1 + math.cos(math.pi * min(k, a.steps) / a.steps)))
    mfile = open(out / "metrics.jsonl", "a")

    def save(name: str, step: int):
        torch.save({"model": model.state_dict(), "arch": arch, "step": step, "init": str(a.init),
                    "note": "air-only fine-tune: pore/material split not trained"}, out / name)

    step, t0 = 0, time.time()
    run = {"loss": 0.0, "ce": 0.0, "dice": 0.0, "air_dice_hard": 0.0}
    for x, y, m in dl:
        x, y, m = x.to(device, non_blocking=True), y.to(device, non_blocking=True), m.to(device, non_blocking=True)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            lo = model(x)
        lo2 = air_logits(lo.float())
        ce = (F.cross_entropy(lo2, y, reduction="none") * m).sum() / m.sum().clamp_min(1)
        pa = lo2.softmax(1)[:, 1] * m
        yt = y.float() * m
        dice = 1 - (2 * (pa * yt).sum() + 1) / (pa.sum() + yt.sum() + 1)
        loss = ce + a.dice_weight * dice
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step(); sched.step(); step += 1
        with torch.no_grad():
            ph = (lo2.argmax(1) == 1).float() * m
            hd = (2 * (ph * yt).sum() / (ph.sum() + yt.sum()).clamp_min(1)).item()
        run["loss"] += loss.item(); run["ce"] += ce.item(); run["dice"] += dice.item(); run["air_dice_hard"] += hd
        if step % a.log_every == 0:
            row = {"step": step, **{k: v / a.log_every for k, v in run.items()}, "lr": sched.get_last_lr()[0],
                   "elapsed_s": time.time() - t0}
            mfile.write(json.dumps(row) + "\n"); mfile.flush()
            print(json.dumps(row), flush=True)
            run = {k: 0.0 for k in run}
        if step % a.save_every == 0:
            save(f"ckpt_{step}.ckpt", step)
        if step >= a.steps:
            break
    save("last.ckpt", step)
    print("done", out, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
