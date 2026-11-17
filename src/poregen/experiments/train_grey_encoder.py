#!/usr/bin/env python
"""Train the grey-only encoder (OQ-72): grey patch -> the production VAE's mu.

Targets are the latent store's posterior means (channels ``0..C-1`` of
``latents.bin``), aligned to the patch memmap through ``index.parquet``'s
``source_row``.  The loss is the MSE on mu in the store's raw units.  The
production decoder is frozen and used only for validation: the predicted mu
is decoded and the 3-class label scored against the reference label
(``poregen.metrics.seg.multiclass_metrics``), next to the teacher (the full
encoder with the true mask planes) on the same patches.

Run directory ``runs/grey_encoder/<exp>-run-NNNN-<date>/``: ``metrics.jsonl``
(train rows every ``--log-every`` steps, val rows every ``--val-every``),
``tb/``, ``last.ckpt``, ``best.ckpt`` (lowest val latent MSE), ``config.json``.

    python -m poregen.experiments.train_grey_encoder --exp ge01 --steps 60000
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
import yaml
from torch.utils.data import DataLoader, Dataset

from poregen.metrics.seg import multiclass_metrics
from poregen.models.vae.base import VAEConfig
from poregen.models.vae.v2.grey_encoder import GreyEncoder

REPO = Path(__file__).resolve().parents[3]
PS = 64


class GreyLatentPairs(Dataset):
    """Row i: the grey patch at ``source_row[i]`` and mu[i] of the store."""

    def __init__(self, data_root: Path, store: Path, split: str) -> None:
        meta = json.loads((data_root / "patches_meta.json").read_text())
        self.xct = np.memmap(data_root / "patches_xct.bin", dtype=np.uint8, mode="r", shape=tuple(meta["shape"]))
        self.label = np.memmap(data_root / "patches_label.bin", dtype=np.uint8, mode="r", shape=tuple(meta["shape"]))
        smeta = json.loads((store / "metadata.json").read_text())
        C, *sp = smeta["latent_shape"]
        self.C = C
        n = smeta["splits"][split]
        self.lat = np.memmap(store / split / "latents.bin", dtype=np.float16, mode="r", shape=(n, 2 * C, *sp))
        idx = pd.read_parquet(store / split / "index.parquet", columns=["source_row"])
        assert len(idx) == n
        self.rows = idx["source_row"].to_numpy(np.int64)

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, i: int):
        r = int(self.rows[i])
        x = torch.from_numpy(np.asarray(self.xct[r], dtype=np.float32)).mul_(1 / 255.0).unsqueeze(0)
        mu = torch.from_numpy(np.asarray(self.lat[i, : self.C], dtype=np.float32))
        lab = torch.from_numpy(np.asarray(self.label[r]).astype(np.int64))
        return x, mu, lab


def load_teacher(run: Path, ckpt: str, device):
    from poregen.experiments.train_vae import build_model  # noqa: PLC0415
    cfg = yaml.safe_load((run / "resolved_config.yaml").read_text())
    model = build_model(cfg, device)
    model.load_state_dict(torch.load(run / ckpt, map_location="cpu", weights_only=False)["model"], strict=True)
    for p in model.parameters():
        p.requires_grad_(False)
    return model.eval(), cfg["model"]


@torch.no_grad()
def decode_scores(teacher, mu: torch.Tensor, x: torch.Tensor, lab: torch.Tensor) -> dict:
    dec = teacher.decoder(mu)
    logits = teacher.class_head(dec)
    grey = teacher.xct_head(dec).clamp(0, 1)
    m = multiclass_metrics(logits, lab)
    m["grey_l1"] = (grey - x).abs().mean().item()
    return m


@torch.no_grad()
def validate(student, teacher, loader, device, max_batches: int) -> dict:
    acc: dict[str, list[float]] = {}
    for b, (x, mu, lab) in enumerate(loader):
        if b >= max_batches:
            break
        x, mu, lab = x.to(device), mu.to(device), lab.to(device)
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
            pred = student(x).float()
        row = {"latent_mse": F.mse_loss(pred, mu).item()}
        row.update({f"student_{k}": v for k, v in decode_scores(teacher, pred, x, lab).items()})
        row.update({f"teacher_{k}": v for k, v in decode_scores(teacher, mu, x, lab).items()})
        for k, v in row.items():
            acc.setdefault(k, []).append(v)
    return {k: float(np.mean(v)) for k, v in acc.items()}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--exp", default="ge01")
    ap.add_argument("--data-root", type=Path, default=REPO / "data/split_v4")
    ap.add_argument("--store", type=Path, default=REPO / "data/split_v4/latents_r08-run-0020-s25000z4")
    ap.add_argument("--vae-run", type=Path, default=None, help="default: runs/vae/r08-run-0020-*")
    ap.add_argument("--vae-ckpt", default=None, help="default: the step 25000 checkpoint")
    ap.add_argument("--steps", type=int, default=60_000)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--warmup", type=int, default=1000)
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--log-every", type=int, default=100)
    ap.add_argument("--val-every", type=int, default=2000)
    ap.add_argument("--val-batches", type=int, default=64)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out-root", type=Path, default=REPO / "runs/grey_encoder")
    a = ap.parse_args()

    torch.manual_seed(a.seed)
    device = torch.device("cuda")
    vae_run = a.vae_run or next((REPO / "runs/vae").glob("r08-run-0020-2026*"))
    vae_ckpt = a.vae_ckpt or next(vae_run.glob("*_step00025000.ckpt")).name
    teacher, mcfg = load_teacher(vae_run, vae_ckpt, device)
    teacher.to(device)
    vcfg = dict(in_channels=1, z_channels=mcfg["z_channels"], base_channels=mcfg["base_channels"],
                n_blocks=mcfg["n_blocks"], patch_size=mcfg["patch_size"])
    student = GreyEncoder(VAEConfig(**vcfg)).to(device)

    n_prev = len(list(a.out_root.glob(f"{a.exp}-run-*")))
    out = a.out_root / f"{a.exp}-run-{n_prev + 1:04d}-{datetime.now():%Y%m%d-%H%M%S}"
    out.mkdir(parents=True)
    (out / "config.json").write_text(json.dumps({**vars(a), "vae_run": str(vae_run), "vae_ckpt": vae_ckpt,
                                                 "vae_config": vcfg, "store": str(a.store),
                                                 "data_root": str(a.data_root), "out_root": str(a.out_root),
                                                 "params": sum(p.numel() for p in student.parameters())}, indent=1, default=str))
    from torch.utils.tensorboard import SummaryWriter  # noqa: PLC0415
    tb = SummaryWriter(str(out / "tb"))

    train = GreyLatentPairs(a.data_root, a.store, "train")
    val = GreyLatentPairs(a.data_root, a.store, "val")
    g = torch.Generator().manual_seed(a.seed)
    tl = DataLoader(train, batch_size=a.batch, shuffle=True, num_workers=a.workers, pin_memory=True,
                    drop_last=True, persistent_workers=True, generator=g)
    vsub = torch.utils.data.Subset(val, torch.randperm(len(val), generator=g)[: a.val_batches * a.batch].tolist())
    vl = DataLoader(vsub, batch_size=a.batch, shuffle=False, num_workers=4, pin_memory=True)

    opt = torch.optim.AdamW(student.parameters(), lr=a.lr, weight_decay=0.01, betas=(0.9, 0.99))
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1.0, (s + 1) / a.warmup) * 0.5 * (1 + math.cos(math.pi * min(s, a.steps) / a.steps)))
    mfile = open(out / "metrics.jsonl", "a")

    def log(row: dict):
        mfile.write(json.dumps(row) + "\n"); mfile.flush()
        for k, v in row.items():
            if isinstance(v, (int, float)) and k != "step":
                tb.add_scalar(f"{row['split']}/{k}", v, row["step"])

    step, best, t0, run_loss = 0, float("inf"), time.time(), 0.0
    student.train()
    while step < a.steps:
        for x, mu, _ in tl:
            x, mu = x.to(device, non_blocking=True), mu.to(device, non_blocking=True)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                pred = student(x)
            loss = F.mse_loss(pred.float(), mu)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(student.parameters(), 1.0)
            opt.step(); sched.step(); step += 1
            run_loss += loss.item()
            if step % a.log_every == 0:
                log({"split": "train", "step": step, "latent_mse": run_loss / a.log_every,
                     "lr": sched.get_last_lr()[0], "elapsed_s": time.time() - t0})
                run_loss = 0.0
            if step % a.val_every == 0 or step == a.steps:
                student.eval()
                v = validate(student, teacher, vl, device, a.val_batches)
                student.train()
                log({"split": "val", "step": step, **v, "elapsed_s": time.time() - t0})
                print(f"step {step} val latent_mse {v['latent_mse']:.4f} pore Dice {v['student_dice_pore']:.3f} "
                      f"(teacher {v['teacher_dice_pore']:.3f}) air Dice {v['student_dice_air']:.3f} "
                      f"porosity MAE {v['student_porosity_mae']:.5f} (teacher {v['teacher_porosity_mae']:.5f})", flush=True)
                state = {"model": student.state_dict(), "step": step, "vae_config": vcfg, "val": v,
                         "vae_run": str(vae_run), "vae_ckpt": vae_ckpt, "store": str(a.store)}
                torch.save(state, out / "last.ckpt")
                if v["latent_mse"] < best:
                    best = v["latent_mse"]; torch.save(state, out / "best.ckpt")
            if step >= a.steps:
                break
    print("done", out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
