"""WGAN-GP training for SliceGAN: one 3-D generator, three 2-D critics.

The loop is the paper's. Per generator step the critics are updated
``n_critic`` times on real and generated slices, with a gradient penalty on
interpolates; then the generator takes one step against all three at once. A
generated volume therefore receives gradient only through its 2-D sections,
which is the claim the method rests on.

Losses are logged per axis, because a collapse is usually one axis first —
averaging the three hides exactly the failure worth catching.
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
from torch import nn

from poregen.baselines.slicegan.data import SliceBank, to_critic_batch
from poregen.baselines.slicegan.networks import (
    Critic2D,
    Generator3D,
    latent_for_shape,
    volume_to_slices,
)

logger = logging.getLogger(__name__)


@dataclass
class TrainConfig:
    """Everything the run is, in one object that goes into the checkpoint."""

    #: Paper values. lr 1e-4 with betas (0.9, 0.99) is what SliceGAN uses; the
    #: WGAN-GP default (0.0, 0.9) is for a different critic schedule.
    lr: float = 1e-4
    beta1: float = 0.9
    beta2: float = 0.99
    gp_lambda: float = 10.0
    n_critic: int = 5
    nz: int = 64
    ngf: int = 64
    ndf: int = 64
    #: Volumes per generator step, and how many slices per axis each yields to
    #: the critics. 4 x 16 = 64 slices per axis, near the paper's batch of 32
    #: per critic while keeping the 3-D batch small enough to be cheap.
    g_batch: int = 4
    slices_per_volume: int = 16
    train_shape: tuple[int, int, int] = (64, 64, 64)
    steps: int = 60_000
    seed: int = 101
    #: Wall-clock cap. The run stops cleanly at the next checkpoint past it.
    max_hours: float = 24.0
    checkpoint_minutes: float = 30.0
    sample_minutes: float = 60.0
    log_every: int = 50


def gradient_penalty(critic: nn.Module, real: torch.Tensor, fake: torch.Tensor,
                     device: torch.device) -> torch.Tensor:
    """WGAN-GP on interpolates between a real and a fake slice."""
    n = real.shape[0]
    eps = torch.rand(n, 1, 1, 1, device=device)
    mix = (eps * real + (1.0 - eps) * fake).requires_grad_(True)
    score = critic(mix)
    grad = torch.autograd.grad(
        outputs=score, inputs=mix,
        grad_outputs=torch.ones_like(score),
        create_graph=True, retain_graph=True, only_inputs=True)[0]
    return ((grad.reshape(n, -1).norm(2, dim=1) - 1.0) ** 2).mean()


def _real_slices(bank: SliceBank, axis: int, n: int, rng: np.random.Generator,
                 device: torch.device) -> torch.Tensor:
    rows = rng.choice(bank.axis_rows(axis), size=n, replace=False)
    return torch.from_numpy(to_critic_batch(bank, rows)).to(device)


def _fake_slices(vol: torch.Tensor, axis: int, k: int,
                 rng: np.random.Generator) -> torch.Tensor:
    """``k`` random slices per volume along ``axis`` — not all of them.

    Taking every slice would make the critic batch 64x the volume batch and
    correlate it heavily; the paper samples too.
    """
    s = volume_to_slices(vol, axis)
    n_per = s.shape[0] // vol.shape[0]
    pick = np.concatenate([
        b * n_per + rng.choice(n_per, size=k, replace=False)
        for b in range(vol.shape[0])])
    return s[torch.from_numpy(pick).to(s.device)]


def train(bank: SliceBank, cfg: TrainConfig, out_dir: Path,
          device: torch.device, resume: Path | None = None) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(cfg.seed)
    rng = np.random.default_rng(cfg.seed)

    gen = Generator3D(nz=cfg.nz, ngf=cfg.ngf).to(device)
    critics = [Critic2D(ndf=cfg.ndf).to(device) for _ in range(3)]
    opt_g = torch.optim.Adam(gen.parameters(), lr=cfg.lr,
                             betas=(cfg.beta1, cfg.beta2))
    opt_d = [torch.optim.Adam(c.parameters(), lr=cfg.lr,
                              betas=(cfg.beta1, cfg.beta2)) for c in critics]

    start = 0
    if resume is not None and Path(resume).exists():
        ck = torch.load(resume, map_location=device, weights_only=False)
        gen.load_state_dict(ck["generator"])
        for c, sd in zip(critics, ck["critics"]):
            c.load_state_dict(sd)
        opt_g.load_state_dict(ck["opt_g"])
        for o, sd in zip(opt_d, ck["opt_d"]):
            o.load_state_dict(sd)
        start = int(ck["step"])
        logger.info("resumed from %s at step %d", resume, start)

    (out_dir / "config.json").write_text(json.dumps(asdict(cfg), indent=2) + "\n")
    hist_path = out_dir / "losses.jsonl"
    t0 = time.time()
    last_ck = last_sample = t0
    k = cfg.slices_per_volume
    lat = latent_for_shape(cfg.train_shape)

    for step in range(start, cfg.steps):
        d_losses, gps = [], []
        for _ in range(cfg.n_critic):
            with torch.no_grad():
                z = torch.randn(cfg.g_batch, cfg.nz, *lat, device=device)
                fake_vol = gen(z)
            for axis in range(3):
                real = _real_slices(bank, axis, cfg.g_batch * k, rng, device)
                fake = _fake_slices(fake_vol, axis, k, rng)
                critic, opt = critics[axis], opt_d[axis]
                opt.zero_grad(set_to_none=True)
                d_real = critic(real).mean()
                d_fake = critic(fake).mean()
                gp = gradient_penalty(critic, real, fake, device)
                loss = d_fake - d_real + cfg.gp_lambda * gp
                loss.backward()
                opt.step()
                d_losses.append(float((d_fake - d_real).detach()))
                gps.append(float(gp.detach()))

        z = torch.randn(cfg.g_batch, cfg.nz, *lat, device=device)
        fake_vol = gen(z)
        opt_g.zero_grad(set_to_none=True)
        g_loss = torch.stack([
            -critics[axis](_fake_slices(fake_vol, axis, k, rng)).mean()
            for axis in range(3)]).sum()
        g_loss.backward()
        opt_g.step()

        now = time.time()
        if step % cfg.log_every == 0:
            row = {"step": step, "g_loss": float(g_loss.detach()),
                   "d_wasserstein": float(np.mean(d_losses)),
                   "gp": float(np.mean(gps)),
                   "d_per_axis": [float(np.mean(d_losses[i::3])) for i in range(3)],
                   "elapsed_s": now - t0}
            with hist_path.open("a") as fh:
                fh.write(json.dumps(row) + "\n")
            logger.info("step %6d  G %8.3f  D(W) %8.3f  GP %6.3f  axes %s  %.1f h",
                        step, row["g_loss"], row["d_wasserstein"], row["gp"],
                        [round(v, 2) for v in row["d_per_axis"]],
                        (now - t0) / 3600)

        if now - last_ck >= cfg.checkpoint_minutes * 60 or step == cfg.steps - 1:
            _save(out_dir / "latest.ckpt", gen, critics, opt_g, opt_d, step, cfg)
            last_ck = now
        if now - last_sample >= cfg.sample_minutes * 60:
            _sample_grid(gen, cfg, device, out_dir / f"sample_step{step:07d}.npy")
            last_sample = now
        if (now - t0) >= cfg.max_hours * 3600:
            logger.info("wall-clock cap of %.1f h reached at step %d", cfg.max_hours, step)
            _save(out_dir / "latest.ckpt", gen, critics, opt_g, opt_d, step, cfg)
            break

    return out_dir / "latest.ckpt"


def _save(path: Path, gen, critics, opt_g, opt_d, step: int, cfg: TrainConfig) -> None:
    torch.save({
        "step": step + 1,
        "generator": gen.state_dict(),
        "critics": [c.state_dict() for c in critics],
        "opt_g": opt_g.state_dict(),
        "opt_d": [o.state_dict() for o in opt_d],
        "config": asdict(cfg),
    }, path)


@torch.no_grad()
def _sample_grid(gen, cfg: TrainConfig, device, path: Path) -> None:
    """One small volume, saved raw for inspection — no plotting on the GPU box."""
    gen.eval()
    z = gen.sample_latent(cfg.train_shape, n=1, device=device)
    v = gen(z)[0].cpu().numpy()
    gen.train()
    np.save(path, v.astype(np.float16))
