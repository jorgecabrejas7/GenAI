"""Training and sampling for the pixel-space DDPM baseline.

THE SCHEDULE IS ldm06's OWN. `DDPMSchedule` is imported from
`poregen.diffusion.noise_schedule`, the same object ldm06 trains against, with
the same cosine schedule, v objective and zero-terminal-SNR correction. That is
deliberate: the comparison this baseline exists for is about the SPACE the
diffusion happens in, and giving it a different schedule would confound the two.

LARGE VOLUMES ARE FUSED, NOT CONDITIONED. ldm06 denoises a big canvas jointly,
with each window told about its six neighbours. This model has no conditioning
path at all, so a volume larger than 64 cubed is made by denoising overlapping
windows INDEPENDENTLY and fusing them with the same tapered window the decoder
uses. Independent windows will not agree where they meet, and the seam metrics
will say so — that disagreement is the measurement, not an implementation defect
to be tuned away. Tuning it away would require neighbour conditioning, which is
the thing being compared against.
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch

from poregen.baselines.ddpm3d.networks import UNet3DPixel
from poregen.diffusion.noise_schedule import DDPMSchedule

logger = logging.getLogger(__name__)

PATCH = 64


@dataclass
class DDPMConfig:
    """The run, in one object that goes into the checkpoint."""

    # Shared with ldm06 so the comparison is about pixel vs latent space.
    T: int = 1000
    objective: str = "v"
    zero_terminal_snr: bool = True
    cosine_s: float = 0.008

    # 25.3 M parameters. Sized on purpose against ldm06's 83 M denoiser: a
    # 5.9 M baseline would invite "the baseline was too small", and the paper's
    # compute row carries parameters beside hours and steps for all three
    # generators so the reader can judge that for themselves.
    base: int = 64
    mults: tuple[int, ...] = (1, 2, 4, 4)
    lr: float = 2e-4
    weight_decay: float = 0.0
    #: Drops to 8 if 16 will not fit — `train_ddpm3d.py --batch-size 8`. A
    #: smaller batch at the same wall-clock cap is the honest trade; shrinking
    #: the model to fit would undo the sizing above.
    batch_size: int = 16
    steps: int = 120_000
    ema_decay: float = 0.999
    seed: int = 101
    max_hours: float = 24.0
    checkpoint_minutes: float = 30.0
    log_every: int = 100
    num_workers: int = 4


def build(cfg: DDPMConfig, device: torch.device):
    model = UNet3DPixel(base=cfg.base, mults=tuple(cfg.mults)).to(device)
    sched = DDPMSchedule(T=cfg.T, s=cfg.cosine_s, device=device,
                         objective=cfg.objective,
                         zero_terminal_snr=cfg.zero_terminal_snr)
    return model, sched


@torch.no_grad()
def _ema_update(ema: dict, model: torch.nn.Module, decay: float) -> None:
    for k, v in model.state_dict().items():
        if v.dtype.is_floating_point:
            ema[k].mul_(decay).add_(v.detach(), alpha=1.0 - decay)
        else:
            ema[k].copy_(v)


def train(dataset, cfg: DDPMConfig, out_dir: Path, device: torch.device,
          resume: Path | None = None) -> Path:
    from torch.utils.data import DataLoader

    out_dir.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(cfg.seed)
    model, sched = build(cfg, device)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr,
                            weight_decay=cfg.weight_decay)
    ema = {k: v.detach().clone() for k, v in model.state_dict().items()}

    start = 0
    if resume is not None and Path(resume).exists():
        ck = torch.load(resume, map_location=device, weights_only=False)
        model.load_state_dict(ck["model"])
        ema = {k: v.to(device) for k, v in ck["ema"].items()}
        opt.load_state_dict(ck["opt"])
        start = int(ck["step"])
        logger.info("resumed %s at step %d", resume, start)

    loader = DataLoader(dataset, batch_size=cfg.batch_size, shuffle=True,
                        num_workers=cfg.num_workers, drop_last=True,
                        pin_memory=device.type == "cuda",
                        persistent_workers=cfg.num_workers > 0)
    (out_dir / "config.json").write_text(json.dumps(asdict(cfg), indent=2) + "\n")
    hist = out_dir / "losses.jsonl"
    # TensorBoard under tb/, as every VAE and LDM run writes. Missing before:
    # this run logged a JSONL history and no event file, so a 24-hour
    # unattended training was invisible while it ran.
    from poregen.baselines.slicegan.train import _summary_writer  # noqa: PLC0415

    writer = _summary_writer(out_dir / "tb")
    t0 = time.time()
    last_ck = t0
    step = start
    model.train()

    while step < cfg.steps:
        for x0 in loader:
            if step >= cfg.steps:
                break
            x0 = x0.to(device, non_blocking=True)
            t = torch.randint(0, cfg.T, (x0.shape[0],), device=device)
            noise = torch.randn_like(x0)
            x_t = sched.q_sample(x0, t, noise)
            target = sched.training_target(x0, noise, t)
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16,
                                enabled=device.type == "cuda"):
                pred = model(x_t, t)
                loss = torch.nn.functional.mse_loss(pred.float(), target.float())
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            _ema_update(ema, model, cfg.ema_decay)
            step += 1

            now = time.time()
            if step % cfg.log_every == 0:
                row = {"step": step, "loss": float(loss.detach()),
                       "elapsed_s": now - t0}
                with hist.open("a") as fh:
                    fh.write(json.dumps(row) + "\n")
                if writer is not None:
                    writer.add_scalar("train/loss", row["loss"], step)
                    writer.flush()
                logger.info("step %7d  loss %.5f  %.2f h", step, row["loss"],
                            (now - t0) / 3600)
            if now - last_ck >= cfg.checkpoint_minutes * 60:
                _save(out_dir / "latest.ckpt", model, ema, opt, step, cfg)
                last_ck = now
            if (now - t0) >= cfg.max_hours * 3600:
                logger.info("wall-clock cap %.1f h at step %d", cfg.max_hours, step)
                _save(out_dir / "latest.ckpt", model, ema, opt, step, cfg)
                return out_dir / "latest.ckpt"

    _save(out_dir / "latest.ckpt", model, ema, opt, step, cfg)
    return out_dir / "latest.ckpt"


def _save(path: Path, model, ema, opt, step: int, cfg: DDPMConfig) -> None:
    torch.save({"step": step, "model": model.state_dict(), "ema": ema,
                "opt": opt.state_dict(), "config": asdict(cfg)}, path)


# ---------------------------------------------------------------------------
# Sampling
# ---------------------------------------------------------------------------

def tukey_window_3d(n: int, floor: float = 0.1):
    from poregen.diffusion.sampler import tukey_window_3d as t3
    return np.asarray(t3(n, floor=floor), np.float32)


@torch.no_grad()
def sample_window(model, sched: DDPMSchedule, n: int, device, steps: int,
                  generator=None) -> torch.Tensor:
    """``n`` independent 64-cubed volumes by DDIM."""
    x = torch.randn(n, 4, PATCH, PATCH, PATCH, device=device, generator=generator)
    # The ladder ends at 0, and the last step targets index 0 itself: the
    # schedule's `ddim_step` takes (x_t, t, t_prev, model_out) in THAT order,
    # and t_prev is a tensor, never None.
    ts = torch.linspace(sched.T - 1, 0, steps, device=device).long()
    for i in range(len(ts)):
        t = ts[i].repeat(n)
        t_prev = (ts[i + 1] if i + 1 < len(ts) else torch.zeros_like(ts[0])).repeat(n)
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16,
                            enabled=device.type == "cuda"):
            out = model(x, t)
        x = sched.ddim_step(x, t, t_prev, out.float())
    return x


@torch.no_grad()
def sample_volume(model, sched: DDPMSchedule, shape: tuple[int, int, int],
                  device, steps: int = 200, stride: int = 32,
                  batch: int = 4, generator=None) -> torch.Tensor:
    """A volume of any shape, by fusing INDEPENDENTLY denoised windows.

    No neighbour conditioning exists in this model, so the windows do not know
    about each other and will disagree where they meet. The tapered fusion
    spreads that disagreement rather than hiding it, and the seam metrics are
    meant to see it: what it costs to have no chunk conditioning is the number
    this baseline is here to provide.
    """
    d, h, w = shape
    if min(shape) < PATCH:
        raise ValueError(f"{shape} is smaller than the {PATCH}-voxel window")

    def starts(n: int) -> list[int]:
        out = list(range(0, n - PATCH + 1, stride))
        if out[-1] != n - PATCH:
            out.append(n - PATCH)
        return out

    origins = [(z, y, x) for z in starts(d) for y in starts(h) for x in starts(w)]
    acc = torch.zeros(4, *shape, dtype=torch.float32)
    wacc = torch.zeros(shape, dtype=torch.float32)
    win = torch.from_numpy(tukey_window_3d(PATCH))
    for i in range(0, len(origins), batch):
        chunk = origins[i:i + batch]
        vols = sample_window(model, sched, len(chunk), device, steps,
                             generator=generator).float().cpu()
        for j, (z, y, x) in enumerate(chunk):
            sl = (slice(None), slice(z, z + PATCH), slice(y, y + PATCH),
                  slice(x, x + PATCH))
            acc[sl] += vols[j] * win
            wacc[sl[1:]] += win
    return acc / wacc.clamp_min(1e-8)
