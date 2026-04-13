"""Shared 3-D convolutional building blocks for all PoreGen architecture generations.

Two families exist:

* **v1** — ``Conv3d`` stride-2 down, ``ConvTranspose3d`` up, ``GroupNorm``, ``SiLU``.
  Used by the first-generation VAE architectures.  Weights are compatible with
  checkpoints trained before the v2 refactor.

* **v2** — ``Conv3d`` stride-2 down, ``Upsample(trilinear) + Conv3d`` up,
  ``BatchNorm3d``, ``GELU``.  Faster on H100-class hardware, no checkerboard
  artefacts, aligned with modern diffusion-model conventions.

Shared utilities:
    :func:`norm_groups` — ``min(32, channels)`` for GroupNorm group count.
    :func:`reparameterize` — float32-stable VAE reparameterization trick.
"""

from __future__ import annotations

import torch
import torch.nn as nn


# ── shared utilities ──────────────────────────────────────────────────────────

def norm_groups(channels: int) -> int:
    """Return the number of GroupNorm groups for *channels* feature maps."""
    return min(32, channels)


def reparameterize(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
    """VAE reparameterization trick computed in float32 for numerical stability.

    ``logvar.exp()`` can overflow in bfloat16 when logvar values are large
    (e.g. at high β).  Casting to float32 is cheap (~1-2 µs) and prevents
    KL instability during warmup.

    Parameters
    ----------
    mu, logvar : (B, C, *spatial)
        Posterior parameters from the encoder.

    Returns
    -------
    z : same dtype as *mu*
        Sampled latent via z = mu + std * ε,  ε ~ N(0, I).
    """
    mu_f = mu.float()
    std = (0.5 * logvar.float()).exp()
    eps = torch.randn_like(std)
    return (mu_f + std * eps).to(mu.dtype)


# ── v1 blocks ─────────────────────────────────────────────────────────────────

def down_block_v1(in_ch: int, out_ch: int) -> nn.Sequential:
    """Stride-2 downsampling block — v1 style (GroupNorm + SiLU).

    Architecture::

        Conv3d(4×4×4, stride=2)  →  GroupNorm  →  SiLU
        Conv3d(3×3×3)            →  GroupNorm  →  SiLU

    Returns an ``nn.Sequential`` whose parameter indices match the layout
    expected by v1 checkpoints (positions 0/1/3/4 carry weights; 2/5 are
    activation layers with no parameters).
    """
    return nn.Sequential(
        nn.Conv3d(in_ch, out_ch, 4, stride=2, padding=1),
        nn.GroupNorm(norm_groups(out_ch), out_ch),
        nn.SiLU(),
        nn.Conv3d(out_ch, out_ch, 3, padding=1),
        nn.GroupNorm(norm_groups(out_ch), out_ch),
        nn.SiLU(),
    )


def up_block_v1(in_ch: int, out_ch: int) -> nn.Sequential:
    """Stride-2 upsampling block — v1 style (ConvTranspose3d + GroupNorm + SiLU).

    Architecture::

        ConvTranspose3d(4×4×4, stride=2)  →  GroupNorm  →  SiLU
        Conv3d(3×3×3)                     →  GroupNorm  →  SiLU

    Returns an ``nn.Sequential`` whose parameter layout matches v1 checkpoints.
    """
    return nn.Sequential(
        nn.ConvTranspose3d(in_ch, out_ch, 4, stride=2, padding=1),
        nn.GroupNorm(norm_groups(out_ch), out_ch),
        nn.SiLU(),
        nn.Conv3d(out_ch, out_ch, 3, padding=1),
        nn.GroupNorm(norm_groups(out_ch), out_ch),
        nn.SiLU(),
    )


# ── v2 blocks ─────────────────────────────────────────────────────────────────

def down_block_v2(in_ch: int, out_ch: int) -> nn.Sequential:
    """Stride-2 downsampling block — v2 style (BatchNorm3d + GELU).

    Architecture::

        Conv3d(4×4×4, stride=2)  →  BatchNorm3d  →  GELU
        Conv3d(3×3×3)            →  BatchNorm3d  →  GELU

    BatchNorm3d is more stable than GroupNorm at batch_size ≥ 32 and has
    fused CUDA kernels on Ampere/Hopper hardware.  GELU is the de-facto
    standard activation in modern diffusion models (DDPM, LDM, DiT).
    """
    return nn.Sequential(
        nn.Conv3d(in_ch, out_ch, 4, stride=2, padding=1),
        nn.BatchNorm3d(out_ch),
        nn.GELU(),
        nn.Conv3d(out_ch, out_ch, 3, padding=1),
        nn.BatchNorm3d(out_ch),
        nn.GELU(),
    )


def up_block_v2(in_ch: int, out_ch: int) -> nn.Sequential:
    """Stride-2 upsampling block — v2 style (trilinear Upsample + Conv + BatchNorm + GELU).

    Architecture::

        Upsample(scale_factor=2, trilinear)  →
        Conv3d(3×3×3)  →  BatchNorm3d  →  GELU  →
        Conv3d(3×3×3)  →  BatchNorm3d  →  GELU

    Trilinear upsample + Conv3d is faster than ``ConvTranspose3d`` on H100
    and avoids checkerboard artefacts at pore boundaries.
    """
    return nn.Sequential(
        nn.Upsample(scale_factor=2, mode="trilinear", align_corners=False),
        nn.Conv3d(in_ch, out_ch, 3, padding=1),
        nn.BatchNorm3d(out_ch),
        nn.GELU(),
        nn.Conv3d(out_ch, out_ch, 3, padding=1),
        nn.BatchNorm3d(out_ch),
        nn.GELU(),
    )
