"""KL divergence with free-bits and beta schedule."""

from __future__ import annotations

import torch


def kl_divergence(
    mu: torch.Tensor,
    logvar: torch.Tensor,
    free_bits: float = 0.0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compute KL(q(z|x) || N(0,I)) with optional free-bits.

    Parameters
    ----------
    mu, logvar : (B, C, d, h, w)
        Posterior parameters from the encoder.
    free_bits : float
        Minimum KL per channel before the channel contributes to the
        total.  Set to 0 to disable.  A typical value is 0.25–2.0.

    Returns
    -------
    kl : scalar tensor
        Sum KL across channels (after free-bits clamping).
    kl_collapsed_fraction : scalar tensor
        Fraction of channels whose raw KL is below ``free_bits`` (i.e. clamped).
    kl_per_channel : (C,) tensor
        Raw (pre-clamp) per-channel KL for monitoring and ablation
        decisions (e.g. z_channels active-units criterion).
    """
    # Per-element KL: 0.5 * (mu^2 + exp(logvar) - logvar - 1)
    kl_elem = 0.5 * (mu.pow(2) + logvar.exp() - logvar - 1.0)

    # Aggregate per channel: mean over batch and spatial dims
    # kl_elem shape: (B, C, d, h, w) → (C,)
    kl_per_channel = kl_elem.mean(dim=(0, *range(2, kl_elem.ndim)))

    if free_bits > 0.0:
        clamped = torch.clamp(kl_per_channel, min=free_bits)
        n_clamped = (kl_per_channel < free_bits).float().mean()
        kl = clamped.sum()
    else:
        kl = kl_per_channel.sum()
        n_clamped = torch.tensor(0.0, device=mu.device)

    return kl, n_clamped, kl_per_channel


def kl_divergence_flat(
    mu: torch.Tensor,
    logvar: torch.Tensor,
    free_bits: float = 0.0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compute KL(q(z|x) || N(0,I)) with optional free-bits, for a flat latent.

    Same math as :func:`kl_divergence`, but for a flat ``(B, C)`` latent
    (e.g. the VRRAE bottleneck's rank-``C`` SVD-coefficient posterior)
    instead of a spatial ``(B, C, d, h, w)`` latent — there are no spatial
    dims to reduce over, so the per-channel reduction is over ``dim=(0,)``
    only.

    Parameters
    ----------
    mu, logvar : (B, C)
        Posterior parameters from the encoder/bottleneck.
    free_bits : float
        Minimum KL per channel before the channel contributes to the
        total.  Set to 0 to disable.

    Returns
    -------
    kl : scalar tensor
        Sum KL across channels (after free-bits clamping).
    kl_collapsed_fraction : scalar tensor
        Fraction of channels whose raw KL is below ``free_bits`` (i.e. clamped).
    kl_per_channel : (C,) tensor
        Raw (pre-clamp) per-channel KL for monitoring and ablation decisions.
    """
    # Force fp32. Under bf16 autocast mu/logvar arrive as bfloat16, where
    # logvar.exp() can overflow and an 8-bit mantissa loses accuracy across a
    # batch-sized mean. reparameterize() already upcasts for the same reason,
    # but the bottleneck casts back down before the loss sees these, so that
    # protection does not reach here.
    mu = mu.float()
    logvar = logvar.float()
    kl_elem = 0.5 * (mu.pow(2) + logvar.exp() - logvar - 1.0)

    # kl_elem shape: (B, C) -> (C,), mean over batch only (no spatial dims).
    kl_per_channel = kl_elem.mean(dim=0)

    if free_bits > 0.0:
        clamped = torch.clamp(kl_per_channel, min=free_bits)
        n_clamped = (kl_per_channel < free_bits).float().mean()
        kl = clamped.sum()
    else:
        kl = kl_per_channel.sum()
        n_clamped = torch.tensor(0.0, device=mu.device)

    return kl, n_clamped, kl_per_channel


def beta_schedule(
    step: int,
    warmup_steps: int = 5000,
    max_beta: float = 1.0,
) -> float:
    """Linear β warm-up from 0 → max_beta over *warmup_steps*.

    Returns *max_beta* for all subsequent steps.
    """
    if warmup_steps <= 0:
        return max_beta
    return min(max_beta, max_beta * step / warmup_steps)
