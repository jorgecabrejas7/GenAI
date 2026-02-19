"""Latent-space diagnostic metrics."""

from __future__ import annotations

import torch


@torch.no_grad()
def active_units(
    mu: torch.Tensor,
    threshold: float = 0.01,
) -> dict[str, float]:
    """Fraction of latent channels whose variance exceeds *threshold*.

    Aggregates mu over the batch and spatial dims → per-channel variance.
    A channel counts as "active" if Var(mu_c) > threshold.

    Parameters
    ----------
    mu : (B, C, d, h, w)
    threshold : float

    Returns
    -------
    dict with ``active_fraction`` and ``n_active`` / ``n_total``.
    """
    # Flatten to (B * spatial, C)
    B, C = mu.shape[:2]
    mu_flat = mu.permute(1, 0, *range(2, mu.ndim)).reshape(C, -1)  # (C, N)
    var = mu_flat.var(dim=1)  # (C,)
    n_active = (var > threshold).sum().item()
    return {
        "active_fraction": n_active / C,
        "n_active": n_active,
        "n_total": C,
    }


@torch.no_grad()
def kl_per_channel(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
    """KL divergence per latent channel, averaged over batch + spatial.

    Returns
    -------
    (C,) tensor.
    """
    kl = 0.5 * (mu.pow(2) + logvar.exp() - logvar - 1.0)
    # mean over batch and spatial, keep channel dim
    return kl.mean(dim=(0, *range(2, kl.ndim)))


@torch.no_grad()
def latent_stats(mu: torch.Tensor, logvar: torch.Tensor) -> dict[str, float]:
    """Summary statistics for mu and logvar."""
    return {
        "mu_mean": mu.mean().item(),
        "mu_std": mu.std().item(),
        "logvar_mean": logvar.mean().item(),
        "logvar_std": logvar.std().item(),
        "std_mean": (0.5 * logvar).exp().mean().item(),
    }
