"""Latent-space diagnostic metrics."""

from __future__ import annotations

from typing import Any, Iterable

import torch


def _flatten_mu_channels(mu: torch.Tensor) -> torch.Tensor:
    """Reshape ``mu`` to ``(C, N)`` so channel-wise statistics are easy to compute."""
    return mu.permute(1, 0, *range(2, mu.ndim)).reshape(mu.shape[1], -1)


def _variance_from_moments(
    count: int,
    sum_: torch.Tensor,
    sum_sq: torch.Tensor,
) -> torch.Tensor:
    """Sample variance from aggregated channel-wise moments."""
    if count <= 1:
        return torch.zeros_like(sum_, dtype=torch.float64)
    numerator = sum_sq - sum_.square() / count
    return torch.clamp(numerator / (count - 1), min=0.0)


@torch.no_grad()
def latent_channel_moments(mu: torch.Tensor) -> dict[str, Any]:
    """Channel-wise first and second moments for aggregating ``mu`` across batches."""
    mu_flat = _flatten_mu_channels(mu).to(dtype=torch.float64)
    return {
        "count": int(mu_flat.shape[1]),
        "sum": mu_flat.sum(dim=1).cpu(),
        "sum_sq": mu_flat.square().sum(dim=1).cpu(),
    }


def merge_latent_channel_moments(
    moment_summaries: Iterable[dict[str, Any]],
) -> dict[str, Any]:
    """Merge channel-wise moment summaries produced by :func:`latent_channel_moments`."""
    total_count = 0
    total_sum: torch.Tensor | None = None
    total_sum_sq: torch.Tensor | None = None

    for summary in moment_summaries:
        count = int(summary["count"])
        sum_ = summary["sum"]
        sum_sq = summary["sum_sq"]
        total_count += count
        total_sum = sum_.clone() if total_sum is None else total_sum + sum_
        total_sum_sq = sum_sq.clone() if total_sum_sq is None else total_sum_sq + sum_sq

    if total_sum is None or total_sum_sq is None:
        raise ValueError("moment_summaries must contain at least one batch summary.")

    return {
        "count": total_count,
        "sum": total_sum,
        "sum_sq": total_sum_sq,
    }


@torch.no_grad()
def active_units_from_moments(
    count: int,
    sum_: torch.Tensor,
    sum_sq: torch.Tensor,
    threshold: float = 0.01,
) -> dict[str, float]:
    """Active-unit metric from aggregated channel-wise moments."""
    var = _variance_from_moments(count, sum_, sum_sq)
    n_total = int(var.numel())
    n_active = int((var > threshold).sum().item())
    return {
        "mu_active_fraction": n_active / n_total if n_total > 0 else 0.0,
        "mu_n_active": n_active,
    }


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
    dict with ``mu_active_fraction`` and ``mu_n_active``.
    """
    moments = latent_channel_moments(mu)
    return active_units_from_moments(
        moments["count"],
        moments["sum"],
        moments["sum_sq"],
        threshold=threshold,
    )


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
