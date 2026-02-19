"""XCT reconstruction losses — operate on logits, apply sigmoid internally."""

from __future__ import annotations

import torch
import torch.nn.functional as F


def l1_loss(pred_logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """L1 loss between sigmoid(pred_logits) and target in [0, 1]."""
    return F.l1_loss(torch.sigmoid(pred_logits), target)


def mse_loss(pred_logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """MSE loss between sigmoid(pred_logits) and target in [0, 1]."""
    return F.mse_loss(torch.sigmoid(pred_logits), target)


def charbonnier_loss(
    pred_logits: torch.Tensor,
    target: torch.Tensor,
    eps: float = 1e-6,
) -> torch.Tensor:
    """Charbonnier (smooth L1) loss: sqrt((pred - target)^2 + eps^2).

    Less sensitive to outliers than MSE while being differentiable at 0
    (unlike plain L1).
    """
    pred = torch.sigmoid(pred_logits)
    diff = pred - target
    return torch.mean(torch.sqrt(diff * diff + eps * eps))


_RECON_LOSSES = {
    "l1": l1_loss,
    "mse": mse_loss,
    "charbonnier": charbonnier_loss,
}


def get_recon_loss(name: str):
    """Look up a reconstruction loss function by name."""
    if name not in _RECON_LOSSES:
        raise KeyError(f"Unknown recon loss '{name}'. Available: {sorted(_RECON_LOSSES)}")
    return _RECON_LOSSES[name]
