"""Reconstruction quality metrics (XCT channel)."""

from __future__ import annotations

import torch
import torch.nn.functional as F


@torch.no_grad()
def mse(pred_logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Mean squared error between sigmoid(pred_logits) and target."""
    return F.mse_loss(torch.sigmoid(pred_logits), target)


@torch.no_grad()
def mae(pred_logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Mean absolute error between sigmoid(pred_logits) and target."""
    return F.l1_loss(torch.sigmoid(pred_logits), target)


@torch.no_grad()
def psnr(pred_logits: torch.Tensor, target: torch.Tensor, max_val: float = 1.0) -> torch.Tensor:
    """Peak signal-to-noise ratio (dB).

    Uses MSE between sigmoid(pred_logits) and target as the error.
    """
    mse_val = mse(pred_logits, target)
    if mse_val == 0:
        return torch.tensor(float("inf"), device=pred_logits.device)
    return 10.0 * torch.log10(torch.tensor(max_val ** 2, device=pred_logits.device) / mse_val)


@torch.no_grad()
def sharpness_proxy(x: torch.Tensor) -> torch.Tensor:
    """Cheap sharpness proxy: mean absolute gradient (finite differences).

    Parameters
    ----------
    x : (B, 1, D, H, W)
        Reconstruction (after sigmoid) or ground-truth volume.

    Returns
    -------
    Scalar mean absolute gradient across all axes.
    """
    gd = (x[:, :, 1:, :, :] - x[:, :, :-1, :, :]).abs().mean()
    gh = (x[:, :, :, 1:, :] - x[:, :, :, :-1, :]).abs().mean()
    gw = (x[:, :, :, :, 1:] - x[:, :, :, :, :-1]).abs().mean()
    return (gd + gh + gw) / 3.0
