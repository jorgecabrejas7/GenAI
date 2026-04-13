"""Reconstruction quality metrics (XCT channel).

These helpers expect prediction/target tensors that are already in the
comparison space used by the caller. In particular, ``sharpness_proxy``
assumes an intensity volume in ``[0, 1]`` and does not apply activations
internally.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


@torch.no_grad()
def mse(pred_logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Mean squared error between pred_logits and target (both in [0, 1])."""
    return F.mse_loss(pred_logits, target)


@torch.no_grad()
def mae(pred_logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Mean absolute error between pred_logits and target (both in [0, 1])."""
    return F.l1_loss(pred_logits, target)


@torch.no_grad()
def psnr(pred_logits: torch.Tensor, target: torch.Tensor, max_val: float = 1.0) -> torch.Tensor:
    """Peak signal-to-noise ratio (dB).

    Uses MSE between pred_logits and target as the error.  ``max_val`` is 1.0
    since both tensors are in [0, 1].
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
        Reconstruction or ground-truth volume in [0, 1].

    Returns
    -------
    Scalar mean absolute gradient across all axes.
    """
    gd = (x[:, :, 1:, :, :] - x[:, :, :-1, :, :]).abs().mean()
    gh = (x[:, :, :, 1:, :] - x[:, :, :, :-1, :]).abs().mean()
    gw = (x[:, :, :, :, 1:] - x[:, :, :, :, :-1]).abs().mean()
    return (gd + gh + gw) / 3.0
