"""Reconstruction quality metrics (XCT channel).

These helpers expect ``pred_logits`` as raw unbounded decoder output and
``target`` in [0, 1].  ``mae``, ``mse``, and ``psnr`` apply
``torch.sigmoid`` to ``pred_logits`` internally so both tensors are in
[0, 1] before computing the error.  ``sharpness_proxy`` expects an
already-activated volume in [0, 1] and does not apply any activation.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


@torch.no_grad()
def mse(pred_logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Mean squared error; sigmoid applied to pred_logits before comparison."""
    return F.mse_loss(torch.sigmoid(pred_logits), target)


@torch.no_grad()
def mae(pred_logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Mean absolute error; sigmoid applied to pred_logits before comparison."""
    return F.l1_loss(torch.sigmoid(pred_logits), target)


@torch.no_grad()
def psnr(pred_logits: torch.Tensor, target: torch.Tensor, max_val: float = 1.0) -> torch.Tensor:
    """Peak signal-to-noise ratio (dB).

    Applies sigmoid to ``pred_logits`` before computing MSE so both tensors
    are in [0, 1].  ``max_val`` is 1.0.
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
