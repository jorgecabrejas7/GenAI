"""Pore-mask losses — all operate on logits (numerically stable).

Performance note
----------------
``dice_loss`` and ``tversky_loss`` accept an optional *sigmoid* argument.
When the caller has already computed ``torch.sigmoid(logits)`` (e.g. for
metrics), passing it here avoids a redundant elementwise op on the full
``(B, 1, D, H, W)`` tensor.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


def bce_logits_loss(
    logits: torch.Tensor,
    target: torch.Tensor,
    pos_weight: torch.Tensor | None = None,
) -> torch.Tensor:
    """Binary cross-entropy with logits (numerically stable)."""
    return F.binary_cross_entropy_with_logits(logits, target, pos_weight=pos_weight)


def dice_loss(
    logits: torch.Tensor,
    target: torch.Tensor,
    smooth: float = 1.0,
    *,
    sigmoid: torch.Tensor | None = None,
) -> torch.Tensor:
    """Soft Dice loss (1 − Dice coefficient).

    Parameters
    ----------
    logits : (B, 1, D, H, W)
        Raw mask logits.
    target : (B, 1, D, H, W)
        Binary ground-truth mask.
    smooth : float
        Laplace smoothing constant.
    sigmoid : optional pre-computed ``torch.sigmoid(logits)``
        Pass this to skip the sigmoid when it has already been computed
        (e.g. when the eval loop caches it for metrics).
    """
    pred = sigmoid if sigmoid is not None else torch.sigmoid(logits)
    pred_flat   = pred.flatten(1)
    target_flat = target.flatten(1)

    intersection = (pred_flat * target_flat).sum(1)
    cardinality  = pred_flat.sum(1) + target_flat.sum(1)
    dice = (2.0 * intersection + smooth) / (cardinality + smooth)
    return 1.0 - dice.mean()


def tversky_loss(
    logits: torch.Tensor,
    target: torch.Tensor,
    alpha: float = 0.3,
    beta: float = 0.7,
    smooth: float = 1.0,
    *,
    sigmoid: torch.Tensor | None = None,
) -> torch.Tensor:
    """Tversky loss — generalises Dice; ``alpha < beta`` penalises FN more.

    Default ``alpha=0.3, beta=0.7`` biases towards recall (fewer missed
    pores), which is generally what we want for small / sparse pores.

    Parameters
    ----------
    sigmoid : optional pre-computed ``torch.sigmoid(logits)``
        Avoids redundant sigmoid computation when the caller already has it.
    """
    pred = sigmoid if sigmoid is not None else torch.sigmoid(logits)
    pred_flat   = pred.flatten(1)
    target_flat = target.flatten(1)

    tp = (pred_flat * target_flat).sum(1)
    fp = (pred_flat * (1.0 - target_flat)).sum(1)
    fn = ((1.0 - pred_flat) * target_flat).sum(1)

    tversky = (tp + smooth) / (tp + alpha * fp + beta * fn + smooth)
    return 1.0 - tversky.mean()


def combined_mask_loss(
    logits: torch.Tensor,
    target: torch.Tensor,
    bce_weight: float = 1.0,
    dice_weight: float = 1.0,
    use_tversky: bool = False,
    tversky_alpha: float = 0.3,
    tversky_beta: float = 0.7,
    pos_weight: torch.Tensor | None = None,
    *,
    sigmoid: torch.Tensor | None = None,
) -> dict[str, torch.Tensor]:
    """Combined mask loss returning a component dict.

    Parameters
    ----------
    sigmoid : optional pre-computed ``torch.sigmoid(logits)``
        When provided, is forwarded to Dice / Tversky so that sigmoid is
        computed at most once across the entire loss + metrics pipeline.

    Returns
    -------
    dict with keys ``mask_bce``, ``mask_dice`` (or ``mask_tversky``),
    and ``mask_total``.
    """
    bce = bce_logits_loss(logits, target, pos_weight=pos_weight)

    if use_tversky:
        region = tversky_loss(
            logits, target,
            alpha=tversky_alpha, beta=tversky_beta,
            sigmoid=sigmoid,
        )
        region_key = "mask_tversky"
    else:
        region = dice_loss(logits, target, sigmoid=sigmoid)
        region_key = "mask_dice"

    total = bce_weight * bce + dice_weight * region
    return {"mask_bce": bce, region_key: region, "mask_total": total}
