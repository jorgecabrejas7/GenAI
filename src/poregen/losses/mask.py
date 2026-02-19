"""Pore-mask losses — all operate on logits (numerically stable)."""

from __future__ import annotations

import torch
import torch.nn.functional as F


def bce_logits_loss(
    logits: torch.Tensor,
    target: torch.Tensor,
    pos_weight: torch.Tensor | None = None,
) -> torch.Tensor:
    """Binary cross-entropy with logits (stable)."""
    return F.binary_cross_entropy_with_logits(logits, target, pos_weight=pos_weight)


def dice_loss(
    logits: torch.Tensor,
    target: torch.Tensor,
    smooth: float = 1.0,
) -> torch.Tensor:
    """Soft Dice loss (1 - Dice coefficient).

    Applies sigmoid to *logits* internally.
    """
    pred = torch.sigmoid(logits)
    # Flatten spatial dims but keep batch dim
    pred_flat = pred.flatten(1)
    target_flat = target.flatten(1)

    intersection = (pred_flat * target_flat).sum(1)
    cardinality = pred_flat.sum(1) + target_flat.sum(1)
    dice = (2.0 * intersection + smooth) / (cardinality + smooth)
    return 1.0 - dice.mean()


def tversky_loss(
    logits: torch.Tensor,
    target: torch.Tensor,
    alpha: float = 0.3,
    beta: float = 0.7,
    smooth: float = 1.0,
) -> torch.Tensor:
    """Tversky loss — generalises Dice; alpha < beta penalises FN more.

    Default ``alpha=0.3, beta=0.7`` biases towards recall (fewer missed
    pores), which is generally what we want for small / sparse pores.
    """
    pred = torch.sigmoid(logits)
    pred_flat = pred.flatten(1)
    target_flat = target.flatten(1)

    tp = (pred_flat * target_flat).sum(1)
    fp = (pred_flat * (1 - target_flat)).sum(1)
    fn = ((1 - pred_flat) * target_flat).sum(1)

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
) -> dict[str, torch.Tensor]:
    """Combined mask loss returning component dict.

    Returns
    -------
    dict with keys ``mask_bce``, ``mask_dice`` (or ``mask_tversky``),
    and ``mask_total``.
    """
    bce = bce_logits_loss(logits, target, pos_weight=pos_weight)

    if use_tversky:
        region = tversky_loss(logits, target, alpha=tversky_alpha, beta=tversky_beta)
        region_key = "mask_tversky"
    else:
        region = dice_loss(logits, target)
        region_key = "mask_dice"

    total = bce_weight * bce + dice_weight * region
    return {"mask_bce": bce, region_key: region, "mask_total": total}
