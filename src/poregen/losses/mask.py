"""Pore-mask losses — all operate on logits (numerically stable).

Performance note
----------------
``dice_loss``, ``tversky_loss``, and ``combined_mask_loss`` accept an optional
*sigmoid* argument.  When the caller has already computed
``torch.sigmoid(logits)`` (e.g. for metrics), passing it here avoids a
redundant elementwise op on the full ``(B, 1, D, H, W)`` tensor.

r08+ — three-class label
------------------------
``multiclass_ce_loss`` / ``multiclass_dice_loss`` / ``combined_class_loss``
operate on the 3-class voxel label (0 material, 1 pore, 2 air) that the
``*_cls`` decoder heads emit, and take class logits rather than a single
binary logit.

R04+ — Focal loss
-----------------
When ``use_focal=True`` in the config, :func:`focal_loss` replaces BCE as the
primary component of the mask loss.  Focal loss down-weights easy negatives
(background voxels) via a ``(1 − p_t)^γ`` modulating factor, which addresses
the false-positive dominance observed in R03 (over-predicting pore voxels).
Tversky/Dice is kept alongside focal loss unchanged.
"""

from __future__ import annotations

from collections.abc import Sequence

import torch
import torch.nn.functional as F


def focal_loss(
    logits: torch.Tensor,
    target: torch.Tensor,
    gamma: float = 2.0,
    alpha: float = 0.25,
    *,
    sigmoid: torch.Tensor | None = None,
) -> torch.Tensor:
    """Focal loss (Lin et al. 2017) for imbalanced binary segmentation.

    ``FL(p_t) = −α_t · (1 − p_t)^γ · log(p_t)``

    Numerically stable: uses ``F.binary_cross_entropy_with_logits`` for the
    log term and applies the focal modulation on top.

    Parameters
    ----------
    logits : (B, 1, D, H, W)
    target : (B, 1, D, H, W) — binary {0, 1}
    gamma : float
        Focusing exponent.  ``γ=0`` recovers standard BCE (plus ``alpha``
        weighting).  Lin et al. default is ``γ=2``.
    alpha : float
        Weight applied to the positive (pore) class.  ``1 − alpha`` is used
        for the negative (background) class.  Lin et al. default is ``0.25``.
    """
    bce = F.binary_cross_entropy_with_logits(logits, target, reduction="none")
    prob   = sigmoid if sigmoid is not None else torch.sigmoid(logits)
    p_t    = prob * target + (1.0 - prob) * (1.0 - target)
    alpha_t = alpha * target + (1.0 - alpha) * (1.0 - target)
    weight  = alpha_t * (1.0 - p_t).pow(gamma)
    return (weight * bce).mean()


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
    use_focal: bool = False,
    focal_gamma: float = 2.0,
    focal_alpha: float = 0.25,
    *,
    sigmoid: torch.Tensor | None = None,
) -> dict[str, torch.Tensor]:
    """Combined mask loss returning a component dict.

    Parameters
    ----------
    use_focal : bool
        When True, replace BCE with Focal loss.  ``bce_weight`` is reused as
        the focal-term weight for backward-compatible loss scaling.
    focal_gamma : float
        Focusing exponent for focal loss (default 2.0).
    focal_alpha : float
        Positive-class weight for focal loss (default 0.25).
    sigmoid : optional pre-computed ``torch.sigmoid(logits)``
        When provided, is forwarded to Dice / Tversky so that sigmoid is
        computed at most once across the entire loss + metrics pipeline.
        *Not* used by focal_loss (which internally computes sigmoid from
        logits for numerical stability).

    Returns
    -------
    dict with keys:
    - ``mask_bce``: BCE or focal loss value (always present for compat logging).
    - ``mask_focal``: focal loss value (only when ``use_focal=True``).
    - ``mask_dice`` or ``mask_tversky``: region loss.
    - ``mask_total``: weighted sum.
    """
    if use_focal:
        primary = focal_loss(logits, target, gamma=focal_gamma, alpha=focal_alpha, sigmoid=sigmoid)
        result_extra = {"mask_focal": primary}
    else:
        primary = bce_logits_loss(logits, target, pos_weight=pos_weight)
        result_extra = {}

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

    total = bce_weight * primary + dice_weight * region
    return {
        "mask_bce":  primary,   # focal when use_focal=True; keeps compat with existing logging
        **result_extra,         # adds "mask_focal" key when use_focal=True
        region_key:  region,
        "mask_total": total,
    }


# ---------------------------------------------------------------------------
# Three-class voxel label (r08+): material / pore / air
# ---------------------------------------------------------------------------

def multiclass_ce_loss(
    logits: torch.Tensor,
    target: torch.Tensor,
    class_weights: torch.Tensor | None = None,
) -> torch.Tensor:
    """Class-weighted cross-entropy over the 3-class voxel label.

    Parameters
    ----------
    logits : (B, C, D, H, W) raw class logits.
    target : (B, D, H, W) int64 class index.
    class_weights : (C,) optional per-class weight, normally inverse to the
        class frequency of the training split.  Air and pore are each a few
        per cent of the voxels, so without it the loss is dominated by
        material and both minority classes collapse.
    """
    return F.cross_entropy(logits, target, weight=class_weights)


def multiclass_dice_loss(
    logits: torch.Tensor,
    target: torch.Tensor,
    classes: Sequence[int],
    smooth: float = 1.0,
    *,
    probs: torch.Tensor | None = None,
) -> dict[str, torch.Tensor]:
    """Soft Dice per class, averaged over *classes*.

    Only the classes that matter are averaged: material is the background
    here, and its Dice is ~1 whatever the model does, so including it would
    dilute the signal from pore and air.

    Parameters
    ----------
    logits : (B, C, D, H, W)
    target : (B, D, H, W) int64
    classes : which class indices to score, e.g. ``(1, 2)`` for pore and air.
    probs : optional pre-computed ``softmax(logits, dim=1)``.

    Returns
    -------
    dict with ``class_dice_{i}`` per class and ``class_dice`` (their mean).
    """
    p = probs if probs is not None else torch.softmax(logits, dim=1)
    out: dict[str, torch.Tensor] = {}
    terms = []
    for c in classes:
        pred_flat = p[:, c].flatten(1)
        tgt_flat = (target == c).to(p.dtype).flatten(1)
        inter = (pred_flat * tgt_flat).sum(1)
        card = pred_flat.sum(1) + tgt_flat.sum(1)
        loss_c = 1.0 - ((2.0 * inter + smooth) / (card + smooth)).mean()
        out[f"class_dice_{c}"] = loss_c
        terms.append(loss_c)
    out["class_dice"] = torch.stack(terms).mean()
    return out


def combined_class_loss(
    logits: torch.Tensor,
    target: torch.Tensor,
    ce_weight: float = 1.0,
    dice_weight: float = 1.0,
    class_weights: torch.Tensor | None = None,
    dice_classes: Sequence[int] = (1, 2),
    *,
    probs: torch.Tensor | None = None,
) -> dict[str, torch.Tensor]:
    """Weighted cross-entropy + soft Dice over the minority classes.

    Returns a component dict with ``class_ce``, ``class_dice``, the per-class
    Dice terms, and ``class_total`` (the weighted sum that goes into the total
    loss).
    """
    ce = multiclass_ce_loss(logits, target, class_weights)
    d = multiclass_dice_loss(logits, target, dice_classes, probs=probs)
    out: dict[str, torch.Tensor] = {"class_ce": ce, **d}
    out["class_total"] = ce_weight * ce + dice_weight * d["class_dice"]
    return out

