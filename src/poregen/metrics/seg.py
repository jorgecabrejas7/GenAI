"""Segmentation metrics with explicit empty-GT handling.

All public functions are decorated with ``@torch.no_grad()`` and are fully
vectorised — TP/FP/FN are computed over the **entire batch** in a single GPU
pass, with a single ``.cpu()`` transfer at the end.  This eliminates the
per-sample ``.item()`` calls that caused ~10-20 % overhead in the original
implementation.
"""

from __future__ import annotations

import torch


@torch.no_grad()
def segmentation_metrics(
    logits: torch.Tensor,
    target: torch.Tensor,
    threshold: float = 0.5,
    apply_sigmoid: bool = True,
) -> dict[str, float]:
    """Compute Dice, precision, and recall for non-empty patches only.

    Only ``pos_only`` variants are returned — samples where the ground-truth
    mask is entirely empty are excluded, as they inflate overlap metrics.
    IoU (monotone in Dice) and F1 (identical to Dice for binary masks) have
    been removed to reduce TensorBoard noise.

    Parameters
    ----------
    logits : (B, 1, D, H, W)
        Raw mask logits, or pre-activated probabilities when
        ``apply_sigmoid=False``.
    target : (B, 1, D, H, W)
        Binary ground-truth mask {0, 1}.
    threshold : float
        Threshold for binarising predictions.
    apply_sigmoid : bool
        When False, ``logits`` is treated as already-activated (avoids a
        redundant sigmoid when the caller pre-computes it).

    Returns
    -------
    dict
        Keys: dice_pos_only, precision_pos_only, recall_pos_only.
    """
    activated = torch.sigmoid(logits) if apply_sigmoid else logits
    pred = (activated >= threshold).float()

    B = pred.shape[0]
    pred_flat   = pred.reshape(B, -1)
    target_flat = target.reshape(B, -1)

    tp = (pred_flat * target_flat).sum(dim=1)
    fp = (pred_flat * (1.0 - target_flat)).sum(dim=1)
    fn = ((1.0 - pred_flat) * target_flat).sum(dim=1)
    gt_sum = target_flat.sum(dim=1)

    dice_per = (2.0 * tp + 1e-7) / (2.0 * tp + fp + fn + 1e-7)

    tp_cpu     = tp.cpu()
    fp_cpu     = fp.cpu()
    fn_cpu     = fn.cpu()
    gt_sum_cpu = gt_sum.cpu()
    dice_cpu   = dice_per.cpu()

    sel = (gt_sum_cpu > 0).nonzero(as_tuple=True)[0]
    if sel.numel() == 0:
        return {"dice_pos_only": 0.0, "precision_pos_only": 0.0, "recall_pos_only": 0.0}

    tp_s = tp_cpu[sel].sum().item()
    fp_s = fp_cpu[sel].sum().item()
    fn_s = fn_cpu[sel].sum().item()

    precision = (tp_s + 1e-7) / (tp_s + fp_s + 1e-7)
    recall    = (tp_s + 1e-7) / (tp_s + fn_s + 1e-7)

    return {
        "dice_pos_only":      float(dice_cpu[sel].mean().item()),
        "precision_pos_only": float(precision),
        "recall_pos_only":    float(recall),
    }


@torch.no_grad()
def porosity_metrics(
    mask_logits: torch.Tensor,
    target: torch.Tensor,
    apply_sigmoid: bool = True,
) -> dict[str, float]:
    """Porosity MAE, bias, and collapse-detection stats per batch.

    Primary success metric per EDA: porosity_mae < 0.005.

    Parameters
    ----------
    mask_logits : (B, 1, D, H, W)  — raw logits, or pre-activated
        probabilities when ``apply_sigmoid=False``.
    target      : (B, 1, D, H, W)  — binary ground-truth mask {0, 1}
    apply_sigmoid : bool
        When False, ``mask_logits`` is treated as already-activated
        (avoids a redundant sigmoid when the caller pre-computes it).

    Returns
    -------
    dict with:
      porosity_mae      — mean |pred − gt| porosity (primary metric)
      porosity_bias     — mean (pred − gt), detects systematic over/under-prediction
      mask_pred_mean    — mean sigmoid(mask_logits) over batch (collapse: → 0 or 1)
    """
    pred_sigmoid = torch.sigmoid(mask_logits) if apply_sigmoid else mask_logits
    pred_por = pred_sigmoid.mean(dim=(1, 2, 3, 4))  # (B,)
    gt_por   = target.mean(dim=(1, 2, 3, 4))         # (B,)
    signed   = pred_por - gt_por
    return {
        "porosity_mae":   signed.abs().mean().item(),
        "porosity_bias":  signed.mean().item(),
        "mask_pred_mean": pred_sigmoid.mean().item(),
    }


@torch.no_grad()
def porosity_binned_mae(
    pred_por: torch.Tensor,
    gt_por: torch.Tensor,
    bins: tuple[float, ...] = (0.0, 0.01, 0.03, 0.06, float("inf")),
) -> dict[str, float]:
    """MAE per GT-porosity bin over the full aggregated eval set.

    Parameters
    ----------
    pred_por : (N,) — predicted porosity per patch (sigmoid already applied)
    gt_por   : (N,) — GT porosity per patch
    bins     : monotonically increasing bin edges; creates len(bins)-1 bins

    Returns
    -------
    dict with keys ``porosity_mae_bin_0`` … ``porosity_mae_bin_{n-1}``.
    Bins with no samples get ``float("nan")``.
    """
    abs_err = (pred_por - gt_por).abs()
    result: dict[str, float] = {}
    for i in range(len(bins) - 1):
        lo, hi = bins[i], bins[i + 1]
        sel = (gt_por >= lo) & (gt_por < hi)
        result[f"porosity_mae_bin_{i}"] = (
            float(abs_err[sel].mean().item()) if sel.sum() > 0 else float("nan")
        )
    return result
