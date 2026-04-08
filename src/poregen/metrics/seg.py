"""Segmentation metrics with explicit empty-GT handling."""

from __future__ import annotations

import torch


@torch.no_grad()
def segmentation_metrics(
    logits: torch.Tensor,
    target: torch.Tensor,
    threshold: float = 0.5,
) -> dict[str, float]:
    """Compute Dice, IoU, precision, recall, F1.

    Returns ``*_all`` (computed over the full batch) and ``*_pos_only``
    (excluding samples where the ground-truth mask is entirely empty).

    Parameters
    ----------
    logits : (B, 1, D, H, W)
        Raw mask logits from the model.
    target : (B, 1, D, H, W)
        Binary ground-truth mask {0, 1}.
    threshold : float
        Sigmoid threshold for binarising predictions.

    Returns
    -------
    dict
        Keys: dice_all, dice_pos_only, iou_all, iou_pos_only,
        precision_all, precision_pos_only, recall_all, recall_pos_only,
        f1_all, f1_pos_only.
    """
    pred = (torch.sigmoid(logits) >= threshold).float()

    results: dict[str, float] = {}
    for suffix, mask_fn in [("all", lambda _: True), ("pos_only", lambda gt_sum: gt_sum > 0)]:
        tp_total = 0.0
        fp_total = 0.0
        fn_total = 0.0
        dice_sum = 0.0
        iou_sum = 0.0
        count = 0

        for b in range(pred.shape[0]):
            p = pred[b].flatten()
            t = target[b].flatten()
            gt_sum = t.sum().item()

            if not mask_fn(gt_sum):
                continue

            tp = (p * t).sum().item()
            fp = (p * (1 - t)).sum().item()
            fn = ((1 - p) * t).sum().item()

            tp_total += tp
            fp_total += fp
            fn_total += fn

            # Per-sample Dice and IoU
            dice_val = (2 * tp + 1e-7) / (2 * tp + fp + fn + 1e-7)
            iou_val = (tp + 1e-7) / (tp + fp + fn + 1e-7)
            dice_sum += dice_val
            iou_sum += iou_val
            count += 1

        if count == 0:
            results[f"dice_{suffix}"] = 0.0
            results[f"iou_{suffix}"] = 0.0
            results[f"precision_{suffix}"] = 0.0
            results[f"recall_{suffix}"] = 0.0
            results[f"f1_{suffix}"] = 0.0
        else:
            precision = (tp_total + 1e-7) / (tp_total + fp_total + 1e-7)
            recall = (tp_total + 1e-7) / (tp_total + fn_total + 1e-7)
            f1 = (2 * precision * recall) / (precision + recall + 1e-7)

            results[f"dice_{suffix}"] = dice_sum / count
            results[f"iou_{suffix}"] = iou_sum / count
            results[f"precision_{suffix}"] = precision
            results[f"recall_{suffix}"] = recall
            results[f"f1_{suffix}"] = f1

    return results


@torch.no_grad()
def porosity_error(
    mask_logits: torch.Tensor,
    target: torch.Tensor,
) -> dict[str, float]:
    """Mean absolute error between predicted and ground-truth porosity per patch.

    Porosity is the mean voxel activation over the patch volume.  Reporting
    per-batch mean ± std lets you track whether the VAE learns pore fraction
    globally, independently of local texture accuracy.

    Parameters
    ----------
    mask_logits : (B, 1, D, H, W)  — raw logits
    target      : (B, 1, D, H, W)  — binary ground-truth mask {0, 1}

    Returns
    -------
    dict with ``porosity_error`` (mean |pred − gt|) and ``porosity_error_std``.
    """
    pred_por = torch.sigmoid(mask_logits).mean(dim=(1, 2, 3, 4))  # (B,)
    gt_por   = target.mean(dim=(1, 2, 3, 4))                       # (B,)
    err = (pred_por - gt_por).abs()
    return {
        "porosity_error":     err.mean().item(),
        "porosity_error_std": err.std().item() if err.numel() > 1 else 0.0,
    }
