"""Compose the total VAE loss from XCT recon + mask + KL components."""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn

from poregen.losses.recon import get_recon_loss
from poregen.losses.mask import combined_class_loss, combined_mask_loss
from poregen.losses.kl import beta_schedule, kl_divergence, kl_divergence_flat, kl_rombach_sum
from poregen.models.vae.base import CLASS_AIR, CLASS_PORE, VAEOutput


def compute_total_loss(
    output: VAEOutput,
    batch: dict[str, torch.Tensor],
    step: int,
    cfg: dict[str, Any],
    *,
    pos_weight: torch.Tensor | None = None,
    mask_sigmoid: torch.Tensor | None = None,
    class_weights: torch.Tensor | None = None,
    class_probs: torch.Tensor | None = None,
    perceptual: nn.Module | None = None,
) -> dict[str, torch.Tensor | float]:
    """Compute combined VAE loss.

    Parameters
    ----------
    output : VAEOutput
    batch : dict
        Must contain ``"xct"``, plus ``"mask"`` for a binary-head variant or
        ``"label"`` (int64 class index) for a 3-class-head one.
    step : int
        Current global training step (for β scheduling).
    cfg : dict
        Config dict as returned by ``load_config()``.  Must contain
        ``loss`` and optionally ``training`` sections.
    pos_weight : torch.Tensor, optional
        Pre-allocated scalar tensor for the BCE positive-class weight.
        **Callers should create this once at training start** and pass it
        here every step to avoid a device allocation on every call.
        If ``None``, the tensor is created from ``cfg["loss"]["mask_bce_pos_weight"]``
        (original behaviour — safe but slightly slower).
    mask_sigmoid : torch.Tensor, optional
        Pre-computed ``torch.sigmoid(output.mask_logits)``.  When the eval
        loop already has the sigmoid (e.g. for metrics), pass it here to
        avoid computing it twice inside the loss.
    class_weights : torch.Tensor, optional
        ``(3,)`` per-class weight for the r08 cross-entropy, normally inverse
        to the training-split class frequency.  **Pre-allocate it once per
        run** and pass it every step; when ``None`` it is built from
        ``cfg["loss"]["class_weights"]`` on every call.
    class_probs : torch.Tensor, optional
        Pre-computed ``softmax(output.class_logits, dim=1)``, for the same
        reason as ``mask_sigmoid``.
    perceptual : nn.Module, optional
        The frozen slice-LPIPS module
        (:func:`poregen.losses.perceptual.build_perceptual`). Required when
        ``loss.perceptual_weight > 0``; the loss refuses to run without it
        rather than silently drop the term.

    Returns
    -------
    dict with keys:
        total, xct_loss, kl, beta, kl_collapsed_fraction, kl_per_channel,
        and (only when ``output.mask_logits is not None``) mask_bce,
        mask_dice (or mask_tversky); or (only when
        ``output.class_logits is not None``) class_ce, class_dice,
        class_dice_<i> per scored class, and class_total.
        Opt-in keys (absent under the default config): ``lpips`` when
        ``loss.perceptual_weight > 0``; ``kl_rombach`` when
        ``loss.kl_reduction == "rombach_sum"``; ``rec_total`` (the
        reconstruction term the adaptive adversarial weight balances against)
        when ``loss.disc_adaptive`` is true.

    Notes
    -----
    The mask term is entirely skipped when ``output.mask_logits is None``
    (e.g. the XCT-only-decoder ``v2.vrrae`` variant, which has no
    ``mask_head``) — this is a guard, not a removal: variants that still
    populate ``mask_logits`` are unaffected and go through the exact same
    mask-loss code path as before.

    KL uses :func:`poregen.losses.kl.kl_divergence` for spatial latents
    (``mu.ndim > 2``, i.e. ``(B, C, d, h, w)``) and
    :func:`poregen.losses.kl.kl_divergence_flat` for flat latents
    (``mu.ndim == 2``, i.e. ``(B, C)`` — the VRRAE bottleneck's rank-``C``
    SVD-coefficient posterior).  Free-bits clamping and the beta schedule
    are unchanged in both cases.

    ``loss.kl_reduction`` picks the KL term that enters ``total``:
    ``per_channel_mean`` (default, the historical recipe — per-channel KL
    averaged over batch and space, summed over channels, free bits) or
    ``rombach_sum`` (Rombach et al. 2022: KL summed over every latent element
    per sample, batch mean, divided by the voxels per patch because our
    reconstruction terms are per-voxel means; no free bits). ``kl``,
    ``kl_per_channel`` and ``kl_collapsed_fraction`` are reported either way.
    """
    c = cfg["loss"]

    # ── XCT reconstruction ────────────────────────────────────────────
    recon_fn = get_recon_loss(c["xct_loss_type"])
    xct_loss = recon_fn(output.xct_out, batch["xct"])

    # ── Perceptual (opt-in): LPIPS on paired grey slices, added to the
    #    per-voxel reconstruction term as Rombach's rec = |x-x'| + p*LPIPS ──
    p_weight = float(c.get("perceptual_weight", 0.0))
    lpips_loss: torch.Tensor | None = None
    if p_weight > 0.0:
        if perceptual is None:
            raise ValueError(
                "loss.perceptual_weight > 0 but no perceptual module was "
                "passed; build it with poregen.losses.perceptual.build_perceptual."
            )
        lpips_loss = perceptual(output.xct_out, batch["xct"])

    # ── Mask — class-balanced BCE + Dice/Tversky (skipped when the model
    #    variant has no mask head, e.g. v2.vrrae) ───────────────────────
    mask_dict: dict[str, Any] = {}
    if output.mask_logits is not None:
        # pos_weight should ideally be pre-allocated once per training run.
        # Falling back to on-the-fly allocation keeps the API backward-compat.
        if pos_weight is None:
            pos_weight = torch.tensor(
                float(c["mask_bce_pos_weight"]),
                dtype=output.mask_logits.dtype,
                device=output.mask_logits.device,
            )

        mask_dict = combined_mask_loss(
            output.mask_logits,
            batch["mask"],
            bce_weight=c["mask_bce_weight"],
            dice_weight=c["mask_dice_weight"],
            use_tversky=c["use_tversky"],
            tversky_alpha=c.get("tversky_alpha", 0.3),
            tversky_beta=c.get("tversky_beta", 0.7),
            pos_weight=pos_weight,
            use_focal=c.get("use_focal", False),
            focal_gamma=c.get("focal_gamma", 2.0),
            focal_alpha=c.get("focal_alpha", 0.25),
            sigmoid=mask_sigmoid,
        )
    mask_total = mask_dict.get("mask_total", torch.zeros((), device=output.xct_out.device))

    # ── Three-class voxel label (r08+): weighted CE + soft Dice over the
    #    minority classes.  Mutually exclusive with the binary mask head —
    #    a variant emits one or the other, never both. ───────────────────
    class_dict: dict[str, Any] = {}
    if output.class_logits is not None:
        if class_weights is None:
            w = c.get("class_weights")
            if not w:
                raise ValueError(
                    "loss.class_weights is required for a 3-class head — "
                    "unweighted cross-entropy is dominated by the material "
                    "class and both minority classes collapse. Compute them "
                    "with `python scripts/build_split_v4.py --segmentation <name> --stage weights` "
                    "and paste the result into the experiment config."
                )
            class_weights = torch.tensor(
                [float(x) for x in w],
                dtype=output.class_logits.dtype,
                device=output.class_logits.device,
            )
        class_dict = combined_class_loss(
            output.class_logits,
            batch["label"],
            ce_weight=c["class_ce_weight"],
            dice_weight=c["class_dice_weight"],
            class_weights=class_weights,
            dice_classes=tuple(c.get("class_dice_classes", (CLASS_PORE, CLASS_AIR))),
            probs=class_probs,
        )
    class_total = class_dict.get(
        "class_total", torch.zeros((), device=output.xct_out.device))

    # ── KL — flat (B, C) latent (VRRAE) vs spatial (B, C, d, h, w) ─────
    kl_fn = kl_divergence_flat if output.mu.ndim == 2 else kl_divergence
    kl, kl_collapsed_fraction, kl_per_channel = kl_fn(
        output.mu, output.logvar, free_bits=c["kl_free_bits"],
    )
    beta = beta_schedule(step, c["kl_warmup_steps"], c["kl_max_beta"])

    kl_reduction = c.get("kl_reduction", "per_channel_mean")
    kl_rombach: torch.Tensor | None = None
    if kl_reduction == "per_channel_mean":
        kl_term = kl
    elif kl_reduction == "rombach_sum":
        if float(c["kl_free_bits"]) != 0.0:
            raise ValueError(
                "loss.kl_reduction 'rombach_sum' has no free bits; set "
                "loss.kl_free_bits to 0."
            )
        kl_rombach = kl_rombach_sum(
            output.mu, output.logvar, n_voxels=batch["xct"][0].numel())
        kl_term = kl_rombach
    else:
        raise ValueError(
            f"loss.kl_reduction must be 'per_channel_mean' or 'rombach_sum', "
            f"got {kl_reduction!r}"
        )

    # ── Total ─────────────────────────────────────────────────────────
    rec_total = c["xct_weight"] * xct_loss
    if lpips_loss is not None:
        rec_total = rec_total + p_weight * lpips_loss
    total = (
        rec_total
        + mask_total
        + class_total
        + beta * kl_term
    )

    result: dict[str, Any] = {
        "total":          total,
        "xct_loss":       xct_loss,
        "kl":                    kl,
        "beta":                  beta,
        "kl_collapsed_fraction": kl_collapsed_fraction,
        "kl_per_channel":        kl_per_channel,   # (C,) — for monitoring and ablation
    }
    if mask_dict:
        result["mask_bce"] = mask_dict["mask_bce"]
        # Add the region loss under its actual key (mask_dice or mask_tversky)
        for k, v in mask_dict.items():
            if k not in ("mask_total", "mask_bce"):
                result[k] = v
    result.update(class_dict)
    if lpips_loss is not None:
        result["lpips"] = lpips_loss
    if kl_rombach is not None:
        result["kl_rombach"] = kl_rombach
    if c.get("disc_adaptive", False):
        result["rec_total"] = rec_total

    return result


def build_vae_loss_fn(cfg: dict[str, Any], device: torch.device):
    """``loss_fn(output, batch, step)`` for this config, on *device*.

    The one place a VAE loss is assembled from a config: it builds the frozen
    LPIPS module when ``loss.perceptual_weight > 0`` (otherwise none) and
    binds it, so training, evaluation and the smoke step score the same total.
    """
    from poregen.losses.perceptual import build_perceptual

    perceptual = build_perceptual(cfg, device)
    return lambda output, batch, step: compute_total_loss(
        output, batch, step, cfg, perceptual=perceptual)
