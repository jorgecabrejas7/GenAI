"""Compose the total VAE loss from XCT recon + mask + KL components."""

from __future__ import annotations

from typing import Any

import torch

from poregen.losses.recon import get_recon_loss
from poregen.losses.mask import combined_mask_loss
from poregen.losses.kl import kl_divergence, kl_divergence_flat, beta_schedule
from poregen.models.vae.base import VAEOutput


def compute_total_loss(
    output: VAEOutput,
    batch: dict[str, torch.Tensor],
    step: int,
    cfg: dict[str, Any],
    *,
    pos_weight: torch.Tensor | None = None,
    mask_sigmoid: torch.Tensor | None = None,
) -> dict[str, torch.Tensor | float]:
    """Compute combined VAE loss.

    Parameters
    ----------
    output : VAEOutput
    batch : dict
        Must contain ``"xct"`` and ``"mask"`` tensors.
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

    Returns
    -------
    dict with keys:
        total, xct_loss, kl, beta, kl_collapsed_fraction, kl_per_channel,
        and (only when ``output.mask_logits is not None``) mask_bce,
        mask_dice (or mask_tversky).

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
    """
    c = cfg["loss"]

    # ── XCT reconstruction ────────────────────────────────────────────
    recon_fn = get_recon_loss(c["xct_loss_type"])
    xct_loss = recon_fn(output.xct_out, batch["xct"])

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

    # ── KL — flat (B, C) latent (VRRAE) vs spatial (B, C, d, h, w) ─────
    kl_fn = kl_divergence_flat if output.mu.ndim == 2 else kl_divergence
    kl, kl_collapsed_fraction, kl_per_channel = kl_fn(
        output.mu, output.logvar, free_bits=c["kl_free_bits"],
    )
    beta = beta_schedule(step, c["kl_warmup_steps"], c["kl_max_beta"])

    # ── Total ─────────────────────────────────────────────────────────
    total = (
        c["xct_weight"] * xct_loss
        + mask_total
        + beta * kl
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

    return result
