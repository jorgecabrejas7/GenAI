"""Compose the total VAE loss from XCT recon + mask + KL components."""

from __future__ import annotations

from typing import Any

import torch

from poregen.losses.recon import get_recon_loss
from poregen.losses.mask import combined_mask_loss
from poregen.losses.kl import kl_divergence, beta_schedule
from poregen.models.vae.base import VAEOutput


def compute_total_loss(
    output: VAEOutput,
    batch: dict[str, torch.Tensor],
    step: int,
    cfg: dict[str, Any],
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
        ``loss`` and optionally ``training`` sections.  No internal
        defaults — all values must be present in *cfg*.

    Returns
    -------
    dict with keys:
        total, xct_loss, mask_bce, mask_dice (or mask_tversky),
        kl, beta, freebits_used, kl_per_channel
    """
    c = cfg["loss"]

    # ── XCT reconstruction ─────────────────────────────────────────
    recon_fn = get_recon_loss(c["xct_loss_type"])
    xct_loss = recon_fn(output.xct_logits, batch["xct"])

    # ── Mask — class-balanced BCE + Dice/Tversky ───────────────────
    # pos_weight from config (EDA: phi_mean=0.019 → correct value ≈ 51).
    # Using a fixed value avoids instability from per-batch estimates on
    # very sparse or near-zero porosity patches.
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
    )

    # ── KL ─────────────────────────────────────────────────────────
    kl, freebits_used, kl_per_channel = kl_divergence(
        output.mu, output.logvar, free_bits=c["kl_free_bits"],
    )
    beta = beta_schedule(step, c["kl_warmup_steps"], c["kl_max_beta"])

    # ── Total ──────────────────────────────────────────────────────
    total = (
        c["xct_weight"] * xct_loss
        + mask_dict["mask_total"]
        + beta * kl
    )

    result: dict[str, Any] = {
        "total": total,
        "xct_loss": xct_loss,
        "mask_bce": mask_dict["mask_bce"],
        "kl": kl,
        "beta": beta,
        "freebits_used": freebits_used,
        "kl_per_channel": kl_per_channel,  # (C,) — for monitoring and ablation
    }
    # Add the region loss under its actual key (mask_dice or mask_tversky)
    for k, v in mask_dict.items():
        if k not in ("mask_total", "mask_bce"):
            result[k] = v

    return result
