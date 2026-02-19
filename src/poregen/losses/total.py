"""Compose the total VAE loss from XCT recon + mask + KL components."""

from __future__ import annotations

from typing import Any

import torch

from poregen.losses.recon import get_recon_loss
from poregen.losses.mask import combined_mask_loss
from poregen.losses.kl import kl_divergence, beta_schedule
from poregen.models.vae.base import VAEOutput


# Default config values — override in your notebook / YAML.
_DEFAULTS: dict[str, Any] = {
    "xct_loss_type": "l1",
    "xct_weight": 1.0,
    "mask_bce_weight": 1.0,
    "mask_dice_weight": 1.0,
    "use_tversky": False,
    "tversky_alpha": 0.3,
    "tversky_beta": 0.7,
    "kl_free_bits": 0.25,
    "kl_warmup_steps": 5000,
    "kl_max_beta": 1.0,
}


def compute_total_loss(
    output: VAEOutput,
    batch: dict[str, torch.Tensor],
    step: int,
    cfg: dict[str, Any] | None = None,
) -> dict[str, torch.Tensor | float]:
    """Compute combined VAE loss.

    Parameters
    ----------
    output : VAEOutput
    batch : dict
        Must contain ``"xct"`` and ``"mask"`` tensors.
    step : int
        Current global training step (for β scheduling).
    cfg : dict, optional
        Override any key from ``_DEFAULTS``.

    Returns
    -------
    dict with keys:
        total, xct_loss, mask_bce, mask_dice (or mask_tversky),
        kl, beta, freebits_used
    """
    c = {**_DEFAULTS, **(cfg or {})}

    # ── XCT reconstruction ─────────────────────────────────────────
    recon_fn = get_recon_loss(c["xct_loss_type"])
    xct_loss = recon_fn(output.xct_logits, batch["xct"])

    # ── Mask ───────────────────────────────────────────────────────
    mask_dict = combined_mask_loss(
        output.mask_logits,
        batch["mask"],
        bce_weight=c["mask_bce_weight"],
        dice_weight=c["mask_dice_weight"],
        use_tversky=c["use_tversky"],
        tversky_alpha=c.get("tversky_alpha", 0.3),
        tversky_beta=c.get("tversky_beta", 0.7),
    )

    # ── KL ─────────────────────────────────────────────────────────
    kl, freebits_used = kl_divergence(
        output.mu, output.logvar, free_bits=c["kl_free_bits"],
    )
    beta = beta_schedule(step, c["kl_warmup_steps"], c["kl_max_beta"])

    # ── Total ──────────────────────────────────────────────────────
    total = (
        c["xct_weight"] * xct_loss
        + mask_dict["mask_total"]
        + beta * kl
    )

    result = {
        "total": total,
        "xct_loss": xct_loss,
        "mask_bce": mask_dict["mask_bce"],
        "kl": kl,
        "beta": beta,
        "freebits_used": freebits_used,
    }
    # Add the region loss under its actual key (mask_dice or mask_tversky)
    for k, v in mask_dict.items():
        if k not in ("mask_total", "mask_bce"):
            result[k] = v

    return result
