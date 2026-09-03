"""Decoded-space auxiliary losses for the LDM (ldm06/aux).

The ε (or v) objective trains entirely in latent space and never opens the
decoder.  That is efficient and it is also blind to a whole class of defect:
two latents that differ only in where they put air, or whether their grey
level agrees with their class label, can carry the SAME residual.  The
diagnostics keep reporting exactly those defects — air inside the specimen
envelope, dark regions the class head calls material, a delivered porosity
that ignores ``cond_por``.

The fix is the one Berrada et al. 2025 (arXiv:2411.04873) use for latent video
diffusion: push x̂₀ through the frozen decoder during training and score the
result in the space the defect is defined in.  Four terms, each measuring one
thing the latent loss cannot see:

============================  ==========================================
``air_outside_material``      p(air) where ``cond_material`` says specimen
``pore_dice``                 soft Dice of p(pore) against the real label
``porosity_consistency``      delivered φ vs the φ the patch was conditioned on
``grey_agreement``            a pore voxel must be dark, a material voxel bright
============================  ==========================================

Three things make this affordable and safe, and all three are load-bearing:

**Only the lowest-t items.**  x̂₀ from a high-t step is a blur — there is no
pore structure in it to score, so every term would be measuring the schedule
rather than the model.  ``t_max_frac`` keeps the lowest quartile of the
schedule.

**A hard sub-batch cap.**  MEASURED on the real r08 decoder (z=4, base 32,
n_blocks 2), the intermediate activations of a 64³ decode with a live autograd
graph come to **0.33 GB per item in float32, ~0.20 GB under bf16 autocast** —
the 64³ stage alone carries a (N, 32, 64³) tensor five times over.  Decoding a
whole ``batch_size`` 256 batch would therefore need ~84 GB (fp32) / ~51 GB
(bf16) on top of the denoiser, which does not fit beside anything else on the
GB10.  ``decoded_max_items`` caps it — 32 items is ~10.5 GB / ~6.3 GB — and
the items chosen are the LOWEST-t ones, i.e. the sharpest x̂₀ available.

**The VAE stays frozen in every sense.**  Its parameters arrive with
``requires_grad_(False)`` so no gradient accumulates on them, and
:meth:`DecodedAuxLoss.__call__` puts it in ``eval()`` — the r08 decoder is
BatchNorm3d, and a decode in train mode would update the frozen VAE's running
statistics from generated latents.  ``requires_grad_(False)`` does NOT prevent
that; only ``eval()`` does.  Gradient still flows back through the decoder to
x̂₀, which is the whole point.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from poregen.models.vae.base import (
    CLASS_AIR,
    CLASS_PORE,
    decode_class_probs,
    decode_xct,
)

__all__ = [
    "AIR_GREY_THRESHOLD",
    "DecodedLossConfig",
    "DecodedAuxLoss",
    "air_outside_material",
    "grey_agreement",
    "pore_dice",
    "porosity_consistency",
    "ramp_weight",
    "select_decoded_items",
]

# Grey level below which a voxel reads as void, on the [0, 1] scale the XCT
# head emits.  182/255 is the real-calibrated ABSOLUTE threshold from the air
# audit (campaign 04): Dice 0.842 against the segmented masks of real volumes.
# It is the same number the eval-v3 detector uses, so the training signal and
# the measurement agree on what "dark" means.
AIR_GREY_THRESHOLD = 182.0 / 255.0

# A cell counts as specimen when the envelope fraction is essentially 1.  The
# envelope is fractional only where the outer surface or a drilled hole cuts a
# cell, and those cells legitimately contain air — scoring them would punish
# the model for the geometry it was told about.
MATERIAL_FULL = 0.99


# ── term (a): air outside the specimen envelope ───────────────────────────────

def air_outside_material(
    p_air: torch.Tensor,
    cond_material: torch.Tensor,
    threshold: float = MATERIAL_FULL,
) -> torch.Tensor:
    """Mean p(air) over the voxels the envelope says are inside the specimen.

    ``cond_material`` lives on the latent grid (one value per ``ds³`` voxels)
    and is upsampled with nearest-neighbour, which is exactly what the value
    means: the fraction of that cell inside the envelope, constant across it.

    Only cells at ``> threshold`` are scored.  A partially-filled cell sits on
    the specimen surface or a drilled hole and genuinely contains air, so
    including it would penalise the model for obeying its conditioning.

    Parameters
    ----------
    p_air         : (B, D, H, W) — decoded air probability
    cond_material : (B, 1, d, h, w) — envelope fraction per latent cell
    """
    up = F.interpolate(cond_material.float(), size=p_air.shape[-3:], mode="nearest")
    mask = (up.squeeze(1) > threshold).to(p_air.dtype)
    return (p_air * mask).sum() / mask.sum().clamp(min=1.0)


# ── term (b): soft Dice against the real pore mask ────────────────────────────

def pore_dice(
    p_pore: torch.Tensor,
    label: torch.Tensor,
    smooth: float = 1.0,
) -> torch.Tensor:
    """``1 − Dice`` between the decoded pore probability and the real pores.

    Pore only.  Material is the background and its Dice sits near 1 whatever
    the model does; air is already covered by
    :func:`air_outside_material`, which scores it where it is actually wrong.

    Parameters
    ----------
    p_pore : (B, D, H, W) — decoded pore probability
    label  : (B, D, H, W) — the source patch's voxel label, 0/1/2
    """
    target = (label == CLASS_PORE).to(p_pore.dtype)
    pred_flat, target_flat = p_pore.flatten(1), target.flatten(1)
    intersection = (pred_flat * target_flat).sum(1)
    cardinality = pred_flat.sum(1) + target_flat.sum(1)
    dice = (2.0 * intersection + smooth) / (cardinality + smooth)
    return 1.0 - dice.mean()


# ── term (c): delivered porosity vs the conditioned porosity ──────────────────

def porosity_consistency(p_pore: torch.Tensor, phi_target: torch.Tensor) -> torch.Tensor:
    """``|mean p(pore) − φ|`` per item, averaged.

    ``cond_por`` is the standardised ``log(φ + 1e-3)`` the patch was
    conditioned on; ``phi_target`` is the raw φ the store's index recorded for
    the same patch, so the comparison happens on the scale the metric
    ``gen/por_cond_mae`` is reported on rather than in log space.

    Parameters
    ----------
    p_pore     : (B, D, H, W)
    phi_target : (B,) — raw pore fraction
    """
    delivered = p_pore.flatten(1).mean(1)
    return (delivered - phi_target.reshape(-1).to(delivered.dtype)).abs().mean()


# ── term (d): grey level vs class label ───────────────────────────────────────

def grey_agreement(
    probs: torch.Tensor,
    grey: torch.Tensor,
    threshold: float = AIR_GREY_THRESHOLD,
) -> torch.Tensor:
    """Penalise a bright voxel called pore and a dark voxel called material.

    Both heads decode from ONE latent, so they cannot legitimately disagree.
    The penalty is one-sided per class and linear in the disagreement::

        p_pore     · relu(grey − threshold)      a pore that is too bright
        p_material · relu(threshold − grey)      material that is too dark

    where ``p_material = 1 − p_pore − p_air``.  Air is deliberately unscored:
    exterior air and the drilled holes are dark AND correct.

    This is the void-mask audit's finding (dark-but-unmasked regions in
    generated volumes) written as a training signal.

    Parameters
    ----------
    probs : (B, 3, D, H, W) — class probabilities, material/pore/air
    grey  : (B, D, H, W) — decoded grey level in [0, 1]
    """
    p_pore = probs[:, CLASS_PORE]
    p_air = probs[:, CLASS_AIR]
    p_material = 1.0 - p_pore - p_air
    too_bright = p_pore * F.relu(grey - threshold)
    too_dark = p_material * F.relu(threshold - grey)
    return (too_bright + too_dark).mean()


# ── sub-batch selection and ramp ──────────────────────────────────────────────

def select_decoded_items(
    t: torch.Tensor,
    T: int,
    t_max_frac: float,
    max_items: int,
) -> torch.Tensor:
    """Indices of the items whose x̂₀ is worth decoding, lowest ``t`` first.

    Eligible items are those with ``t < t_max_frac · T`` — above that the x̂₀
    estimate is a blur and every decoded term would measure the schedule
    instead of the model.  Of those, the ``max_items`` with the LOWEST ``t``
    are kept, because they carry the sharpest structure per unit of decode
    memory.

    Returns an empty tensor when no item in the batch qualifies, which is a
    normal outcome for a small batch and not an error.

    Parameters
    ----------
    t          : (B,) long — the timestep drawn for each item
    T          : total diffusion steps
    t_max_frac : fraction of the schedule that counts as "low t"
    max_items  : hard cap on the number of items decoded
    """
    t_max = t_max_frac * float(T)
    eligible = (t.float() < t_max).nonzero(as_tuple=False).flatten()
    if eligible.numel() == 0 or max_items <= 0:
        return eligible[:0]
    order = torch.argsort(t[eligible])
    return eligible[order[: int(max_items)]]


def ramp_weight(step: int, ramp_steps: int) -> float:
    """Linear 0 → 1 ramp over the first ``ramp_steps`` optimiser steps.

    The latent objective has to find the manifold before a decoded critique of
    x̂₀ means anything; a decoded loss at full weight from step 0 would be
    scoring noise.  ``ramp_steps <= 0`` disables the ramp.
    """
    if ramp_steps <= 0:
        return 1.0
    return min(1.0, float(step + 1) / float(ramp_steps))


# ── configuration + orchestration ─────────────────────────────────────────────

@dataclass(frozen=True)
class DecodedLossConfig:
    """The ``loss.decoded`` block of a resolved experiment config."""

    t_max_frac: float = 0.25
    ramp_steps: int = 10000
    max_items: int = 32
    w_air_outside_material: float = 1.0
    w_pore_dice: float = 1.0
    w_porosity_consistency: float = 1.0
    w_grey_agreement: float = 0.5

    @classmethod
    def from_cfg(cls, cfg: dict[str, Any]) -> "DecodedLossConfig | None":
        """Build from ``cfg["loss"]["decoded"]``, or None when it is off."""
        block = (cfg.get("loss") or {}).get("decoded") or {}
        if not bool(block.get("enabled", False)):
            return None
        weights = block.get("weights") or {}
        return cls(
            t_max_frac=float(block.get("t_max_frac", 0.25)),
            ramp_steps=int(block.get("ramp_steps", 10000)),
            max_items=int(block.get("decoded_max_items", 32)),
            w_air_outside_material=float(weights.get("air_outside_material", 1.0)),
            w_pore_dice=float(weights.get("pore_dice", 1.0)),
            w_porosity_consistency=float(weights.get("porosity_consistency", 1.0)),
            w_grey_agreement=float(weights.get("grey_agreement", 0.5)),
        )


class DecodedAuxLoss:
    """Decode the lowest-t x̂₀ estimates and score them.

    Holds everything the decode needs that does not change per step: the
    frozen VAE, the latent store's denormalisation stats, and the weights.

    Parameters
    ----------
    config      : the resolved ``loss.decoded`` block
    vae         : the frozen 3-class VAE (``decoder``, ``xct_head``, ``class_head``)
    latent_mean, latent_std : (C, 1, 1, 1) per-channel normalisation stats.
        The LDM trains in normalised space; the decoder expects raw latents.
    """

    def __init__(
        self,
        config: DecodedLossConfig,
        vae: nn.Module,
        latent_mean: torch.Tensor | float,
        latent_std: torch.Tensor | float,
    ) -> None:
        for attr in ("decoder", "xct_head", "class_head"):
            if not hasattr(vae, attr):
                raise TypeError(
                    f"The decoded auxiliary loss needs a 3-class VAE with .{attr}; "
                    f"got {type(vae).__name__}.  It scores material/pore/air "
                    f"probabilities, not a binary mask."
                )
        if any(p.requires_grad for p in vae.parameters()):
            raise ValueError(
                "The decoded auxiliary loss requires a FROZEN VAE — call "
                "requires_grad_(False) on its parameters first.  Back-propagating "
                "into the decoder would let the LDM move the target it is being "
                "scored against."
            )
        self.config = config
        self.vae = vae
        self.latent_mean = latent_mean
        self.latent_std = latent_std

    def __call__(
        self,
        *,
        model_out: torch.Tensor,
        z_t: torch.Tensor,
        t: torch.Tensor,
        schedule: Any,
        batch: dict[str, Any],
        step: int,
        autocast_dtype: torch.dtype = torch.bfloat16,
    ) -> tuple[torch.Tensor | None, dict[str, float]]:
        """Score the decoded x̂₀ of the lowest-t items in this batch.

        Parameters
        ----------
        model_out : (B, C, d, h, w) — the denoiser's raw output (ε or v)
        z_t       : (B, C, d, h, w) — the noisy latent it was given
        t         : (B,) long — the timestep of each item
        schedule  : DDPMSchedule — supplies the objective-aware ``predict_x0``
        batch     : the device batch; needs ``cond_material``, ``label``, ``phi``
        step      : optimiser step, for the ramp

        Returns
        -------
        ``(loss, metrics)``.  ``loss`` is None when no item qualified, in which
        case the caller adds nothing.  ``metrics`` is always populated so a
        skipped step is visible in the logs rather than silently absent.
        """
        cfg = self.config
        idx = select_decoded_items(t, schedule.T, cfg.t_max_frac, cfg.max_items)
        scale = ramp_weight(step, cfg.ramp_steps)
        metrics: dict[str, float] = {"aux_items": float(idx.numel()), "aux_ramp": scale}
        if idx.numel() == 0:
            return None, metrics

        missing = [k for k in ("cond_material", "label", "phi") if k not in batch]
        if missing:
            raise KeyError(
                f"The decoded auxiliary loss needs {missing} in the batch. "
                f"`label` is served only when loss.decoded.enabled is true — "
                f"check that build_latent_dataloaders saw the same config."
            )

        # x̂₀ is NOT clamped here.  The ±10 clamp guards a reverse process
        # against one bad step; this loss only ever looks at the lowest
        # quartile of the schedule, where sqrt(ᾱ) is large and predict_x0 is
        # well conditioned — and a clamp would zero the gradient exactly on
        # the items that are most wrong.
        x0 = schedule.predict_x0(z_t[idx], t[idx], model_out[idx])
        probs, grey = self._decode(x0, autocast_dtype)

        p_pore, p_air = probs[:, CLASS_PORE], probs[:, CLASS_AIR]
        terms = {
            "aux_air_outside_material": (
                cfg.w_air_outside_material,
                air_outside_material(p_air, batch["cond_material"][idx]),
            ),
            "aux_pore_dice": (
                cfg.w_pore_dice,
                pore_dice(p_pore, batch["label"][idx]),
            ),
            "aux_porosity_consistency": (
                cfg.w_porosity_consistency,
                porosity_consistency(p_pore, batch["phi"][idx]),
            ),
            "aux_grey_agreement": (
                cfg.w_grey_agreement,
                grey_agreement(probs, grey),
            ),
        }

        total = z_t.new_zeros(())
        for name, (weight, value) in terms.items():
            total = total + weight * value
            metrics[name] = float(value.detach())
        total = scale * total
        metrics["aux_total"] = float(total.detach())
        return total, metrics

    def _decode(
        self,
        x0: torch.Tensor,
        autocast_dtype: torch.dtype,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Denormalise and decode, returning ``(class probs, grey)``.

        ``eval()`` is mandatory, not hygiene: the r08 decoder is BatchNorm3d,
        so a decode in train mode would fold generated latents into the frozen
        VAE's running statistics.  Freezing the parameters does not stop that.
        Gradient still reaches ``x0`` — only the VAE's own parameters are out
        of the graph.
        """
        self.vae.eval()
        mean, std = self.latent_mean, self.latent_std
        if isinstance(mean, torch.Tensor):
            mean = mean.to(x0.device)
        if isinstance(std, torch.Tensor):
            std = std.to(x0.device)
        z = x0 * std + mean
        with torch.autocast(device_type=x0.device.type, dtype=autocast_dtype,
                            enabled=x0.device.type == "cuda"):
            dec = self.vae.decoder(z)
            xct_out = self.vae.xct_head(dec)
            class_logits = self.vae.class_head(dec)
        probs = decode_class_probs(class_logits.float())
        grey = decode_xct(xct_out.float()).squeeze(1)
        return probs, grey
