"""Loss functions for PoreGen training.

The names re-exported here are the VAE ones — the pieces
``compute_total_loss`` composes.  The LDM's decoded-space auxiliary loss
lives in :mod:`poregen.losses.decoded` and is imported from there
directly: it needs a frozen VAE and a noise schedule, so it is not a
drop-in term and should not look like one.
"""

from poregen.losses.recon import l1_loss, mse_loss, charbonnier_loss
from poregen.losses.mask import bce_logits_loss, dice_loss, focal_loss, tversky_loss, combined_mask_loss
from poregen.losses.kl import kl_divergence, beta_schedule
from poregen.losses.total import compute_total_loss

__all__ = [
    "l1_loss", "mse_loss", "charbonnier_loss",
    "bce_logits_loss", "dice_loss", "focal_loss", "tversky_loss", "combined_mask_loss",
    "kl_divergence", "beta_schedule",
    "compute_total_loss",
]
