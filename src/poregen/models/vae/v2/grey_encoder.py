"""GreyEncoder: a grey-only front end for the production VAE (OQ-72).

The production encoder (``v2.conv_noattn_dualbranch_cls``) takes
``cat([xct, pore, air])`` and the decoder renders grey AND the 3-class label
from the latent.  Given no mask planes it encodes "no pores" (pore Dice 0.000
on real val patches, 2026-11-17), so it cannot segment a new scan.

This module is the same two-branch trunk with ``in_channels = 1``, trained by
regression onto the stored posterior mean ``mu`` of the production encoder
(the latent store's channels ``0..C-1``).  The decoder, the LDM and every
result stay untouched: for a new scan,
``grey -> GreyEncoder -> mu -> production decoder -> grey + label``.

Privileged information: the mask planes are available at training time only
(they define the target latent); the student learns to reach the same latent
from grey alone.
"""
from __future__ import annotations

import torch
import torch.nn as nn

from poregen.models.nn.blocks import down_block_v2
from poregen.models.vae.base import VAEConfig


class GreyEncoder(nn.Module):
    """Two-branch grey-only encoder, ``(B, 1, 64, 64, 64) -> mu (B, z, 16, 16, 16)``."""

    def __init__(self, cfg: VAEConfig) -> None:
        super().__init__()
        if cfg.in_channels != 1:
            raise ValueError(f"GreyEncoder takes the grey channel only; got in_channels={cfg.in_channels}")
        self.cfg = cfg
        ch = cfg.channel_schedule()
        enc_out_ch = ch[cfg.n_blocks - 1]

        def _branch() -> nn.Sequential:
            blocks, in_ch = [], 1
            for i in range(cfg.n_blocks):
                blocks.append(down_block_v2(in_ch, ch[i]))
                in_ch = ch[i]
            return nn.Sequential(*blocks)

        self.encoder_a = _branch()
        self.encoder_b = _branch()
        self.fusion = nn.Conv3d(2 * enc_out_ch, enc_out_ch, kernel_size=1, bias=False)
        self.to_mu = nn.Conv3d(enc_out_ch, cfg.z_channels, 1)

    def forward(self, xct: torch.Tensor) -> torch.Tensor:
        """``xct``: grey level in [0, 1]. Returns the predicted posterior mean."""
        h = self.fusion(torch.cat([self.encoder_a(xct), self.encoder_b(xct)], dim=1))
        return self.to_mu(h)


def load_grey_encoder(ckpt: str, device: torch.device) -> GreyEncoder:
    """A trained GreyEncoder from ``train_grey_encoder.py``'s checkpoint."""
    state = torch.load(ckpt, map_location="cpu", weights_only=False)
    model = GreyEncoder(VAEConfig(**state["vae_config"]))
    model.load_state_dict(state["model"], strict=True)
    return model.eval().to(device)
