"""ConvVAE3DNoAttn — purely convolutional 3-D VAE (no skip connections, no attention).

First-generation architecture.  Checkpoint-compatible with all runs trained
before the v2 refactor — parameter names are unchanged.

Use this to ablate the contribution of bottleneck attention relative to
:class:`poregen.models.vae.v1.conv.ConvVAE3D`.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from poregen.models.nn.blocks import down_block_v1, up_block_v1, reparameterize
from poregen.models.vae.base import VAEConfig, VAEOutput
from poregen.models.vae.registry import register_vae


@register_vae("conv_noattn")
class ConvVAE3DNoAttn(nn.Module):
    """Purely convolutional 3-D VAE — no bottleneck attention, no skip connections.

    With default ``VAEConfig(n_blocks=2, base_channels=32)`` the encoder
    path is::

        (B,  2, 64, 64, 64)  →  down  →  (B,  32, 32, 32, 32)
                              →  down  →  (B,  64, 16, 16, 16)
                              →  1×1   →  mu / logvar  (B, 8, 16, 16, 16)

    Decoder mirrors in reverse, ending with two 1×1 heads (logits).
    """

    def __init__(self, cfg: VAEConfig) -> None:
        super().__init__()
        self.cfg = cfg
        ch = cfg.channel_schedule()

        # ── encoder ──
        enc: list[nn.Module] = []
        in_ch = cfg.in_channels
        for i in range(cfg.n_blocks):
            enc.append(down_block_v1(in_ch, ch[i]))
            in_ch = ch[i]
        self.encoder = nn.Sequential(*enc)

        # ── bottleneck → latent ──
        self.to_mu     = nn.Conv3d(ch[cfg.n_blocks - 1], cfg.z_channels, 1)
        self.to_logvar = nn.Conv3d(ch[cfg.n_blocks - 1], cfg.z_channels, 1)

        # ── decoder ──
        dec: list[nn.Module] = []
        in_ch = cfg.z_channels
        for i in range(cfg.n_blocks - 1, -1, -1):
            dec.append(up_block_v1(in_ch, ch[i]))
            in_ch = ch[i]
        self.decoder = nn.Sequential(*dec)

        # ── heads (logits — no activation) ──
        self.xct_head  = nn.Conv3d(ch[0], 1, 1)
        self.mask_head = nn.Conv3d(ch[0], 1, 1)

    def forward(self, xct: torch.Tensor, mask: torch.Tensor) -> VAEOutput:
        """
        Parameters
        ----------
        xct  : (B, 1, D, H, W) float32 in [0, 1]
        mask : (B, 1, D, H, W) float32 {0, 1}
        """
        x = torch.cat([xct, mask], dim=1)   # (B, 2, D, H, W)
        h = self.encoder(x)

        mu     = self.to_mu(h)
        logvar = self.to_logvar(h)
        z      = reparameterize(mu, logvar)

        dec = self.decoder(z)
        return VAEOutput(
            xct_logits=self.xct_head(dec),
            mask_logits=self.mask_head(dec),
            mu=mu,
            logvar=logvar,
            z=z,
        )
