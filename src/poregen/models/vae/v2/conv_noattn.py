"""ConvVAE3DNoAttnV2 — second-generation convolutional VAE, no attention, no skip connections.

See :mod:`poregen.models.vae.v2.conv` for the full list of v2 architecture
improvements over v1.  Use this variant to ablate the contribution of
bottleneck attention in the v2 family.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from poregen.models.nn.blocks import down_block_v2, up_block_v2, reparameterize
from poregen.models.vae.base import VAEConfig, VAEOutput
from poregen.models.vae.registry import register_vae


@register_vae("v2.conv_noattn")
class ConvVAE3DNoAttnV2(nn.Module):
    """Second-generation purely convolutional VAE — no attention, no skip connections.

    With default ``VAEConfig(n_blocks=2, base_channels=32)``::

        (B,  2, 64, 64, 64)
          → down_v2 → (B,  32, 32, 32, 32)
          → down_v2 → (B,  64, 16, 16, 16)
          → 1×1     → mu / logvar  (B, 8, 16, 16, 16)
          → up_v2   → (B,  64, 32, 32, 32)
          → up_v2   → (B,  32, 64, 64, 64)
          → 1×1 heads → xct_logits, mask_logits
    """

    def __init__(self, cfg: VAEConfig) -> None:
        super().__init__()
        self.cfg = cfg
        ch = cfg.channel_schedule()

        enc: list[nn.Module] = []
        in_ch = cfg.in_channels
        for i in range(cfg.n_blocks):
            enc.append(down_block_v2(in_ch, ch[i]))
            in_ch = ch[i]
        self.encoder = nn.Sequential(*enc)

        self.to_mu     = nn.Conv3d(ch[cfg.n_blocks - 1], cfg.z_channels, 1)
        self.to_logvar = nn.Conv3d(ch[cfg.n_blocks - 1], cfg.z_channels, 1)

        dec: list[nn.Module] = []
        in_ch = cfg.z_channels
        for i in range(cfg.n_blocks - 1, -1, -1):
            dec.append(up_block_v2(in_ch, ch[i]))
            in_ch = ch[i]
        self.decoder = nn.Sequential(*dec)

        self.xct_head  = nn.Conv3d(ch[0], 1, 1)
        self.mask_head = nn.Conv3d(ch[0], 1, 1)

    def forward(self, xct: torch.Tensor, mask: torch.Tensor) -> VAEOutput:
        """
        Parameters
        ----------
        xct  : (B, 1, D, H, W) float32 in [0, 1]
        mask : (B, 1, D, H, W) float32 {0, 1}
        """
        x = torch.cat([xct, mask], dim=1)
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
