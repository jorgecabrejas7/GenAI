"""UNetVAE3DV2 — second-generation UNet-style VAE with skip connections.

See :mod:`poregen.models.vae.v2.conv` for the full list of v2 improvements.
Skip connections produce sharper reconstructions; the v2 block family
eliminates checkerboard artefacts in the process.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from poregen.models.nn.blocks import down_block_v2, up_block_v2, norm_groups, reparameterize
from poregen.models.vae.base import VAEConfig, VAEOutput
from poregen.models.vae.registry import register_vae


# ── building blocks ───────────────────────────────────────────────────────────

class DownBlockV2(nn.Module):
    """Stride-2 downsampling block (wraps :func:`down_block_v2` for ModuleList)."""

    def __init__(self, in_ch: int, out_ch: int) -> None:
        super().__init__()
        self.net = down_block_v2(in_ch, out_ch)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class UpBlockV2(nn.Module):
    """Trilinear upsample → concat skip → refine with two 3×3×3 convs (v2 style).

    Uses BatchNorm3d and GELU throughout.  The skip tensor is expected to have
    the same spatial size as the upsampled feature map.
    """

    def __init__(self, in_ch: int, skip_ch: int, out_ch: int) -> None:
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode="trilinear", align_corners=False)
        self.conv = nn.Sequential(
            nn.Conv3d(in_ch + skip_ch, out_ch, 3, padding=1),
            nn.BatchNorm3d(out_ch),
            nn.GELU(),
            nn.Conv3d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm3d(out_ch),
            nn.GELU(),
        )

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        h = self.up(x)
        h = torch.cat([h, skip], dim=1)
        return self.conv(h)


# ── model ─────────────────────────────────────────────────────────────────────

@register_vae("v2.unet")
class UNetVAE3DV2(nn.Module):
    """Second-generation UNet-style VAE with decoder skip connections.

    Architecture (default ``n_blocks=2``, ``base_channels=32``)::

        Encoder:
          (B,  2, 64,64,64)
            → DownBlockV2 → (B, 32, 32,32,32)  skip[0]
            → DownBlockV2 → (B, 64, 16,16,16)  skip[1]

        Bottleneck:
          1×1 → mu, logvar → z  (B, 8, 16,16,16)

        Decoder:
          z → from_z  (B, 64, 16,16,16)
            → merge skip[1] at 16³ (concat + conv)
            → UpBlockV2 → concat skip[0] → (B, 32, 32,32,32)
            → UpBlockV2 →                  (B, 32, 64,64,64)

        Heads: 1×1 → xct_logits, mask_logits
    """

    def __init__(self, cfg: VAEConfig) -> None:
        super().__init__()
        self.cfg = cfg
        ch = cfg.channel_schedule()

        # ── encoder ──
        self.enc_blocks = nn.ModuleList()
        in_ch = cfg.in_channels
        for i in range(cfg.n_blocks):
            self.enc_blocks.append(DownBlockV2(in_ch, ch[i]))
            in_ch = ch[i]

        # ── latent ──
        bottleneck_ch = ch[cfg.n_blocks - 1]
        self.to_mu     = nn.Conv3d(bottleneck_ch, cfg.z_channels, 1)
        self.to_logvar = nn.Conv3d(bottleneck_ch, cfg.z_channels, 1)

        # ── decoder stem ──
        self.from_z = nn.Sequential(
            nn.Conv3d(cfg.z_channels, bottleneck_ch, 3, padding=1),
            nn.BatchNorm3d(bottleneck_ch),
            nn.GELU(),
        )

        # ── merge deepest skip at same resolution ──
        if cfg.n_blocks >= 2:
            self.deep_merge = nn.Sequential(
                nn.Conv3d(bottleneck_ch + bottleneck_ch, bottleneck_ch, 3, padding=1),
                nn.BatchNorm3d(bottleneck_ch),
                nn.GELU(),
            )
        else:
            self.deep_merge = None

        # ── upsampling blocks ──
        self.dec_blocks = nn.ModuleList()
        in_ch = bottleneck_ch
        for i in range(cfg.n_blocks):
            skip_idx = cfg.n_blocks - 2 - i
            if skip_idx >= 0:
                skip_ch = ch[skip_idx]
                out_ch  = ch[skip_idx]
                self.dec_blocks.append(UpBlockV2(in_ch, skip_ch, out_ch))
            else:
                out_ch = ch[0]
                self.dec_blocks.append(up_block_v2(in_ch, out_ch))
            in_ch = out_ch

        # ── heads ──
        self.xct_head  = nn.Conv3d(ch[0], 1, 1)
        self.mask_head = nn.Conv3d(ch[0], 1, 1)

    def forward(self, xct: torch.Tensor, mask: torch.Tensor) -> VAEOutput:
        x = torch.cat([xct, mask], dim=1)

        skips: list[torch.Tensor] = []
        h = x
        for block in self.enc_blocks:
            h = block(h)
            skips.append(h)

        mu     = self.to_mu(h)
        logvar = self.to_logvar(h)
        z      = reparameterize(mu, logvar)

        h = self.from_z(z)

        if self.deep_merge is not None:
            h = torch.cat([h, skips[-1]], dim=1)
            h = self.deep_merge(h)

        for i, block in enumerate(self.dec_blocks):
            skip_idx = self.cfg.n_blocks - 2 - i
            if skip_idx >= 0 and isinstance(block, UpBlockV2):
                h = block(h, skips[skip_idx])
            else:
                h = block(h)

        return VAEOutput(
            xct_logits=self.xct_head(h),
            mask_logits=self.mask_head(h),
            mu=mu,
            logvar=logvar,
            z=z,
        )
