"""UNetVAE3D — skip-connection VAE for sharper reconstructions."""

from __future__ import annotations

import torch
import torch.nn as nn

from poregen.models.vae.base import VAEConfig, VAEOutput
from poregen.models.vae.registry import register_vae


# ── building blocks ─────────────────────────────────────────────────────

class DownBlock(nn.Module):
    """Stride-2 downsampling + two convs."""

    def __init__(self, in_ch: int, out_ch: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv3d(in_ch, out_ch, 4, stride=2, padding=1),
            nn.GroupNorm(min(32, out_ch), out_ch),
            nn.SiLU(inplace=True),
            nn.Conv3d(out_ch, out_ch, 3, padding=1),
            nn.GroupNorm(min(32, out_ch), out_ch),
            nn.SiLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class UpBlock(nn.Module):
    """Upsample → concat skip → refine with two 3×3 convs.

    The skip is expected to have the **same spatial size** as the
    *upsampled* ``x`` (i.e. the skip is from the encoder stage that
    was at the target resolution).
    """

    def __init__(self, in_ch: int, skip_ch: int, out_ch: int) -> None:
        super().__init__()
        self.up = nn.ConvTranspose3d(in_ch, in_ch, 4, stride=2, padding=1)
        self.conv = nn.Sequential(
            nn.Conv3d(in_ch + skip_ch, out_ch, 3, padding=1),
            nn.GroupNorm(min(32, out_ch), out_ch),
            nn.SiLU(inplace=True),
            nn.Conv3d(out_ch, out_ch, 3, padding=1),
            nn.GroupNorm(min(32, out_ch), out_ch),
            nn.SiLU(inplace=True),
        )

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        h = self.up(x)
        h = torch.cat([h, skip], dim=1)
        return self.conv(h)


# ── model ────────────────────────────────────────────────────────────────

@register_vae("unet")
class UNetVAE3D(nn.Module):
    """UNet-style VAE with decoder skip connections.

    Architecture (default ``n_blocks=2``, ``base_channels=32``)::

        Encoder:
          (B,  2, 64,64,64)
            → DownBlock → (B, 32, 32,32,32)  skip[0]
            → DownBlock → (B, 64, 16,16,16)  skip[1]

        Bottleneck:
          1×1 → mu, logvar → z  (B, 8, 16,16,16)

        Decoder:
          z → from_z (B, 64, 16,16,16)
            → merge skip[1] at 16³ (concat + conv, no spatial change)
            → UpBlock  → concat skip[0] → (B, 32, 32,32,32)
            → UpBlock  →                  (B, 32, 64,64,64)

        Heads: 1×1 conv → xct_logits, mask_logits
    """

    def __init__(self, cfg: VAEConfig) -> None:
        super().__init__()
        self.cfg = cfg
        ch = cfg.channel_schedule()  # e.g. [32, 64, 128] for n_blocks=2

        # ── encoder ──
        self.enc_blocks = nn.ModuleList()
        in_ch = cfg.in_channels
        for i in range(cfg.n_blocks):
            self.enc_blocks.append(DownBlock(in_ch, ch[i]))
            in_ch = ch[i]

        # ── latent ──
        bottleneck_ch = ch[cfg.n_blocks - 1]
        self.to_mu = nn.Conv3d(bottleneck_ch, cfg.z_channels, 1)
        self.to_logvar = nn.Conv3d(bottleneck_ch, cfg.z_channels, 1)

        # ── decoder ──
        # Project z back to bottleneck channel width
        self.from_z = nn.Sequential(
            nn.Conv3d(cfg.z_channels, bottleneck_ch, 3, padding=1),
            nn.GroupNorm(min(32, bottleneck_ch), bottleneck_ch),
            nn.SiLU(inplace=True),
        )

        # Merge deepest skip at same resolution (no upsample)
        # from_z (bottleneck_ch) + skip[-1] (bottleneck_ch) → bottleneck_ch
        if cfg.n_blocks >= 2:
            self.deep_merge = nn.Sequential(
                nn.Conv3d(bottleneck_ch + bottleneck_ch, bottleneck_ch, 3, padding=1),
                nn.GroupNorm(min(32, bottleneck_ch), bottleneck_ch),
                nn.SiLU(inplace=True),
            )
        else:
            self.deep_merge = None

        # UpBlocks: upsample + concat with earlier skips
        # For n_blocks=2: one UpBlock upsamples from 16→32 and concats skip[0]
        # Then a final UpBlock upsamples from 32→64 (no skip)
        self.dec_blocks = nn.ModuleList()
        in_ch = bottleneck_ch
        n_up = cfg.n_blocks
        for i in range(n_up):
            # Which skip to concat with? We go from deepest to shallowest.
            # After deep_merge consumed skip[-1], remaining skips are [0..n_blocks-2]
            # So up block i concats with skip[n_blocks-2-i] (if it exists)
            skip_idx = cfg.n_blocks - 2 - i
            if skip_idx >= 0:
                skip_ch = ch[skip_idx]
            else:
                skip_ch = 0  # no skip for this level

            out_ch = ch[skip_idx] if skip_idx >= 0 else ch[0]

            if skip_ch > 0:
                self.dec_blocks.append(UpBlock(in_ch, skip_ch, out_ch))
            else:
                # Pure upsample with no skip
                self.dec_blocks.append(self._make_pure_up(in_ch, out_ch))
            in_ch = out_ch

        # ── heads ──
        self.xct_head = nn.Conv3d(ch[0], 1, 1)
        self.mask_head = nn.Conv3d(ch[0], 1, 1)

    @staticmethod
    def _make_pure_up(in_ch: int, out_ch: int) -> nn.Sequential:
        return nn.Sequential(
            nn.ConvTranspose3d(in_ch, out_ch, 4, stride=2, padding=1),
            nn.GroupNorm(min(32, out_ch), out_ch),
            nn.SiLU(inplace=True),
            nn.Conv3d(out_ch, out_ch, 3, padding=1),
            nn.GroupNorm(min(32, out_ch), out_ch),
            nn.SiLU(inplace=True),
        )

    @staticmethod
    def _reparameterize(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        std = (0.5 * logvar).exp()
        return mu + std * torch.randn_like(std)

    def forward(self, xct: torch.Tensor, mask: torch.Tensor) -> VAEOutput:
        x = torch.cat([xct, mask], dim=1)

        # ── encode ──
        skips: list[torch.Tensor] = []
        h = x
        for block in self.enc_blocks:
            h = block(h)
            skips.append(h)

        mu = self.to_mu(h)
        logvar = self.to_logvar(h)
        z = self._reparameterize(mu, logvar)

        # ── decode ──
        h = self.from_z(z)

        # Merge deepest skip at same resolution
        if self.deep_merge is not None:
            h = torch.cat([h, skips[-1]], dim=1)
            h = self.deep_merge(h)

        # UpBlocks with remaining skips
        for i, block in enumerate(self.dec_blocks):
            skip_idx = self.cfg.n_blocks - 2 - i
            if skip_idx >= 0 and isinstance(block, UpBlock):
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
