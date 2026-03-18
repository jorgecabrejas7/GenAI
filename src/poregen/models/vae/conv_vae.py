"""ConvVAE3D — 3-D convolutional VAE with bottleneck self-attention (no skip connections)."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from poregen.models.vae.base import VAEConfig, VAEOutput
from poregen.models.vae.registry import register_vae


# ── building blocks ─────────────────────────────────────────────────────

def _down_block(in_ch: int, out_ch: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv3d(in_ch, out_ch, 4, stride=2, padding=1),
        nn.GroupNorm(min(32, out_ch), out_ch),
        nn.SiLU(inplace=True),
        nn.Conv3d(out_ch, out_ch, 3, padding=1),
        nn.GroupNorm(min(32, out_ch), out_ch),
        nn.SiLU(inplace=True),
    )


def _up_block(in_ch: int, out_ch: int) -> nn.Sequential:
    return nn.Sequential(
        nn.ConvTranspose3d(in_ch, out_ch, 4, stride=2, padding=1),
        nn.GroupNorm(min(32, out_ch), out_ch),
        nn.SiLU(inplace=True),
        nn.Conv3d(out_ch, out_ch, 3, padding=1),
        nn.GroupNorm(min(32, out_ch), out_ch),
        nn.SiLU(inplace=True),
    )


# ── bottleneck attention ─────────────────────────────────────────────────

class BottleneckAttention3D(nn.Module):
    """Multi-head self-attention at the encoder bottleneck.

    Captures long-range pore correlations without bypassing the information
    bottleneck via skip connections. Uses ``F.scaled_dot_product_attention``
    (flash-attention kernel on CUDA, torch ≥ 2.0) so memory cost is O(N)
    rather than O(N²) for the attention weights.

    Parameters
    ----------
    channels:
        Number of feature channels at the bottleneck (= last encoder channel count).
    num_heads:
        Number of attention heads. Must divide *channels* evenly.
    """

    def __init__(self, channels: int, num_heads: int = 4) -> None:
        super().__init__()
        assert channels % num_heads == 0, (
            f"channels ({channels}) must be divisible by num_heads ({num_heads})"
        )
        self.num_heads = num_heads
        self.head_dim = channels // num_heads
        self.norm = nn.GroupNorm(min(32, channels), channels)
        self.qkv = nn.Linear(channels, channels * 3, bias=False)
        self.proj = nn.Linear(channels, channels, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : (B, C, D, H, W)

        Returns
        -------
        (B, C, D, H, W) — residual connection included.
        """
        B, C, D, H, W = x.shape
        residual = x

        h = self.norm(x)
        h = h.flatten(2).transpose(1, 2)          # (B, N, C), N = D*H*W

        qkv = self.qkv(h)                          # (B, N, 3*C)
        qkv = qkv.reshape(B, -1, 3, self.num_heads, self.head_dim)
        q, k, v = qkv.unbind(2)                    # each (B, N, heads, head_dim)
        q = q.transpose(1, 2)                      # (B, heads, N, head_dim)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        h = F.scaled_dot_product_attention(q, k, v)  # (B, heads, N, head_dim)
        h = h.transpose(1, 2).reshape(B, -1, C)      # (B, N, C)
        h = self.proj(h)
        h = h.transpose(1, 2).reshape(B, C, D, H, W)

        return h + residual


# ── model ────────────────────────────────────────────────────────────────

@register_vae("conv")
class ConvVAE3D(nn.Module):
    """Simple 3-D convolutional VAE.

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
        # ch = [32, 64, 128, ...] with length n_blocks + 1
        ch = cfg.channel_schedule()

        # ── encoder ──
        enc: list[nn.Module] = []
        in_ch = cfg.in_channels
        for i in range(cfg.n_blocks):
            enc.append(_down_block(in_ch, ch[i]))
            in_ch = ch[i]
        self.encoder = nn.Sequential(*enc)

        # ── bottleneck attention (D8 — 2026-03-18) ──
        self.bottleneck_attn = BottleneckAttention3D(ch[cfg.n_blocks - 1])

        # ── bottleneck → latent ──
        self.to_mu = nn.Conv3d(ch[cfg.n_blocks - 1], cfg.z_channels, 1)
        self.to_logvar = nn.Conv3d(ch[cfg.n_blocks - 1], cfg.z_channels, 1)

        # ── decoder ──
        dec: list[nn.Module] = []
        in_ch = cfg.z_channels
        for i in range(cfg.n_blocks - 1, -1, -1):
            dec.append(_up_block(in_ch, ch[i]))
            in_ch = ch[i]
        self.decoder = nn.Sequential(*dec)

        # ── heads (logits — no activation) ──
        self.xct_head = nn.Conv3d(ch[0], 1, 1)
        self.mask_head = nn.Conv3d(ch[0], 1, 1)

    # ── helpers ──────────────────────────────────────────────────────

    @staticmethod
    def _reparameterize(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        std = (0.5 * logvar).exp()
        return mu + std * torch.randn_like(std)

    def forward(self, xct: torch.Tensor, mask: torch.Tensor) -> VAEOutput:
        """
        Parameters
        ----------
        xct  : (B, 1, D, H, W) float32 in [0, 1]
        mask : (B, 1, D, H, W) float32 {0, 1}
        """
        x = torch.cat([xct, mask], dim=1)  # (B, 2, D, H, W)
        h = self.encoder(x)
        h = self.bottleneck_attn(h)

        mu = self.to_mu(h)
        logvar = self.to_logvar(h)
        z = self._reparameterize(mu, logvar)

        dec = self.decoder(z)
        return VAEOutput(
            xct_logits=self.xct_head(dec),
            mask_logits=self.mask_head(dec),
            mu=mu,
            logvar=logvar,
            z=z,
        )
