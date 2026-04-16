"""ConvVAE3DV2 — second-generation convolutional VAE with bottleneck self-attention.

Architecture differences from v1:

* **BatchNorm3d** instead of GroupNorm — more stable gradient statistics at
  batch_size ≥ 32 and faster fused CUDA kernels on Ampere / Hopper hardware.
* **GELU** instead of SiLU — de-facto standard in modern diffusion models
  (DDPM, LDM, DiT); negligibly faster, identical gradient safety.
* **Upsample(trilinear) + Conv3d** instead of ConvTranspose3d — eliminates
  checkerboard artefacts at pore boundaries and achieves higher throughput on
  H100-class GPUs.
* **float32-safe reparameterization** (shared with all architectures).

Not checkpoint-compatible with v1 ``"conv"`` weights.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from poregen.models.nn.blocks import down_block_v2, up_block_v2, norm_groups, reparameterize
from poregen.models.vae.base import VAEConfig, VAEOutput
from poregen.models.vae.registry import register_vae


# ── bottleneck attention ──────────────────────────────────────────────────────

class BottleneckAttention3D(nn.Module):
    """Multi-head self-attention at the encoder bottleneck (v2).

    Uses ``F.scaled_dot_product_attention`` (Flash-Attention kernel on
    CUDA ≥ sm_80).  Normalisation uses GroupNorm (attention over the spatial
    token sequence does not benefit from BatchNorm).
    """

    def __init__(self, channels: int, num_heads: int = 4) -> None:
        super().__init__()
        assert channels % num_heads == 0, (
            f"channels ({channels}) must be divisible by num_heads ({num_heads})"
        )
        self.num_heads = num_heads
        self.head_dim = channels // num_heads
        self.norm = nn.GroupNorm(norm_groups(channels), channels)
        self.qkv  = nn.Linear(channels, channels * 3, bias=False)
        self.proj = nn.Linear(channels, channels, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, D, H, W = x.shape
        residual = x

        h = self.norm(x)
        h = h.flatten(2).transpose(1, 2)

        qkv = self.qkv(h).reshape(B, -1, 3, self.num_heads, self.head_dim)
        q, k, v = qkv.unbind(2)
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        h = F.scaled_dot_product_attention(q, k, v)
        h = h.transpose(1, 2).reshape(B, -1, C)
        h = self.proj(h).transpose(1, 2).reshape(B, C, D, H, W)

        return h + residual


# ── model ─────────────────────────────────────────────────────────────────────

@register_vae("v2.conv")
class ConvVAE3DV2(nn.Module):
    """Second-generation convolutional VAE with bottleneck attention.

    R03+ design: encoder receives XCT only (``in_channels=1``).  See
    :class:`poregen.models.vae.v2.conv_noattn.ConvVAE3DNoAttnV2` for the
    full rationale.

    With ``VAEConfig(n_blocks=2, base_channels=32, in_channels=1, z_channels=16)``::

        (B,  1, 64, 64, 64)
          → down_v2 → (B,  32, 32, 32, 32)
          → down_v2 → (B,  64, 16, 16, 16)
          → attn
          → 1×1     → mu / logvar  (B, 16, 16, 16, 16)
          → up_v2   → (B,  64, 32, 32, 32)
          → up_v2   → (B,  32, 64, 64, 64)
          → 1×1 heads → xct_logits, mask_logits  (B, 1, 64, 64, 64) each
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

        self.bottleneck_attn = BottleneckAttention3D(ch[cfg.n_blocks - 1])

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
        xct  : (B, 1, D, H, W) float32 — XCT intensity, z-scored
        mask : (B, 1, D, H, W) float32 {0, 1} — pore mask (reconstruction
               target only; not fed to the encoder)
        """
        h = self.encoder(xct)  # encoder sees XCT only
        h = self.bottleneck_attn(h)

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
