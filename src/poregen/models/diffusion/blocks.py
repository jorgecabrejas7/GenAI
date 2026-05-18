"""3-D building blocks for the LDM denoising U-Net.

Uses GroupNorm + SiLU throughout — BatchNorm is unsuitable for diffusion models
because activation statistics shift across noise levels within a batch.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def norm_groups(channels: int) -> int:
    """Number of GroupNorm groups for *channels* feature maps."""
    for g in range(32, 0, -1):
        if channels % g == 0:
            return g
    return 1


# ── Time embedding ────────────────────────────────────────────────────────────

class SinusoidalPositionEmbedding(nn.Module):
    """Sinusoidal timestep embedding (Vaswani et al. 2017, DDPM convention).

    Maps integer timestep t → (B, dim) continuous embedding using
    half-sine, half-cosine log-spaced frequencies.
    """

    def __init__(self, dim: int) -> None:
        super().__init__()
        self.dim = dim

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        t : (B,) long — timestep indices

        Returns
        -------
        (B, dim) float32
        """
        device = t.device
        half = self.dim // 2
        freqs = torch.exp(
            -math.log(10000.0) * torch.arange(half, device=device).float() / half
        )
        args = t.float().unsqueeze(1) * freqs.unsqueeze(0)   # (B, half)
        return torch.cat([args.sin(), args.cos()], dim=-1)    # (B, dim)


# ── Adaptive Group Norm ───────────────────────────────────────────────────────

class AdaGN(nn.Module):
    """Adaptive Group Normalisation with conditioning vector.

    Given a conditioning vector ``cond`` of shape ``(B, cond_dim)``, produces
    per-channel scale and shift parameters that modulate the GroupNorm output:

        h = GN(x)
        scale, shift = Linear(cond_dim, 2*channels).split(channels, dim=1)
        out = h * (1 + scale) + shift

    The ``(1 + scale)`` form initialises close to identity (scale≈0 at init
    when the linear layer is zero-initialised) which stabilises early training.
    """

    def __init__(self, channels: int, cond_dim: int) -> None:
        super().__init__()
        self.gn   = nn.GroupNorm(norm_groups(channels), channels)
        self.proj = nn.Linear(cond_dim, 2 * channels)
        # Zero-initialise so early conditioning is identity
        nn.init.zeros_(self.proj.weight)
        nn.init.zeros_(self.proj.bias)

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x    : (B, C, D, H, W)
        cond : (B, cond_dim)

        Returns
        -------
        (B, C, D, H, W)
        """
        h = self.gn(x)
        scale, shift = self.proj(cond).chunk(2, dim=1)          # each (B, C)
        scale = scale.view(-1, scale.shape[1], 1, 1, 1)
        shift = shift.view(-1, shift.shape[1], 1, 1, 1)
        return h * (1.0 + scale) + shift


# ── Residual block ────────────────────────────────────────────────────────────

class ResBlock3D(nn.Module):
    """3-D residual block with adaptive group-norm conditioning.

    Architecture::

        [Conv3d(3³) → AdaGN → SiLU] × 2 + residual skip

    A 1×1×1 projection is added when ``in_ch != out_ch``.
    """

    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        cond_dim: int,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.conv1  = nn.Conv3d(in_ch,  out_ch, 3, padding=1)
        self.ada1   = AdaGN(out_ch, cond_dim)
        self.conv2  = nn.Conv3d(out_ch, out_ch, 3, padding=1)
        self.ada2   = AdaGN(out_ch, cond_dim)
        self.act    = nn.SiLU()
        self.drop   = nn.Dropout3d(dropout) if dropout > 0.0 else nn.Identity()
        self.skip   = nn.Conv3d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x    : (B, in_ch, D, H, W)
        cond : (B, cond_dim)
        """
        h = self.act(self.ada1(self.conv1(x), cond))
        h = self.drop(h)
        h = self.ada2(self.conv2(h), cond)
        return h + self.skip(x)


# ── Self-attention ────────────────────────────────────────────────────────────

class SelfAttention3D(nn.Module):
    """Multi-head self-attention over 3-D spatial tokens.

    Flattens the spatial dimensions (D, H, W) into a sequence of length
    D×H×W, applies scaled dot-product attention, then reshapes back.
    Used only at the bottleneck (4³ = 64 tokens) where the sequence is short.
    """

    def __init__(self, channels: int, n_heads: int = 8) -> None:
        super().__init__()
        assert channels % n_heads == 0, f"channels={channels} must be divisible by n_heads={n_heads}"
        self.n_heads   = n_heads
        self.norm      = nn.GroupNorm(norm_groups(channels), channels)
        self.qkv       = nn.Conv1d(channels, 3 * channels, 1)
        self.proj_out  = nn.Conv1d(channels, channels, 1)
        nn.init.zeros_(self.proj_out.weight)
        nn.init.zeros_(self.proj_out.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : (B, C, D, H, W)

        Returns
        -------
        (B, C, D, H, W)
        """
        B, C, D, H, W = x.shape
        h = self.norm(x).view(B, C, -1)                         # (B, C, L)
        qkv = self.qkv(h)                                        # (B, 3C, L)
        q, k, v = qkv.chunk(3, dim=1)                           # each (B, C, L)

        # Reshape for multi-head attention: (B, n_heads, head_dim, L)
        head_dim = C // self.n_heads
        def split_heads(t: torch.Tensor) -> torch.Tensor:
            return t.view(B, self.n_heads, head_dim, -1).transpose(2, 3)  # (B, H, L, d)

        q, k, v = split_heads(q), split_heads(k), split_heads(v)
        attn_out = F.scaled_dot_product_attention(q, k, v)      # (B, H, L, d)
        attn_out = attn_out.transpose(2, 3).reshape(B, C, -1)   # (B, C, L)

        out = self.proj_out(attn_out).view(B, C, D, H, W)
        return x + out


# ── Up / Down sampling ────────────────────────────────────────────────────────

class Downsample3D(nn.Module):
    """Stride-2 downsampling via a 4×4×4 convolution (no checkerboard)."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.conv = nn.Conv3d(channels, channels, kernel_size=4, stride=2, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class Upsample3D(nn.Module):
    """Trilinear upsample × 2 followed by a 3×3×3 convolution.

    Matches the v2 VAE upsampling style (no checkerboard artefacts at pore
    boundaries that ConvTranspose3d can introduce).
    """

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.up   = nn.Upsample(scale_factor=2, mode="trilinear", align_corners=False)
        self.conv = nn.Conv3d(channels, channels, kernel_size=3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(self.up(x))
