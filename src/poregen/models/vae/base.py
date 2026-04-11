"""VAE dataclasses: configuration and forward-pass output."""

from __future__ import annotations

from dataclasses import dataclass, field

import torch


@dataclass
class VAEConfig:
    """Configuration shared by all VAE architectures.

    Parameters
    ----------
    in_channels : int
        Number of input channels (XCT + mask = 2).
    z_channels : int
        Number of latent channels.
    base_channels : int
        Width of the first encoder stage; subsequent stages double.
    n_blocks : int
        Number of down/up-sampling stages.  With ``patch_size=64`` and
        ``n_blocks=2`` the spatial dims go 64 → 32 → 16 (factor 4).
    patch_size : int
        Expected cubic patch side length (for shape validation only).
    """

    in_channels: int = 2
    z_channels: int = 8
    base_channels: int = 32
    n_blocks: int = 2
    patch_size: int = 64

    @property
    def downsample_factor(self) -> int:
        return 2 ** self.n_blocks

    @property
    def latent_spatial(self) -> int:
        return self.patch_size // self.downsample_factor

    def channel_schedule(self) -> list[int]:
        """Return per-stage channel counts (encoder order)."""
        return [self.base_channels * (2 ** i) for i in range(self.n_blocks + 1)]


@dataclass
class VAEOutput:
    """Standardised output produced by every VAE ``forward()`` call.

    All spatial tensors keep the same batch dimension as the input.

    - **xct_logits**: raw decoder output for XCT in z-score space (unbounded).
      The decoder predicts z-scored values directly — no activation needed.
    - **mask_logits**: raw decoder output for the pore mask; use with
      ``BCEWithLogitsLoss``.
    - **mu**, **logvar**: posterior parameters.
    - **z**: sampled latent (after reparameterization).
    """

    xct_logits: torch.Tensor    # (B, 1, D, H, W)
    mask_logits: torch.Tensor   # (B, 1, D, H, W)
    mu: torch.Tensor            # (B, z_channels, d, h, w)
    logvar: torch.Tensor        # (B, z_channels, d, h, w)
    z: torch.Tensor             # (B, z_channels, d, h, w)
