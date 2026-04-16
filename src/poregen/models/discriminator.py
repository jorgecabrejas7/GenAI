"""2D Multi-Plane Patch Discriminator for adversarial XCT reconstruction training.

Used from R04+: provides adversarial supervision for the XCT decoder head by
classifying real vs. reconstructed 2D slices extracted from 3D XCT volumes.

Design decisions
----------------
- **2D, not 3D**: a 3D PatchGAN on 64³ volumes would require ~8× more activations
  than a 2D PatchGAN on 64×64 slices while adding little discriminative power for
  texture artefacts which are most visible per-plane.
- **Multi-plane**: one random axial, coronal, and sagittal slice is sampled per
  sample per step, so the discriminator sees all orientations and cannot be fooled
  by blurry reconstructions in cross-sectional planes.
- **Spectral norm**: bounds the Lipschitz constant of each convolutional layer
  without extra hyperparameters, stabilising GAN training.
- **No batch norm**: BN statistics are unreliable for small mini-batches
  (3×batch_size slices) and can trigger training collapse.
- **LSGAN**: least-squares objective avoids vanishing gradients when the
  discriminator is saturated, and is stable at the ``disc_weight=0.01`` scale.

Architecture (input 1×64×64)::

    SN-Conv2d(1→64,   4×4, stride=2, pad=1) → LeakyReLU(0.2) →  (B, 64,  32, 32)
    SN-Conv2d(64→128, 4×4, stride=2, pad=1) → LeakyReLU(0.2) →  (B, 128, 16, 16)
    SN-Conv2d(128→256,4×4, stride=2, pad=1) → LeakyReLU(0.2) →  (B, 256,  8,  8)
    SN-Conv2d(256→1,  4×4, stride=1, pad=1)                   →  (B,   1,  7,  7)
    mean(dim=(1,2,3))                                          →  (B,)

Total parameters ≈ 661 K (all in float32; discriminator does not use AMP).
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torch.nn.utils import spectral_norm


class PatchDiscriminator2D(nn.Module):
    """2D PatchGAN discriminator with spectral normalisation.

    Parameters
    ----------
    in_channels : int
        Number of input channels.  Use ``1`` for single-channel XCT slices.
    base_channels : int
        Feature-map width at the first layer; each subsequent layer doubles.
    """

    def __init__(self, in_channels: int = 1, base_channels: int = 64) -> None:
        super().__init__()
        c = base_channels

        def _sn(module: nn.Module) -> nn.Module:
            return spectral_norm(module)

        self.net = nn.Sequential(
            _sn(nn.Conv2d(in_channels, c,     4, stride=2, padding=1, bias=True)),  # →(B,  64, 32, 32)
            nn.LeakyReLU(0.2, inplace=True),
            _sn(nn.Conv2d(c,      c * 2, 4, stride=2, padding=1, bias=True)),       # →(B, 128, 16, 16)
            nn.LeakyReLU(0.2, inplace=True),
            _sn(nn.Conv2d(c * 2,  c * 4, 4, stride=2, padding=1, bias=True)),       # →(B, 256,  8,  8)
            nn.LeakyReLU(0.2, inplace=True),
            _sn(nn.Conv2d(c * 4, 1,      4, stride=1, padding=1, bias=True)),        # →(B,   1,  7,  7)
        )

        total = sum(p.numel() for p in self.parameters())
        import logging as _log
        _log.getLogger(__name__).info("PatchDiscriminator2D: %d params", total)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : (B, 1, H, W) float32

        Returns
        -------
        (B,) real/fake score per sample (un-activated; LSGAN uses these raw).
        """
        return self.net(x).mean(dim=(1, 2, 3))


# ── multi-plane slice extraction ──────────────────────────────────────────────

def extract_multiplane_slices(vol: torch.Tensor) -> torch.Tensor:
    """Extract one random axial, coronal, and sagittal slice per sample.

    Parameters
    ----------
    vol : (B, 1, D, H, W) — expects D == H == W (cubic patches).

    Returns
    -------
    (3B, 1, D, D) — three 64×64 slices per sample for 64³ patches.

    Uses vectorised fancy indexing — no Python loop over the batch dimension.
    Each call samples independent random positions for every sample in the
    batch, exposing the discriminator to the full spatial distribution.
    """
    B, _C, D, H, W = vol.shape
    device = vol.device
    b = torch.arange(B, device=device)

    # Axial: slice along D → (B, 1, H, W)
    d_rand = torch.randint(0, D, (B,), device=device)
    axial   = vol.permute(0, 2, 1, 3, 4)[b, d_rand]   # (B, 1, H, W)

    # Coronal: slice along H → (B, 1, D, W)
    h_rand  = torch.randint(0, H, (B,), device=device)
    coronal  = vol.permute(0, 3, 1, 2, 4)[b, h_rand]  # (B, 1, D, W)

    # Sagittal: slice along W → (B, 1, D, H)
    w_rand  = torch.randint(0, W, (B,), device=device)
    sagittal = vol.permute(0, 4, 1, 2, 3)[b, w_rand]  # (B, 1, D, H)

    # All three are (B, 1, 64, 64) since D=H=W; cat → (3B, 1, 64, 64)
    return torch.cat([axial, coronal, sagittal], dim=0)


# ── LSGAN objectives ──────────────────────────────────────────────────────────

def lsgan_gen_loss(d_fake: torch.Tensor) -> torch.Tensor:
    """Generator LSGAN loss: ``0.5 * E[(D(fake) − 1)²]``.

    The generator wants the discriminator to output 1 for reconstructions.
    """
    return 0.5 * (d_fake - 1.0).pow(2).mean()


def lsgan_disc_loss(d_real: torch.Tensor, d_fake: torch.Tensor) -> torch.Tensor:
    """Discriminator LSGAN loss: ``0.5 * (E[(D(real)−1)²] + E[D(fake)²])``.

    The discriminator wants to output 1 for real slices and 0 for fakes.
    """
    return 0.5 * ((d_real - 1.0).pow(2).mean() + d_fake.pow(2).mean())
