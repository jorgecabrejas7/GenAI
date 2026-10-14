"""Patch discriminators for adversarial XCT reconstruction training.

Two kinds, chosen by ``discriminator.kind`` in the config:

- ``slices2d`` (default, used from R04 on): a 2D PatchGAN that scores one random
  axial, coronal and sagittal slice per sample, each slice on its own.
- ``volume3d`` (the author, 2026-10-14, vault D69): a 3D PatchGAN that scores
  the whole 64³ grey patch. The 2D version cannot see whether the three planes
  agree in 3D, and the material is anisotropic (plies), so the adversarial
  rungs ``r08/advperc-rf*`` use the 3D one. Everything else about the
  adversarial term (LSGAN, spectral norm, adaptive weight) is the same.

Each discriminator has ``prepare(vol)``: it turns a ``(B, 1, D, H, W)`` grey
volume into the discriminator's own input, so the training step does not need
to know which kind it holds.

Shared design
-------------
- **Spectral norm**: bounds the Lipschitz constant of each convolutional layer
  without extra hyperparameters, stabilising GAN training.
- **No batch norm**: BN statistics are unreliable for small mini-batches and can
  trigger training collapse.
- **LSGAN**: least-squares objective avoids vanishing gradients when the
  discriminator is saturated.

2D architecture (input 1×64×64)::

    SN-Conv2d(1→64,   4, stride=2, pad=1) → LeakyReLU(0.2) →  (B, 64,  32, 32)
    SN-Conv2d(64→128, 4, stride=2, pad=1) → LeakyReLU(0.2) →  (B, 128, 16, 16)
    SN-Conv2d(128→256,4, stride=2, pad=1) → LeakyReLU(0.2) →  (B, 256,  8,  8)
    SN-Conv2d(256→1,  4, stride=1, pad=1)                   →  (B,   1,  7,  7)
    mean                                                      →  (B,)

3D architecture (input 1×64×64×64): the same four layers with 3D convolutions,
output (B, 1, 7, 7, 7) averaged to (B,).
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

    @staticmethod
    def prepare(vol: torch.Tensor) -> torch.Tensor:
        """(B, 1, D, H, W) grey volume → (3B, 1, D, D) random multi-plane slices."""
        return extract_multiplane_slices(vol)


class PatchDiscriminator3D(nn.Module):
    """3D PatchGAN discriminator with spectral normalisation.

    The 2D design with every convolution made 3D: it scores a whole grey patch,
    so it sees texture and its consistency across the three planes together.
    """

    def __init__(self, in_channels: int = 1, base_channels: int = 64) -> None:
        super().__init__()
        c = base_channels
        sn = spectral_norm
        self.net = nn.Sequential(
            sn(nn.Conv3d(in_channels, c,     4, stride=2, padding=1, bias=True)),  # →(B,  64, 32³)
            nn.LeakyReLU(0.2, inplace=True),
            sn(nn.Conv3d(c,      c * 2, 4, stride=2, padding=1, bias=True)),       # →(B, 128, 16³)
            nn.LeakyReLU(0.2, inplace=True),
            sn(nn.Conv3d(c * 2,  c * 4, 4, stride=2, padding=1, bias=True)),       # →(B, 256,  8³)
            nn.LeakyReLU(0.2, inplace=True),
            sn(nn.Conv3d(c * 4, 1,      4, stride=1, padding=1, bias=True)),        # →(B,   1,  7³)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """(B, 1, D, H, W) float32 → (B,) un-activated real/fake score."""
        return self.net(x).mean(dim=(1, 2, 3, 4))

    @staticmethod
    def prepare(vol: torch.Tensor) -> torch.Tensor:
        """The 3D discriminator scores the volume itself."""
        return vol


DISCRIMINATORS = {"slices2d": PatchDiscriminator2D, "volume3d": PatchDiscriminator3D}


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


# ── adaptive generator weight (Rombach et al. 2022) ───────────────────────────

def adaptive_disc_weight(
    rec_loss: torch.Tensor,
    gen_loss: torch.Tensor,
    last_layer_weight: torch.Tensor,
    eps: float = 1e-4,
    max_weight: float = 1e4,
) -> torch.Tensor:
    """``||grad_w rec_loss|| / (||grad_w gen_loss|| + eps)``, clamped, detached.

    The LDM autoencoder's ``calculate_adaptive_weight``: the adversarial term
    is scaled so its gradient at the decoder's last layer matches the
    reconstruction gradient there. The caller multiplies by the base
    ``disc_weight``. *last_layer_weight* is the weight of the conv that
    produces the grey output (``model.xct_head``).

    The ratio is scale-free, so it gives the same balance whether the
    reconstruction is summed over voxels (Rombach) or averaged (ours). Only
    the absolute clamp at ``max_weight`` and ``eps`` act in our units.
    """
    rec_grad = torch.autograd.grad(rec_loss, last_layer_weight, retain_graph=True)[0]
    gen_grad = torch.autograd.grad(gen_loss, last_layer_weight, retain_graph=True)[0]
    weight = torch.norm(rec_grad) / (torch.norm(gen_grad) + eps)
    return torch.clamp(weight, 0.0, max_weight).detach()
