"""VRRAE with the rank reduction applied PER LATENT GRID CELL, one shared basis.

THE IDEA, AND WHY IT IS STILL THE PAPER'S LAYER. The flat VRRAE hands the RR
layer one column per PATCH: a 64-cubed patch becomes a single 2048- or
4096-vector, and the SVD's sample axis is the batch. Here the encoder stops
two stride-2 blocks down instead of five or six, leaving a (B, 64, 16, 16, 16)
grid, and each of the 4096 CELLS becomes a column. The SVD then sees
``N = batch x cells`` samples of a 64-dimensional space and returns ONE shared
basis for all of them.

Every RRAE/VRRAE identity holds under that reading — the layer is unmodified
and its mechanics are untouched; only what counts as a "sample" changes. That
is the whole design, and it is why this reuses `VRRAEBottleneck` rather than
reimplementing it.

WHAT IT BUYS. The flat variants throw away spatial extent: six stride-2 blocks
leave a 1x1x1 latent, and the decoder has to invent a 64-cubed volume from one
vector. Here the latent keeps a 16-cubed grid with k* coefficients per cell, so
at k*=8 a patch carries 8 x 16^3 = 32 768 numbers — EXACTLY r08's latent
budget, which makes the comparison with the spatial baseline a like-for-like
one rather than a capacity argument.

WHAT IS SHARED AND WHAT IS NOT. One basis for every cell of every patch: the
basis is a property of the layer, not of a position, so the decoder sees a
consistent target. The coefficients are per cell. `fc_in` is Identity by
construction here (`vrrae_dim` = the encoder's channel count), so the RR layer
operates on the encoder's own features.

THE RANK CONSTRAINT STOPS BITING. `RRLayer` clips the effective rank to
``min(rank, vrrae_dim, n_samples)``, and n_samples is now ``batch x 4096``
rather than ``batch`` — so a rank of 8 needs a batch of 1, not a batch of 768.
That is what lets this run at batch 128 where the flat variants needed 1024.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from poregen.models.nn.blocks import (
    down_block_v2,
    reparameterize,
    up_block_v2_mirror,
)
from poregen.models.vae.base import VAEConfig, VAEOutput
from poregen.models.vae.registry import register_vae
from poregen.models.vae.v2.vrrae_bottleneck import VRRAEBottleneck


@register_vae("v2.vrrae_conv")
class ConvVAE3DVRRAEConvV2(nn.Module):
    """Encoder -> per-cell RR bottleneck -> decoder, with one shared basis.

    Data flow with the defaults (``n_blocks=2, base_channels=32,
    vrrae_dim=64, vrrae_rank=8, patch_size=64``):

        x            (B,  1, 64, 64, 64)
        encoder  ->  (B, 64, 16, 16, 16)        # ch[1] = 64, 64/2^2 = 16
        as cells ->  (B*4096, 64)               # each CELL is an SVD column
        bottleneck-> mu, logvar (B*4096, 8), basis (64, 8)
        z        ->  (B*4096, 8)
        y = z@U^T -> (B*4096, 64)
        as grid  ->  (B, 64, 16, 16, 16)
        decoder  ->  (B,  1, 64, 64, 64)

    There is no ``dec_b``: the reconstructed cell vectors already have the
    encoder's channel width, so they reshape straight back onto the grid.
    """

    def __init__(self, cfg: VAEConfig) -> None:
        super().__init__()
        self.cfg = cfg
        ch = cfg.channel_schedule()
        self.enc_channels = ch[cfg.n_blocks - 1]
        self.latent_spatial = cfg.latent_spatial

        enc: list[nn.Module] = []
        in_ch = cfg.in_channels
        for i in range(cfg.n_blocks):
            enc.append(down_block_v2(in_ch, ch[i]))
            in_ch = ch[i]
        self.encoder = nn.Sequential(*enc)

        # THE ONE DEPARTURE FROM THE FLAT VARIANT, AND IT IS IN THE CALLER, NOT
        # THE LAYER: in_flat_dim is the CHANNEL count, not channels x cells,
        # because a column is now one cell rather than one patch.
        if cfg.vrrae_dim != self.enc_channels:
            raise ValueError(
                f"vrrae_dim ({cfg.vrrae_dim}) must equal the encoder's channel "
                f"count ({self.enc_channels}) for the per-cell variant: a column "
                f"of the SVD is one cell, so its length is the channel width. "
                f"Set model.vrrae_dim to {self.enc_channels} or change n_blocks."
            )
        self.bottleneck = VRRAEBottleneck(
            in_flat_dim=self.enc_channels,
            vrrae_dim=cfg.vrrae_dim,
            rank=cfg.vrrae_rank,
            basis_history_size=cfg.vrrae_basis_history_size,
        )

        dec: list[nn.Module] = []
        in_ch = ch[cfg.n_blocks - 1]
        for i in range(cfg.n_blocks - 1, 0, -1):
            dec.append(up_block_v2_mirror(in_ch, ch[i - 1]))
            in_ch = ch[i - 1]
        dec.append(up_block_v2_mirror(in_ch, cfg.in_channels, final=True))
        self.decoder = nn.Sequential(*dec)

    @property
    def latents_per_patch(self) -> int:
        """k* x cells — the number this variant is meant to match r08 on."""
        return self.cfg.vrrae_rank * self.latent_spatial ** 3

    def finalize_inference_basis(self, train_loader, *, device, autocast_dtype):
        """Construct and install U_f from a complete training-dataset pass."""
        from poregen.models.vae.v2.vrrae_finetune import (  # noqa: PLC0415
            finalize_basis_from_dataloader,
        )

        return finalize_basis_from_dataloader(
            self, train_loader, device=device, autocast_dtype=autocast_dtype)

    def forward(self, xct: torch.Tensor, mask: torch.Tensor | None = None) -> VAEOutput:
        """
        Parameters
        ----------
        xct  : (B, 1, D, H, W) float32 — XCT intensity, z-scored
        mask : accepted for call-site compatibility; unused, as in `v2.vrrae`.
        """
        h = self.encoder(xct)                       # (B, C, L, L, L)
        b, c = h.shape[0], h.shape[1]

        # (B, C, L, L, L) -> (B*cells, C). permute first so that each ROW is
        # one cell's channel vector; a bare reshape would interleave channels
        # with positions and hand the SVD columns that are not cells at all.
        cells = h.permute(0, 2, 3, 4, 1).reshape(-1, c)

        mu, logvar, basis = self.bottleneck(cells)  # (N, k), (N, k), (C, k)
        z = reparameterize(mu, logvar)              # (N, k)

        # Y_tilde = U @ alpha_tilde, exactly as the flat variant does it, and
        # for the same reason: the coefficients only mean anything in the basis
        # they were produced in, and that basis rotates every batch.
        y = z @ basis.transpose(0, 1)               # (N, C)
        grid = y.view(b, self.latent_spatial, self.latent_spatial,
                      self.latent_spatial, c).permute(0, 4, 1, 2, 3).contiguous()
        xct_out = self.decoder(grid)

        # mu is (N, k) with ndim 2, so `compute_total_loss` routes it to
        # `kl_divergence_flat`: mean over B*cells, sum over k*. That is the
        # reduction this variant wants, and it comes out of the existing code
        # rather than a new branch.
        return VAEOutput(xct_out=xct_out, mask_logits=None,
                         mu=mu, logvar=logvar, z=z)
