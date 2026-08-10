"""ConvVAE3DVRRAEV2 — VRRAE-bottleneck VAE, XCT-only encoder AND decoder.

Forked from :mod:`poregen.models.vae.v2.conv_noattn` (the cleanest v2
template — no dual-branch fusion complexity).  Replaces the spatial
Conv3d-mu/logvar bottleneck with a VRRAE (Variational Rank-Reduction
Autoencoder) bottleneck: flatten -> FC -> truncated-SVD RR layer ->
identity posterior mean -> reparameterize -> KL.  See
:mod:`poregen.models.vae.v2.vrrae_bottleneck` for the bottleneck itself.

**New for this architecture: the decoder is also XCT-only.**  No mask
segmentation head, no ``mask_logits`` output (``VAEOutput.mask_logits`` is
``None`` for this variant) — the encoder was already XCT-only in R03+; this
variant additionally drops the mask reconstruction target from the decoder
side, since VRRAE is about aggregate-posterior structure and decoder-head
design, not the segmentation auxiliary task.

The decoder is an exact mirror of the encoder, stage for stage
----------------------------------------------------------------
Every encoder stage is ``down_block_v2(in_ch, out_ch)`` — see
:func:`poregen.models.nn.blocks.down_block_v2` — whose stride-2 conv runs
*first*, so both its convs execute at the *halved* (output) resolution.
Every decoder stage is :func:`poregen.models.nn.blocks.up_block_v2_mirror`,
the literal reverse of that block: the resolution-preserving conv runs
first (at the block's *input*, i.e. still-low, resolution), and a
``ConvTranspose3d`` — the adjoint of the encoder's strided ``Conv3d`` — does
the upsample-and-channel-projection in a single op last. Decoder stage
``j`` mirrors encoder stage ``n_blocks - 1 - j`` exactly: same channel
counts, same resolutions, reversed order. The final decoder stage collapses
straight to 1 channel at full (patch_size³) resolution with no trailing
BatchNorm/GELU, absorbing what used to be a separate ``xct_head`` — this is
the mirror of the encoder's very first stage, which projects 1 -> ch[0]
channels at the input's full resolution before ever downsampling.

This symmetry means the encoder and decoder share the same largest
per-stage tensor (``ch[-1]`` channels @ 32³, i.e. the *first* encoder
stage's output / the *last* decoder stage's input) rather than the decoder
alone carrying a wide-channel tensor at full 64³ resolution — which is what
the previous (asymmetric, ``Upsample``-based) decoder did, and which was
the direct cause of a batch-size throughput cliff (cuDNN's 3-D convolution
backward falls back to a much slower 64-bit-indexed kernel once a tensor
crosses 2**31 elements; the old decoder's final stage crossed that
threshold at batch_size=256).

Data flow (with the defaults ``vrrae_dim=2048, vrrae_rank=300,
base_channels=32, n_blocks=5, patch_size=64`` — ``z_channels`` is unused by
this variant, kept only for cross-variant config compatibility)::

    (B,   1, 64, 64, 64)
      → down_v2 → (B,  32, 32, 32, 32)                       # ch[0]
      → down_v2 → (B,  64, 16, 16, 16)                       # ch[1]
      → down_v2 → (B, 128,  8,  8,  8)                       # ch[2]
      → down_v2 → (B, 256,  4,  4,  4)                       # ch[3]
      → down_v2 → (B, 512,  2,  2,  2)                       # ch[4] = ch[n_blocks-1]
      → flatten → (B, 512*2*2*2=4096)                        # enc_flat_dim
      → VRRAEBottleneck: FC(4096->2048) -> RRLayer(rank=300)  # mu (B,300)
                          + logvar_head(300->300)              # logvar (B,300)
      → reparameterize   → z (B, 300)
      → y = z @ basis.T   -> (B, 2048)                        # Y = U @ alpha
      → dec_b: Linear(2048 -> 4096) -> reshape                # (B,512,2,2,2)
      → up_v2_mirror → (B, 256,  4,  4,  4)
      → up_v2_mirror → (B, 128,  8,  8,  8)
      → up_v2_mirror → (B,  64, 16, 16, 16)
      → up_v2_mirror → (B,  32, 32, 32, 32)
      → up_v2_mirror(final=True) → xct_logits (B, 1, 64, 64, 64);  mask_logits = None
"""

from __future__ import annotations

import torch
import torch.nn as nn

from poregen.models.nn.blocks import down_block_v2, up_block_v2_mirror, reparameterize
from poregen.models.vae.base import VAEConfig, VAEOutput
from poregen.models.vae.registry import register_vae
from poregen.models.vae.v2.vrrae_bottleneck import VRRAEBottleneck


@register_vae("v2.vrrae")
class ConvVAE3DVRRAEV2(nn.Module):
    """VRRAE-bottleneck VAE — XCT-only encoder AND decoder, no mask head.

    See module docstring for the full data-flow diagram and the
    encoder/decoder symmetry design decision.

    Small trailing batches: the (rank -> vrrae_dim) leg uses the basis
    returned by the bottleneck rather than a fixed-width layer, so if RRLayer
    clips the effective rank (``batch_size < rank`` — see
    ``vrrae_bottleneck`` docstring) ``z`` and ``basis`` shrink together and
    the decode still works.  A large ``batch_size`` is still strongly
    preferred on statistical grounds — the SVD basis is estimated from the
    batch, so a small batch yields a poorly-conditioned one — but it is no
    longer a hard shape constraint.
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

        self.enc_flat_dim = ch[cfg.n_blocks - 1] * cfg.latent_spatial ** 3

        self.bottleneck = VRRAEBottleneck(
            in_flat_dim=self.enc_flat_dim,
            vrrae_dim=cfg.vrrae_dim,
            rank=cfg.vrrae_rank,
            basis_history_size=cfg.vrrae_basis_history_size,
        )

        # Decoder input projection. The (rank -> vrrae_dim) leg is NOT a
        # learned layer: per Fig. 1 of the VRRAE paper (arXiv:2505.09458) the
        # sampled coefficients are mapped back through the *same* truncated
        # basis they were produced in (``Y_tilde = U @ alpha_tilde``), and
        # ``U`` is re-derived from every batch's SVD. A fixed
        # ``nn.Linear(rank, vrrae_dim)`` here — which is what this used to be
        # — applies a stale basis to coefficients expressed in a different,
        # constantly-rotating one, so the decoder can never learn more than
        # the batch mean. Only the (vrrae_dim -> enc_flat_dim) leg is learned,
        # mirroring the encoder's fc_in.
        # Mirror fc_in. If the RR ambient width already equals the flattened
        # encoder width, its reconstructed vector can be reshaped directly.
        self.dec_b = (
            nn.Identity()
            if cfg.vrrae_dim == self.enc_flat_dim
            else nn.Linear(cfg.vrrae_dim, self.enc_flat_dim)
        )

        dec: list[nn.Module] = []
        in_ch = ch[cfg.n_blocks - 1]
        for i in range(cfg.n_blocks - 1, 0, -1):
            dec.append(up_block_v2_mirror(in_ch, ch[i - 1]))
            in_ch = ch[i - 1]
        # Final stage mirrors the encoder's first stage (in_channels -> ch[0]):
        # collapses straight to 1 channel at full resolution, raw logits.
        dec.append(up_block_v2_mirror(in_ch, cfg.in_channels, final=True))
        self.decoder = nn.Sequential(*dec)
        # No separate xct_head — absorbed into the final decoder stage above.
        # No mask_head — this variant's decoder is XCT-only.

    def finalize_inference_basis(
        self,
        train_loader,
        *,
        device: torch.device,
        autocast_dtype: torch.dtype,
    ) -> dict[str, int]:
        """Construct and install U_f from a complete training-dataset pass."""
        from poregen.models.vae.v2.vrrae_finetune import (
            finalize_basis_from_dataloader,
        )

        return finalize_basis_from_dataloader(
            self,
            train_loader,
            device=device,
            autocast_dtype=autocast_dtype,
        )

    def forward(self, xct: torch.Tensor, mask: torch.Tensor | None = None) -> VAEOutput:
        """
        Parameters
        ----------
        xct  : (B, 1, D, H, W) float32 — XCT intensity, z-scored
        mask : accepted for call-site signature compatibility with the
               other VAE variants (``model(xct, mask)`` in
               ``poregen.training.engine``), but **unused** — this variant
               has no mask reconstruction target/head.
        """
        h = self.encoder(xct)  # (B, ch[n_blocks-1], latent_spatial, ...)
        h_flat = h.flatten(1)  # (B, enc_flat_dim)

        # basis: (vrrae_dim, rank) — dynamic before finalization, then U_f.
        mu, logvar, basis = self.bottleneck(h_flat)  # (B, rank) each
        z = reparameterize(mu, logvar)

        ch_last = self.cfg.channel_schedule()[self.cfg.n_blocks - 1]
        ls = self.cfg.latent_spatial
        # Y_tilde = U @ alpha_tilde  -> (B, vrrae_dim). Basis-independent, so
        # the decoder sees a stable target even though U itself rotates.
        # Using the returned basis (rather than a fixed-width layer) also makes
        # this correct under RRLayer's small-batch rank clipping for free: z
        # and basis shrink together.
        y = z @ basis.transpose(0, 1)
        dec_in = self.dec_b(y).view(z.shape[0], ch_last, ls, ls, ls)
        xct_logits = self.decoder(dec_in)

        return VAEOutput(
            xct_logits=xct_logits,
            mask_logits=None,
            mu=mu,
            logvar=logvar,
            z=z,
        )
