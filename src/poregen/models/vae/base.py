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
        Number of encoder input channels.  Use 1 for XCT-only (R03+); 2 for
        legacy XCT+mask runs (R00–R02, v1 checkpoints).
    z_channels : int
        Number of latent channels.
    base_channels : int
        Width of the first encoder stage; subsequent stages double.
    n_blocks : int
        Number of down/up-sampling stages.  With ``patch_size=64`` and
        ``n_blocks=2`` the spatial dims go 64 → 32 → 16 (factor 4).
    patch_size : int
        Expected cubic patch side length (for shape validation only).
    vrrae_dim : int
        [``v2.vrrae`` only] Width of the VRRAE bottleneck's shared FC-in
        projection (``L``) — the dimensionality the truncated-SVD RR layer
        operates over.  Set to a 2x reduction from the encoder's flattened
        output (``channel_schedule()[n_blocks-1] * latent_spatial**3`` = 4096
        at production defaults, so ``L``=2048). Ignored by all other VAE
        variants.
    vrrae_rank : int
        [``v2.vrrae`` only] Target SVD truncation rank (``k``).  Must
        satisfy ``k <= min(vrrae_dim, batch_size)`` or the RR layer silently
        clips the effective rank per batch (see
        ``poregen.models.vae.v2.vrrae_bottleneck`` docstring) — every
        training batch must therefore have ``batch_size >= vrrae_rank``.
        Ignored by all other VAE variants.
    vrrae_basis_history_size : int
        [``v2.vrrae`` only] Number of recent per-batch training SVD bases
        the RR layer retains for later fixed-basis (``Uf``) extraction at
        inference time.  Default matches RRLayer's own default (20).
        Ignored by all other VAE variants.
    vrrae_conv_channels : int
        **Legacy / currently unused.**  An earlier ``v2.vrrae`` revision
        inserted an extra convolutional stage between the normal encoder and
        the flatten+FC+RR-layer bottleneck, reducing to this channel count.
        That stage broke encoder/decoder symmetry (the decoder's mirrored
        stage ran two full-width convolutions at 64^3 resolution, which
        crossed cuDNN's 2**31-element indexing threshold at batch_size=256
        and caused a ~5x backward-pass throughput cliff) and has been
        removed — the encoder now flattens directly from its last
        ``down_block_v2`` stage. Field is kept only so old configs/
        checkpoints referencing it don't hard-fail; ``v2.vrrae`` no longer
        reads it.
    """

    in_channels: int = 2
    z_channels: int = 8
    base_channels: int = 32
    n_blocks: int = 2
    patch_size: int = 64
    vrrae_dim: int = 2048
    vrrae_rank: int = 32
    vrrae_basis_history_size: int = 20
    vrrae_conv_channels: int = 8

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
      ``BCEWithLogitsLoss``.  ``None`` for XCT-only decoder variants (e.g.
      ``v2.vrrae``) that have no ``mask_head`` — all other variants still
      populate it and depend on it being required-by-convention, so it
      keeps its position in the field list; only the default makes it
      optional.  Uses ``kw_only`` (Python 3.10+) so it can carry a default
      without forcing defaults on the required fields that follow it
      (``mu``/``logvar``/``z``).  Every existing construction call site
      uses keyword arguments already (verified), so this is safe.
    - **mu**, **logvar**: posterior parameters.  Spatial ``(B, C, d, h, w)``
      for conv-bottleneck variants; flat ``(B, rank)`` for ``v2.vrrae``.
    - **z**: sampled latent (after reparameterization).
    """

    xct_logits: torch.Tensor                                     # (B, 1, D, H, W)
    mask_logits: torch.Tensor | None = field(default=None, kw_only=True)  # (B, 1, D, H, W) or None
    mu: torch.Tensor                            # (B, z_channels, d, h, w) or (B, rank) — required
    logvar: torch.Tensor                        # same shape as mu — required
    z: torch.Tensor                             # same shape as mu — required
