"""VRRAE bottleneck — flatten + FC head + vendored RR_layer + identity-mean posterior.

Implements the "VRRAE" (Variational Rank-Reduction Autoencoder) bottleneck:
flatten the pre-bottleneck conv-encoder features -> one FC layer to a target
flat dimension ``vrrae_dim`` -> truncated-SVD rank-reduction layer (vendored
from https://github.com/JadM133/RR_layer, ``third_party/RR_layer``, pinned at
commit ``de2e8a5035cf0d5897a59f7f743fe8c3655a7caa``) -> the SVD coefficients
themselves ARE the posterior mean (``f=identity``, no trainable layer sits
between the RR layer's coefficient output and ``mu`` — required per the
VRRAE paper's ablation, not a style choice).

Design notes
------------
- **mu = SVD coefficients, unmodified.**  ``RRLayer(..., return_factors=True)``
  returns ``(reconstruction, basis, coeffs)`` where ``coeffs`` has shape
  ``(rank, B)`` — the per-sample low-rank coordinates.  We transpose to
  ``(B, rank)`` and use that directly as ``mu``.  No projection/cleanup layer
  is applied to it.
- **logvar is a separate learned head over the SVD coefficients.**  The JAX
  reference defines ``lin_logvar = Linear(k_max, k_max)`` and applies it to
  ``coeffs``. Therefore both ``f`` and ``g`` consume ``alpha_bar``; ``f`` is
  identity and ``g`` is ``nn.Linear(rank, rank)``.
- **bf16-autocast vs SVD numerics.**  ``torch.linalg.svd`` (used inside
  ``RRLayer``'s ``stable_SVD``) is numerically unstable in bfloat16.  The
  production training loop on the target GB10 hardware runs the whole model
  forward under ``torch.autocast(dtype=torch.bfloat16)``
  (``poregen.training.device.get_autocast_dtype``).  We therefore force
  float32 for the FC-in projection and the entire RR layer call by wrapping
  them in ``torch.autocast(..., enabled=False)`` with an explicit
  ``.float()`` cast, then cast ``mu``/``logvar`` back to the input dtype.

Known gotcha — inference_basis device placement
-------------------------------------------------
``RRLayer.finalize_basis()`` (called automatically the first time the model
switches to ``eval()``, or explicitly via
``poregen.models.vae.v2.vrrae_finetune.refinalize_basis``) tries to detect
the correct device via ``next(self.parameters(), None)`` — but ``RRLayer``
itself has no trainable parameters, so that always returns ``None`` and it
falls back to the stored basis bank's device. The basis bank is deliberately
kept on CPU (``self._basis_bank.append(basis_used.detach().cpu())`` inside
``RRLayer.forward``, presumably to save GPU memory across the retained
history), so ``inference_basis`` ends up on CPU even when the rest of the
model is on CUDA — the very first real eval pass then crashes with a
device-mismatch ``RuntimeError`` inside ``RRLayer.forward``'s
``basis_used.T @ X``. Rather than patch the vendored file, we defensively
re-home ``inference_basis`` onto the input's device on every forward call
below (a no-op once it's already correctly placed).

Small-batch rank clipping
-------------------------
``RRLayer.forward`` computes ``r = min(rank, vrrae_dim, batch_size)``. For a
non-``drop_last`` trailing batch this wrapper pads the missing coefficient
and basis columns with zeros. The exact rank-``r`` reconstruction is unchanged,
while ``mu``, ``logvar``, and the returned basis retain fixed configured-rank
shapes required by the JAX ``g: R^k -> R^k`` head.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from RR_layer import RRLayer


class VRRAEBottleneck(nn.Module):
    """Flatten-agnostic VRRAE bottleneck: FC-in -> RRLayer -> (mu, logvar).

    Parameters
    ----------
    in_flat_dim : int
        Size of the flattened pre-bottleneck feature vector fed to
        ``forward`` (e.g. ``enc_out_ch * latent_spatial**3`` for a 3-D conv
        encoder).
    vrrae_dim : int
        Width of the shared FC-in projection (``L`` in the design doc) —
        this is the dimensionality the RR layer's SVD operates over.
    rank : int
        Target SVD truncation rank (``k``).  Effective rank per batch is
        ``min(rank, vrrae_dim, batch_size)`` — see module docstring.
    basis_history_size : int
        Size of RRLayer's compatibility rolling bank. Production ``U_f``
        construction uses a separate complete-dataset pass. Default matches
        RRLayer's own default (20).
    """

    def __init__(
        self,
        in_flat_dim: int,
        vrrae_dim: int,
        rank: int,
        basis_history_size: int = 20,
    ) -> None:
        super().__init__()
        self.in_flat_dim = in_flat_dim
        self.vrrae_dim = vrrae_dim
        self.rank = rank

        # When L equals the flattened encoder width there is no projection to
        # learn: run the RR layer directly on the encoder representation.
        self.fc_in = (
            nn.Identity()
            if in_flat_dim == vrrae_dim
            else nn.Linear(in_flat_dim, vrrae_dim)
        )
        self.rr = RRLayer(rank=rank, basis_history_size=basis_history_size)
        self.logvar_head = nn.Linear(rank, rank)
        self.register_buffer(
            "_basis_finalized",
            torch.tensor(False, dtype=torch.bool),
            persistent=True,
        )

    def train(self, mode: bool = True):
        """Select per-batch SVD or the explicitly finalized inference basis.

        JAX validation during optimization still uses a per-batch SVD;
        ``U_f`` is constructed only in a separate pass after training.
        PyTorch's recursive ``eval()`` would otherwise trigger RRLayer's
        rolling-bank auto-finalization, so an unfinalized bottleneck keeps
        only RRLayer in training mode while the encoder remains in eval mode.
        Resuming training invalidates a finalized basis.
        """
        if mode:
            super().train(True)
            if bool(self._basis_finalized.item()):
                self.rr.inference_basis = None
                self._basis_finalized.fill_(False)
            return self

        if bool(self._basis_finalized.item()):
            super().train(False)
            return self

        # Avoid RRLayer.train(False): it auto-finalizes from its short rolling
        # bank. Only the stateless SVD operation remains in training mode.
        self.training = False
        self.fc_in.train(False)
        self.logvar_head.train(False)
        self.rr.train(True)
        return self

    @property
    def basis_finalized(self) -> bool:
        """Whether evaluation is using an explicitly constructed ``U_f``."""
        return bool(self._basis_finalized.item())

    def set_finalized_basis(self, basis: torch.Tensor) -> None:
        """Install a full-training-set inference basis produced externally."""
        if basis.ndim != 2 or basis.shape != (self.vrrae_dim, self.rank):
            raise ValueError(
                "Finalized basis must have shape "
                f"({self.vrrae_dim}, {self.rank}), got {tuple(basis.shape)}."
            )
        self.rr.inference_basis = basis
        self._basis_finalized.fill_(True)
        self.rr.train(False)

    def projected_features(self, h: torch.Tensor) -> torch.Tensor:
        """Apply the learned pre-SVD projection with training-time dtypes."""
        return self.fc_in(h)

    @torch._dynamo.disable()
    def forward(
        self,
        h: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Parameters
        ----------
        h : (B, in_flat_dim)
            Already-flattened pre-bottleneck encoder features.  This module
            is conv-agnostic — the caller (the VAE model class) is
            responsible for flattening the conv encoder's spatial output.

        Notes
        -----
        Decorated with ``torch._dynamo.disable()`` — under
        ``torch.compile(mode="max-autotune")`` (the production training
        config on the GB10, per CLAUDE.md), ``RRLayer`` keeps Python-level
        mutable state (``self._basis_bank``, a ``deque``) that grows every
        training step until ``basis_history_size`` is reached. Dynamo guards
        traced graphs on that length, so *every* step where the length
        changes triggers a full max-autotune recompile — extremely
        expensive (10s of seconds each) — until it hits
        ``torch._dynamo.config.recompile_limit`` and gives up, by which
        point training has been running at ~20s/step instead of the
        expected sub-second pace. Disabling tracing here creates a clean
        graph break at the bottleneck boundary: the (compute-heavy) conv
        encoder/decoder still get compiled and fused normally, only this
        small, cheap FC+SVD bottleneck runs eagerly.

        Returns
        -------
        mu, logvar : (B, rank) each
        basis : (vrrae_dim, rank)
            The truncated basis ``U`` these coefficients are expressed in.
            **Callers MUST map the sampled latent back through this before
            decoding** (``Y = U @ alpha``, per Fig. 1 of the VRRAE paper,
            arXiv:2505.09458). The coefficients alone are NOT a stable
            per-sample code: in train mode ``U`` is re-derived from each
            batch's own SVD, so a sample's coordinates depend on which
            other samples share its batch, and singular vectors carry an
            arbitrary sign (plus free rotation within near-degenerate
            singular values). Measured on structured low-rank data, the
            same samples encoded alongside different companions gave
            coefficients that were effectively uncorrelated (relative
            difference 1.41, per-channel |corr| 0.17, 166/300 channels
            sign-flipped) while ``U @ alpha`` was stable to 0.001. Decoding
            the raw coefficients through fixed weights therefore trains
            against a coordinate system that is re-randomised every step.
        """
        in_dtype = h.dtype
        x = self.fc_in(h)  # (B, vrrae_dim) — may run under bf16 autocast

        # SVD is numerically unstable in bf16 — force float32 across the RR
        # layer's forward pass, then cast back.
        with torch.autocast(device_type=x.device.type, enabled=False):
            x32 = x.float()
            # RRLayer.finalize_basis() has no parameters to detect device from
            # and falls back to the (deliberately CPU-resident) basis bank's
            # device — re-home defensively before use (see module docstring).
            if (
                self.rr.inference_basis is not None
                and self.rr.inference_basis.device != x32.device
            ):
                self.rr.inference_basis = self.rr.inference_basis.to(x32.device)
            _reconstruction, basis, coeffs = self.rr(x32, return_factors=True)
            mu = coeffs.transpose(0, 1).to(in_dtype)  # (rank, B) -> (B, rank)
            basis = basis.to(in_dtype)                # (vrrae_dim, rank)

        # A non-drop-last evaluation batch can contain fewer than ``rank``
        # samples. Embed its exact coefficients in the configured k-space.
        # Zero basis columns make absent directions reconstruction-neutral.
        if mu.shape[-1] < self.rank:
            missing = self.rank - mu.shape[-1]
            mu = torch.nn.functional.pad(mu, (0, missing))
            basis = torch.nn.functional.pad(basis, (0, missing))

        # JAX: logvar = lin_logvar(coeffs). Keeping log-variance locally is
        # equivalent to sigma because reparameterize uses exp(0.5 * logvar).
        logvar = self.logvar_head(mu)
        return mu, logvar, basis
