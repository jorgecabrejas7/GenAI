"""ConvVAE3DVRRAELinearV2 — SVD-ablation twin of ``v2.vrrae``.

Controlled ablation of the VRRAE bottleneck (see
:mod:`poregen.models.vae.v2.vrrae`): identical encoder, decoder, flatten,
``fc_in``/``dec_b`` projections and latent width, but the truncated-SVD RR
layer is replaced by plain learned ``Linear`` heads. Comparing this variant
against ``v2.vrrae`` at the same config (vrrae_dim=2048, vrrae_rank=300,
same batch/steps/LR/KL) isolates whether the batch-SVD machinery (rotating
per-batch basis, ill-conditioned SVD gradients, train/eval basis mismatch)
or the flat low-dimensional bottleneck itself is responsible for the
reconstruction gap vs. the spatial-latent baselines.

Mapping from the VRRAE bottleneck to this one:

===========================  =====================================
``v2.vrrae``                 ``v2.vrrae_linear``
===========================  =====================================
``fc_in`` (L -> vrrae_dim)   same
SVD coeffs -> ``mu``         ``mu_head = Linear(vrrae_dim, rank)``
``logvar_head(mu)``          same (``Linear(rank, rank)``)
``y = z @ basis.T``          ``expand = Linear(rank, vrrae_dim)``
``dec_b`` (vrrae_dim -> L)   same
===========================  =====================================

No basis finalization, no train/eval special-casing, no
``torch._dynamo.disable`` and no autocast escapes — everything here is
bf16-safe dense algebra. ``VAEOutput.mu`` is flat ``(B, vrrae_rank)``, so
``compute_total_loss`` routes through ``kl_divergence_flat`` exactly as for
``v2.vrrae``.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from poregen.models.nn.blocks import down_block_v2, up_block_v2_mirror, reparameterize
from poregen.models.vae.base import VAEConfig, VAEOutput
from poregen.models.vae.registry import register_vae


@register_vae("v2.vrrae_linear")
class ConvVAE3DVRRAELinearV2(nn.Module):
    """Flat-bottleneck VAE with learned Linear heads in place of the SVD.

    XCT-only encoder AND decoder, no mask head — see module docstring and
    :mod:`poregen.models.vae.v2.vrrae` for the shared architecture.
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

        # Same identity-when-widths-match rule as VRRAEBottleneck.fc_in /
        # ConvVAE3DVRRAEV2.dec_b, so the vrrae03-style direct path is
        # reproducible here too.
        self.fc_in = (
            nn.Identity()
            if self.enc_flat_dim == cfg.vrrae_dim
            else nn.Linear(self.enc_flat_dim, cfg.vrrae_dim)
        )
        self.mu_head = nn.Linear(cfg.vrrae_dim, cfg.vrrae_rank)
        self.logvar_head = nn.Linear(cfg.vrrae_rank, cfg.vrrae_rank)
        self.expand = nn.Linear(cfg.vrrae_rank, cfg.vrrae_dim)
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
        dec.append(up_block_v2_mirror(in_ch, cfg.in_channels, final=True))
        self.decoder = nn.Sequential(*dec)

    def forward(self, xct: torch.Tensor, mask: torch.Tensor | None = None) -> VAEOutput:
        """
        Parameters
        ----------
        xct  : (B, 1, D, H, W) float32 — XCT intensity, z-scored
        mask : accepted for call-site signature compatibility, unused.
        """
        h = self.encoder(xct)
        h_flat = self.fc_in(h.flatten(1))  # (B, vrrae_dim)

        mu = self.mu_head(h_flat)          # (B, rank)
        logvar = self.logvar_head(mu)      # (B, rank)
        z = reparameterize(mu, logvar)

        ch_last = self.cfg.channel_schedule()[self.cfg.n_blocks - 1]
        ls = self.cfg.latent_spatial
        y = self.expand(z)                 # (B, vrrae_dim)
        dec_in = self.dec_b(y).view(z.shape[0], ch_last, ls, ls, ls)
        xct_logits = self.decoder(dec_in)

        return VAEOutput(
            xct_logits=xct_logits,
            mask_logits=None,
            mu=mu,
            logvar=logvar,
            z=z,
        )
