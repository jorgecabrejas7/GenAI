"""ConvVAE3DNoAttnDualBranchV2 — dual-branch encoder variant of the v2 no-attention VAE.

R04 design: the encoder processes the same 1-channel XCT input through two
**independent** branches (A and B), each with the same architecture as the
single encoder in R03.  Branch outputs are concatenated along the channel
dimension and reduced via a 1×1×1 fusion conv before the mu/logvar projection.

Motivation
----------
R03 showed fast XCT convergence but an over-predicting mask head (false
positives dominate), suggesting the latent features are heavily biased toward
global XCT intensity structure and lack fine-grained texture variation.  Two
parallel encoder branches provide ~2× the representational capacity in the
latent path without changing the latent space (z_channels, mu/logvar
parameterisation) or the decoder, while keeping the total parameter count
manageable (~800 K vs ~510 K for R03).

Architecture (n_blocks=2, base_channels=32, z_channels=16, in_channels=1)::

    XCT input (B, 1, 64, 64, 64)
         ┌──────────────────────────┬───────────────────────────┐
         │ Branch A (structural)    │ Branch B (texture/detail) │
    down_v2(1→32)  →(B,32,32³)  down_v2(1→32)  →(B,32,32³)
    down_v2(32→64) →(B,64,16³)  down_v2(32→64) →(B,64,16³)
         └──────────────┬───────────────────────┘
              cat([A,B]) →(B,128,16³)
         fusion Conv3d(128→64, 1×1×1) →(B,64,16³)
                        ↓
              to_mu / to_logvar → (B,16,16³)
                        ↓ decoder (unchanged from R03)
           up_v2(16→64) →(B,64,32³)
           up_v2(64→32) →(B,32,64³)
                    xct_head / mask_head →(B,1,64³) each

Memory guidance
---------------
With bs=128 and float16, peak GPU memory increases by roughly +300–400 MB vs
R03 (two encoder branches run simultaneously; decoder unchanged).  This is
typically safe on 40 GB / 80 GB GPUs.  If OOM occurs, reduce batch_size first
(try bs=96 or bs=64) and re-scale LR accordingly:

    Linear rule: lr_new = lr_old × (bs_new / bs_old)
    Sqrt rule:   lr_new = lr_old × sqrt(bs_new / bs_old)   ← recommended

The peak-GPU-memory log emitted after the first training step (``train/peak_gpu_mem_gb``)
will confirm the actual usage.

Parameter count is logged to INFO at init time.
"""

from __future__ import annotations

import logging

import torch
import torch.nn as nn

from poregen.models.nn.blocks import down_block_v2, up_block_v2, reparameterize
from poregen.models.vae.base import VAEConfig, VAEOutput
from poregen.models.vae.registry import register_vae

logger = logging.getLogger(__name__)


@register_vae("v2.conv_noattn_dualbranch")
class ConvVAE3DNoAttnDualBranchV2(nn.Module):
    """Dual-branch encoder VAE — no attention, no skip connections.

    See module docstring for full design rationale and architecture diagram.

    The ``conv_noattn`` (R03) variant is **unchanged** and can be used for
    exact reproducibility of R03 results.  Switch back by setting
    ``model.name: v2.conv_noattn`` in the config.
    """

    def __init__(self, cfg: VAEConfig) -> None:
        super().__init__()
        self.cfg = cfg
        ch = cfg.channel_schedule()      # [32, 64, 128] for n_blocks=2, base=32
        enc_out_ch = ch[cfg.n_blocks - 1]  # 64 for the default config

        # ── Branch A: primary structural branch (same as R03 encoder) ────────
        enc_a: list[nn.Module] = []
        in_ch = cfg.in_channels
        for i in range(cfg.n_blocks):
            enc_a.append(down_block_v2(in_ch, ch[i]))
            in_ch = ch[i]
        self.encoder_a = nn.Sequential(*enc_a)

        # ── Branch B: parallel detail branch (independent params, same arch) ──
        enc_b: list[nn.Module] = []
        in_ch = cfg.in_channels
        for i in range(cfg.n_blocks):
            enc_b.append(down_block_v2(in_ch, ch[i]))
            in_ch = ch[i]
        self.encoder_b = nn.Sequential(*enc_b)

        # ── Fusion: cat(A, B) [2×enc_out_ch] → enc_out_ch via 1×1×1 conv ─────
        # bias=False because the subsequent BatchNorm (inside to_mu / to_logvar
        # 1×1 convs) will re-introduce a learnable bias equivalent.
        self.fusion = nn.Conv3d(2 * enc_out_ch, enc_out_ch, kernel_size=1, bias=False)

        # ── Bottleneck projections ────────────────────────────────────────────
        self.to_mu     = nn.Conv3d(enc_out_ch, cfg.z_channels, 1)
        self.to_logvar = nn.Conv3d(enc_out_ch, cfg.z_channels, 1)

        # ── Decoder (identical to R03 — unchanged by design) ──────────────────
        dec: list[nn.Module] = []
        in_ch = cfg.z_channels
        for i in range(cfg.n_blocks - 1, -1, -1):
            dec.append(up_block_v2(in_ch, ch[i]))
            in_ch = ch[i]
        self.decoder = nn.Sequential(*dec)

        self.xct_head  = nn.Conv3d(ch[0], 1, 1)
        self.mask_head = nn.Conv3d(ch[0], 1, 1)

        # ── Parameter count audit ─────────────────────────────────────────────
        n_enc_a   = sum(p.numel() for p in self.encoder_a.parameters())
        n_enc_b   = sum(p.numel() for p in self.encoder_b.parameters())
        n_fusion  = sum(p.numel() for p in self.fusion.parameters())
        n_bottleneck = (
            sum(p.numel() for p in self.to_mu.parameters())
            + sum(p.numel() for p in self.to_logvar.parameters())
        )
        n_decoder = (
            sum(p.numel() for p in self.decoder.parameters())
            + sum(p.numel() for p in self.xct_head.parameters())
            + sum(p.numel() for p in self.mask_head.parameters())
        )
        total = sum(p.numel() for p in self.parameters())
        logger.info(
            "ConvVAE3DNoAttnDualBranchV2 (z=%d, base=%d, n_blocks=%d): "
            "total=%d params  [enc_a=%d, enc_b=%d, fusion=%d, bottleneck=%d, decoder+heads=%d]",
            cfg.z_channels, cfg.base_channels, cfg.n_blocks,
            total, n_enc_a, n_enc_b, n_fusion, n_bottleneck, n_decoder,
        )

    def forward(self, xct: torch.Tensor, mask: torch.Tensor) -> VAEOutput:
        """
        Parameters
        ----------
        xct  : (B, 1, D, H, W) float32 — XCT intensity, z-scored.
        mask : (B, 1, D, H, W) float32 {0, 1} — pore mask (decoder target only;
               NOT fed to the encoder, consistent with the R03 XCT-only design).
        """
        h_a = self.encoder_a(xct)                         # (B, enc_out_ch, d, h, w)
        h_b = self.encoder_b(xct)                         # (B, enc_out_ch, d, h, w)
        h   = self.fusion(torch.cat([h_a, h_b], dim=1))   # (B, enc_out_ch, d, h, w)

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
