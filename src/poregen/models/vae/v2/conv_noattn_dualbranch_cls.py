"""ConvVAE3DNoAttnDualBranchClsV2 — the r08 variant: 3-class decoder head.

Same trunk as ``v2.conv_noattn_dualbranch`` (r04/r05/r07): two independent
encoder branches, a 1×1×1 fusion, a conv bottleneck and a mirrored decoder.
Two things change, and both come from ``split_v3``'s 3-class voxel label
(0 material, 1 pore, 2 air):

**Encoder input is three channels.**  ``cat([xct, pore, air])``, where
``pore = label == 1`` and ``air = label == 2``.  Material is implied by the
other two being zero, so it carries no channel of its own.  Exterior air and
the drilled registration holes are therefore something the encoder can see,
instead of being an unexplained dark region inside a "material" patch.

**The binary mask head is gone.**  In its place a 3-channel logit head over the
same classes, read back with :func:`~poregen.models.vae.base.decode_label`
(argmax) or :func:`~poregen.models.vae.base.decode_class_probs` (softmax).
``VAEOutput.mask_logits`` stays ``None`` here: a two-valued head and a
three-valued one are different contracts, and silently emitting both would let
a caller consume a pore mask that disagrees with the label.

The older ``v2.conv_noattn_dualbranch`` is registered unchanged and stays in
use — the r07 checkpoint behind ``data/split_v2/latents_r07z4`` is still loaded
by the campaign 05 and 08 analysis scripts.

Architecture (n_blocks=2, base_channels=32, z_channels=4, in_channels=3)::

    cat([xct, pore, air]) (B, 3, 64, 64, 64)
         ┌──────────────────────────┬───────────────────────────┐
         │ Branch A (structural)    │ Branch B (texture/detail) │
    down_v2(3→32)  →(B,32,32³)  down_v2(3→32)  →(B,32,32³)
    down_v2(32→64) →(B,64,16³)  down_v2(32→64) →(B,64,16³)
         └──────────────┬───────────────────────┘
              cat([A,B]) →(B,128,16³)
         fusion Conv3d(128→64, 1×1×1) →(B,64,16³)
                        ↓
              to_mu / to_logvar → (B,4,16³)
                        ↓ decoder
           up_v2(4→64)  →(B,64,32³)
           up_v2(64→32) →(B,32,64³)
              xct_head →(B,1,64³)   class_head →(B,3,64³)
"""

from __future__ import annotations

import logging

import torch
import torch.nn as nn

from poregen.models.nn.blocks import down_block_v2, up_block_v2, reparameterize
from poregen.models.vae.base import (
    CLASS_AIR, CLASS_PORE, N_CLASSES, VAEConfig, VAEOutput,
)
from poregen.models.vae.registry import register_vae

logger = logging.getLogger(__name__)

ENCODER_IN_CHANNELS = 3


def label_to_channels(label: torch.Tensor) -> torch.Tensor:
    """``(B, D, H, W)`` class index → ``(B, 2, D, H, W)`` float pore/air planes.

    Material needs no plane: it is where both are zero.
    """
    if label.ndim == 5:                       # tolerate a (B, 1, D, H, W) label
        label = label.squeeze(1)
    return torch.stack([(label == CLASS_PORE), (label == CLASS_AIR)],
                       dim=1).to(torch.float32)


@register_vae("v2.conv_noattn_dualbranch_cls")
class ConvVAE3DNoAttnDualBranchClsV2(nn.Module):
    """Dual-branch encoder VAE with a 3-class decoder head (r08).

    ``cfg.in_channels`` must be 3 — XCT plus the pore and air planes.  The
    value is not a switch: there is no other input layout for this variant.
    """

    # The batch keys this variant's forward() consumes, in order.  The training
    # engine reads this instead of hard-coding ("xct", "mask").
    encoder_inputs: tuple[str, ...] = ("xct", "label")

    def __init__(self, cfg: VAEConfig) -> None:
        super().__init__()
        if cfg.in_channels != ENCODER_IN_CHANNELS:
            raise ValueError(
                f"{type(self).__name__} takes cat([xct, pore, air]); set "
                f"model.in_channels: {ENCODER_IN_CHANNELS} "
                f"(got {cfg.in_channels})."
            )
        self.cfg = cfg
        ch = cfg.channel_schedule()
        enc_out_ch = ch[cfg.n_blocks - 1]

        def _branch() -> nn.Sequential:
            blocks, in_ch = [], cfg.in_channels
            for i in range(cfg.n_blocks):
                blocks.append(down_block_v2(in_ch, ch[i]))
                in_ch = ch[i]
            return nn.Sequential(*blocks)

        self.encoder_a = _branch()
        self.encoder_b = _branch()
        self.fusion = nn.Conv3d(2 * enc_out_ch, enc_out_ch, kernel_size=1, bias=False)

        self.to_mu     = nn.Conv3d(enc_out_ch, cfg.z_channels, 1)
        self.to_logvar = nn.Conv3d(enc_out_ch, cfg.z_channels, 1)

        dec: list[nn.Module] = []
        in_ch = cfg.z_channels
        for i in range(cfg.n_blocks - 1, -1, -1):
            dec.append(up_block_v2(in_ch, ch[i]))
            in_ch = ch[i]
        self.decoder = nn.Sequential(*dec)

        self.xct_head   = nn.Conv3d(ch[0], 1, 1)
        self.class_head = nn.Conv3d(ch[0], N_CLASSES, 1)

        logger.info(
            "ConvVAE3DNoAttnDualBranchClsV2 (z=%d, base=%d, n_blocks=%d, "
            "in=%d, classes=%d): total=%d params",
            cfg.z_channels, cfg.base_channels, cfg.n_blocks, cfg.in_channels,
            N_CLASSES, sum(p.numel() for p in self.parameters()),
        )

    def forward(self, xct: torch.Tensor, label: torch.Tensor) -> VAEOutput:
        """
        Parameters
        ----------
        xct   : (B, 1, D, H, W) float32 — grey level in [0, 1].
        label : (B, D, H, W) int64 — 0 material, 1 pore, 2 air.  Split into
                two binary planes and concatenated to the XCT.
        """
        enc_in = torch.cat([xct, label_to_channels(label).to(xct.dtype)], dim=1)
        h_a = self.encoder_a(enc_in)
        h_b = self.encoder_b(enc_in)
        h   = self.fusion(torch.cat([h_a, h_b], dim=1))

        mu     = self.to_mu(h)
        logvar = self.to_logvar(h)
        z      = reparameterize(mu, logvar)

        dec = self.decoder(z)
        return VAEOutput(
            xct_out=self.xct_head(dec),
            class_logits=self.class_head(dec),
            mu=mu,
            logvar=logvar,
            z=z,
        )
