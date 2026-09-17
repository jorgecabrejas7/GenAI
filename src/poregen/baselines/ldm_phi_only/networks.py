"""The phi-only denoiser: ldm06's trunk, conditioned on porosity alone.

WHY THIS IS A SUBCLASS AND NOT A COPY. The comparison this baseline exists to
make is "what do the OTHER conditioning signals buy?", and that question is
only answerable if the two models are the same network apart from those
signals. A copied trunk would answer it on the day it was written and then
drift silently the first time either file was touched. Subclassing means the
encoder, bottleneck, decoder, output head, timestep embedding, porosity MLP and
learned null are not merely equivalent to ldm06's — they are ldm06's, built by
the same code from the same config. `tests/test_ldm_phi_only.py` asserts that,
so the claim is checked rather than asserted in a docstring.

WHAT IS REMOVED, AND WHAT THAT COSTS. The stem narrows from 155 input channels
to 8 (the latent alone), and the depth, face-distance and neighbour-pooling
MLPs go. That is 1.33 M of 83.00 M: the phi-only model is 81.67 M parameters,
98.4 % of ldm06. CAPACITY IS THEREFORE NOT THE VARIABLE — a gap between the two
is the conditioning.

THE FORWARD SIGNATURE IS DELIBERATELY UNCHANGED. Every call site in this
project passes the conditioning positionally in one fixed order — the training
engine, the guided and unguided sampler paths, eval_v4's generator. Keeping the
signature and IGNORING the arguments this model has no use for makes it a
drop-in at all of them, so not one line of the engine, the sampler, the dataset
or the evaluation suite changes to run this baseline. The ignored arguments are
accepted and dropped on the floor on purpose, and are named in the signature so
a reader can see exactly which inputs the model is blind to.

ONE CONSEQUENCE WORTH KNOWING BEFORE READING ANY NUMBER FROM IT. The guided
sampler computes three passes — unconditional, porosity-guided, and
neighbour-guided. With neighbours ignored the third is identical to the second,
so `s_nb` has no effect at all and its term is exactly zero. Run this model with
the s_por arm only; an s_nb sweep on it would produce a flat line that looks
like a measurement and is an identity.
"""

from __future__ import annotations

import logging

import torch
import torch.nn as nn

from poregen.models.diffusion.unet import UNet3DConfig, UNet3DDenoiser

logger = logging.getLogger(__name__)

#: The conditioning ldm06 has and this model does not. Named here so the test
#: that guards the trunk knows exactly what it is allowed to find missing.
REMOVED_MODULES = ("depth_mlp", "dist6_mlp", "nb_pool_mlp", "nb_avail_emb", "nb_t_sin")


class PhiOnlyDenoiser(UNet3DDenoiser):
    """ldm06's U-Net conditioned on ``cond_por`` and the timestep only."""

    def __init__(self, cfg: UNet3DConfig) -> None:
        super().__init__(cfg)
        # The trunk above is ldm06's, unchanged. Only the stem is rebuilt, and
        # only because its width is set by the number of conditioning channels.
        self.input_proj = nn.Conv3d(cfg.z_channels, cfg.channels[0],
                                    kernel_size=3, padding=1)
        for name in REMOVED_MODULES:
            delattr(self, name)
        total = sum(p.numel() for p in self.parameters())
        logger.info(
            "PhiOnlyDenoiser: z_ch=%d  in_ch=%d (latent only)  base_ch=%d  "
            "mult=%s — %d params (%.2fM). Conditioned on porosity and t alone; "
            "orientation, material, neighbours, availability, nb_t, depth and "
            "the six face distances are ACCEPTED AND IGNORED.",
            cfg.z_channels, cfg.z_channels, cfg.base_channels,
            cfg.channel_mult, total, total / 1e6,
        )

    def _build_cond(                                     # type: ignore[override]
        self,
        t: torch.Tensor,
        cond_por: torch.Tensor,
        cond_depth: torch.Tensor,
        cond_dist6: torch.Tensor,
        nb_latents: torch.Tensor,
        nb_avail: torch.Tensor,
        drop_por: "torch.Tensor | None" = None,
    ) -> torch.Tensor:
        """``t_emb + por_emb``, with ldm06's own learned null for CFG dropout.

        ``cond_depth``, ``cond_dist6``, ``nb_latents`` and ``nb_avail`` are
        accepted and ignored — see the module docstring.
        """
        cond = self.t_mlp(self.t_sin(t))
        por_emb = self.por_mlp(cond_por.float().view(-1, 1))
        if drop_por is not None:
            null = self.null_por.to(por_emb.dtype).unsqueeze(0).expand_as(por_emb)
            por_emb = torch.where(drop_por.view(-1, 1), null, por_emb)
        return cond + por_emb

    def forward(                                         # type: ignore[override]
        self,
        z_t: torch.Tensor,
        t: torch.Tensor,
        nb_latents: torch.Tensor,
        nb_avail: torch.Tensor,
        nb_t: torch.Tensor,
        cond_por: torch.Tensor,
        cond_depth: torch.Tensor,
        cond_dist6: torch.Tensor,
        cond_orient: torch.Tensor,
        cond_material: torch.Tensor,
        drop_por: "torch.Tensor | None" = None,
    ) -> torch.Tensor:
        """Predict v in ``z_t`` from the latent, the timestep and the porosity.

        The signature is ldm06's exactly, so this is a drop-in at every call
        site. ``nb_latents``, ``nb_avail``, ``nb_t``, ``cond_depth``,
        ``cond_dist6``, ``cond_orient`` and ``cond_material`` are ignored.
        """
        cond = self._build_cond(t, cond_por, cond_depth, cond_dist6,
                                nb_latents, nb_avail, drop_por)
        return self._trunk(self.input_proj(z_t), cond)
