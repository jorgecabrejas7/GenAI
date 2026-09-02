"""3-D U-Net denoiser for the PoreGen Latent Diffusion Model (ldm06).

Architecture
------------
Input tensor assembly (concatenated along channel dim before input_proj):

    z_t              (B, z_ch,       D, H, W) — noisy latent at step t
    cond_orient      (B, 2,          D, H, W) — (cos2θ, sin2θ) ply orientation profile
    cond_material    (B, 1,          D, H, W) — material fraction per latent cell
    nb_latents_flat  (B, 6×z_ch,     D, H, W) — the 6 face-adjacent neighbour latents,
                                                each noised to its OWN timestep
                                                (zero where not EXISTS)
    nb_avail_spatial (B, 6×avail_d,  D, H, W) — learned availability embeddings
    nb_t_spatial     (B, 6×nb_t_d,   D, H, W) — sinusoidal embeddings of nb_t
    Total input channels
        = z_ch + 2 + 1 + 6×z_ch + 6×avail_d + 6×nb_t_d
        = 127 for z_ch=4, avail_d=8, nb_t_d=8.

Every neighbour arrives with its own noise level ``nb_t[i]``.  Training draws
it (see ``poregen.training.ldm_engine``), the sampler sets it to the canvas
timestep, and the availability embedding says whether there is a neighbour at
all.  A denoiser that only ever saw CLEAN neighbours could not be used inside a
joint reverse process, where every neighbour is itself half-denoised — that
mismatch is what ``nb_t`` removes.

Conditioning vector (scalar → AdaGN/FiLM injected into every ResBlock):

    t_emb      = SinusoidalPositionEmbedding(256)(t) → MLP → (B, cond_dim)
    por_emb    = MLP(cond_por   [1])   (learned null token for CFG)
    depth_emb  = MLP(cond_depth [1])
    dist6_emb  = MLP(cond_dist6 [6])   — ONE MLP over the whole 6-vector
    nb_pool    = MLP(pooled EXISTS neighbours [6×z_ch])
    cond       = t_emb + por_emb + depth_emb + dist6_emb + nb_pool

``cond_por`` is ``log(phi + 1e-3)`` standardised with the store's stats,
``cond_depth`` is the relative depth t ∈ [0,1], and ``cond_dist6`` holds the
six per-face distances to the specimen box (z-, z+, y-, y+, x-, x+), each
``min(d, 64)/64``.  There is NO global-porosity condition — the per-patch
porosity field carries volume-level control.

U-Net structure (N = len(channel_mult)):

    input_proj     Conv3d(in_ch, chs[0], 3)
    enc level 0    n_res × ResBlock3D(chs[0])                       → skip[0]
    enc level 1    Downsample + n_res × ResBlock3D(chs[1])          → skip[1]
    ...
    enc level N-2  Downsample + n_res × ResBlock3D(chs[N-2])        → skip[N-2]
    bottleneck     [Downsample if bottleneck_downsample] +
                   n_res × ResBlock3D(chs[N-1]) +
                   [SelfAttn3D if attn_at_bottleneck] +
                   ResBlock3D(chs[N-1])

    When bottleneck_downsample=True  (deepest = half of last enc level):
      dec level N-2  Upsample + ResBlocks(skip[N-2] cat)
      ...
      dec level 0    Upsample + ResBlocks(skip[0] cat)

    When bottleneck_downsample=False (deepest = same res as last enc level):
      dec merge      cat(bottleneck, skip[N-2]) + ResBlocks  — no upsample
      dec level N-3  Upsample + ResBlocks(skip[N-3] cat)
      ...
      dec level 0    Upsample + ResBlocks(skip[0] cat)

    output_proj    GN + SiLU + Conv3d(chs[0], z_ch, 1)

Default config (channel_mult=[1,2,4,4], bottleneck_downsample=False):
    16³(128) → 8³(256) → 4³(512) → 4³(512,bot) → 4³(merge) → 8³ → 16³
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import torch
import torch.nn as nn

from poregen.diffusion.conditioning import (
    N_AVAIL_STATES,
    N_DIST6,
    N_NEIGHBOURS,
    NB_EXISTS,
)
from poregen.models.diffusion.blocks import (
    Downsample3D,
    ResBlock3D,
    SelfAttention3D,
    SinusoidalPositionEmbedding,
    Upsample3D,
    norm_groups,
)

logger = logging.getLogger(__name__)

_ORIENT_CHANNELS = 2     # (cos2θ, sin2θ)
_MATERIAL_CHANNELS = 1   # material fraction per latent cell


@dataclass
class UNet3DConfig:
    """Hyperparameters for :class:`UNet3DDenoiser`.

    Every conditioning path is unconditional-by-omission free: orientation,
    material, neighbours, availability, neighbour timesteps, porosity (with
    its learned null), depth and the six face distances are all always on.
    ldm05 carried a switch per signal for pre-registered ablations; those runs
    are finished, and a switch that no config ever flips is a way for training
    and generation to disagree.
    """

    z_channels: int = 16
    base_channels: int = 128
    channel_mult: tuple[int, ...] = (1, 2, 4, 4)
    n_res_blocks: int = 2
    attn_at_bottleneck: bool = False
    bottleneck_downsample: bool = False  # if False, bottleneck stays at same res as last enc level
    cond_embed_dim: int = 512
    nb_avail_embed_dim: int = 8
    nb_t_embed_dim: int = 8
    n_attn_heads: int = 8
    dropout: float = 0.0

    @property
    def channels(self) -> list[int]:
        return [self.base_channels * m for m in self.channel_mult]

    @property
    def in_channels(self) -> int:
        return (
            self.z_channels
            + _ORIENT_CHANNELS
            + _MATERIAL_CHANNELS
            + N_NEIGHBOURS * self.z_channels
            + N_NEIGHBOURS * self.nb_avail_embed_dim
            + N_NEIGHBOURS * self.nb_t_embed_dim
        )

    @classmethod
    def from_cfg(cls, cfg: dict) -> "UNet3DConfig":
        m = cfg.get("model", cfg)
        return cls(
            z_channels            = int(m.get("z_channels", 16)),
            base_channels         = int(m.get("base_channels", 128)),
            channel_mult          = tuple(m.get("channel_mult", (1, 2, 4, 4))),
            n_res_blocks          = int(m.get("n_res_blocks", 2)),
            attn_at_bottleneck    = bool(m.get("attn_at_bottleneck", False)),
            bottleneck_downsample = bool(m.get("bottleneck_downsample", False)),
            cond_embed_dim        = int(m.get("cond_embed_dim", 512)),
            nb_avail_embed_dim    = int(m.get("nb_avail_embed_dim", 8)),
            nb_t_embed_dim        = int(m.get("nb_t_embed_dim", 8)),
            n_attn_heads          = int(m.get("n_attn_heads", 8)),
            dropout               = float(m.get("dropout", 0.0)),
        )


def _make_mlp(in_dim: int, out_dim: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Linear(in_dim, out_dim),
        nn.SiLU(),
        nn.Linear(out_dim, out_dim),
    )


def _res_stack(in_ch: int, out_ch: int, C: int, n: int, drop: float) -> nn.ModuleList:
    """n ResBlock3Ds: first takes in_ch, rest take out_ch."""
    return nn.ModuleList([
        ResBlock3D(in_ch if i == 0 else out_ch, out_ch, C, drop)
        for i in range(n)
    ])


class UNet3DDenoiser(nn.Module):
    """3-D U-Net for DDPM latent patch denoising.

    Depth and channel widths are driven entirely by ``cfg.channel_mult``;
    ``cfg.bottleneck_downsample`` controls whether the bottleneck adds one
    more spatial halving (True → deeper) or stays at the same resolution as
    the deepest encoder level (False → skip connection at deepest scale).

    Parameters
    ----------
    cfg : UNet3DConfig
    """

    def __init__(self, cfg: UNet3DConfig) -> None:
        super().__init__()
        self.cfg = cfg
        chs  = cfg.channels        # e.g. [128, 256, 512, 512]
        C    = cfg.cond_embed_dim
        n_r  = cfg.n_res_blocks
        drop = cfg.dropout
        n    = len(chs) - 1        # number of encoder levels (each with a skip)

        if n < 1:
            raise ValueError("channel_mult must have at least 2 entries.")

        # ── Neighbour availability + timestep embeddings ─────────────────────
        # One embedding row per (neighbour position, state) so that "EXISTS at
        # neighbour 0" is distinct from "EXISTS at neighbour 3".
        self.nb_avail_emb = nn.Embedding(
            N_NEIGHBOURS * N_AVAIL_STATES, cfg.nb_avail_embed_dim
        )
        self.nb_t_sin = SinusoidalPositionEmbedding(cfg.nb_t_embed_dim)

        # ── Conditioning MLPs ────────────────────────────────────────────────
        self.t_sin = SinusoidalPositionEmbedding(256)
        self.t_mlp = _make_mlp(256, C)
        self.depth_mlp = _make_mlp(1, C)
        # ONE MLP over the whole 6-vector: the faces are not independent (a
        # patch cannot be near both z faces of a 500-voxel specimen), so the
        # geometry is a shape, not six unrelated scalars.
        self.dist6_mlp = _make_mlp(N_DIST6, C)
        self.por_mlp = _make_mlp(1, C)
        # Learned null porosity embedding for CFG — replaces por_mlp output
        # when drop_por is set, so 0 (a real porosity) is never used as the null.
        self.null_por = nn.Parameter(torch.zeros(C))
        # Global neighbour summary: pool each neighbour latent → (B, 6·z_ch) → MLP → (B, C).
        # Directions with OOB/UNKNOWN availability are zeroed before pooling so only
        # existing neighbours contribute.
        self.nb_pool_mlp = _make_mlp(N_NEIGHBOURS * cfg.z_channels, C)

        # ── Input projection ─────────────────────────────────────────────────
        self.input_proj = nn.Conv3d(cfg.in_channels, chs[0], kernel_size=3, padding=1)

        # ── Encoder ──────────────────────────────────────────────────────────
        # enc_blocks[0] : ResBlocks at full res, no preceding downsample
        # enc_blocks[i] : ResBlocks at half^i res, preceded by enc_downs[i-1]
        self.enc_blocks: nn.ModuleList = nn.ModuleList()
        self.enc_downs:  nn.ModuleList = nn.ModuleList()

        self.enc_blocks.append(_res_stack(chs[0], chs[0], C, n_r, drop))
        for i in range(1, n):
            self.enc_downs.append(Downsample3D(chs[i - 1]))
            self.enc_blocks.append(_res_stack(chs[i - 1], chs[i], C, n_r, drop))

        # ── Bottleneck ───────────────────────────────────────────────────────
        bot_ch     = chs[n]
        enc_top_ch = chs[n - 1]   # channels of the deepest encoder level

        self.bot_down = Downsample3D(enc_top_ch) if cfg.bottleneck_downsample else nn.Identity()
        self.bot_in   = _res_stack(enc_top_ch, bot_ch, C, n_r, drop)
        self.bot_attn = (
            SelfAttention3D(bot_ch, cfg.n_attn_heads)
            if cfg.attn_at_bottleneck else nn.Identity()
        )
        self.bot_out  = ResBlock3D(bot_ch, bot_ch, C, drop)

        # ── Decoder ──────────────────────────────────────────────────────────
        # bottleneck_downsample=True  → n up-steps, each with Upsample + cat + ResBlocks
        # bottleneck_downsample=False → first step is a same-resolution merge (no Upsample),
        #                               followed by n-1 normal up-steps
        self.dec_ups:    nn.ModuleList = nn.ModuleList()
        self.dec_blocks: nn.ModuleList = nn.ModuleList()

        if not cfg.bottleneck_downsample:
            # Merge step at deepest enc resolution (no spatial upsample)
            self.dec_merge = _res_stack(bot_ch + enc_top_ch, enc_top_ch, C, n_r, drop)
            # Remaining n-1 up-steps
            for i in range(1, n):
                up_ch   = chs[n - i]
                skip_ch = chs[n - 1 - i]
                self.dec_ups.append(Upsample3D(up_ch))
                self.dec_blocks.append(_res_stack(up_ch + skip_ch, skip_ch, C, n_r, drop))
        else:
            # n up-steps, all with Upsample
            for i in range(n):
                up_ch   = chs[n - i]
                skip_ch = chs[n - 1 - i]
                self.dec_ups.append(Upsample3D(up_ch))
                self.dec_blocks.append(_res_stack(up_ch + skip_ch, skip_ch, C, n_r, drop))

        # ── Output projection ────────────────────────────────────────────────
        self.out_norm = nn.GroupNorm(norm_groups(chs[0]), chs[0])
        self.out_act  = nn.SiLU()
        self.out_conv = nn.Conv3d(chs[0], cfg.z_channels, kernel_size=1)
        nn.init.zeros_(self.out_conv.weight)
        nn.init.zeros_(self.out_conv.bias)

        total = sum(p.numel() for p in self.parameters())
        bot_res = "same" if not cfg.bottleneck_downsample else "halved"
        logger.info(
            "UNet3DDenoiser: z_ch=%d  in_ch=%d  base_ch=%d  mult=%s  n_res=%d  "
            "bot_res=%s  attn=%s — %d params (%.1fM)",
            cfg.z_channels, cfg.in_channels, cfg.base_channels, cfg.channel_mult,
            cfg.n_res_blocks, bot_res, cfg.attn_at_bottleneck,
            total, total / 1e6,
        )

    # ── conditioning helpers ──────────────────────────────────────────────────

    def _build_cond(
        self,
        t: torch.Tensor,
        cond_por: torch.Tensor,
        cond_depth: torch.Tensor,
        cond_dist6: torch.Tensor,
        nb_latents: torch.Tensor,
        nb_avail: torch.Tensor,
        drop_por: "torch.Tensor | None" = None,
    ) -> torch.Tensor:
        cond = self.t_mlp(self.t_sin(t))
        cond = cond + self.depth_mlp(cond_depth.float().view(-1, 1))
        cond = cond + self.dist6_mlp(cond_dist6.float().view(-1, N_DIST6))

        por_emb = self.por_mlp(cond_por.float().view(-1, 1))
        if drop_por is not None:
            # CFG: replace por_mlp output with the learned null for dropped
            # samples.  null_por is (C,); broadcast to (B, C) for torch.where.
            null = self.null_por.to(por_emb.dtype).unsqueeze(0).expand_as(por_emb)
            por_emb = torch.where(drop_por.view(-1, 1), null, por_emb)
        cond = cond + por_emb

        # Pool each neighbour over spatial dims; zero out OOB/UNKNOWN directions.
        # nb_latents : (B, 6, z_ch, D, H, W);  nb_avail : (B, 6) long
        nb_pooled = nb_latents.float().mean(dim=(-3, -2, -1))             # (B, 6, z_ch)
        exists    = (nb_avail == NB_EXISTS).float().unsqueeze(-1)         # (B, 6, 1)
        nb_flat   = (nb_pooled * exists).view(nb_pooled.shape[0], -1)     # (B, 6·z_ch)
        return cond + self.nb_pool_mlp(nb_flat)

    def _build_nb_spatial(
        self,
        nb_latents: torch.Tensor,
        nb_avail: torch.Tensor,
        nb_t: torch.Tensor,
        spatial_shape: tuple[int, int, int],
    ) -> torch.Tensor:
        B = nb_latents.shape[0]
        D, H, W = spatial_shape
        # Zero out non-EXISTS neighbours so the spatial conv path is invariant to
        # latent values when availability is UNKNOWN or OOB.  This is a no-op on
        # training data (non-EXISTS entries are already zero by construction), but is
        # required for CFG correctness: the guided sampler reuses one nb_latents
        # tensor across all three passes and flips nb_avail only.
        exists_mask = (nb_avail == NB_EXISTS).view(B, N_NEIGHBOURS, 1, 1, 1, 1)
        nb_latents  = nb_latents * exists_mask.to(nb_latents.dtype)
        nb_flat = nb_latents.view(B, N_NEIGHBOURS * self.cfg.z_channels, D, H, W)

        offsets      = torch.arange(N_NEIGHBOURS, device=nb_avail.device).unsqueeze(0)
        avail_idx    = (nb_avail + offsets * N_AVAIL_STATES).long()
        avail_emb    = self.nb_avail_emb(avail_idx)
        avail_flat   = avail_emb.view(B, N_NEIGHBOURS * self.cfg.nb_avail_embed_dim, 1, 1, 1)

        # Per-neighbour noise level: one sinusoidal embedding each, flattened
        # in the same neighbour order as nb_latents.
        t_emb  = self.nb_t_sin(nb_t.reshape(-1).long())                  # (B·6, nb_t_d)
        t_flat = t_emb.view(B, N_NEIGHBOURS * self.cfg.nb_t_embed_dim, 1, 1, 1)

        broadcast = torch.cat([avail_flat, t_flat.to(avail_flat.dtype)], dim=1)
        return torch.cat(
            [nb_flat, broadcast.expand(B, -1, D, H, W).to(nb_flat.dtype)], dim=1
        )

    # ── forward ──────────────────────────────────────────────────────────────

    def forward(
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
        """Predict noise ε in z_t.

        Parameters
        ----------
        z_t           : (B, z_ch, D, H, W)
        t             : (B,) long — the target's timestep
        nb_latents    : (B, 6, z_ch, D, H, W) — whole, unshifted neighbours,
                        each already noised to its own ``nb_t``
        nb_avail      : (B, 6) long — 0=OOB, 1=EXISTS, 2=UNKNOWN
        nb_t          : (B, 6) long — per-neighbour noise level (0 where the
                        neighbour is not EXISTS)
        cond_por      : (B,) float — log(phi+1e-3), standardised
        cond_depth    : (B,) float — relative depth t in [0,1]
        cond_dist6    : (B, 6) float — per-face distance to the specimen box,
                        ordered (z-, z+, y-, y+, x-, x+), each min(d,64)/64
        cond_orient   : (B, 2, D, H, W) float — (cos2θ, sin2θ) profile
        cond_material : (B, 1, D, H, W) float — material fraction per cell
        drop_por      : (B,) bool or None — per-sample porosity dropout for CFG.
                        When True for a sample, por_mlp output is replaced with
                        null_por.  None = no dropout.

        Returns
        -------
        eps_pred : (B, z_ch, D, H, W)
        """
        B, _, D, H, W = z_t.shape
        cond = self._build_cond(t, cond_por, cond_depth, cond_dist6,
                                nb_latents, nb_avail, drop_por)

        x = torch.cat(
            [
                z_t,
                cond_orient.to(z_t.dtype),
                cond_material.to(z_t.dtype),
                self._build_nb_spatial(nb_latents, nb_avail, nb_t, (D, H, W)).to(z_t.dtype),
            ],
            dim=1,
        )
        x = self.input_proj(x)

        # ── Encoder ──────────────────────────────────────────────────────────
        skips: list[torch.Tensor] = []
        for blk in self.enc_blocks[0]:
            x = blk(x, cond)
        skips.append(x)

        for down, level_blocks in zip(self.enc_downs, self.enc_blocks[1:]):
            x = down(x)
            for blk in level_blocks:
                x = blk(x, cond)
            skips.append(x)

        # ── Bottleneck ────────────────────────────────────────────────────────
        x = self.bot_down(x)      # Identity or Downsample depending on config
        for blk in self.bot_in:
            x = blk(x, cond)
        if self.cfg.attn_at_bottleneck:
            x = self.bot_attn(x)
        x = self.bot_out(x, cond)

        # ── Decoder ──────────────────────────────────────────────────────────
        if not self.cfg.bottleneck_downsample:
            # Merge bottleneck with deepest enc skip at same resolution
            x = torch.cat([x, skips[-1]], dim=1)
            for blk in self.dec_merge:
                x = blk(x, cond)
            # Upsample back through the remaining encoder levels
            for up, level_blocks, skip in zip(
                self.dec_ups, self.dec_blocks, reversed(skips[:-1])
            ):
                x = up(x)
                x = torch.cat([x, skip], dim=1)
                for blk in level_blocks:
                    x = blk(x, cond)
        else:
            for up, level_blocks, skip in zip(
                self.dec_ups, self.dec_blocks, reversed(skips)
            ):
                x = up(x)
                x = torch.cat([x, skip], dim=1)
                for blk in level_blocks:
                    x = blk(x, cond)

        return self.out_conv(self.out_act(self.out_norm(x)))
