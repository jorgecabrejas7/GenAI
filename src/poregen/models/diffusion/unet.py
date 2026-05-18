"""3-D U-Net denoiser for the PoreGen Latent Diffusion Model.

Architecture
------------
Input tensor assembly (all concatenated along the channel dim before the
first Conv3d projection):

    z_t              (B, z_ch,      16, 16, 16) — noisy latent at step t
    nb_latents_flat  (B, 6×z_ch,   16, 16, 16) — 6 neighbor latents (zero if unavail)
    nb_avail_spatial (B, 6×avail_d, 16, 16, 16) — learned availability embeddings

    Total input channels = z_ch + 6×z_ch + 6×avail_d  (160 with defaults)

Conditioning vector (scalar → AdaGN in every ResBlock):

    t_emb   = SinusoidalPositionEmbedding(256)(t)  → MLP → (B, cond_dim)
    pos_emb = MLP(pos_frac [3])                    → (B, cond_dim)
    por_emb = MLP(cat(global_por, local_por) [2])  → (B, cond_dim)
    cond    = t_emb + pos_emb + por_emb            (element-wise sum)

U-Net levels (spatial: 16³ → 8³ → 4³ → 8³ → 16³):

    input_proj  Conv3d(in_ch, base_ch, 3)
    down0       n_res_blocks × ResBlock3D(base_ch)           → skip0
    down1       Downsample + n_res_blocks × ResBlock3D(2×)   → skip1
    bottleneck  Downsample + n_res_blocks × ResBlock3D(4×) + SelfAttn3D + ResBlock3D(4×)
    up1         Upsample + n_res_blocks × ResBlock3D(skip1 cat, 2×)
    up0         Upsample + n_res_blocks × ResBlock3D(skip0 cat, 1×)
    output_proj GN + SiLU + Conv3d(base_ch, z_ch)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import torch
import torch.nn as nn

from poregen.models.diffusion.blocks import (
    AdaGN,
    Downsample3D,
    ResBlock3D,
    SelfAttention3D,
    SinusoidalPositionEmbedding,
    Upsample3D,
    norm_groups,
)

logger = logging.getLogger(__name__)

_N_NEIGHBORS = 6


@dataclass
class UNet3DConfig:
    """Hyperparameters for :class:`UNet3DDenoiser`."""

    z_channels: int = 16
    base_channels: int = 128
    channel_mult: tuple[int, ...] = (1, 2, 4)
    n_res_blocks: int = 2
    attn_at_bottleneck: bool = True
    cond_embed_dim: int = 512
    nb_avail_embed_dim: int = 8
    n_attn_heads: int = 8
    dropout: float = 0.0
    # Ablation flags — control which conditioning components are active
    use_neighbor_cond: bool = True    # concatenate neighbor latents to input
    use_avail_embedding: bool = True  # per-direction availability state embedding
    use_pos_cond: bool = True         # spatial position MLP in scalar conditioning
    use_por_cond: bool = True         # porosity MLP in scalar conditioning

    @property
    def channels(self) -> list[int]:
        return [self.base_channels * m for m in self.channel_mult]

    @property
    def in_channels(self) -> int:
        ch = self.z_channels
        if self.use_neighbor_cond:
            ch += _N_NEIGHBORS * self.z_channels
            if self.use_avail_embedding:
                ch += _N_NEIGHBORS * self.nb_avail_embed_dim
        return ch

    @classmethod
    def from_cfg(cls, cfg: dict) -> "UNet3DConfig":
        m = cfg.get("model", cfg)
        return cls(
            z_channels          = int(m.get("z_channels", 16)),
            base_channels       = int(m.get("base_channels", 128)),
            channel_mult        = tuple(m.get("channel_mult", (1, 2, 4))),
            n_res_blocks        = int(m.get("n_res_blocks", 2)),
            attn_at_bottleneck  = bool(m.get("attn_at_bottleneck", True)),
            cond_embed_dim      = int(m.get("cond_embed_dim", 512)),
            nb_avail_embed_dim  = int(m.get("nb_avail_embed_dim", 8)),
            n_attn_heads        = int(m.get("n_attn_heads", 8)),
            dropout             = float(m.get("dropout", 0.0)),
            use_neighbor_cond   = bool(m.get("use_neighbor_cond",   True)),
            use_avail_embedding = bool(m.get("use_avail_embedding", True)),
            use_pos_cond        = bool(m.get("use_pos_cond",        True)),
            use_por_cond        = bool(m.get("use_por_cond",        True)),
        )


def _make_mlp(in_dim: int, out_dim: int) -> nn.Sequential:
    """Two-layer MLP with SiLU activation."""
    return nn.Sequential(
        nn.Linear(in_dim, out_dim),
        nn.SiLU(),
        nn.Linear(out_dim, out_dim),
    )


class UNet3DDenoiser(nn.Module):
    """3-D U-Net for DDPM latent patch denoising.

    See module docstring for full architecture description.

    Parameters
    ----------
    cfg : UNet3DConfig
    """

    def __init__(self, cfg: UNet3DConfig) -> None:
        super().__init__()
        self.cfg   = cfg
        chs        = cfg.channels           # e.g. [128, 256, 512]
        C          = cfg.cond_embed_dim     # 512
        n_res      = cfg.n_res_blocks
        drop       = cfg.dropout

        # ── Neighbor availability embedding (optional) ───────────────────────
        if cfg.use_neighbor_cond and cfg.use_avail_embedding:
            self.nb_avail_emb = nn.Embedding(_N_NEIGHBORS * 3, cfg.nb_avail_embed_dim)

        # ── Conditioning MLPs ────────────────────────────────────────────────
        sin_dim = 256
        self.t_sin = SinusoidalPositionEmbedding(sin_dim)
        self.t_mlp = _make_mlp(sin_dim, C)
        if cfg.use_pos_cond:
            self.pos_mlp = _make_mlp(3, C)
        if cfg.use_por_cond:
            self.por_mlp = _make_mlp(2, C)

        # ── Input projection ─────────────────────────────────────────────────
        self.input_proj = nn.Conv3d(cfg.in_channels, chs[0], kernel_size=3, padding=1)

        # ── Encoder ─────────────────────────────────────────────────────────
        self.down0_blocks = nn.ModuleList(
            [ResBlock3D(chs[0], chs[0], C, drop) for _ in range(n_res)]
        )
        self.down1_sample = Downsample3D(chs[0])
        self.down1_blocks = nn.ModuleList(
            [ResBlock3D(chs[0] if i == 0 else chs[1], chs[1], C, drop) for i in range(n_res)]
        )

        # ── Bottleneck ───────────────────────────────────────────────────────
        self.bot_sample  = Downsample3D(chs[1])
        self.bot_in      = nn.ModuleList(
            [ResBlock3D(chs[1] if i == 0 else chs[2], chs[2], C, drop) for i in range(n_res)]
        )
        self.bot_attn    = SelfAttention3D(chs[2], cfg.n_attn_heads) if cfg.attn_at_bottleneck else nn.Identity()
        self.bot_out     = ResBlock3D(chs[2], chs[2], C, drop)

        # ── Decoder ─────────────────────────────────────────────────────────
        self.up1_sample  = Upsample3D(chs[2])
        self.up1_blocks  = nn.ModuleList(
            [ResBlock3D((chs[2] + chs[1]) if i == 0 else chs[1], chs[1], C, drop) for i in range(n_res)]
        )
        self.up0_sample  = Upsample3D(chs[1])
        self.up0_blocks  = nn.ModuleList(
            [ResBlock3D((chs[1] + chs[0]) if i == 0 else chs[0], chs[0], C, drop) for i in range(n_res)]
        )

        # ── Output projection ────────────────────────────────────────────────
        self.out_norm = nn.GroupNorm(norm_groups(chs[0]), chs[0])
        self.out_act  = nn.SiLU()
        self.out_conv = nn.Conv3d(chs[0], cfg.z_channels, kernel_size=1)

        # Zero-init output conv so predictions start near zero (common DDPM practice)
        nn.init.zeros_(self.out_conv.weight)
        nn.init.zeros_(self.out_conv.bias)

        total = sum(p.numel() for p in self.parameters())
        logger.info(
            "UNet3DDenoiser: z_ch=%d, base_ch=%d, mult=%s, n_res=%d — %d params (%.1fM)",
            cfg.z_channels, cfg.base_channels, cfg.channel_mult, cfg.n_res_blocks,
            total, total / 1e6,
        )

    # ------------------------------------------------------------------

    def _build_cond(
        self,
        t: torch.Tensor,
        pos_frac: torch.Tensor,
        global_por: torch.Tensor,
        local_por: torch.Tensor,
    ) -> torch.Tensor:
        """Assemble the scalar conditioning vector.

        Returns
        -------
        (B, cond_embed_dim)
        """
        cond = self.t_mlp(self.t_sin(t))                             # (B, C) — always
        if self.cfg.use_pos_cond:
            cond = cond + self.pos_mlp(pos_frac.float())
        if self.cfg.use_por_cond:
            por  = torch.stack([global_por, local_por], dim=1).float()
            cond = cond + self.por_mlp(por)
        return cond

    def _build_nb_spatial(
        self,
        nb_latents: torch.Tensor,
        nb_avail: torch.Tensor,
        spatial_shape: tuple[int, int, int],
    ) -> torch.Tensor:
        """Build the spatial neighbor conditioning tensor.

        Parameters
        ----------
        nb_latents : (B, 6, z_ch, D, H, W) — neighbor latents (zeros for unavail)
        nb_avail   : (B, 6) long — availability state {0, 1, 2} per neighbor
        spatial_shape : (D, H, W)

        Returns
        -------
        (B, 6×z_ch + 6×avail_d, D, H, W)
        """
        B = nb_latents.shape[0]
        D, H, W = spatial_shape

        # Flatten neighbor latents: (B, 6*z_ch, D, H, W)
        nb_flat = nb_latents.view(B, _N_NEIGHBORS * self.cfg.z_channels, D, H, W)

        if not self.cfg.use_avail_embedding:
            return nb_flat

        # Availability embeddings: (B, 6) → (B, 6, avail_d) → spatial broadcast
        # Per-neighbor offset keeps embeddings distinct across the 6 directions
        offsets = torch.arange(_N_NEIGHBORS, device=nb_avail.device).unsqueeze(0) * 3  # (1, 6)
        avail_idx = (nb_avail + offsets).long()                     # (B, 6)
        avail_emb = self.nb_avail_emb(avail_idx)                    # (B, 6, avail_d)
        avail_flat = avail_emb.view(B, _N_NEIGHBORS * self.cfg.nb_avail_embed_dim, 1, 1, 1)
        avail_spatial = avail_flat.expand(B, -1, D, H, W)           # (B, 6*avail_d, D, H, W)

        return torch.cat([nb_flat, avail_spatial], dim=1)           # (B, 6*z_ch + 6*avail_d, D, H, W)

    def forward(
        self,
        z_t: torch.Tensor,
        t: torch.Tensor,
        nb_latents: torch.Tensor,
        nb_avail: torch.Tensor,
        pos_frac: torch.Tensor,
        global_por: torch.Tensor,
        local_por: torch.Tensor,
    ) -> torch.Tensor:
        """Predict the noise ε in z_t.

        Parameters
        ----------
        z_t        : (B, z_ch, 16, 16, 16) — noisy latent
        t          : (B,) long — diffusion timestep
        nb_latents : (B, 6, z_ch, 16, 16, 16) — neighbor latents
        nb_avail   : (B, 6) long — 0=OOB, 1=exists, 2=unknown
        pos_frac   : (B, 3) float — normalised patch position in volume
        global_por : (B,) float — volume-level void volume fraction
        local_por  : (B,) float — patch-level void volume fraction

        Returns
        -------
        eps_pred : (B, z_ch, 16, 16, 16) — predicted noise
        """
        B, _, D, H, W = z_t.shape
        cond = self._build_cond(t, pos_frac, global_por, local_por)                  # (B, C)
        if self.cfg.use_neighbor_cond:
            nb = self._build_nb_spatial(nb_latents, nb_avail, (D, H, W))
            x  = torch.cat([z_t, nb], dim=1)
        else:
            x = z_t
        x    = self.input_proj(x)                                                     # (B, ch0, D,H,W)

        # ── Encoder ────────────────────────────────────────────────────────
        for blk in self.down0_blocks:
            x = blk(x, cond)
        skip0 = x                                                                      # 16³

        x = self.down1_sample(x)
        for blk in self.down1_blocks:
            x = blk(x, cond)
        skip1 = x                                                                      # 8³

        # ── Bottleneck ─────────────────────────────────────────────────────
        x = self.bot_sample(x)
        for blk in self.bot_in:
            x = blk(x, cond)
        if self.cfg.attn_at_bottleneck:
            x = self.bot_attn(x)
        x = self.bot_out(x, cond)                                                     # 4³

        # ── Decoder ────────────────────────────────────────────────────────
        x = self.up1_sample(x)
        x = torch.cat([x, skip1], dim=1)
        for blk in self.up1_blocks:
            x = blk(x, cond)                                                          # 8³

        x = self.up0_sample(x)
        x = torch.cat([x, skip0], dim=1)
        for blk in self.up0_blocks:
            x = blk(x, cond)                                                          # 16³

        return self.out_conv(self.out_act(self.out_norm(x)))                          # (B, z_ch, D,H,W)
