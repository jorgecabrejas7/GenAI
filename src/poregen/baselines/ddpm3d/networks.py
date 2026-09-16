"""A small 3-D U-Net denoiser that works on VOXELS.

Deliberately modest. The point of the baseline is the COST of pixel space, so
the architecture is the smallest thing that can denoise a 64-cubed patch
competently; making it large would confound "pixel space is expensive" with
"this particular net is expensive".

Four output channels, as everywhere in this project: grey plus the 3-class
label, diffused together so the pair is generated jointly rather than
assembled.
"""

from __future__ import annotations

import math

import torch
from torch import nn

IN_CHANNELS = 4
OUT_CHANNELS = 4


def timestep_embedding(t: torch.Tensor, dim: int) -> torch.Tensor:
    """Sinusoidal embedding, the standard DDPM one."""
    half = dim // 2
    freqs = torch.exp(
        -math.log(10000.0) * torch.arange(half, dtype=torch.float32, device=t.device) / half
    )
    a = t.float()[:, None] * freqs[None]
    return torch.cat([torch.cos(a), torch.sin(a)], dim=-1)


class Block(nn.Module):
    def __init__(self, cin: int, cout: int, tdim: int) -> None:
        super().__init__()
        self.norm1 = nn.GroupNorm(8, cin)
        self.conv1 = nn.Conv3d(cin, cout, 3, padding=1)
        self.temb = nn.Linear(tdim, cout)
        self.norm2 = nn.GroupNorm(8, cout)
        self.conv2 = nn.Conv3d(cout, cout, 3, padding=1)
        self.skip = nn.Conv3d(cin, cout, 1) if cin != cout else nn.Identity()

    def forward(self, x: torch.Tensor, temb: torch.Tensor) -> torch.Tensor:
        h = self.conv1(torch.nn.functional.silu(self.norm1(x)))
        h = h + self.temb(torch.nn.functional.silu(temb))[:, :, None, None, None]
        h = self.conv2(torch.nn.functional.silu(self.norm2(h)))
        return h + self.skip(x)


class UNet3DPixel(nn.Module):
    """(B, 4, 64, 64, 64) + t -> (B, 4, 64, 64, 64)."""

    def __init__(self, base: int = 48, mults: tuple[int, ...] = (1, 2, 4),
                 tdim: int = 192) -> None:
        super().__init__()
        self.tdim = tdim
        self.tmlp = nn.Sequential(nn.Linear(tdim, tdim), nn.SiLU(), nn.Linear(tdim, tdim))
        chs = [base * m for m in mults]
        self.stem = nn.Conv3d(IN_CHANNELS, chs[0], 3, padding=1)

        self.downs = nn.ModuleList()
        self.pools = nn.ModuleList()
        for i in range(len(chs) - 1):
            self.downs.append(Block(chs[i], chs[i], tdim))
            self.pools.append(nn.Conv3d(chs[i], chs[i + 1], 3, stride=2, padding=1))
        self.mid = Block(chs[-1], chs[-1], tdim)
        self.ups = nn.ModuleList()
        self.unpools = nn.ModuleList()
        for i in reversed(range(len(chs) - 1)):
            self.unpools.append(nn.ConvTranspose3d(chs[i + 1], chs[i], 4, 2, 1))
            self.ups.append(Block(chs[i] * 2, chs[i], tdim))
        self.out_norm = nn.GroupNorm(8, chs[0])
        self.out = nn.Conv3d(chs[0], OUT_CHANNELS, 3, padding=1)

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        temb = self.tmlp(timestep_embedding(t, self.tdim))
        h = self.stem(x)
        skips = []
        for blk, pool in zip(self.downs, self.pools):
            h = blk(h, temb)
            skips.append(h)
            h = pool(h)
        h = self.mid(h, temb)
        for unpool, blk in zip(self.unpools, self.ups):
            h = unpool(h)
            h = blk(torch.cat([h, skips.pop()], dim=1), temb)
        return self.out(torch.nn.functional.silu(self.out_norm(h)))
