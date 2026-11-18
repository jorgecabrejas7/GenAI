"""A plain 3D U-Net for voxel-wise 3-class segmentation (material, pore, air).

Standalone segmenter for NEW grey scans (author 2026-11-18: "Train the 3
class unet segmenter"); not part of the VAE/LDM pipeline.  Following the
composite-CT literature (Galvez-Hernandez & Kratz 2023; Composites A 2026),
a small encoder-decoder with full-resolution skip connections: pore edges pass
straight through, which is what a 16^3 latent cannot keep.

``UNet3D(base=24, levels=3)``: 64^3 -> 32^3 -> 16^3 -> 8^3 bottleneck, about
2.9 M parameters.  GroupNorm, SiLU, no attention, no residuals.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def _block(cin: int, cout: int) -> nn.Sequential:
    g = max(1, cout // 8)
    return nn.Sequential(
        nn.Conv3d(cin, cout, 3, padding=1, bias=False), nn.GroupNorm(g, cout), nn.SiLU(inplace=True),
        nn.Conv3d(cout, cout, 3, padding=1, bias=False), nn.GroupNorm(g, cout), nn.SiLU(inplace=True),
    )


class UNet3D(nn.Module):
    def __init__(self, in_channels: int = 1, n_classes: int = 3, base: int = 24, levels: int = 3) -> None:
        super().__init__()
        chs = [base * 2 ** i for i in range(levels + 1)]
        self.enc = nn.ModuleList()
        cin = in_channels
        for c in chs[:-1]:
            self.enc.append(_block(cin, c)); cin = c
        self.bottleneck = _block(chs[-2], chs[-1])
        self.up = nn.ModuleList(); self.dec = nn.ModuleList()
        for i in range(levels - 1, -1, -1):
            self.up.append(nn.ConvTranspose3d(chs[i + 1], chs[i], 2, stride=2))
            self.dec.append(_block(2 * chs[i], chs[i]))
        self.head = nn.Conv3d(chs[0], n_classes, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        skips = []
        for blk in self.enc:
            x = blk(x); skips.append(x); x = F.max_pool3d(x, 2)
        x = self.bottleneck(x)
        for up, blk in zip(self.up, self.dec):
            x = blk(torch.cat([up(x), skips.pop()], dim=1))
        return self.head(x)


def load_unet3d(ckpt: str, device: torch.device) -> UNet3D:
    state = torch.load(ckpt, map_location="cpu", weights_only=False)
    m = UNet3D(**state["arch"])
    m.load_state_dict(state["model"], strict=True)
    return m.eval().to(device)
