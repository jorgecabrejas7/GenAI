"""The SliceGAN generator and critics.

Shapes follow the paper: a 4^3 latent becomes a 64^3 volume through four
transposed convolutions, each doubling the side. The kernel/stride/padding
triple is the paper's (4, 2, 2) rather than the usual (4, 2, 1): with s=2 and
k=4 the kernel overlaps each output position the same number of times, which is
what removes the checkerboard artefact a (4, 2, 1) stack leaves.
"""

from __future__ import annotations

import torch
from torch import nn

#: The paper's generator: five transposed convolutions, k=4, s=2, and the
#: padding schedule below. Kench & Cooper's anti-checkerboard rule is s < k,
#: k mod s == 0 and p >= k - s, so p = 2 is the SMALLEST legal padding for
#: (k=4, s=2) — p = 1 would double the side neatly and violate the rule, which
#: is the whole point of their Section 2.3.
GEN_KERNEL = 4
GEN_STRIDE = 2
GEN_PADDING = (2, 2, 2, 2, 3)

#: Because p > k - s, a layer maps n -> 2n - 2 (and the last, p=3, n -> 2n - 4),
#: so the generator is AFFINE in the latent side and not a clean multiple:
#:
#:     out = 32 * n - 64
#:
#: Verified against the paper's own figure: a 4-cell latent gives 64 voxels.
#: 8 gives 192 and 34 gives 1024, which are the shapes this campaign samples.
#: Assuming out = 16 * n — the obvious reading of "4^3 -> 64^3" — is wrong and
#: silently produces the wrong volume size; it produced 34^3 here before this
#: was derived.
LATENT_SCALE = 32
LATENT_OFFSET = 64
#: Output channels: grey, then the three label classes.
OUT_CHANNELS = 4
N_CLASSES = 3


def latent_for_shape(shape: tuple[int, int, int]) -> tuple[int, int, int]:
    """Latent grid that generates ``shape`` exactly.

    The generator is fully convolutional, so a larger latent gives a larger
    volume — the paper's route to volumes bigger than the training patch, and
    why a 1024-wide sample needs no retraining.

    ``n = (out + 64) / 32``, so a side must be a multiple of 32. A shape with no
    exact latent is refused rather than rounded: rounding would change the
    volume being reported and every shape-dependent metric with it.
    """
    n = []
    for side in shape:
        num = side + LATENT_OFFSET
        if side <= 0 or num % LATENT_SCALE:
            raise ValueError(
                f"SliceGAN cannot generate a side of {side} exactly: it needs "
                f"(side + {LATENT_OFFSET}) divisible by {LATENT_SCALE}, i.e. a "
                f"multiple of {LATENT_SCALE}. {shape} is not generatable and "
                "rounding it would misreport the volume.")
        n.append(num // LATENT_SCALE)
    return tuple(n)


def shape_for_latent(n: tuple[int, int, int]) -> tuple[int, int, int]:
    """The inverse, for tests and for logging what a latent will cost."""
    return tuple(LATENT_SCALE * int(v) - LATENT_OFFSET for v in n)


class Generator3D(nn.Module):
    """z (B, nz, d, h, w) -> (B, 4, 16d, 16h, 16w).

    Output heads are split: a tanh on the grey channel, mapped to [0, 1] by the
    caller, and a softmax over the three label logits. Keeping them in one
    tensor is what makes the critics see a PAIRED sample — a grey slice and its
    own label, not two independently plausible things.
    """

    def __init__(self, nz: int = 64, ngf: int = 64) -> None:
        super().__init__()
        ch = (nz, ngf * 8, ngf * 4, ngf * 2, ngf, ngf)
        layers: list[nn.Module] = []
        for i, pad in enumerate(GEN_PADDING):
            layers += [
                nn.ConvTranspose3d(ch[i], ch[i + 1], GEN_KERNEL, GEN_STRIDE,
                                   pad, bias=False),
                nn.BatchNorm3d(ch[i + 1]),
                nn.ReLU(inplace=True),
            ]
        self.body = nn.Sequential(*layers)
        # A final stride-1 convolution carries the heads. Putting them on the
        # last transposed conv instead would tie the grey and label resolutions
        # to its stride.
        self.head = nn.Conv3d(ngf, OUT_CHANNELS, 3, 1, 1)
        self.nz = nz

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        raw = self.head(self.body(z))
        grey = torch.tanh(raw[:, :1])
        label = torch.softmax(raw[:, 1:], dim=1)
        return torch.cat([grey, label], dim=1)

    @torch.no_grad()
    def sample_latent(self, shape: tuple[int, int, int], n: int = 1,
                      device=None, generator=None) -> torch.Tensor:
        d, h, w = latent_for_shape(shape)  # raises on a shape it cannot make
        return torch.randn(n, self.nz, d, h, w, device=device, generator=generator)


class Critic2D(nn.Module):
    """A 2-D WGAN critic on 64x64 slices of four channels.

    No batch norm: WGAN-GP penalises the gradient of the critic with respect to
    each sample individually, and batch norm makes that gradient depend on the
    rest of the batch. The paper and the WGAN-GP paper both drop it here.
    """

    def __init__(self, ndf: int = 64, in_channels: int = OUT_CHANNELS) -> None:
        super().__init__()
        ch = (in_channels, ndf, ndf * 2, ndf * 4, ndf * 8)
        layers: list[nn.Module] = []
        for i in range(4):
            layers += [nn.Conv2d(ch[i], ch[i + 1], 4, 2, 1, bias=False),
                       nn.LeakyReLU(0.2, inplace=True)]
        layers.append(nn.Conv2d(ch[4], 1, 4, 1, 0, bias=False))
        self.body = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.body(x).reshape(x.shape[0], -1).mean(dim=1)


def volume_to_slices(vol: torch.Tensor, axis: int) -> torch.Tensor:
    """(B, C, D, H, W) -> (B*n, C, a, b): every slice along ``axis``.

    This is the whole method in one function. A generated volume is judged only
    through its sections, so the generator never receives a 3-D gradient — each
    critic sees a stack of 2-D images and the volume is whatever satisfies all
    three at once.
    """
    if vol.ndim != 5:
        raise ValueError(f"expected (B, C, D, H, W), got {tuple(vol.shape)}")
    perm = {0: (0, 2, 1, 3, 4), 1: (0, 3, 1, 2, 4), 2: (0, 4, 1, 2, 3)}[axis]
    x = vol.permute(*perm).contiguous()
    b, n, c, a, bb = x.shape
    return x.reshape(b * n, c, a, bb)
