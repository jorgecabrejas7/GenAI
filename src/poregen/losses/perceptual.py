"""LPIPS perceptual loss on 2D slices of the grey (XCT) volume.

The Rombach-aligned first stage (vault D68, ``r08/rombach-rf16``) adds
``perceptual_weight * LPIPS(x, x_rec)`` to the per-voxel reconstruction term,
as the LDM autoencoder's ``LPIPSWithDiscriminator`` does
(``rec = |x - x_rec| + p_weight * LPIPS``). LPIPS is a 2D image metric, so it
scores slices: ``n_per_axis`` random slices along each of the three axes per
patch, the SAME positions in the reconstruction and the input.

The LPIPS network (``lpips`` 0.1.4, ``net='vgg'``) is frozen and held in
``eval()``. It is a separate module, not a child of the VAE, so it is never in
the optimiser and never in a checkpoint; ``resolved_config.yaml`` records that
it was used through ``loss.perceptual_*``.

Constructing :class:`SlicePerceptualLoss` loads torchvision's ImageNet VGG16
weights, which torchvision downloads to ``~/.cache/torch/hub/checkpoints`` on
first use (``vgg16-397923af.pth``). The LPIPS linear-layer weights ship inside
the ``lpips`` package.
"""

from __future__ import annotations

import torch
import torch.nn as nn


def paired_axis_slices(
    a: torch.Tensor,
    b: torch.Tensor,
    n_per_axis: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Take the same random 2D slices from two cubic volumes.

    Parameters
    ----------
    a, b : (B, 1, D, D, D)
        Two volumes of the same shape (reconstruction and input).
    n_per_axis : int
        Slices per axis per patch. Positions are drawn independently per
        patch and per axis, with replacement.

    Returns
    -------
    (3 * n_per_axis * B, 1, D, D) slices of ``a`` and of ``b``, in the same
    order: all axis-0 (depth) slices, then axis-1 (height), then axis-2
    (width).
    """
    if a.shape != b.shape:
        raise ValueError(f"paired slices need equal shapes, got {tuple(a.shape)} and {tuple(b.shape)}")
    B, C, D, H, W = a.shape
    if not (D == H == W):
        raise ValueError(f"paired slices need a cubic patch, got {D}x{H}x{W}")
    device = a.device
    bidx = torch.arange(B, device=device).repeat_interleave(n_per_axis)   # (B*n,)

    out_a: list[torch.Tensor] = []
    out_b: list[torch.Tensor] = []
    # Move the sliced axis to dim 1, then index (batch, position) pairs.
    for perm in ((0, 2, 1, 3, 4), (0, 3, 1, 2, 4), (0, 4, 1, 2, 3)):
        pos = torch.randint(0, D, (B * n_per_axis,), device=device)
        out_a.append(a.permute(*perm)[bidx, pos])                        # (B*n, C, D, D)
        out_b.append(b.permute(*perm)[bidx, pos])
    return torch.cat(out_a, dim=0), torch.cat(out_b, dim=0)


def grey_to_lpips_input(x: torch.Tensor) -> torch.Tensor:
    """Grey in ``[0, 1]`` (``xct / 255``) → 3-channel image in ``[-1, 1]``."""
    return (2.0 * x - 1.0).expand(-1, 3, -1, -1)


class SlicePerceptualLoss(nn.Module):
    """Mean LPIPS over paired random slices of reconstruction and input.

    Returns a scalar: LPIPS averaged over every slice of every patch in the
    batch, i.e. a per-patch mean averaged over the batch.
    """

    def __init__(self, n_slices_per_axis: int = 4, net: str = "vgg") -> None:
        super().__init__()
        import lpips  # optional dependency: only the perceptual recipe needs it

        self.n_slices_per_axis = int(n_slices_per_axis)
        self.lpips = lpips.LPIPS(net=net, version="0.1", verbose=False)
        self.lpips.requires_grad_(False)
        self.lpips.eval()

    def train(self, mode: bool = True) -> "SlicePerceptualLoss":
        # Frozen in both senses: no gradient to its weights, and always eval.
        return super().train(False)

    def forward(self, recon: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        r, t = paired_axis_slices(recon, target, self.n_slices_per_axis)
        return self.lpips(grey_to_lpips_input(r), grey_to_lpips_input(t.detach())).mean()


def build_perceptual(cfg: dict, device: torch.device) -> SlicePerceptualLoss | None:
    """The configured perceptual module on *device*, or ``None`` when off.

    Off unless ``loss.perceptual_weight > 0``.
    """
    c = cfg["loss"]
    if float(c.get("perceptual_weight", 0.0)) <= 0.0:
        return None
    net = c.get("perceptual_net", "vgg")
    return SlicePerceptualLoss(
        n_slices_per_axis=int(c.get("perceptual_slices_per_axis", 4)), net=net,
    ).to(device)
