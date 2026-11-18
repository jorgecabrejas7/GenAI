import torch

from poregen.models.unet3d import UNet3D


def test_unet3d_shapes_and_size():
    m = UNet3D(base=24, levels=3)
    out = m(torch.rand(2, 1, 64, 64, 64))
    assert out.shape == (2, 3, 64, 64, 64)
    n = sum(p.numel() for p in m.parameters())
    assert 1e6 < n < 6e6, n


def test_unet3d_any_multiple_of_eight():
    m = UNet3D(base=8, levels=3)
    assert m(torch.rand(1, 1, 32, 48, 64)).shape == (1, 3, 32, 48, 64)
