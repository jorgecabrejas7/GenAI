import torch

from poregen.models.vae.base import VAEConfig
from poregen.models.vae.v2.grey_encoder import GreyEncoder


def test_grey_encoder_shape_matches_production_latent():
    cfg = VAEConfig(in_channels=1, z_channels=4, base_channels=32, n_blocks=2, patch_size=64)
    m = GreyEncoder(cfg)
    mu = m(torch.rand(2, 1, 64, 64, 64))
    assert mu.shape == (2, 4, 16, 16, 16)


def test_grey_encoder_rejects_mask_channels():
    try:
        GreyEncoder(VAEConfig(in_channels=3, z_channels=4))
    except ValueError:
        return
    raise AssertionError("in_channels=3 must be rejected")
