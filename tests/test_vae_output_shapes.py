"""Smoke test: instantiate VAEs and verify output shapes."""

import pytest
import torch

from poregen.models.vae import VAEConfig, VAEOutput, build_vae


BATCH = 2
PATCH = 64


@pytest.fixture(params=["conv", "unet"])
def model(request):
    return build_vae(request.param)


class TestVAEOutputShapes:

    def test_output_type(self, model):
        xct = torch.randn(BATCH, 1, PATCH, PATCH, PATCH)
        mask = torch.randn(BATCH, 1, PATCH, PATCH, PATCH)
        out = model(xct, mask)
        assert isinstance(out, VAEOutput)

    def test_xct_logits_shape(self, model):
        xct = torch.randn(BATCH, 1, PATCH, PATCH, PATCH)
        mask = torch.randn(BATCH, 1, PATCH, PATCH, PATCH)
        out = model(xct, mask)
        assert out.xct_logits.shape == (BATCH, 1, PATCH, PATCH, PATCH)

    def test_mask_logits_shape(self, model):
        xct = torch.randn(BATCH, 1, PATCH, PATCH, PATCH)
        mask = torch.randn(BATCH, 1, PATCH, PATCH, PATCH)
        out = model(xct, mask)
        assert out.mask_logits.shape == (BATCH, 1, PATCH, PATCH, PATCH)

    def test_latent_shapes(self, model):
        cfg = model.cfg
        xct = torch.randn(BATCH, 1, PATCH, PATCH, PATCH)
        mask = torch.randn(BATCH, 1, PATCH, PATCH, PATCH)
        out = model(xct, mask)

        lat_spatial = cfg.latent_spatial  # 64 / 4 = 16
        assert out.mu.shape == (BATCH, cfg.z_channels, lat_spatial, lat_spatial, lat_spatial)
        assert out.logvar.shape == out.mu.shape
        assert out.z.shape == out.mu.shape

    def test_registry_build(self):
        m1 = build_vae("conv")
        m2 = build_vae("unet")
        assert type(m1).__name__ == "ConvVAE3D"
        assert type(m2).__name__ == "UNetVAE3D"

    def test_registry_unknown_raises(self):
        with pytest.raises(KeyError, match="Unknown VAE"):
            build_vae("nonexistent")

    def test_registry_bad_override_raises(self):
        with pytest.raises(TypeError, match="Invalid VAEConfig"):
            build_vae("conv", fake_param=123)
