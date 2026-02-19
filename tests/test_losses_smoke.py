"""Smoke test: losses compute and return finite values."""

import torch
import pytest

from poregen.models.vae.base import VAEOutput
from poregen.losses.total import compute_total_loss


def _make_fake_output(batch_size=2, patch=64, z_ch=8, lat_s=16):
    return VAEOutput(
        xct_logits=torch.randn(batch_size, 1, patch, patch, patch),
        mask_logits=torch.randn(batch_size, 1, patch, patch, patch),
        mu=torch.randn(batch_size, z_ch, lat_s, lat_s, lat_s),
        logvar=torch.randn(batch_size, z_ch, lat_s, lat_s, lat_s) - 1.0,
        z=torch.randn(batch_size, z_ch, lat_s, lat_s, lat_s),
    )


def _make_fake_batch(batch_size=2, patch=64):
    return {
        "xct": torch.rand(batch_size, 1, patch, patch, patch),
        "mask": torch.randint(0, 2, (batch_size, 1, patch, patch, patch)).float(),
    }


class TestLossesSmoke:

    def test_total_loss_finite(self):
        output = _make_fake_output()
        batch = _make_fake_batch()
        result = compute_total_loss(output, batch, step=100)

        assert "total" in result
        assert torch.isfinite(result["total"])
        assert result["total"].item() > 0

    def test_all_components_present(self):
        output = _make_fake_output()
        batch = _make_fake_batch()
        result = compute_total_loss(output, batch, step=100)

        for key in ["total", "xct_loss", "mask_bce", "kl", "beta", "freebits_used"]:
            assert key in result, f"Missing key: {key}"

    def test_dice_key_present(self):
        output = _make_fake_output()
        batch = _make_fake_batch()
        result = compute_total_loss(output, batch, step=100)
        assert "mask_dice" in result

    def test_tversky_mode(self):
        output = _make_fake_output()
        batch = _make_fake_batch()
        result = compute_total_loss(output, batch, step=100, cfg={"use_tversky": True})
        assert "mask_tversky" in result

    def test_beta_zero_at_step_zero(self):
        output = _make_fake_output()
        batch = _make_fake_batch()
        result = compute_total_loss(output, batch, step=0)
        assert result["beta"] == 0.0

    def test_all_components_finite(self):
        output = _make_fake_output()
        batch = _make_fake_batch()
        result = compute_total_loss(output, batch, step=1000)
        for k, v in result.items():
            val = v.item() if isinstance(v, torch.Tensor) else v
            assert not (val != val), f"{k} is NaN"  # NaN check
