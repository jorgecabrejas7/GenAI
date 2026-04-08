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


def _default_cfg(use_tversky=False):
    return {
        "loss": {
            "xct_loss_type": "l1",
            "xct_weight": 1.0,
            "mask_bce_weight": 1.0,
            "mask_bce_pos_weight": 51.0,
            "mask_dice_weight": 1.0,
            "use_tversky": use_tversky,
            "tversky_alpha": 0.3,
            "tversky_beta": 0.7,
            "kl_free_bits": 0.25,
            "kl_warmup_steps": 5000,
            "kl_max_beta": 0.05,
        }
    }


class TestLossesSmoke:

    def test_total_loss_finite(self):
        output = _make_fake_output()
        batch = _make_fake_batch()
        result = compute_total_loss(output, batch, step=100, cfg=_default_cfg())

        assert "total" in result
        assert torch.isfinite(result["total"])
        assert result["total"].item() > 0

    def test_all_components_present(self):
        output = _make_fake_output()
        batch = _make_fake_batch()
        result = compute_total_loss(output, batch, step=100, cfg=_default_cfg())

        for key in ["total", "xct_loss", "mask_bce", "kl", "beta", "freebits_used"]:
            assert key in result, f"Missing key: {key}"

    def test_dice_key_present(self):
        output = _make_fake_output()
        batch = _make_fake_batch()
        result = compute_total_loss(output, batch, step=100, cfg=_default_cfg())
        assert "mask_dice" in result

    def test_tversky_mode(self):
        output = _make_fake_output()
        batch = _make_fake_batch()
        result = compute_total_loss(output, batch, step=100, cfg=_default_cfg(use_tversky=True))
        assert "mask_tversky" in result

    def test_beta_zero_at_step_zero(self):
        output = _make_fake_output()
        batch = _make_fake_batch()
        result = compute_total_loss(output, batch, step=0, cfg=_default_cfg())
        assert result["beta"] == 0.0

    def test_all_components_finite(self):
        output = _make_fake_output()
        batch = _make_fake_batch()
        result = compute_total_loss(output, batch, step=1000, cfg=_default_cfg())
        for k, v in result.items():
            if isinstance(v, torch.Tensor):
                assert torch.isfinite(v).all(), f"{k} contains non-finite values"
            else:
                assert v == v, f"{k} is NaN"  # NaN check for Python float
