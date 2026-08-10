"""Shape / gradient / registry / config tests for the v2.vrrae VAE variant."""

from __future__ import annotations

import torch
import pytest

from poregen.models.vae import build_vae, list_vaes
from poregen.models.vae.base import VAEConfig, VAEOutput
from poregen.losses.total import compute_total_loss
from poregen.configuration import resolve_experiment

# Small dims for fast tests — production defaults are much larger.
_KW = dict(
    in_channels=1, z_channels=4, base_channels=8, n_blocks=2, patch_size=32,
    vrrae_dim=32, vrrae_rank=8, vrrae_basis_history_size=5,
)
_BATCH = 16  # >= vrrae_rank, avoids the documented small-batch gotcha


def _build():
    torch.manual_seed(0)
    return build_vae("v2.vrrae", **_KW)


def test_registered():
    assert "v2.vrrae" in list_vaes()


def test_build_vae_constructs():
    model = _build()
    assert not hasattr(model, "mask_head")


def test_forward_shapes_and_no_mask_logits():
    model = _build()
    model.train()
    xct = torch.randn(_BATCH, 1, 32, 32, 32)
    out = model(xct, torch.zeros(_BATCH, 1, 32, 32, 32))
    assert isinstance(out, VAEOutput)
    assert out.xct_logits.shape == (_BATCH, 1, 32, 32, 32)
    assert out.mask_logits is None
    assert out.mu.shape == (_BATCH, _KW["vrrae_rank"])
    assert out.logvar.shape == (_BATCH, _KW["vrrae_rank"])
    assert out.z.shape == (_BATCH, _KW["vrrae_rank"])


def test_forward_without_mask_arg():
    """mask argument is optional/unused for this XCT-only-decoder variant."""
    model = _build()
    xct = torch.randn(_BATCH, 1, 32, 32, 32)
    out = model(xct)
    assert out.mask_logits is None


def test_gradients_reach_encoder_and_decoder():
    model = _build()
    model.train()
    xct = torch.randn(_BATCH, 1, 32, 32, 32)
    out = model(xct)
    loss = out.xct_logits.pow(2).mean() + out.mu.pow(2).mean()
    loss.backward()

    enc_grad = next(model.encoder.parameters()).grad
    dec_grad = next(model.decoder.parameters()).grad
    assert enc_grad is not None and torch.isfinite(enc_grad).all()
    assert dec_grad is not None and torch.isfinite(dec_grad).all()
    assert model.dec_b.weight.grad is not None
    assert model.bottleneck.fc_in.weight.grad is not None


def test_compute_total_loss_no_crash_and_no_mask_keys():
    model = _build()
    model.train()
    xct = torch.randn(_BATCH, 1, 32, 32, 32)
    mask = torch.randint(0, 2, (_BATCH, 1, 32, 32, 32)).float()
    out = model(xct, mask)

    cfg = {
        "loss": {
            "xct_loss_type": "charbonnier",
            "xct_weight": 1.0,
            "kl_free_bits": 0.25,
            "kl_warmup_steps": 0,
            "kl_max_beta": 0.05,
        }
    }
    result = compute_total_loss(out, {"xct": xct, "mask": mask}, step=10, cfg=cfg)

    assert "mask_bce" not in result
    assert "mask_dice" not in result
    assert "mask_tversky" not in result
    assert torch.isfinite(result["total"])
    result["total"].backward()


def test_compute_total_loss_uses_flat_kl():
    """mu.ndim == 2 must route through kl_divergence_flat (reduction dims
    differ from the spatial kl_divergence — sanity-check the KL value is
    consistent with the flat formula by comparing against a direct call."""
    from poregen.losses.kl import kl_divergence_flat

    model = _build()
    model.eval()  # rr will auto-finalize if bank has entries; here it's fresh so falls back
    # Use train mode to avoid requiring a finalized inference_basis on a fresh model.
    model.train()
    xct = torch.randn(_BATCH, 1, 32, 32, 32)
    out = model(xct)

    cfg = {"loss": {"xct_loss_type": "l1", "xct_weight": 1.0, "kl_free_bits": 0.0,
                     "kl_warmup_steps": 0, "kl_max_beta": 1.0}}
    result = compute_total_loss(out, {"xct": xct, "mask": torch.zeros_like(xct)}, step=0, cfg=cfg)

    expected_kl, _, _ = kl_divergence_flat(out.mu, out.logvar, free_bits=0.0)
    assert torch.allclose(result["kl"], expected_kl)


def test_vae_config_has_vrrae_fields():
    cfg = VAEConfig(vrrae_dim=128, vrrae_rank=16)
    assert cfg.vrrae_dim == 128
    assert cfg.vrrae_rank == 16
    assert cfg.vrrae_basis_history_size == 20  # default


def test_build_vae_rejects_unknown_kwarg_still_works():
    """Registry's strict-kwarg validation must still reject genuinely unknown
    keys (regression guard — adding vrrae_* fields shouldn't loosen this)."""
    with pytest.raises(TypeError):
        build_vae("v2.vrrae", not_a_real_field=123)


def test_experiment_config_resolves():
    resolved = resolve_experiment("vrrae/base")
    assert resolved.cfg["model"]["name"] == "v2.vrrae"
    assert "vrrae_dim" in resolved.cfg["model"]
    assert "vrrae_rank" in resolved.cfg["model"]

def test_vrrae03_uses_direct_encoder_and_decoder_paths():
    resolved = resolve_experiment("vrrae03/base")
    cfg = resolved.cfg["model"]
    model = build_vae(
        cfg["name"],
        in_channels=cfg["in_channels"],
        base_channels=cfg["base_channels"],
        n_blocks=cfg["n_blocks"],
        patch_size=cfg["patch_size"],
        vrrae_dim=cfg["vrrae_dim"],
        vrrae_rank=cfg["vrrae_rank"],
        vrrae_basis_history_size=cfg["vrrae_basis_history_size"],
    )
    assert model.enc_flat_dim == 65536
    assert isinstance(model.bottleneck.fc_in, torch.nn.Identity)
    assert isinstance(model.dec_b, torch.nn.Identity)


def test_other_variants_unaffected():
    """v2.conv_noattn (and friends) must still populate mask_logits and be
    otherwise unaffected by the VAEOutput.mask_logits default / new
    VAEConfig fields."""
    model = build_vae(
        "v2.conv_noattn", in_channels=1, z_channels=4, base_channels=8,
        n_blocks=2, patch_size=32,
    )
    xct = torch.randn(4, 1, 32, 32, 32)
    mask = torch.zeros(4, 1, 32, 32, 32)
    out = model(xct, mask)
    assert out.mask_logits is not None
    assert hasattr(model, "mask_head")
