"""Shape / gradient / registry / config tests for the v2.vrrae_linear
SVD-ablation variant."""

from __future__ import annotations

import torch

from poregen.models.vae import build_vae, list_vaes
from poregen.models.vae.base import VAEOutput
from poregen.losses.total import compute_total_loss
from poregen.configuration import resolve_experiment

# Small dims for fast tests — production defaults are much larger.
_KW = dict(
    in_channels=1, base_channels=8, n_blocks=2, patch_size=32,
    vrrae_dim=64, vrrae_rank=8,
)
_BATCH = 4  # no batch >= rank constraint in this variant — that's the point


def _build():
    torch.manual_seed(0)
    return build_vae("v2.vrrae_linear", **_KW)


def test_registered():
    assert "v2.vrrae_linear" in list_vaes()


def test_forward_shapes_and_no_mask_logits():
    model = _build()
    model.train()
    xct = torch.randn(_BATCH, 1, 32, 32, 32)
    out = model(xct, torch.zeros(_BATCH, 1, 32, 32, 32))
    assert isinstance(out, VAEOutput)
    assert out.xct_out.shape == (_BATCH, 1, 32, 32, 32)
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
    loss = out.xct_out.pow(2).mean() + out.mu.pow(2).mean()
    loss.backward()

    enc_grad = next(model.encoder.parameters()).grad
    dec_grad = next(model.decoder.parameters()).grad
    assert enc_grad is not None and torch.isfinite(enc_grad).all()
    assert dec_grad is not None and torch.isfinite(dec_grad).all()
    assert model.mu_head.weight.grad is not None
    assert model.expand.weight.grad is not None


def test_compute_total_loss_uses_flat_kl():
    """mu.ndim == 2 must route through kl_divergence_flat, same as v2.vrrae."""
    from poregen.losses.kl import kl_divergence_flat

    model = _build()
    model.train()
    xct = torch.randn(_BATCH, 1, 32, 32, 32)
    out = model(xct)

    cfg = {"loss": {"xct_loss_type": "charbonnier", "xct_weight": 1.0,
                    "kl_free_bits": 0.0, "kl_warmup_steps": 0,
                    "kl_max_beta": 1.0}}
    result = compute_total_loss(out, {"xct": xct, "mask": torch.zeros_like(xct)}, step=0, cfg=cfg)

    expected_kl, _, _ = kl_divergence_flat(out.mu, out.logvar, free_bits=0.0)
    assert torch.allclose(result["kl"], expected_kl)
    assert torch.isfinite(result["total"])
    result["total"].backward()


def test_no_finalize_hook():
    """The engine's duck-typed basis-finalization hook must not fire —
    this variant has no basis to finalize."""
    model = _build()
    assert getattr(model, "finalize_inference_basis", None) is None


def test_identity_paths_when_widths_match():
    """vrrae_dim == enc_flat_dim makes fc_in and dec_b identities (the
    vrrae03-style direct path), same rule as v2.vrrae."""
    kw = dict(_KW)
    # n_blocks=2, base_channels=8, patch_size=32 -> ch[1]=16 @ 8^3 = 8192
    kw["vrrae_dim"] = 8192
    model = build_vae("v2.vrrae_linear", **kw)
    assert model.enc_flat_dim == 8192
    assert isinstance(model.fc_in, torch.nn.Identity)
    assert isinstance(model.dec_b, torch.nn.Identity)


def test_experiment_config_resolves():
    resolved = resolve_experiment("vrrae04/base")
    cfg = resolved.cfg["model"]
    assert cfg["name"] == "v2.vrrae_linear"
    assert cfg["vrrae_dim"] == 2048
    assert cfg["vrrae_rank"] == 300
    # Ablation must inherit the SVD runs' training setup unchanged.
    assert resolved.cfg["data"]["batch_size"] == 768
    assert resolved.cfg["training"]["total_steps"] == 11948
    assert resolved.cfg["training"]["max_grad_norm"] == 1.0
