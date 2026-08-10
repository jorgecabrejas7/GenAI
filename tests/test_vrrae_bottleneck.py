"""Shape / gradient / numerical-stability tests for the VRRAE bottleneck.

Style follows tests/test_latent_backends.py (plain pytest functions, small
synthetic tensors, no GPU required).
"""

from __future__ import annotations

import torch
import pytest

from poregen.models.vae.v2.vrrae_bottleneck import VRRAEBottleneck
from poregen.losses.kl import kl_divergence_flat
from RR_layer.rr_layer import stable_SVD


# ---------------------------------------------------------------------------
# Shape / basic forward
# ---------------------------------------------------------------------------

def test_forward_shape_train_mode():
    torch.manual_seed(0)
    bn = VRRAEBottleneck(in_flat_dim=1024, vrrae_dim=64, rank=8)
    bn.train()
    h = torch.randn(16, 1024)
    mu, logvar, _basis = bn(h)
    assert mu.shape == (16, 8)
    assert logvar.shape == (16, 8)


def test_forward_dtype_preserved():
    torch.manual_seed(0)
    bn = VRRAEBottleneck(in_flat_dim=256, vrrae_dim=32, rank=4)
    h = torch.randn(16, 256, dtype=torch.float32)
    mu, logvar, _basis = bn(h)
    assert mu.dtype == torch.float32
    assert logvar.dtype == torch.float32


def test_logvar_head_is_rank_to_rank_and_consumes_coefficients():
    torch.manual_seed(7)
    bn = VRRAEBottleneck(in_flat_dim=32, vrrae_dim=16, rank=4)
    assert bn.logvar_head.in_features == 4
    assert bn.logvar_head.out_features == 4

    seen = []
    handle = bn.logvar_head.register_forward_pre_hook(
        lambda _module, args: seen.append(args[0].detach().clone())
    )
    mu, _logvar, _basis = bn(torch.randn(8, 32))
    handle.remove()

    assert len(seen) == 1
    assert torch.equal(seen[0], mu)


def test_identity_mean_equals_raw_svd_coefficients():
    torch.manual_seed(8)
    bn = VRRAEBottleneck(in_flat_dim=32, vrrae_dim=16, rank=4)
    h = torch.randn(8, 32)

    with torch.no_grad():
        y = bn.fc_in(h).float()
        _u, s, vh = torch.linalg.svd(y.T, full_matrices=False)
        expected = (s[:4, None] * vh[:4]).T
        mu, _logvar, _basis = bn(h)

    assert torch.allclose(mu.float(), expected, atol=1e-6, rtol=1e-5)


def test_stable_svd_gradient_matches_native_exactly():
    torch.manual_seed(9)
    source = torch.randn(12, 7, dtype=torch.float64)
    weights = [torch.randn_like(t) for t in torch.linalg.svd(source, full_matrices=False)]

    def gradient(svd_fn):
        x = source.clone().requires_grad_(True)
        factors = svd_fn(x)
        loss = sum((factor * weight).sum() for factor, weight in zip(factors, weights))
        return torch.autograd.grad(loss, x)[0]

    patched = gradient(stable_SVD)
    native = gradient(lambda x: torch.linalg.svd(x, full_matrices=False))
    relative_error = (patched - native).norm() / native.norm()

    assert relative_error.item() == 0.0


# ---------------------------------------------------------------------------
# Gradient flow
# ---------------------------------------------------------------------------

def test_gradients_reach_input_and_heads():
    torch.manual_seed(0)
    bn = VRRAEBottleneck(in_flat_dim=256, vrrae_dim=32, rank=4)
    h = torch.randn(16, 256, requires_grad=True)

    mu, logvar, _basis = bn(h)
    loss = mu.pow(2).sum() + logvar.pow(2).sum()
    loss.backward()

    assert h.grad is not None
    assert torch.isfinite(h.grad).all()
    assert bn.fc_in.weight.grad is not None
    assert torch.isfinite(bn.fc_in.weight.grad).all()
    assert bn.logvar_head.weight.grad is not None
    assert torch.isfinite(bn.logvar_head.weight.grad).all()


# ---------------------------------------------------------------------------
# Numerical stability
# ---------------------------------------------------------------------------

def test_no_nan_plain_float32():
    torch.manual_seed(1)
    bn = VRRAEBottleneck(in_flat_dim=512, vrrae_dim=64, rank=8)
    h = torch.randn(16, 512)
    mu, logvar, _basis = bn(h)
    assert torch.isfinite(mu).all()
    assert torch.isfinite(logvar).all()


def test_no_nan_under_simulated_bf16_autocast():
    """Bottleneck is called from inside a bf16 autocast context (as it will
    be in the real training loop on GB10) — SVD must still be numerically
    stable because the bottleneck forces float32 internally."""
    torch.manual_seed(2)
    bn = VRRAEBottleneck(in_flat_dim=512, vrrae_dim=64, rank=8)
    h = torch.randn(16, 512, requires_grad=True)

    with torch.autocast(device_type="cpu", dtype=torch.bfloat16, enabled=True):
        mu, logvar, _basis = bn(h)
        loss = mu.pow(2).sum() + logvar.pow(2).sum()

    loss.backward()

    assert torch.isfinite(mu).all()
    assert torch.isfinite(logvar).all()
    assert h.grad is not None
    assert torch.isfinite(h.grad).all()


# ---------------------------------------------------------------------------
# Small-batch rank-clipping — mu/logvar widths must always agree
# ---------------------------------------------------------------------------

def test_mu_logvar_widths_agree_on_clipped_batch():
    """When batch_size < rank, RRLayer silently clips the effective rank.
    logvar_head must be sliced to match, not just independently shrink."""
    torch.manual_seed(3)
    bn = VRRAEBottleneck(in_flat_dim=256, vrrae_dim=64, rank=32)
    h = torch.randn(8, 256)  # batch_size=8 < rank=32

    with pytest.warns(UserWarning, match="maximum achievable rank"):
        mu, logvar, _basis = bn(h)

    assert mu.shape == logvar.shape
    assert mu.shape[1] <= 32


# ---------------------------------------------------------------------------
# Train -> eval transition (fixed-basis auto-finalization, inherited from
# RRLayer itself)
# ---------------------------------------------------------------------------

def test_train_to_eval_does_not_freeze_an_early_basis():
    torch.manual_seed(4)
    bn = VRRAEBottleneck(in_flat_dim=256, vrrae_dim=32, rank=4, basis_history_size=5)
    bn.train()
    for _ in range(5):
        bn(torch.randn(16, 256))

    assert bn.rr.inference_basis is None

    # JAX validation continues to use a per-batch SVD. Merely switching the
    # PyTorch module to eval must not manufacture U_f from the rolling bank.
    bn.eval()
    assert bn.rr.inference_basis is None
    assert bn.rr.training
    assert not bn.basis_finalized

    mu, logvar, _basis = bn(torch.randn(16, 256))
    assert mu.shape == (16, 4)
    assert logvar.shape == (16, 4)


# ---------------------------------------------------------------------------
# kl_divergence_flat
# ---------------------------------------------------------------------------

def test_kl_divergence_flat_zero_when_matching_prior():
    mu = torch.zeros(4, 8)
    logvar = torch.zeros(4, 8)  # exp(0)=1 -> matches N(0,I) exactly
    kl, collapsed, per_ch = kl_divergence_flat(mu, logvar, free_bits=0.0)
    assert torch.allclose(kl, torch.tensor(0.0), atol=1e-6)
    assert per_ch.shape == (8,)


def test_kl_divergence_flat_positive_for_nontrivial_posterior():
    torch.manual_seed(5)
    mu = torch.randn(4, 8)
    logvar = torch.randn(4, 8) * 0.1
    kl, _, _ = kl_divergence_flat(mu, logvar, free_bits=0.0)
    assert kl.item() > 0.0
    assert torch.isfinite(kl)


def test_kl_divergence_flat_free_bits_clamping():
    mu = torch.zeros(4, 8)      # KL per-channel would be 0 without free-bits
    logvar = torch.zeros(4, 8)
    kl, collapsed, per_ch = kl_divergence_flat(mu, logvar, free_bits=0.5)
    # all 8 channels clamped up to 0.5 -> kl == 8 * 0.5
    assert torch.allclose(kl, torch.tensor(4.0), atol=1e-5)
    assert torch.allclose(collapsed, torch.tensor(1.0))


def test_basis_returned_maps_coeffs_back_to_encoder_space():
    """The returned basis must reconstruct the RR layer's own output.

    Guards the contract the decoder relies on: ``U @ alpha`` is what gets
    decoded (Fig. 1 of arXiv:2505.09458), so the basis handed back must be
    the one the coefficients are actually expressed in.
    """
    bn = VRRAEBottleneck(in_flat_dim=512, vrrae_dim=256, rank=32).train()
    h = torch.randn(64, 512)
    mu, _logvar, basis = bn(h)

    assert basis.shape[0] == 256, "basis rows must span vrrae_dim"
    assert basis.shape[1] == mu.shape[1], "basis cols must match coeff width"
    # orthonormal columns (it is a truncated SVD basis)
    gram = basis.T @ basis
    assert torch.allclose(gram, torch.eye(gram.shape[0]), atol=1e-4)


def test_decoder_input_is_batch_stable_unlike_raw_coeffs():
    """Same samples + different batch companions => stable ``U @ alpha``.

    Raw coefficients are NOT stable (the per-batch SVD re-derives the basis,
    with arbitrary sign and rotation within near-degenerate singular values).
    Decoding them through fixed weights was the original bug; this pins the
    property that made the fix necessary.
    """
    torch.manual_seed(0)
    bn = VRRAEBottleneck(in_flat_dim=128, vrrae_dim=64, rank=48).train()

    # genuinely low-rank data, so a stable subspace exists to be found
    base = torch.linalg.qr(torch.randn(128, 40))[0]
    make = lambda n: (base @ torch.randn(40, n)).T
    tracked = make(32)

    def encode(companions):
        mu, _lv, basis = bn(torch.cat([tracked, companions], 0))
        return mu[:32], (mu @ basis.transpose(0, 1))[:32]

    with torch.no_grad():
        mu1, y1 = encode(make(96))
        mu2, y2 = encode(make(96))

    rel = lambda a, b: ((a - b).norm() / a.norm()).item()
    assert rel(y1, y2) < 0.05, f"decoder input unstable across batches: {rel(y1, y2)}"
    assert rel(y1, y2) < rel(mu1, mu2), "U@alpha must be more stable than raw coeffs"
