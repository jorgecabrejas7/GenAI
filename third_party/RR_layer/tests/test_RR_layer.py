import pytest
import torch

from RR_layer import RRLayer


# ============================================================
# Constructor
# ============================================================

def test_invalid_rank():
    with pytest.raises(ValueError):
        RRLayer(rank=0)


def test_invalid_basis_history_size():
    with pytest.raises(ValueError):
        RRLayer(
            rank=4,
            basis_history_size=0,
        )


# ============================================================
# Input validation
# ============================================================

def test_input_must_be_tensor():
    rr = RRLayer(rank=4)

    with pytest.raises(TypeError):
        rr([1, 2, 3])


def test_input_must_have_batch_dimension():
    rr = RRLayer(rank=4)

    x = torch.randn(10)

    with pytest.raises(ValueError):
        rr(x)


def test_empty_batch():
    rr = RRLayer(rank=4)

    x = torch.randn(0, 10)

    with pytest.raises(ValueError):
        rr(x)


def test_nan_input():
    rr = RRLayer(rank=4)

    x = torch.randn(8, 16)
    x[0, 0] = torch.nan

    with pytest.raises(ValueError):
        rr(x)


def test_invalid_basis_ndim():
    rr = RRLayer(rank=4)

    x = torch.randn(8, 16)

    basis = torch.randn(16)

    with pytest.raises(ValueError):
        rr(x, basis=basis)


def test_invalid_basis_rows():
    rr = RRLayer(rank=4)

    x = torch.randn(8, 16)

    basis = torch.randn(17, 4)

    with pytest.raises(ValueError):
        rr(x, basis=basis)


def test_empty_basis_columns():
    rr = RRLayer(rank=4)

    x = torch.randn(8, 16)

    basis = torch.empty(16, 0)

    with pytest.raises(ValueError):
        rr(x, basis=basis)


# ============================================================
# Forward pass
# ============================================================

def test_output_shape_preserved():
    rr = RRLayer(rank=4)

    x = torch.randn(8, 32)

    y = rr(x)

    assert y.shape == x.shape


def test_output_shape_preserved_3d():
    rr = RRLayer(rank=4)

    x = torch.randn(16, 3, 32)

    y = rr(x)

    assert y.shape == x.shape


def test_return_factors_shapes():
    rr = RRLayer(rank=4)

    x = torch.randn(8, 16)

    y, basis, coeffs = rr(
        x,
        return_factors=True,
    )

    assert y.shape == x.shape

    assert basis.shape == (16, 4)

    assert coeffs.shape == (4, 8)


def test_factorization_reconstructs_output():
    rr = RRLayer(rank=4)

    x = torch.randn(8, 16)

    y, basis, coeffs = rr(
        x,
        return_factors=True,
    )

    reconstructed = basis @ coeffs

    reconstructed = reconstructed.reshape(
        16,
        8,
    )

    reconstructed = reconstructed.T

    assert torch.allclose(
        y,
        reconstructed,
        atol=1e-5,
    )


# ============================================================
# Explicit basis projection
# ============================================================

def test_projection_branch_matches_formula():
    rr = RRLayer(rank=4)

    x = torch.randn(8, 16)

    basis = torch.randn(16, 4)
    basis, _ = torch.linalg.qr(basis)

    y = rr(
        x,
        basis=basis,
    )

    X = x.T

    expected = basis @ (basis.T @ X)

    expected = expected.T

    assert torch.allclose(
        y,
        expected,
        atol=1e-5,
    )


# ============================================================
# Basis collection
# ============================================================

def test_basis_collection_occurs_during_training():
    rr = RRLayer(
        rank=4,
        basis_history_size=10,
    )

    rr.train()

    for _ in range(3):
        rr(torch.randn(8, 16))

    assert len(rr._basis_bank) == 3


def test_basis_history_size_respected():
    rr = RRLayer(
        rank=4,
        basis_history_size=5,
    )

    rr.train()

    for _ in range(20):
        rr(torch.randn(8, 16))

    assert len(rr._basis_bank) == 5


# ============================================================
# Finalization
# ============================================================

def test_finalize_basis_creates_inference_basis():
    rr = RRLayer(rank=4)

    rr.train()

    for _ in range(5):
        rr(torch.randn(8, 16))

    rr.finalize_basis()

    assert rr.inference_basis is not None

    assert rr.inference_basis.shape == (
        16,
        4,
    )


def test_finalize_without_bases_raises():
    rr = RRLayer(rank=4)

    with pytest.raises(RuntimeError):
        rr.finalize_basis()


# ============================================================
# Eval mode
# ============================================================

def test_eval_auto_finalizes_basis():
    rr = RRLayer(rank=4)

    rr.train()

    for _ in range(5):
        rr(torch.randn(8, 16))

    rr.eval()

    assert rr.inference_basis is not None


def test_eval_uses_inference_basis():
    rr = RRLayer(rank=4)

    rr.train()

    for _ in range(5):
        rr(torch.randn(8, 16))

    rr.eval()

    x = torch.randn(8, 16)

    y1 = rr(x)

    basis = rr.inference_basis

    y2 = rr(
        x,
        basis=basis,
    )

    assert torch.allclose(
        y1,
        y2,
        atol=1e-5,
    )


# ============================================================
# Gradients
# ============================================================

def test_backward_pass():
    rr = RRLayer(rank=4)

    x = torch.randn(
        8,
        16,
        requires_grad=True,
    )

    y = rr(x)

    loss = y.pow(2).mean()

    loss.backward()

    assert x.grad is not None

    assert torch.isfinite(x.grad).all()

# ============================================================
# Train -> Eval transition
# ============================================================

def test_eval_uses_inference_basis_after_finalize():
    rr = RRLayer(rank=4)

    rr.train()

    for _ in range(10):
        rr(torch.randn(8, 16))

    assert rr.inference_basis is None

    rr.eval()

    assert rr.inference_basis is not None

    x = torch.randn(8, 16)

    y_eval = rr(x)

    y_manual = rr(
        x,
        basis=rr.inference_basis,
    )

    assert torch.allclose(
        y_eval,
        y_manual,
        atol=1e-4,
        rtol=1e-4,
    )
    