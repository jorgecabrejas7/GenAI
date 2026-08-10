"""Tests for the fixed-basis extraction / checkpoint round-trip / decoder-only
fine-tune mechanism (poregen.models.vae.v2.vrrae_finetune)."""

from __future__ import annotations

import torch
import pytest

from poregen.models.vae import build_vae
from poregen.losses.total import compute_total_loss
from poregen.models.vae.v2.vrrae_bottleneck import VRRAEBottleneck
from poregen.models.vae.v2.vrrae_finetune import (
    refinalize_basis,
    finalize_basis_from_dataloader,
    load_vrrae_state_dict,
    finetune_decoder_on_fixed_basis,
    FinetuneConfig,
)

_KW = dict(
    in_channels=1, z_channels=4, base_channels=8, n_blocks=2, patch_size=32,
    vrrae_dim=32, vrrae_rank=8, vrrae_basis_history_size=5,
)
_BATCH = 16

_LOSS_CFG = {
    "loss": {
        "xct_loss_type": "l1",
        "xct_weight": 1.0,
        "kl_free_bits": 0.0,
        "kl_warmup_steps": 0,
        "kl_max_beta": 0.01,
    }
}


def _loss_fn(output, batch, step):
    return compute_total_loss(output, batch, step, cfg=_LOSS_CFG)


def _make_batch(batch_size=_BATCH):
    return {
        "xct": torch.randn(batch_size, 1, 32, 32, 32),
        "mask": torch.zeros(batch_size, 1, 32, 32, 32),
    }


def _train_a_bit(model, n_steps=6):
    model.train()
    for _ in range(n_steps):
        out = model(*_make_batch().values())


# ---------------------------------------------------------------------------
# refinalize_basis
# ---------------------------------------------------------------------------

def test_refinalize_basis_is_explicit_not_triggered_by_eval():
    torch.manual_seed(0)
    model = build_vae("v2.vrrae", **_KW)

    _train_a_bit(model, n_steps=3)
    model.eval()
    assert model.bottleneck.rr.inference_basis is None
    assert not model.bottleneck.basis_finalized

    n = refinalize_basis(model)
    assert n == 1
    basis = model.bottleneck.rr.inference_basis
    assert basis is not None
    assert basis.shape == (_KW["vrrae_dim"], _KW["vrrae_rank"])
    assert model.bottleneck.basis_finalized



def test_full_dataset_finalization_matches_concatenated_basis_svd():
    torch.manual_seed(10)
    model = build_vae("v2.vrrae", **_KW)

    class Dataset(torch.utils.data.Dataset):
        def __init__(self):
            self.xct = torch.randn(18, 1, 32, 32, 32)

        def __len__(self):
            return len(self.xct)

        def __getitem__(self, idx):
            return {"xct": self.xct[idx], "mask": torch.zeros_like(self.xct[idx])}

    dataset = Dataset()
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=8,
        shuffle=False,
        drop_last=True,
    )

    model.eval()
    batch_bases = []
    with torch.no_grad():
        for start in range(0, len(dataset), 8):
            xct = dataset.xct[start : start + 8]
            y = model.bottleneck.fc_in(model.encoder(xct).flatten(1)).float()
            u, _, _ = torch.linalg.svd(y.T, full_matrices=False)
            batch_bases.append(u[:, : min(_KW["vrrae_rank"], u.shape[1])])
        expected, _, _ = torch.linalg.svd(
            torch.cat(batch_bases, dim=1),
            full_matrices=False,
        )
        expected = expected[:, : _KW["vrrae_rank"]]

    stats = finalize_basis_from_dataloader(
        model,
        loader,
        device=torch.device("cpu"),
        autocast_dtype=torch.float32,
    )

    actual = model.bottleneck.rr.inference_basis
    assert stats == {"n_batches": 3, "n_samples": 18}
    assert model.bottleneck.basis_finalized
    assert not model.bottleneck.rr.training
    assert torch.allclose(
        actual @ actual.T,
        expected @ expected.T,
        atol=2e-4,
        rtol=2e-4,
    )

    # A finalized U_f gives the same projection to a tracked sample regardless
    # of its batch companions.
    tracked = dataset.xct[:1]
    with torch.no_grad():
        mu1 = model(torch.cat([tracked, dataset.xct[1:8]])).mu[0]
        mu2 = model(torch.cat([tracked, dataset.xct[8:15]])).mu[0]
    assert torch.allclose(mu1, mu2, atol=1e-5, rtol=1e-5)

    # Resuming optimization invalidates U_f; the next eval returns to dynamic
    # per-batch SVD until finalization is explicitly run again.
    model.train()
    assert not model.bottleneck.basis_finalized
    assert model.bottleneck.rr.inference_basis is None
    model.eval()
    assert model.bottleneck.rr.training

def test_full_dataset_finalization_sketched_path(monkeypatch):
    """The sketched (large-dim) path recovers the top-rank eigenspace of the
    exact accumulated covariance: the sketch basis must capture ~all of the
    eigenvalue mass the exact top-rank eigenvectors capture."""
    import poregen.models.vae.v2.vrrae_finetune as vf

    torch.manual_seed(11)
    model = build_vae("v2.vrrae", **_KW)

    class Dataset(torch.utils.data.Dataset):
        def __init__(self):
            self.xct = torch.randn(24, 1, 32, 32, 32)

        def __len__(self):
            return len(self.xct)

        def __getitem__(self, idx):
            return {"xct": self.xct[idx], "mask": torch.zeros_like(self.xct[idx])}

    loader = torch.utils.data.DataLoader(
        Dataset(), batch_size=8, shuffle=False, drop_last=True,
    )

    # Exact covariance from the same batches (vrrae_dim=32 is tiny here).
    model.eval()
    covariance = torch.zeros(_KW["vrrae_dim"], _KW["vrrae_dim"])
    with torch.no_grad():
        for start in range(0, 24, 8):
            y = model.bottleneck.fc_in(
                model.encoder(loader.dataset.xct[start : start + 8]).flatten(1)
            ).float()
            u, _, _ = torch.linalg.svd(y.T, full_matrices=False)
            b = u[:, : _KW["vrrae_rank"]]
            covariance += b @ b.T
    eigvals = torch.linalg.eigvalsh(covariance)
    exact_topk_mass = eigvals[-_KW["vrrae_rank"] :].sum()

    # Force the sketched branch despite the tiny dim.
    monkeypatch.setattr(vf, "_EXACT_COVARIANCE_DIM_LIMIT", 0)
    stats = vf.finalize_basis_from_dataloader(
        model, loader, device=torch.device("cpu"), autocast_dtype=torch.float32,
    )
    assert stats == {"n_batches": 3, "n_samples": 24}

    basis = model.bottleneck.rr.inference_basis
    assert basis.shape == (_KW["vrrae_dim"], _KW["vrrae_rank"])
    captured_mass = torch.trace(basis.T @ covariance @ basis)
    assert captured_mass >= 0.999 * exact_topk_mass


def test_refinalize_basis_raises_if_never_trained():
    model = build_vae("v2.vrrae", **_KW)
    with pytest.raises(RuntimeError):
        refinalize_basis(model)


# ---------------------------------------------------------------------------
# checkpoint round-trip workaround
# ---------------------------------------------------------------------------

def test_load_vrrae_state_dict_round_trips_inference_basis():
    torch.manual_seed(1)
    model = build_vae("v2.vrrae", **_KW)
    _train_a_bit(model, n_steps=8)
    refinalize_basis(model)
    sd = model.state_dict()

    fresh = build_vae("v2.vrrae", **_KW)
    assert fresh.bottleneck.rr.inference_basis is None

    # Plain load_state_dict must fail (documented library bug).
    with pytest.raises(RuntimeError, match="inference_basis"):
        fresh.load_state_dict(sd)

    # Workaround must succeed and reproduce identical eval-mode output.
    fresh2 = build_vae("v2.vrrae", **_KW)
    load_vrrae_state_dict(fresh2, sd)
    assert fresh2.bottleneck.rr.inference_basis is not None
    assert torch.equal(fresh2.bottleneck.rr.inference_basis, model.bottleneck.rr.inference_basis)

    model.eval()
    fresh2.eval()
    xct = torch.randn(_BATCH, 1, 32, 32, 32)
    out1 = model(xct)
    out2 = fresh2(xct)
    assert torch.allclose(out1.mu, out2.mu, atol=1e-5)
    # A pre-alignment checkpoint can contain an early inference_basis but no
    # explicit finalization marker. It must load, but evaluation must ignore
    # that stale basis and continue with per-batch SVDs.
    legacy_sd = {
        key: value
        for key, value in sd.items()
        if not key.endswith("_basis_finalized")
    }
    fresh3 = build_vae("v2.vrrae", **_KW)
    load_vrrae_state_dict(fresh3, legacy_sd)
    assert not fresh3.bottleneck.basis_finalized
    fresh3.eval()
    assert fresh3.bottleneck.rr.training



# ---------------------------------------------------------------------------
# decoder-only fine-tune
# ---------------------------------------------------------------------------

def test_finetune_decoder_freezes_encoder_and_bottleneck():
    torch.manual_seed(2)
    model = build_vae("v2.vrrae", **_KW)
    _train_a_bit(model, n_steps=8)
    refinalize_basis(model)

    enc_before = {k: v.clone() for k, v in model.encoder.state_dict().items()}
    fcin_before = model.bottleneck.fc_in.weight.clone()
    dec_before = next(model.decoder.parameters()).clone()

    data_iter = iter(_make_batch() for _ in range(100))
    losses = finetune_decoder_on_fixed_basis(
        model, data_iter, _loss_fn, steps=10, lr=1e-2, device=torch.device("cpu"),
    )

    assert len(losses) == 10
    assert all(l == l for l in losses)  # no NaN
    assert all(l < 1e6 for l in losses)  # no blow-up

    for k, v in model.encoder.state_dict().items():
        assert torch.equal(v, enc_before[k]), f"encoder param {k} changed during decoder-only fine-tune"
    assert torch.equal(model.bottleneck.fc_in.weight, fcin_before)
    assert not torch.equal(next(model.decoder.parameters()), dec_before), "decoder params did not change"


def test_finetune_requires_finalized_basis():
    """A model that never ran a single training step has an empty basis
    bank — RRLayer.eval()'s own auto-finalize can't run either (nothing
    banked), so this must fail loudly rather than silently fine-tune
    against an undefined/absent basis.

    (Note: if even a FEW training steps have run, RRLayer's eval()
    transition auto-finalizes from whatever's banked — see
    test_refinalize_basis_overrides_early_auto_finalization — so this test
    specifically covers the zero-steps-ever-trained case.)"""
    model = build_vae("v2.vrrae", **_KW)
    data_iter = iter(_make_batch() for _ in range(10))
    with pytest.raises(RuntimeError):
        finetune_decoder_on_fixed_basis(model, data_iter, _loss_fn, steps=2)


def test_finetune_config_defaults():
    cfg = FinetuneConfig()
    assert cfg.enabled is False
    assert cfg.steps == 50
    assert cfg.lr == 1.0e-4
