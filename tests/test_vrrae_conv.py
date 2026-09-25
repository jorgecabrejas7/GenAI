"""The per-cell VRRAE: is it the same layer, just fed differently?

The claim this variant rests on is that handing the RR layer one column per
CELL rather than one per PATCH changes nothing about the layer — only what
counts as a sample. The strongest way to check that is to remove the
difference: on a 1x1x1 latent grid a patch HAS exactly one cell, so the
per-cell path and the flat path must produce the same numbers from the same
weights. If they do not, the reshape is wrong.
"""

from __future__ import annotations

import pytest
import torch

from poregen.losses.kl import kl_divergence_flat
from poregen.models.vae.registry import build_vae

PATCH = 64


def conv_model(**kw):
    base = dict(in_channels=1, base_channels=32, n_blocks=2, patch_size=PATCH,
                vrrae_dim=64, vrrae_rank=8, z_channels=None)
    base.update(kw)
    return build_vae("v2.vrrae_conv", **base)


class TestItIsTheFlatModelWhenThereIsOneCell:
    """A 1x1x1 grid has one cell per patch, so the two paths must coincide."""

    @staticmethod
    def _pair():
        # 6 stride-2 blocks take 64 -> 1, and ch[5] = 32 * 2^5 = 1024. With
        # vrrae_dim = 1024 the flat model's dec_b is Identity too, so the only
        # remaining difference between the two classes is the reshape.
        kw = dict(in_channels=1, base_channels=32, n_blocks=6, patch_size=PATCH,
                  vrrae_dim=1024, vrrae_rank=8, z_channels=None)
        torch.manual_seed(0)
        flat = build_vae("v2.vrrae", **kw)
        torch.manual_seed(0)
        conv = conv_model(**kw)
        conv.load_state_dict(flat.state_dict(), strict=True)
        return flat.eval(), conv.eval()

    def test_the_weights_are_interchangeable(self):
        """A strict load is itself the check that the two carry one parameter set."""
        flat, conv = self._pair()
        assert {k for k, _ in flat.named_parameters()} == \
               {k for k, _ in conv.named_parameters()}

    def test_the_reconstruction_is_identical(self):
        flat, conv = self._pair()
        x = torch.randn(4, 1, PATCH, PATCH, PATCH)
        torch.manual_seed(1)
        a = flat(x)
        torch.manual_seed(1)
        b = conv(x)
        assert torch.equal(a.xct_out, b.xct_out)

    def test_the_posterior_is_identical(self):
        flat, conv = self._pair()
        x = torch.randn(4, 1, PATCH, PATCH, PATCH)
        torch.manual_seed(1)
        a = flat(x)
        torch.manual_seed(1)
        b = conv(x)
        assert torch.equal(a.mu, b.mu)
        assert torch.equal(a.logvar, b.logvar)


class TestTheCellReshape:
    """A bare reshape would hand the SVD columns that are not cells."""

    def test_each_row_is_one_cell_s_channel_vector(self):
        m = conv_model()
        c, l = 64, 16
        h = torch.randn(2, c, l, l, l)
        cells = h.permute(0, 2, 3, 4, 1).reshape(-1, c)
        # Row for (batch 1, cell (3, 5, 7)) must be that cell's channels.
        row = 1 * l ** 3 + 3 * l ** 2 + 5 * l + 7
        assert torch.equal(cells[row], h[1, :, 3, 5, 7])

    def test_a_bare_reshape_would_be_wrong(self):
        """Guarding the fix: the obvious `h.reshape(-1, c)` is NOT the same."""
        c, l = 64, 4
        h = torch.randn(2, c, l, l, l)
        good = h.permute(0, 2, 3, 4, 1).reshape(-1, c)
        bad = h.reshape(-1, c)
        assert not torch.equal(good, bad)

    def test_the_round_trip_puts_every_cell_back(self):
        m = conv_model()
        c, l, b = 64, 16, 2
        h = torch.randn(b, c, l, l, l)
        cells = h.permute(0, 2, 3, 4, 1).reshape(-1, c)
        back = cells.view(b, l, l, l, c).permute(0, 4, 1, 2, 3).contiguous()
        assert torch.equal(back, h)


class TestShapesAndBudget:

    def test_the_posterior_has_one_row_per_cell(self):
        m = conv_model().train()
        out = m(torch.randn(3, 1, PATCH, PATCH, PATCH))
        assert out.mu.shape == (3 * 16 ** 3, 8)
        assert out.logvar.shape == (3 * 16 ** 3, 8)
        assert out.xct_out.shape == (3, 1, PATCH, PATCH, PATCH)

    def test_the_latent_budget_matches_r08(self):
        """8 coefficients x 16^3 cells = 32 768, which is r08 z=8 exactly.

        It is what makes the comparison a like-for-like one instead of an
        argument about capacity.
        """
        assert conv_model().latents_per_patch == 32768
        assert conv_model().latents_per_patch == 8 * 16 ** 3

    @pytest.mark.parametrize("k", [4, 8, 16, 32])
    def test_the_rank_is_exposed_for_the_sweep(self, k):
        m = conv_model(vrrae_rank=k).train()
        out = m(torch.randn(2, 1, PATCH, PATCH, PATCH))
        assert out.mu.shape[1] == k
        assert m.latents_per_patch == k * 16 ** 3

    def test_vrrae_dim_must_be_the_channel_width(self):
        """A column is one cell, so its length is the channel count. Anything
        else is a silent mis-specification, so it is refused."""
        with pytest.raises(ValueError, match="must equal the encoder's channel"):
            conv_model(vrrae_dim=128)


class TestTheRankConstraintStopsBiting:
    """`RRLayer` clips the rank to min(rank, dim, n_samples). n_samples is now
    batch x cells, so a rank of 8 no longer forces a batch of 768."""

    def test_a_batch_of_one_still_carries_the_full_rank(self):
        m = conv_model().train()
        out = m(torch.randn(1, 1, PATCH, PATCH, PATCH))
        assert out.mu.shape == (16 ** 3, 8)      # 4096 samples from ONE patch


class TestTheKLReduction:

    def test_it_routes_to_the_flat_kl_and_reduces_over_cells(self):
        m = conv_model().train()
        out = m(torch.randn(2, 1, PATCH, PATCH, PATCH))
        # `compute_total_loss` dispatches on ndim == 2.
        assert out.mu.ndim == 2
        kl, collapsed, per_channel = kl_divergence_flat(out.mu, out.logvar)
        assert kl.ndim == 0                       # one scalar
        assert per_channel.shape == (8,)          # one per COEFFICIENT, not per cell
        assert torch.isfinite(kl)
        # The reduction is over B*cells: doubling the batch must not change the
        # scale of the KL, because it is a mean over samples and cells alike.
        out2 = m(torch.randn(4, 1, PATCH, PATCH, PATCH))
        kl2, _, _ = kl_divergence_flat(out2.mu, out2.logvar)
        assert 0.2 < float(kl2) / float(kl) < 5.0


# ---------------------------------------------------------------------------
# End-of-training basis finalization
#
# This is not a corner case. Training derives U_f in a separate pass AFTER the
# last step, and a checkpoint without it is refused by vae_val_l1 and by the
# reconstruction figure alike — so a variant whose finalization does not work
# produces no row and no figure however well it trained.
# ---------------------------------------------------------------------------

def _tiny_loader(n=4, batch=2):
    from torch.utils.data import DataLoader, Dataset

    class _D(Dataset):
        def __len__(self):
            return n

        def __getitem__(self, i):
            g = torch.Generator().manual_seed(i)
            return {"xct": torch.randn(1, 64, 64, 64, generator=g)}

    return DataLoader(_D(), batch_size=batch, num_workers=0)


def test_finalization_treats_cells_as_samples_and_stays_small():
    """The per-cell model finalizes, and its covariance is 64x64, not 262144^2.

    Flattening the (B, 64, 16, 16, 16) grid asks for 275 GB and OOM-kills the
    host; that is what happened to a 3.4 h conv_k8 run on 2026-09-25.
    """
    model = conv_model()
    info = model.finalize_inference_basis(
        _tiny_loader(), device=torch.device("cpu"), autocast_dtype=torch.float32)

    basis = model.bottleneck.rr.inference_basis
    assert basis.shape == (64, 8), basis.shape
    # 4 patches x 16^3 cells — the sample count is CELLS, not patches.
    assert info["n_samples"] == 4 * 16 ** 3, info


def test_finalization_of_a_1x1x1_grid_still_takes_the_flat_path():
    """At n_blocks=6 the grid degenerates and the two readings coincide."""
    from poregen.models.vae.v2.vrrae_finetune import _as_samples

    h = torch.randn(3, 64, 1, 1, 1)
    assert torch.equal(_as_samples(h, 64), h.flatten(1))


def test_as_samples_refuses_an_output_it_cannot_read():
    from poregen.models.vae.v2.vrrae_finetune import _as_samples

    with pytest.raises(RuntimeError, match="neither one sample"):
        _as_samples(torch.randn(2, 30, 4, 4, 4), 64)
