"""The SliceGAN baseline, against the things that make it that method.

Two of these exist because the obvious reading of the paper is wrong. "A 4^3
latent becomes 64^3" suggests each latent cell is 16 voxels, so a 12-cell latent
would give 192. It does not: the padding schedule makes the map AFFINE,
out = 32n - 64, and the obvious reading produced a 34^3 volume from a 4^3 latent
the first time this was written.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from unittest import mock

import numpy as np
import pytest
import torch

from poregen.baselines.slicegan.networks import (
    Critic2D,
    Generator3D,
    latent_for_shape,
    shape_for_latent,
    volume_to_slices,
)


def _load_sampler():
    """The sampler lives in scripts/, which is not a package."""
    path = Path(__file__).resolve().parents[1] / "scripts" / "analysis" / "slicegan_sample.py"
    spec = importlib.util.spec_from_file_location("slicegan_sample", path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["slicegan_sample"] = mod
    spec.loader.exec_module(mod)
    return mod


sg_module = _load_sampler()
sg_generate = sg_module.generate


class TestGeneratorGeometry:

    def test_the_paper_shape_comes_out_exactly(self):
        g = Generator3D()
        v = g(torch.randn(1, g.nz, 4, 4, 4))
        assert tuple(v.shape) == (1, 4, 64, 64, 64)

    @pytest.mark.parametrize("shape", [(64, 64, 64), (192, 192, 192),
                                       (192, 1024, 1024), (256, 256, 256)])
    def test_latent_and_shape_are_inverses(self, shape):
        assert shape_for_latent(latent_for_shape(shape)) == shape

    def test_the_affine_relation_is_not_a_multiple(self):
        """The trap. If this ever reads 16*n, the padding schedule has changed
        and every sampled volume is the wrong size."""
        assert latent_for_shape((192, 192, 192)) == (8, 8, 8)
        assert latent_for_shape((192, 1024, 1024)) == (8, 34, 34)
        # 192 / 16 = 12 is the obvious-and-wrong answer.
        assert latent_for_shape((192, 192, 192))[0] != 12

    def test_a_shape_it_cannot_make_is_refused(self):
        with pytest.raises(ValueError, match="cannot generate"):
            latent_for_shape((100, 100, 100))

    def test_a_larger_latent_really_gives_a_larger_volume(self):
        """Fully convolutional: this is the paper's route to volumes bigger
        than the training patch, so it is load-bearing, not incidental."""
        g = Generator3D(ngf=8)
        v = g(torch.randn(1, g.nz, 8, 4, 4))
        assert tuple(v.shape)[2:] == (192, 64, 64)


class TestHaloTiling:
    """The tiled sampler must equal the single forward pass, EXACTLY.

    Tiling exists because a 192x1024x1024 volume in one pass took this machine
    to 119 GB of its 121 GB unified pool. It is only legitimate if it changes
    nothing: the generator is fully convolutional, so a tile carrying enough
    halo reproduces the full pass in its interior bit for bit. "Close enough"
    is not the claim being made, so the test is equality and not a tolerance.
    """

    @staticmethod
    def _single_pass(gen, shape, seed):
        g = torch.Generator(device="cpu").manual_seed(seed)
        with torch.no_grad():
            out = gen(gen.sample_latent(shape, n=1, generator=g))[0]
        grey = ((out[0].clamp(-1, 1) + 1) * 127.5).round().clamp(0, 255)
        return grey.to(torch.uint8).numpy(), out[1:].argmax(dim=0).to(torch.uint8).numpy()

    @pytest.mark.parametrize("shape,core", [((128, 128, 128), 2), ((192, 192, 192), 3)])
    def test_the_tiled_volume_equals_the_single_pass(self, shape, core):
        gen = Generator3D(ngf=8).eval()
        torch.manual_seed(0)
        tiled_g, tiled_l = sg_generate(gen, shape, seed=7,
                                       device=torch.device("cpu"), core=core)
        full_g, full_l = self._single_pass(gen, shape, 7)
        assert np.array_equal(tiled_g, full_g)
        assert np.array_equal(tiled_l, full_l)

    def test_the_cores_cover_the_volume_and_nothing_more(self):
        """The bug this caught: a latent of n cells makes 32(n-2) voxels.

        Tiling all n cells asks the generator for output past the end of the
        volume. The first version did exactly that and the guard stopped it at
        cells (0,0,4) rather than writing a truncated volume.
        """
        gen = Generator3D(ngf=8).eval()
        shape = (128, 192, 256)
        n_cells = latent_for_shape(shape)
        grey, _ = sg_generate(gen, shape, seed=1, device=torch.device("cpu"), core=2)
        assert grey.shape == shape
        for axis in range(3):
            assert 32 * (n_cells[axis] - 2) == shape[axis]

    def test_a_smaller_halo_would_not_reproduce_the_full_pass(self):
        """The guard on the guard: 2 cells is the MINIMUM, not a safe margin.

        If a halo of 1 also reproduced the full pass, the measured receptive
        field would be wrong and the constant would be carrying no weight.
        """
        gen = Generator3D(ngf=8).eval()
        torch.manual_seed(0)
        shape = (128, 128, 128)
        full_g, _ = self._single_pass(gen, shape, 7)
        with mock.patch.object(sg_module, "HALO_CELLS", 1):
            try:
                thin_g, _ = sg_generate(gen, shape, seed=7,
                                        device=torch.device("cpu"), core=2)
            except RuntimeError:
                return          # refused outright, which is also a failure to tile
        assert not np.array_equal(thin_g, full_g)


class TestOutputContract:

    def test_grey_is_tanh_ranged_and_the_label_is_a_distribution(self):
        g = Generator3D(ngf=8)
        v = g(torch.randn(2, g.nz, 4, 4, 4))
        assert float(v[:, 0].min()) >= -1.0 and float(v[:, 0].max()) <= 1.0
        sums = v[:, 1:].sum(dim=1)
        assert torch.allclose(sums, torch.ones_like(sums), atol=1e-5)

    def test_four_channels_so_grey_and_label_are_paired(self):
        """The baseline must emit a paired grey+label volume, or it cannot be
        scored on the same tables as ldm06."""
        g = Generator3D(ngf=8)
        assert g(torch.randn(1, g.nz, 4, 4, 4)).shape[1] == 4


class TestSliceExtraction:

    @pytest.mark.parametrize("axis", [0, 1, 2])
    def test_every_axis_yields_square_slices_of_all_channels(self, axis):
        vol = torch.randn(2, 4, 8, 12, 16)
        s = volume_to_slices(vol, axis)
        n = vol.shape[2 + axis]
        assert s.shape[0] == 2 * n
        assert s.shape[1] == 4
        other = [vol.shape[2 + a] for a in range(3) if a != axis]
        assert tuple(s.shape[2:]) == tuple(other)

    def test_the_slices_are_the_actual_slices(self):
        """Not a reshape that scrambles them — the critic would still train and
        the volume would mean nothing."""
        vol = torch.arange(2 * 1 * 3 * 4 * 5, dtype=torch.float32).reshape(2, 1, 3, 4, 5)
        s = volume_to_slices(vol, 0)
        assert torch.equal(s[0, 0], vol[0, 0, 0])
        assert torch.equal(s[1, 0], vol[0, 0, 1])
        assert torch.equal(s[3, 0], vol[1, 0, 0])

    def test_a_critic_accepts_slices_from_every_axis(self):
        g, d = Generator3D(ngf=8), Critic2D(ndf=8)
        v = g(torch.randn(1, g.nz, 4, 4, 4))
        for axis in range(3):
            assert d(volume_to_slices(v, axis)).shape == (64,)


class TestTrainingStep:

    def test_one_wgan_gp_step_updates_both_networks(self, tmp_path):
        from poregen.baselines.slicegan.data import SliceBank
        from poregen.baselines.slicegan.train import TrainConfig, train

        rng = np.random.default_rng(0)
        n = 120
        bank = SliceBank(
            grey=rng.integers(0, 255, (n, 64, 64), dtype=np.uint8),
            label=rng.integers(0, 3, (n, 64, 64)).astype(np.uint8),
            axis=np.repeat([0, 1, 2], n // 3).astype(np.int8),
        )
        cfg = TrainConfig(steps=1, g_batch=1, slices_per_volume=2, n_critic=1,
                          ngf=8, ndf=8, log_every=1)
        ck = train(bank, cfg, tmp_path, torch.device("cpu"))
        assert ck.exists()
        state = torch.load(ck, map_location="cpu", weights_only=False)
        assert state["step"] == 1
        assert len(state["critics"]) == 3, "one critic per axis, always"
        assert (tmp_path / "losses.jsonl").exists()

    def test_the_gradient_penalty_is_finite_and_positive(self):
        from poregen.baselines.slicegan.train import gradient_penalty

        d = Critic2D(ndf=8)
        real = torch.randn(4, 4, 64, 64)
        fake = torch.randn(4, 4, 64, 64)
        gp = gradient_penalty(d, real, fake, torch.device("cpu"))
        assert torch.isfinite(gp) and float(gp) >= 0.0


class TestRealSlices:

    def test_the_critic_batch_is_in_the_generator_s_own_space(self):
        """Real and fake must be encoded identically or the critic separates
        them on encoding rather than on content."""
        from poregen.baselines.slicegan.data import SliceBank, to_critic_batch

        bank = SliceBank(
            grey=np.array([[[0, 255], [128, 64]]], np.uint8),
            label=np.array([[[0, 1], [2, 0]]], np.uint8),
            axis=np.array([0], np.int8),
        )
        x = to_critic_batch(bank, np.array([0]))
        assert x.shape == (1, 4, 2, 2)
        assert x[0, 0].min() == pytest.approx(-1.0)
        assert x[0, 0].max() == pytest.approx(1.0)
        assert np.allclose(x[0, 1:].sum(axis=0), 1.0)
        assert x[0, 1, 0, 0] == 1.0 and x[0, 2, 0, 1] == 1.0 and x[0, 3, 1, 0] == 1.0
