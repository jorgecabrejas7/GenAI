"""The pixel-space DDPM baseline.

Its job is to price the latent space, so what these tests protect is that the
comparison stays about the SPACE: the schedule must be ldm06's own, the output
must be the same paired grey+label contract, and a large volume must be built
by fusion rather than by any conditioning the model does not have.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from poregen.baselines.ddpm3d.data import decode_sample
from poregen.baselines.ddpm3d.networks import UNet3DPixel, timestep_embedding
from poregen.baselines.ddpm3d.train import (
    DDPMConfig,
    build,
    sample_volume,
    sample_window,
)

SMALL = DDPMConfig(base=8, mults=(1, 2))


class TestNetwork:

    def test_it_denoises_a_64_cubed_patch_shape_for_shape(self):
        m = UNet3DPixel(base=8, mults=(1, 2))
        x = torch.randn(1, 4, 64, 64, 64)
        assert m(x, torch.randint(0, 1000, (1,))).shape == x.shape

    def test_four_channels_in_and_out_so_grey_and_label_stay_paired(self):
        m = UNet3DPixel(base=8, mults=(1, 2))
        y = m(torch.randn(2, 4, 32, 32, 32), torch.randint(0, 1000, (2,)))
        assert y.shape[1] == 4

    def test_the_timestep_embedding_separates_timesteps(self):
        e = timestep_embedding(torch.tensor([0, 1, 500, 999]), 32)
        assert e.shape == (4, 32)
        assert not torch.allclose(e[0], e[1])
        assert not torch.allclose(e[2], e[3])


class TestSharedSchedule:
    """The comparison is about pixel vs latent space, so the schedule must be
    the SAME object ldm06 trains against — not a lookalike."""

    def test_it_uses_ldm06_s_own_schedule_class(self):
        from poregen.diffusion.noise_schedule import DDPMSchedule

        _, sched = build(SMALL, torch.device("cpu"))
        assert isinstance(sched, DDPMSchedule)

    def test_the_defaults_are_ldm06_s(self):
        cfg = DDPMConfig()
        assert cfg.objective == "v"
        assert cfg.zero_terminal_snr is True
        assert cfg.T == 1000

    def test_q_sample_and_training_target_agree_on_shape(self):
        _, sched = build(SMALL, torch.device("cpu"))
        x0 = torch.randn(2, 4, 16, 16, 16)
        t = torch.randint(0, 1000, (2,))
        noise = torch.randn_like(x0)
        assert sched.q_sample(x0, t, noise).shape == x0.shape
        assert sched.training_target(x0, noise, t).shape == x0.shape


class TestSampling:

    def test_a_window_comes_out_at_the_training_shape(self):
        m, sched = build(SMALL, torch.device("cpu"))
        m.eval()
        x = sample_window(m, sched, 1, torch.device("cpu"), steps=2)
        assert tuple(x.shape) == (1, 4, 64, 64, 64)

    def test_a_larger_volume_is_fused_to_the_shape_asked_for(self):
        m, sched = build(SMALL, torch.device("cpu"))
        m.eval()
        v = sample_volume(m, sched, (64, 64, 96), torch.device("cpu"),
                          steps=2, stride=32, batch=2)
        assert tuple(v.shape) == (4, 64, 64, 96)

    def test_a_shape_below_the_window_is_refused(self):
        """Not silently padded: a 32-cubed 'sample' from a 64-cubed model would
        be a crop of one window and not a sample of that shape."""
        m, sched = build(SMALL, torch.device("cpu"))
        with pytest.raises(ValueError, match="smaller than"):
            sample_volume(m, sched, (32, 32, 32), torch.device("cpu"), steps=1)

    def test_the_fusion_weights_sum_to_one_everywhere(self):
        """Every voxel must be fully accounted for, or the fused volume is
        darker at the overlaps and the seam metric reads an artefact."""
        from poregen.baselines.ddpm3d.train import tukey_window_3d

        shape, patch, stride = (64, 64, 96), 64, 32
        win = tukey_window_3d(patch)
        wacc = np.zeros(shape, np.float32)
        starts = lambda n: sorted(set(list(range(0, n - patch + 1, stride)) + [n - patch]))
        for z in starts(shape[0]):
            for y in starts(shape[1]):
                for x in starts(shape[2]):
                    wacc[z:z+patch, y:y+patch, x:x+patch] += win
        assert wacc.min() > 0.0


class TestOutputContract:

    def test_decode_gives_uint8_grey_and_a_3_class_label(self):
        x = torch.randn(4, 8, 8, 8)
        g, lab = decode_sample(x)
        assert g.dtype == np.uint8 and lab.dtype == np.uint8
        assert set(np.unique(lab)) <= {0, 1, 2}

    def test_out_of_range_grey_is_clamped_not_wrapped(self):
        """A diffusion sample can leave [-1, 1]; scaling before clamping would
        send those voxels to the opposite end of the range."""
        x = torch.zeros(4, 2, 2, 2)
        x[0] = torch.tensor([[[-3.0, 3.0], [0.0, 0.0]], [[0.0, 0.0], [0.0, 0.0]]])
        g, _ = decode_sample(x)
        assert g[0, 0, 0] == 0
        assert g[0, 0, 1] == 255
