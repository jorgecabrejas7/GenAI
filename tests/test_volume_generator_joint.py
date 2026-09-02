"""Tests for the MultiDiffusion-style JOINT volume denoising mode.

Covers:
  1. Window/canvas index math — every canvas cell covered, overlap counts
     correct at corners/edges/faces/interior, coverage guard raises.
  2. Fusion weights — strictly positive, normalised weights sum to 1 at every
     canvas voxel.
  3. The per-timestep average is applied on ε PREDICTIONS inside the reverse
     process (one shared canvas, one DDIM step per timestep), not on finished
     per-window samples.
  4. Neighbour treatment — every window runs with availability all-UNKNOWN and
     zero neighbour latents.
  5. generate(mode=...) dispatch, stats keys, and the DDIM-only guard.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from poregen.diffusion.conditioning import NB_OOB, NB_UNKNOWN
from poregen.diffusion.noise_schedule import DDPMSchedule
from poregen.diffusion.sampler import (
    VolumeGenerator,
    joint_window_origins,
    joint_window_weight,
)


# ── Minimal stubs ─────────────────────────────────────────────────────────────

class _FakeModelCfg:
    z_channels = 2
    use_por_cond = False
    use_orient_cond = False


class _FakeModel:
    cfg = _FakeModelCfg()

    def eval(self) -> "_FakeModel":
        return self


class _SpyDDIMSampler:
    """DDIM-shaped sampler with a deterministic, window-dependent ε function.

    eps(x_w) = 0.1·x_w + cond_depth  — depends on both the window's current
    canvas slice and its per-window conditioning, so the test can tell a
    shared-canvas recursion apart from independent per-window trajectories.
    """

    def __init__(self, T: int = 100, steps: tuple[int, ...] = (99, 49, 0)) -> None:
        self.model = _FakeModel()
        self.schedule = DDPMSchedule(T=T)
        self._timesteps = list(steps)
        self.recorded_avail: list[torch.Tensor] = []
        self.recorded_nb: list[torch.Tensor] = []

    @staticmethod
    def eps_fn(xw: torch.Tensor, depth: torch.Tensor) -> torch.Tensor:
        return 0.1 * xw + depth.view(-1, 1, 1, 1, 1)

    def predict_eps(self, x, t, nb_latents, nb_avail, cond_por, cond_depth,
                    cond_dist, cond_orient, autocast_dtype=torch.bfloat16):
        self.recorded_avail.append(nb_avail.clone())
        self.recorded_nb.append(nb_latents.clone())
        return self.eps_fn(x, cond_depth)


class _FakeVAE:
    """Decodes latents to constant patches (logit = the latent's mean)."""

    class _MeanHead:
        def __init__(self, patch_size: int) -> None:
            self.P = patch_size

        def __call__(self, dec: torch.Tensor) -> torch.Tensor:
            B = dec.shape[0]
            per_sample = dec.mean(dim=(1, 2, 3, 4)).view(B, 1, 1, 1, 1)
            return per_sample.expand(B, 1, self.P, self.P, self.P)

    def __init__(self, patch_size: int) -> None:
        self.xct_head = self._MeanHead(patch_size)
        self.mask_head = self._MeanHead(patch_size)

    def eval(self) -> "_FakeVAE":
        return self

    def decoder(self, z: torch.Tensor) -> torch.Tensor:
        return z


def _make_generator(sampler, patch_size: int = 8, latent_size: int = 4) -> VolumeGenerator:
    return VolumeGenerator(
        sampler=sampler,
        vae=_FakeVAE(patch_size=patch_size),
        device=torch.device("cpu"),
        patch_size=patch_size,
        generation_stride=patch_size,
        neighbour_offset=patch_size,
        latent_size=latent_size,
        latent_std=1.0,
        voxel_size_mm=0.025,
    )


# ── 1 + 2: window/canvas index math and fusion weights ───────────────────────

def test_window_origins_cover_the_whole_canvas() -> None:
    """192³ voxels → 48³ latent cells, 16-cell windows at 8-cell stride."""
    canvas, L, s = (48, 48, 48), 16, 8
    origins = joint_window_origins(canvas, L, s)
    assert len(origins) == 5 ** 3 == 125            # (48-16)/8 + 1 = 5 per axis

    counts = np.zeros(canvas, dtype=np.int32)
    for z, y, x in origins:
        counts[z:z + L, y:y + L, x:x + L] += 1
    assert (counts >= 1).all(), "every canvas cell must be covered"
    # 1-D coverage is 1 in the first/last stride band and 2 elsewhere, so the
    # 3-D counts are the products: corner 1, edge 2, face 4, interior 8.
    assert counts[0, 0, 0] == 1
    assert counts[24, 0, 0] == 2
    assert counts[24, 24, 0] == 4
    assert counts[24, 24, 24] == 8
    assert set(np.unique(counts)) == {1, 2, 4, 8}


def test_window_origins_reject_uncoverable_canvas() -> None:
    with pytest.raises(ValueError, match="cannot be covered"):
        joint_window_origins((50, 48, 48), 16, 8)   # (50-16) % 8 != 0
    with pytest.raises(ValueError, match="cannot be covered"):
        joint_window_origins((8, 48, 48), 16, 8)    # axis smaller than a window
    with pytest.raises(ValueError, match="stride_cells"):
        joint_window_origins((48, 48, 48), 16, 20)  # stride > window


def test_fusion_weights_positive_and_normalised() -> None:
    canvas, L, s = (32, 32, 32), 16, 8
    weight = joint_window_weight(L)
    assert weight.shape == (L, L, L)
    assert (weight > 0).all(), "weights must be strictly positive everywhere"

    origins = joint_window_origins(canvas, L, s)
    weight_sum = torch.zeros(canvas)
    for z, y, x in origins:
        weight_sum[z:z + L, y:y + L, x:x + L] += weight
    assert (weight_sum > 0).all()

    # After normalisation the per-voxel weights of all covering windows sum to
    # 1 everywhere — corners (1 window), edges, faces and interior alike.
    norm_total = torch.zeros(canvas)
    for z, y, x in origins:
        norm_total[z:z + L, y:y + L, x:x + L] += (
            weight / weight_sum[z:z + L, y:y + L, x:x + L]
        )
    assert torch.allclose(norm_total, torch.ones(canvas), atol=1e-5)


# ── 3 + 4: the average is on predictions inside the reverse process ──────────

def test_joint_average_is_on_predictions_not_samples() -> None:
    """The joint path must equal the canvas recursion with per-step ε fusion.

    Expected (MultiDiffusion): one canvas x; per timestep, every window's ε is
    computed FROM THE SHARED CANVAS, fused by the normalised cosine weights,
    and a single DDIM step advances the canvas.  The wrong algorithm (per-
    window full trajectories, finished samples averaged at the end) gives a
    different result because ε here depends on the evolving window content.
    """
    P, L = 8, 4                      # downsample 2
    vol = (16, 16, 16)               # canvas 8³ cells
    window_stride = 4                # 2 cells → 3 origins per axis, 27 windows
    sampler = _SpyDDIMSampler(T=100, steps=(99, 49, 0))
    gen = _make_generator(sampler, patch_size=P, latent_size=L)

    torch.manual_seed(1234)
    generated, grid_origins = gen._generate_latents_joint(
        volume_shape=vol, window_stride=window_stride, window_batch=5,
    )

    # ── manual reference: canvas recursion with ε fused per timestep ─────────
    ds = P // L
    canvas = tuple(v // ds for v in vol)
    origins = joint_window_origins(canvas, L, window_stride // ds)
    weight = joint_window_weight(L)
    weight_sum = torch.zeros((1, 1, *canvas))
    for z, y, x in origins:
        weight_sum[0, 0, z:z + L, y:y + L, x:x + L] += weight
    depths = torch.tensor(
        [gen._patch_position(tuple(c * ds for c in oc), vol)[0] for oc in origins]
    )

    schedule = DDPMSchedule(T=100)
    torch.manual_seed(1234)
    x = torch.randn(1, 2, *canvas)
    steps = [99, 49, 0]
    for i, t_val in enumerate(steps[:-1]):
        eps_sum = torch.zeros_like(x)
        for j, (z, y, xx) in enumerate(origins):
            xw = x[:, :, z:z + L, y:y + L, xx:xx + L]
            eps = _SpyDDIMSampler.eps_fn(xw, depths[j:j + 1])
            eps_sum[0, :, z:z + L, y:y + L, xx:xx + L] += weight * eps[0]
        t = torch.tensor([t_val])
        t_prev = torch.tensor([steps[i + 1]])
        x = schedule.ddim_step(x, t, t_prev, eps_sum / weight_sum)

    for gi, (z0, y0, x0) in grid_origins.items():
        cz, cy, cx = z0 // ds, y0 // ds, x0 // ds
        Lc = P // ds
        expected = x[0, :, cz:cz + Lc, cy:cy + Lc, cx:cx + Lc]
        assert torch.allclose(generated[gi], expected, atol=1e-5), (
            f"tile {gi} diverges from the shared-canvas ε-fusion recursion"
        )

    # ── negative control: averaging FINISHED per-window samples differs ──────
    torch.manual_seed(1234)
    x0_init = torch.randn(1, 2, *canvas)
    final_sum = torch.zeros_like(x0_init)
    for j, (z, y, xx) in enumerate(origins):
        xw = x0_init[:, :, z:z + L, y:y + L, xx:xx + L].clone()
        for i, t_val in enumerate(steps[:-1]):
            eps = _SpyDDIMSampler.eps_fn(xw, depths[j:j + 1])
            xw = schedule.ddim_step(xw, torch.tensor([t_val]),
                                    torch.tensor([steps[i + 1]]), eps)
        final_sum[0, :, z:z + L, y:y + L, xx:xx + L] += weight * xw[0]
    wrong = final_sum / weight_sum
    joint_canvas = torch.zeros(2, *canvas)
    for gi, (z0, y0, x0v) in grid_origins.items():
        cz, cy, cx = z0 // ds, y0 // ds, x0v // ds
        Lc = P // ds
        joint_canvas[:, cz:cz + Lc, cy:cy + Lc, cx:cx + Lc] = generated[gi]
    assert not torch.allclose(joint_canvas, wrong[0], atol=1e-3), (
        "joint result must NOT equal the average of finished per-window samples"
    )


def test_joint_windows_neighbour_semantics() -> None:
    # Default "specimen": out-of-volume faces OOB (as training saw at real
    # specimen boundaries), in-volume faces UNKNOWN, never EXISTS.
    sampler = _SpyDDIMSampler(T=100, steps=(99, 0))
    gen = _make_generator(sampler, patch_size=8, latent_size=4)
    gen._generate_latents_joint(volume_shape=(16, 16, 16),
                                window_stride=4, window_batch=32)
    assert sampler.recorded_avail, "the sampler was never called"
    seen_oob = False
    for avail in sampler.recorded_avail:
        assert ((avail == NB_UNKNOWN) | (avail == NB_OOB)).all()
        seen_oob = seen_oob or bool((avail == NB_OOB).any())
    assert seen_oob, "a 16-vox volume has boundary windows — OOB expected"
    for nb in sampler.recorded_nb:
        assert (nb == 0).all()

    # "legacy" (original joint) and "interior": all-UNKNOWN everywhere.
    for semantics in ("legacy", "interior"):
        spy = _SpyDDIMSampler(T=100, steps=(99, 0))
        g = _make_generator(spy, patch_size=8, latent_size=4)
        g.conditioning_semantics = semantics
        g._generate_latents_joint(volume_shape=(16, 16, 16),
                                  window_stride=4, window_batch=32)
        for avail in spy.recorded_avail:
            assert (avail == NB_UNKNOWN).all(), semantics


# ── 5: generate() dispatch and API surface ───────────────────────────────────

def test_generate_joint_mode_end_to_end() -> None:
    sampler = _SpyDDIMSampler(T=100, steps=(99, 49, 0))
    gen = _make_generator(sampler, patch_size=8, latent_size=4)
    xct, mask, stats = gen.generate(
        volume_size_mm=(0.4, 0.4, 0.4),        # 16 vox per axis, 2³ tiles
        target_porosity=0.02,
        mode="joint",
        joint_window_stride=4,
        joint_window_batch=8,
    )
    assert xct.shape == (16, 16, 16)
    assert mask.shape == (16, 16, 16)
    assert stats["generation_mode"] == "joint"
    assert stats["joint_window_stride"] == 4
    assert "seam_xct_ratio" in stats and "seam_mask_ratio" in stats


def test_generate_sequential_stats_report_mode() -> None:
    class _SeqSampler:
        model = _FakeModel()

        def sample_batch(self, nb_latents, nb_avail, cond_por, cond_depth,
                         cond_dist, cond_orient=None,
                         autocast_dtype=torch.bfloat16):
            B, C, D = nb_latents.shape[0], nb_latents.shape[2], nb_latents.shape[3]
            return torch.zeros(B, C, D, D, D)

    gen = _make_generator(_SeqSampler(), patch_size=8, latent_size=4)
    _, _, stats = gen.generate(volume_size_mm=(0.4, 0.4, 0.4))
    assert stats["generation_mode"] == "sequential"
    assert "joint_window_stride" not in stats


def test_generate_rejects_unknown_mode_and_non_ddim_sampler() -> None:
    sampler = _SpyDDIMSampler(T=100, steps=(99, 0))
    gen = _make_generator(sampler, patch_size=8, latent_size=4)
    with pytest.raises(ValueError, match="Unknown generation mode"):
        gen.generate(volume_size_mm=(0.4, 0.4, 0.4), mode="bogus")

    class _DDPMLike:
        model = _FakeModel()

    gen2 = _make_generator(_DDPMLike(), patch_size=8, latent_size=4)
    with pytest.raises(TypeError, match="DDIMSampler"):
        gen2.generate(volume_size_mm=(0.4, 0.4, 0.4), mode="joint")


def test_joint_rejects_bad_window_stride() -> None:
    sampler = _SpyDDIMSampler(T=100, steps=(99, 0))
    gen = _make_generator(sampler, patch_size=8, latent_size=4)
    with pytest.raises(ValueError, match="divisor"):
        gen._generate_latents_joint(volume_shape=(16, 16, 16), window_stride=6)
    with pytest.raises(ValueError, match="multiple"):
        gen._generate_latents_joint(volume_shape=(16, 16, 16), window_stride=1)
