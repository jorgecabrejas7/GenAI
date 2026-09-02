"""Audit tests: porosity-conditioning map construction, training vs generation.

These tests PIN DOWN CURRENT BEHAVIOR of the conditioning paths (they are
documentation, not aspiration).  The bugs originally pinned by ``AUDIT:``
comments have been fixed: local phi is clamped to the training range at the
point of use in VolumeGenerator, drop_por on a null-less model raises, and
generate() returns post-generation porosity stats.  Tests marked ``FIXED``
assert the corrected behavior.  Do not change a test here without also
changing (or consciously accepting) the production behavior it documents.

Covered paths
-------------
1. VolumeGenerator._generate_latents — per-patch (global_por, local_por,
   pos_frac) assignment at generation time.
2. VolumeGenerator.generate — direct-tiling assembly at stride == patch_size
   and the mask threshold semantics with no blending.
3. UNet3DDenoiser null-porosity token engagement rules.
4. Local-porosity map builders (_gaussian_por_grid in ldm_engine,
   _build_local_por_map in scripts/generate_volumes.py).
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

from poregen.diffusion.sampler import (
    _POR_MAX,
    _POR_MIN,
    VolumeGenerator,
)
from poregen.models.diffusion.unet import UNet3DConfig, UNet3DDenoiser
from poregen.diffusion.conditioning import PARITY_GROUP_ORDER, parity_group
from poregen.training.ldm_engine import _gaussian_por_grid

# ── shared stubs ──────────────────────────────────────────────────────────────


class _FakeModelCfg:
    z_channels = 2
    use_por_cond = True
    use_orient_cond = False


class _FakeModel:
    cfg = _FakeModelCfg()


class _RecordingSampler:
    """Records the conditioning values every patch is sampled with.

    Latents are filled with a per-patch constant chosen by ``fill_fn(cond_por)``
    so that a fake VAE downstream can emit per-patch constant logits.
    """

    def __init__(self, fill_fn=None) -> None:
        self.model = _FakeModel()
        self.por: list[float] = []
        self.depth: list[float] = []
        self.dist: list[float] = []
        self._fill_fn = fill_fn or (lambda cond_por: 0.0)

    def sample_batch(self, nb_latents, nb_avail, cond_por, cond_depth, cond_dist,
                     cond_orient=None, autocast_dtype=torch.bfloat16):
        B = nb_latents.shape[0]
        C, D = nb_latents.shape[2], nb_latents.shape[3]
        out = torch.empty(B, C, D, D, D)
        for i in range(B):
            self.por.append(float(cond_por[i]))
            self.depth.append(float(cond_depth[i]))
            self.dist.append(float(cond_dist[i]))
            out[i] = self._fill_fn(float(cond_por[i]))
        return out


class _ConstHead:
    """Head returning a (B,1,P,P,P) tensor equal to the per-sample latent mean."""

    def __init__(self, patch_size: int) -> None:
        self.P = patch_size

    def __call__(self, dec: torch.Tensor) -> torch.Tensor:
        B = dec.shape[0]
        per_sample = dec.mean(dim=(1, 2, 3, 4)).view(B, 1, 1, 1, 1)
        return per_sample.expand(B, 1, self.P, self.P, self.P)


class _ConstVAE:
    """Fake VAE: decoder passes latents through; heads emit per-patch constant
    logits equal to the latent fill value."""

    def __init__(self, patch_size: int) -> None:
        self.decoder = lambda z: z
        self.xct_head = _ConstHead(patch_size)
        self.mask_head = _ConstHead(patch_size)

    def eval(self):
        return self


# Identity porosity transform: cond_por == log(phi + 1e-3), so the tests read
# the raw clamp behaviour straight off the recorded conditioning values.
_POR_STATS = (0.0, 1.0)


def _cond(phi: float) -> float:
    return float(np.log(phi + 1e-3))


def _make_generator(sampler, vae=None, stride=64):
    return VolumeGenerator(
        sampler=sampler,
        vae=vae,
        device=torch.device("cpu"),
        patch_size=64,
        generation_stride=stride,
        neighbour_offset=stride,
        latent_size=4,
        latent_std=1.0,
        voxel_size_mm=0.025,
        por_log_stats=_POR_STATS,
    )


# ── 1. per-patch conditioning value assignment at generation time ─────────────


class TestGenerationConditioningValues:
    def test_training_range_clamp_constants(self):
        # Pinned: EDA ground-truth VVF range hard-coded in sampler.py.
        assert _POR_MIN == 0.002
        assert _POR_MAX == 0.107

    def test_por_clamped_high(self):
        sampler = _RecordingSampler()
        gen = _make_generator(sampler)
        gen._generate_latents(volume_shape=(64, 64, 128), target_porosity=0.30)
        assert all(p == pytest.approx(_cond(_POR_MAX)) for p in sampler.por)

    def test_por_clamped_low(self):
        sampler = _RecordingSampler()
        gen = _make_generator(sampler)
        gen._generate_latents(volume_shape=(64, 64, 128), target_porosity=0.0005)
        assert all(p == pytest.approx(_cond(_POR_MIN)) for p in sampler.por)

    def test_target_none_falls_back_to_0p05(self):
        sampler = _RecordingSampler()
        gen = _make_generator(sampler)
        gen._generate_latents(volume_shape=(64, 64, 128), target_porosity=None)
        assert all(p == pytest.approx(_cond(0.05)) for p in sampler.por)

    def test_por_defaults_to_the_clamped_uniform_target(self):
        """Without a local_por_map every patch is conditioned to the same
        (clamped) target.  D32 §4 warns this is atypical — only 12.9 % of real
        patches sit within 0.001 of their volume mean — so real runs should
        pass a coherent per-patch field instead."""
        sampler = _RecordingSampler()
        gen = _make_generator(sampler)
        gen._generate_latents(volume_shape=(64, 64, 128), target_porosity=0.30)
        assert all(p == pytest.approx(_cond(_POR_MAX)) for p in sampler.por)

    def test_local_por_map_values_clamped_to_training_range(self):
        """local_por_map values are clamped to [_POR_MIN, _POR_MAX] at the
        choke point in _generate_latents, so OOD map values ('center'/'edges'
        peaks) can never reach the porosity embedding verbatim."""
        sampler = _RecordingSampler()
        gen = _make_generator(sampler)
        # (64,64,128) with stride 64 → grid indices (0,0,0) and (0,0,1)
        lmap = {(0, 0, 0): 0.50, (0, 0, 1): 0.0001}
        gen._generate_latents(
            volume_shape=(64, 64, 128), target_porosity=0.05, local_por_map=lmap,
        )
        assert sorted(sampler.por) == pytest.approx(
            sorted([_cond(_POR_MAX), _cond(_POR_MIN)])
        )

    def test_depth_and_dist_follow_the_documented_geometry(self):
        """cond_depth is the patch centre's fractional depth in z; cond_dist is
        the distance from that centre to the nearest outer surface over ALL
        THREE axes, capped at 64 voxels and divided by 64 — the same definition
        scripts/build_conditioning.py uses (D32 §1, signals 2 and 3)."""
        sampler = _RecordingSampler()
        gen = _make_generator(sampler)
        vol = (192, 64, 64)
        _, grid_origins = gen._generate_latents(volume_shape=vol, target_porosity=0.05)

        ordered = [
            gi
            for g in PARITY_GROUP_ORDER
            for gi in sorted(x for x in grid_origins if parity_group(x) == g)
        ]
        for gi, depth, dist in zip(ordered, sampler.depth, sampler.dist):
            centres = [o + 32.0 for o in grid_origins[gi]]
            assert depth == pytest.approx(centres[0] / vol[0])
            d_sur = min(min(c, e - c) for c, e in zip(centres, vol))
            assert dist == pytest.approx(min(d_sur, 64.0) / 64.0)

    def test_there_is_no_global_porosity_input(self):
        """D32 §1 drops the global-porosity condition entirely: the per-patch
        field carries volume-level control, and the local/global conflation was
        the D31 bug.  The porosity MLP therefore takes ONE scalar."""
        model = _mini_model(use_por_null=False)
        assert model.por_mlp[0].in_features == 1
        assert not hasattr(model, "pos_mlp")          # pos_frac is gone
        assert model.depth_mlp[0].in_features == 1
        assert model.dist_mlp[0].in_features == 1


# ── 2. direct-tiling assembly arithmetic ──────────────────────────────────────


class TestTiledAssemblyArithmetic:
    def test_constant_logits_preserved_everywhere(self):
        """Patches tile, so a decoded logit reaches the output untouched.

        There is no window and no weighted mean any more: a voxel belongs to
        exactly one patch.  Index 0 on each axis is a normal voxel — under the
        old Hann blending it was a zero-weight artifact forced to 0.
        """
        fill = 0.2
        sampler = _RecordingSampler(fill_fn=lambda cond_por: fill)
        vae = _ConstVAE(patch_size=64)
        gen = _make_generator(sampler, vae=vae)
        xct, mask, stats = gen.generate(volume_size_mm=(1.6, 1.6, 3.2), target_porosity=0.05)
        assert xct.shape == (64, 64, 128)

        # The XCT head regresses xct/255 directly — decode is clamp-and-scale,
        # no activation (models.vae.base.decode_xct).
        expected_xct = int(round(float(np.clip(fill, 0.0, 1.0)) * 255.0))
        assert np.all(np.abs(xct.astype(int) - expected_xct) <= 1)
        assert np.all(mask == 255)                   # logit 0.2 > 0 → pore
        assert np.all(xct[0] == xct[32])             # no zero-weight boundary plane

        # Post-generation self-audit stats returned by generate().
        assert stats["actual_mask_porosity"] == pytest.approx(float((mask > 0).mean()))
        assert stats["target_porosity"] == pytest.approx(0.05)
        assert stats["conditioned_porosity"] == pytest.approx(0.05)

    def test_mask_is_a_per_patch_decision(self):
        """Each patch decides its own block: per-patch porosity carries through.

        The old 50 %-overlap assembly thresholded a Hann-WEIGHTED MEAN LOGIT,
        so a voxel that patch A called a pore could be voted down by patch B.
        With touching patches there is no vote: patch A owns x in [0,64) and
        patch B owns x in [64,128), and volume porosity is the patch mean.
        """

        # Patches are told apart by their porosity conditioning: grid index
        # (0,0,0) asks for phi=0.01, (0,0,1) for phi=0.05.
        def fill_fn(cond_por):
            return 1.0 if cond_por < _cond(0.02) else -3.0

        sampler = _RecordingSampler(fill_fn=fill_fn)
        vae = _ConstVAE(patch_size=64)
        gen = _make_generator(sampler, vae=vae)
        _, mask, stats = gen.generate(
            volume_size_mm=(1.6, 1.6, 3.2), target_porosity=0.05,
            local_por_map={(0, 0, 0): 0.01, (0, 0, 1): 0.05},
        )

        assert np.all(mask[:, :, :64] == 255)        # patch A: logit +1 → pore
        assert np.all(mask[:, :, 64:] == 0)          # patch B: logit -3 → not pore
        # Volume porosity is exactly the mean of the per-patch decisions.
        assert stats["actual_mask_porosity"] == pytest.approx(0.5)

    def test_seam_diagnostic_reports_the_disagreement(self):
        """Two blocks that disagree at their shared face give a large ratio.

        This replaces the pre-blend overlap-disagreement metric: at stride 64
        there is no overlap, so the discontinuity across the shared face is the
        assembly-quality signal.
        """
        def fill_fn(cond_por):
            return 1.0 if cond_por < _cond(0.02) else -3.0

        sampler = _RecordingSampler(fill_fn=fill_fn)
        vae = _ConstVAE(patch_size=64)
        gen = _make_generator(sampler, vae=vae)
        _, _, stats = gen.generate(
            volume_size_mm=(1.6, 1.6, 3.2), target_porosity=0.05,
            local_por_map={(0, 0, 0): 0.01, (0, 0, 1): 0.05},
        )
        assert stats["seam_xct_x_planes"] == 1
        assert stats["seam_xct_x_mad"] > 10.0            # a hard step in grey level
        assert stats["seam_xct_x_interior_mad"] == pytest.approx(0.0, abs=1e-6)
        assert stats["seam_xct_z_planes"] == 0           # one patch deep


# ── 3. null-porosity token engagement ─────────────────────────────────────────


_MINI = dict(
    z_channels=2, base_channels=8, channel_mult=(1, 2), n_res_blocks=1,
    cond_embed_dim=16, nb_avail_embed_dim=4, n_attn_heads=2, dropout=0.0,
)


def _mini_model(use_por_null: bool) -> UNet3DDenoiser:
    torch.manual_seed(0)
    model = UNet3DDenoiser(UNet3DConfig(**_MINI, use_por_null=use_por_null)).eval()
    # out_conv and every AdaGN projection are zero-initialised in production,
    # which makes an untrained model conditioning-blind.  Randomise them so the
    # output actually reflects conditioning differences in these tests.
    with torch.no_grad():
        torch.nn.init.normal_(model.out_conv.weight, std=0.1)
        torch.nn.init.normal_(model.out_conv.bias, std=0.1)
        for mod in model.modules():
            if isinstance(mod, torch.nn.Linear):
                torch.nn.init.normal_(mod.weight, std=0.05)
                torch.nn.init.normal_(mod.bias, std=0.05)
    return model


def _mini_inputs(B=2, D=4):
    torch.manual_seed(0)
    return dict(
        z_t=torch.randn(B, 2, D, D, D),
        t=torch.tensor([3, 7]),
        nb_latents=torch.zeros(B, 6, 2, D, D, D),
        nb_avail=torch.full((B, 6), 2, dtype=torch.long),
        cond_por=torch.tensor([-3.0, -2.5]),
        cond_depth=torch.tensor([0.25, 0.75]),
        cond_dist=torch.tensor([0.5, 1.0]),
        cond_orient=torch.randn(B, 2, D, D, D),
    )


class TestNullTokenEngagement:
    def test_null_inactive_when_drop_por_is_none(self):
        """Cleared for failure mode (e), unguided path: with drop_por=None the
        learned null cannot influence the output — perturbing null_por leaves
        the forward pass bit-identical.  The unguided samplers (DDPM always,
        DDIM at s_por=s_nb=1) pass no drop mask, so no accidental null."""
        model = _mini_model(use_por_null=True)
        inp = _mini_inputs()
        with torch.no_grad():
            out1 = model(**inp)
            model.null_por.add_(100.0)
            out2 = model(**inp)
        assert torch.equal(out1, out2)

    def test_null_engaged_when_dropped(self):
        """Sanity: with use_por_null=True, drop_por=True actually switches the
        porosity embedding to the null."""
        model = _mini_model(use_por_null=True)
        with torch.no_grad():
            model.null_por.add_(1.0)  # make null distinguishable from por_mlp out
        inp = _mini_inputs()
        drop = torch.tensor([True, True])
        with torch.no_grad():
            out_cond = model(**inp)
            out_drop = model(**inp, drop_por=drop)
        assert not torch.allclose(out_cond, out_drop)

    def test_drop_por_raises_without_use_por_null(self):
        """FIXED (failure mode e, inverse leakage): when a checkpoint has
        use_por_null=False (ldm01/ldm02) and a caller passes drop_por (as the
        guided DDIM sampler does for its eps_uncond pass), _build_cond now
        raises ValueError instead of silently ignoring the drop."""
        model = _mini_model(use_por_null=False)
        inp = _mini_inputs()
        drop = torch.tensor([True, True])
        with torch.no_grad():
            with pytest.raises(ValueError, match="use_por_null"):
                model(**inp, drop_por=drop)


# ── 4. local-porosity map builders ────────────────────────────────────────────


def _load_generate_volumes_module():
    path = Path(__file__).resolve().parents[1] / "scripts" / "generate_volumes.py"
    spec = importlib.util.spec_from_file_location("generate_volumes_audit", path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


class TestLocalPorMapBuilders:
    def test_gaussian_por_grid_mean_matches_but_peak_is_ood(self):
        """_gaussian_por_grid (training-time sample logging) normalises so the
        patch MEAN equals global_por, but does not clip individual values.

        The builder itself still emits OOD peaks (> _POR_MAX); this is now
        harmless because VolumeGenerator._generate_latents clamps every local
        phi to [_POR_MIN, _POR_MAX] at the point of use."""
        grid = _gaussian_por_grid((3, 3, 3), global_por=0.05)
        vals = np.array(list(grid.values()))
        assert vals.mean() == pytest.approx(0.05, rel=1e-6)
        assert vals.max() > _POR_MAX  # pinned: no clipping applied

    def test_build_local_por_map_uniform_is_exact(self):
        mod = _load_generate_volumes_module()
        m = mod._build_local_por_map(2, 3, 4, target_por=0.03, distribution="uniform")
        assert len(m) == 24
        assert all(v == pytest.approx(0.03) for v in m.values())

    def test_build_local_por_map_center_peak_exceeds_training_clamp(self):
        """'center'/'edges' maps are clipped to [0.001, 0.999] — NOT to the
        sampler's training range [0.002, 0.107].  The builder is unchanged;
        the OOD peaks are now clamped downstream at the point of use (see
        test_local_por_map_values_clamped_to_training_range)."""
        mod = _load_generate_volumes_module()
        m = mod._build_local_por_map(4, 8, 8, target_por=0.05, distribution="center")
        vals = np.array(list(m.values()))
        # Mean over patches is preserved (up to the clip):
        assert vals.mean() == pytest.approx(0.05, rel=0.05)
        # ... but the peak is far outside the training-range clamp.
        assert vals.max() > _POR_MAX
        # Pinned clip bounds:
        assert vals.min() >= 0.001 and vals.max() <= 0.999
