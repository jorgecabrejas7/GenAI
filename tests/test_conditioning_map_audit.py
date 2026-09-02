"""Audit tests: porosity-conditioning map construction, training vs generation.

These tests PIN DOWN CURRENT BEHAVIOR of the conditioning paths (they are
documentation, not aspiration).  Where the pinned behavior is a suspected
contributor to the generated-volume porosity overshoot, the assertion carries
an ``AUDIT:`` comment describing the issue.  Do not "fix" a test here without
also fixing (or consciously accepting) the production behavior it documents.

Covered paths
-------------
1. VolumeGenerator._generate_latents — per-patch (global_por, local_por,
   pos_frac) assignment at generation time.
2. VolumeGenerator.generate — cosine-taper logit-space blending and the
   mask threshold semantics under 50% overlap.
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
from poregen.training.ldm_engine import _gaussian_por_grid

# ── shared stubs ──────────────────────────────────────────────────────────────


class _FakeModelCfg:
    z_channels = 2


class _FakeModel:
    cfg = _FakeModelCfg()


class _RecordingSampler:
    """Records the conditioning values every patch is sampled with.

    Latents are filled with a per-patch constant chosen by ``fill_fn(pos_frac_row)``
    so that a fake VAE downstream can emit per-patch constant logits.
    """

    def __init__(self, fill_fn=None) -> None:
        self.model = _FakeModel()
        self.g_por: list[float] = []
        self.l_por: list[float] = []
        self.pos: list[torch.Tensor] = []
        self._fill_fn = fill_fn or (lambda pos_row: 0.0)

    def sample_batch(self, nb_latents, nb_avail, pos_frac, global_por, local_por,
                     autocast_dtype=torch.bfloat16):
        B = nb_latents.shape[0]
        C, D = nb_latents.shape[2], nb_latents.shape[3]
        out = torch.empty(B, C, D, D, D)
        for i in range(B):
            self.g_por.append(float(global_por[i]))
            self.l_por.append(float(local_por[i]))
            self.pos.append(pos_frac[i].clone())
            out[i] = self._fill_fn(pos_frac[i])
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


def _make_generator(sampler, vae=None, stride=32):
    return VolumeGenerator(
        sampler=sampler,
        vae=vae,
        device=torch.device("cpu"),
        patch_size=64,
        patch_stride=stride,
        latent_size=4,
        latent_std=1.0,
        voxel_size_mm=0.025,
    )


# ── 1. per-patch conditioning value assignment at generation time ─────────────


class TestGenerationConditioningValues:
    def test_training_range_clamp_constants(self):
        # Pinned: EDA ground-truth VVF range hard-coded in sampler.py.
        assert _POR_MIN == 0.002
        assert _POR_MAX == 0.107

    def test_global_por_clamped_high(self):
        sampler = _RecordingSampler()
        gen = _make_generator(sampler)
        gen._generate_latents(volume_shape=(64, 64, 96), target_porosity=0.30)
        assert all(g == pytest.approx(_POR_MAX) for g in sampler.g_por)

    def test_global_por_clamped_low(self):
        sampler = _RecordingSampler()
        gen = _make_generator(sampler)
        gen._generate_latents(volume_shape=(64, 64, 96), target_porosity=0.0005)
        assert all(g == pytest.approx(_POR_MIN) for g in sampler.g_por)

    def test_target_none_falls_back_to_0p05(self):
        sampler = _RecordingSampler()
        gen = _make_generator(sampler)
        gen._generate_latents(volume_shape=(64, 64, 96), target_porosity=None)
        assert all(g == pytest.approx(0.05) for g in sampler.g_por)
        assert all(l == pytest.approx(0.05) for l in sampler.l_por)

    def test_local_por_defaults_to_clamped_global(self):
        """Without a local_por_map every patch's local_por channel receives the
        VOLUME-level target.  Training feeds this channel the PATCH-level mask
        fraction, whose distribution inside a real volume is highly non-uniform.

        AUDIT (failure mode b): the value semantics of the local channel differ
        between training (patch phi) and default generation (volume phi for
        every patch).  This is a documented conflation, not an arithmetic bug:
        conditioning every patch to phi_target requests a uniformly porous
        material, which is in-distribution per patch but atypical jointly.
        """
        sampler = _RecordingSampler()
        gen = _make_generator(sampler)
        gen._generate_latents(volume_shape=(64, 64, 96), target_porosity=0.30)
        # local inherits the CLAMPED global — consistent with each other.
        assert all(l == pytest.approx(_POR_MAX) for l in sampler.l_por)

    def test_local_por_map_values_bypass_training_range_clamp(self):
        """AUDIT (failure modes a/d): local_por_map values are forwarded to the
        model UNCLAMPED, while global_por is clamped to [_POR_MIN, _POR_MAX].
        A map built by scripts/generate_volumes.py ('center'/'edges') routinely
        contains per-patch values several times the target — far outside the
        training patch-phi distribution — and they reach por_mlp verbatim.
        """
        sampler = _RecordingSampler()
        gen = _make_generator(sampler)
        # (64,64,96) with stride 32 → grid indices (0,0,0) and (0,0,1)
        lmap = {(0, 0, 0): 0.50, (0, 0, 1): 0.50}
        gen._generate_latents(
            volume_shape=(64, 64, 96), target_porosity=0.05, local_por_map=lmap,
        )
        assert all(l == pytest.approx(0.50) for l in sampler.l_por)  # NOT clamped
        assert all(g == pytest.approx(0.05) for g in sampler.g_por)  # clamped path

    def test_pos_frac_matches_training_formula(self):
        """pos_frac at generation is z0/(vol_d-1) etc. — same formula as
        LatentPatchDataset.__getitem__.  Cleared for failure mode (a)."""
        sampler = _RecordingSampler()
        gen = _make_generator(sampler)
        vol = (64, 96, 128)
        _, grid_origins = gen._generate_latents(volume_shape=vol, target_porosity=0.05)

        # Reconstruct the phase ordering used by _generate_latents.
        phase0 = sorted(gi for gi in grid_origins if sum(gi) % 2 == 0)
        phase1 = sorted(gi for gi in grid_origins if sum(gi) % 2 == 1)
        for gi, pos in zip(phase0 + phase1, sampler.pos):
            z0, y0, x0 = grid_origins[gi]
            expected = torch.tensor([
                z0 / (vol[0] - 1), y0 / (vol[1] - 1), x0 / (vol[2] - 1),
            ], dtype=torch.float32).clamp(0.0, 1.0)
            assert torch.allclose(pos, expected, atol=1e-6)

    def test_every_patch_gets_identical_global_por(self):
        """global_por is a single volume-level scalar for all patches — the same
        semantics as training's vol_porosity.  Cleared for failure mode (b) on
        the GLOBAL channel."""
        sampler = _RecordingSampler()
        gen = _make_generator(sampler)
        gen._generate_latents(volume_shape=(64, 96, 96), target_porosity=0.05)
        assert len(set(sampler.g_por)) == 1


# ── 2. overlap-add blending arithmetic ────────────────────────────────────────


class TestOverlapBlendArithmetic:
    def test_constant_logits_preserved_in_interior(self):
        """Partition-of-unity: when every patch decodes to the same constant
        logit, interior voxels of the blended volume carry exactly that logit
        (xct = sigmoid(logit), mask = logit > 0).  Cleared for failure mode (c)
        for the CONTINUOUS field: no double-counting, the weighted mean is
        unbiased for identical overlapping patches."""
        fill = 0.2
        sampler = _RecordingSampler(fill_fn=lambda pos: fill)
        vae = _ConstVAE(patch_size=64)
        gen = _make_generator(sampler, vae=vae)
        xct, mask = gen.generate(volume_size_mm=(1.6, 1.6, 3.2), target_porosity=0.05)
        assert xct.shape == (64, 64, 128)

        from scipy.special import expit
        expected_xct = int(np.clip(expit(fill) * 255.0, 0, 255))
        interior = xct[10:50, 10:50, 10:110]
        assert np.all(np.abs(interior.astype(int) - expected_xct) <= 1)
        assert np.all(mask[10:50, 10:50, 10:110] == 255)  # logit 0.2 > 0 → pore

        # Zero-weight min-face planes are forced to 0 (documented artifact).
        assert np.all(xct[0] == 0) and np.all(mask[0] == 0)

    def test_mask_is_weighted_logit_vote_not_per_patch_decision(self):
        """AUDIT (failure mode c, mask channel): the final mask thresholds the
        Hann-weighted MEAN LOGIT at 0.  In overlap regions this is a weight
        vote between patches, not a union or a per-patch decision, so per-patch
        porosity does NOT carry linearly to volume porosity for the mask.

        Setup: x-overlapping patches at x0 = 0, 32, 64; patch A (x0=0) decodes
        constant logit +1 (pore everywhere), patches B and C (x0=32, 64)
        constant logit -3.  Along x in [32,64) the y/z window factors cancel,
        leaving
            mean_logit(x) = (wA(x)*1 + wB(x)*(-3)) / (wA(x)+wB(x)),
        wA(x)=sin^2(pi*x/64), wB(x)=sin^2(pi*(x-32)/64).
        """

        def fill_fn(pos):
            # pos[2] is x0/(128-1): 0.0 for patch A, ~0.252 / ~0.504 for B / C.
            return 1.0 if float(pos[2]) < 0.1 else -3.0

        sampler = _RecordingSampler(fill_fn=fill_fn)
        vae = _ConstVAE(patch_size=64)
        gen = _make_generator(sampler, vae=vae)
        _, mask = gen.generate(volume_size_mm=(1.6, 1.6, 3.2), target_porosity=0.05)

        z, y = 32, 32  # interior in z/y
        # A-only region: pore, exactly as patch A decided.
        assert mask[z, y, 8] == 255
        # Overlap, near patch A's fringe but where wA >> 3*wB: still pore.
        assert mask[z, y, 33] == 255
        # Overlap midpoint: wA == wB → mean logit = -1 → NOT pore, although
        # patch A alone claims pore at this voxel.  Pinned behavior.
        assert mask[z, y, 48] == 0
        # B/C-only region: not pore.
        assert mask[z, y, 100] == 0


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
        pos_frac=torch.rand(B, 3),
        global_por=torch.tensor([0.05, 0.05]),
        local_por=torch.tensor([0.02, 0.08]),
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

    def test_drop_por_silently_ignored_without_use_por_null(self):
        """AUDIT (failure mode e, inverse leakage): when a checkpoint has
        use_por_null=False (ldm01/ldm02) but the guided DDIM sampler passes
        drop_por=True for its eps_uncond pass, _build_cond IGNORES the drop
        (unet.py: `if drop_por is not None and self.cfg.use_por_null`).
        Consequence: eps_uncond == eps_por, the porosity-guidance term
        vanishes, and s_por has no effect — silently, with no warning."""
        model = _mini_model(use_por_null=False)
        inp = _mini_inputs()
        drop = torch.tensor([True, True])
        with torch.no_grad():
            out_none = model(**inp)
            out_drop = model(**inp, drop_por=drop)
        assert torch.equal(out_none, out_drop)  # drop request is a no-op


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

        AUDIT (failure mode a): for grid (3,3,3), sigma=1, global 0.05 the
        centre patch is conditioned to > _POR_MAX — outside the clamp range
        the sampler enforces for global_por and outside the bulk of the
        training patch-phi distribution."""
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
        """AUDIT (failure modes a/d): 'center'/'edges' maps are clipped to
        [0.001, 0.999] — NOT to the sampler's training range [0.002, 0.107].
        With sigma_norm=0.3 the centre patches receive several times the
        target; these OOD values reach por_mlp unclamped (see
        test_local_por_map_values_bypass_training_range_clamp)."""
        mod = _load_generate_volumes_module()
        m = mod._build_local_por_map(4, 8, 8, target_por=0.05, distribution="center")
        vals = np.array(list(m.values()))
        # Mean over patches is preserved (up to the clip):
        assert vals.mean() == pytest.approx(0.05, rel=0.05)
        # ... but the peak is far outside the training-range clamp.
        assert vals.max() > _POR_MAX
        # Pinned clip bounds:
        assert vals.min() >= 0.001 and vals.max() <= 0.999
