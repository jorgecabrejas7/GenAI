"""The ldm06 hybrid chunked sampler: neighbour sources, blended decode, seams.

Everything runs on a synthetic denoiser and decoder, so the numbers are exact
rather than plausible: the model records what it was handed and returns a fixed
epsilon, and the decoders emit either a constant (which makes the blend
normalisation checkable to the last bit) or a trilinear upsample of the latent
(which gives the seam metrics real structure to measure).  Geometry is the
production one — patch 64, latent 16, downsample 4 — over volumes of two or
three tiles per axis.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch
import torch.nn as nn

from poregen.diffusion.conditioning import (
    NB_EXISTS,
    NB_OOB,
    NB_UNKNOWN,
    NEIGHBOUR_DIRS,
)
from poregen.diffusion.noise_schedule import DDPMSchedule
from poregen.diffusion.sampler import (
    DDIMSampler,
    VolumeGenerator,
    region_noise_field,
    seam_discontinuity,
    theta_from_layup,
    window_origins,
    window_weight,
)

P, LAT, DS = 64, 16, 4      # production geometry: patch, latent cells, factor
C = 2
VOX_MM = 0.025


class _Cfg:
    z_channels = C


class _SpyModel(nn.Module):
    """Records every call and returns a fixed epsilon."""

    cfg = _Cfg()

    def __init__(self, eps_value: float = 0.0) -> None:
        super().__init__()
        self.eps_value = eps_value
        self.calls: list[dict] = []

    def forward(self, z_t, t, nb_latents, nb_avail, nb_t, cond_por, cond_depth,
                cond_dist6, cond_orient, cond_material, drop_por=None):
        self.calls.append({
            "t": int(t[0]),
            "z_t": z_t.detach().clone(),
            "nb_latents": nb_latents.detach().clone(),
            "nb_avail": nb_avail.detach().clone(),
            "nb_t": nb_t.detach().clone(),
            "cond_por": cond_por.detach().clone(),
            "cond_depth": cond_depth.detach().clone(),
            "cond_dist6": cond_dist6.detach().clone(),
            "cond_material": cond_material.detach().clone(),
            "drop_por": None if drop_por is None else drop_por.detach().clone(),
        })
        return torch.full_like(z_t, self.eps_value)


class _ConstVAE(nn.Module):
    """Decoder that emits a constant grey level and constant class logits."""

    def __init__(self, grey: float = 0.4, logits=(2.0, 0.5, -1.0)) -> None:
        super().__init__()
        self.grey = grey
        self.logits = logits

    def decoder(self, z):
        return z

    def xct_head(self, dec):
        b = dec.shape[0]
        return torch.full((b, 1, P, P, P), self.grey)

    def class_head(self, dec):
        b = dec.shape[0]
        out = torch.empty(b, 3, P, P, P)
        for k, v in enumerate(self.logits):
            out[:, k] = v
        return out


class _LatentVAE(nn.Module):
    """Decoder that upsamples the latent, so the decode carries real structure."""

    def decoder(self, z):
        return z

    def xct_head(self, dec):
        up = torch.nn.functional.interpolate(dec[:, :1], scale_factor=DS,
                                             mode="trilinear", align_corners=False)
        return up * 0.05 + 0.5

    def class_head(self, dec):
        up = torch.nn.functional.interpolate(dec[:, :1], scale_factor=DS,
                                             mode="trilinear", align_corners=False)
        return torch.cat([torch.zeros_like(up), up, -torch.ones_like(up)], dim=1)


def _generator(model, vae, chunk_tiles, n_steps=2, tiles=(2, 2, 2), **kw):
    dev = torch.device("cpu")
    sch = DDPMSchedule(T=100, device=dev)
    sampler = DDIMSampler(model, sch, dev, n_steps=n_steps,
                          s_por=kw.pop("s_por", 1.0), s_nb=kw.pop("s_nb", 1.0))
    theta = theta_from_layup(tiles[0] * P, [0, 45, 90], 19.6)
    gen = VolumeGenerator(
        sampler, vae, dev, patch_size=P, latent_size=LAT,
        latent_mean=0.0, latent_std=1.0, voxel_size_mm=VOX_MM,
        por_log_stats=(-3.0, 1.0), theta_deg=theta,
        chunk_tiles=chunk_tiles, **kw,
    )
    size_mm = tuple(t * P * VOX_MM for t in tiles)
    return gen, size_mm


# ── window helpers ───────────────────────────────────────────────────────────

class TestWindowGeometry:

    def test_origins_cover_the_whole_canvas(self):
        canvas, w, s = (48, 32, 32), 16, 8
        origins = window_origins(canvas, w, s)
        covered = np.zeros(canvas, bool)
        for o in origins:
            covered[o[0]:o[0] + w, o[1]:o[1] + w, o[2]:o[2] + w] = True
        assert covered.all()
        assert len(origins) == 5 * 3 * 3

    def test_a_canvas_that_cannot_be_tiled_is_refused(self):
        with pytest.raises(ValueError, match="cannot be covered"):
            window_origins((50, 48, 48), 16, 8)     # (50-16) % 8 != 0
        with pytest.raises(ValueError, match="cannot be covered"):
            window_origins((8, 48, 48), 16, 8)      # axis smaller than a window
        with pytest.raises(ValueError, match="stride_cells"):
            window_origins((48, 48, 48), 16, 20)    # stride > window

    def test_fusion_weight_is_strictly_positive_and_normalisable(self):
        w = window_weight(16)
        assert w.shape == (16, 16, 16)
        assert float(w.min()) > 0.0
        canvas = (48, 32, 32)
        acc = torch.zeros(canvas)
        for o in window_origins(canvas, 16, 8):
            acc[o[0]:o[0] + 16, o[1]:o[1] + 16, o[2]:o[2] + 16] += w
        assert float(acc.min()) > 0.0        # no cell divides by zero


# ── neighbour source logic ───────────────────────────────────────────────────

class TestNeighbourPlan:
    """`_neighbour_plan` decides where each of a window's six faces comes from."""

    @staticmethod
    def _plan(visible, origins, canvas=(64, 64, 64), ctx_lo=(0, 0, 0)):
        gen, _ = _generator(_SpyModel(), _ConstVAE(), (1, 1, 1))
        return gen._neighbour_plan(origins, visible, canvas, ctx_lo)

    def test_a_face_leaving_the_volume_is_oob(self):
        visible = np.ones((64, 64, 64), bool)
        states, slices = self._plan(visible, [(0, 0, 0)])
        for f, d in enumerate(NEIGHBOUR_DIRS):
            leaves = any(dd < 0 for dd in d)
            assert states[0, f] == (NB_OOB if leaves else NB_EXISTS)
            assert (slices[0][f] is None) == leaves

    def test_a_face_reaching_an_ungenerated_chunk_is_unknown(self):
        visible = np.zeros((64, 64, 64), bool)
        visible[:32] = True                       # only the low-z half is done
        states, slices = self._plan(visible, [(16, 16, 16)])
        by_dir = dict(zip(NEIGHBOUR_DIRS, states[0]))
        assert by_dir[(1, 0, 0)] == NB_UNKNOWN    # +z reaches z=32.. which is not
        assert by_dir[(-1, 0, 0)] == NB_EXISTS    # -z reaches z=0.. which is
        assert by_dir[(0, 1, 0)] == NB_EXISTS
        assert slices[0][NEIGHBOUR_DIRS.index((1, 0, 0))] is None

    def test_a_block_straddling_two_visible_chunks_is_exists(self):
        """Both sources sit at the same noise level, so a straddle is coherent."""
        visible = np.zeros((64, 64, 64), bool)
        visible[:40] = True
        states, _ = self._plan(visible, [(16, 16, 16)])
        # +z block spans z 32..48: 32..39 visible, 40..47 not -> UNKNOWN
        assert states[0][NEIGHBOUR_DIRS.index((1, 0, 0))] == NB_UNKNOWN
        visible[:48] = True
        states, _ = self._plan(visible, [(16, 16, 16)])
        assert states[0][NEIGHBOUR_DIRS.index((1, 0, 0))] == NB_EXISTS

    def test_out_of_volume_beats_ungenerated(self):
        """OOB says the specimen ENDS; UNKNOWN says 'not yet'.  Never both."""
        visible = np.zeros((64, 64, 64), bool)
        states, _ = self._plan(visible, [(0, 0, 0)])
        assert states[0][NEIGHBOUR_DIRS.index((-1, 0, 0))] == NB_OOB

    def test_slices_are_offset_into_the_context_canvas(self):
        visible = np.ones((64, 64, 64), bool)
        states, slices = self._plan(visible, [(32, 32, 32)], ctx_lo=(16, 16, 16))
        sl = slices[0][NEIGHBOUR_DIRS.index((1, 0, 0))]
        assert sl[0].start == 32 + LAT - 16
        assert sl[0].stop - sl[0].start == LAT


class TestNeighbourSourcesEndToEnd:

    def test_states_seen_by_the_model_match_the_geometry(self):
        """chunk_tiles (1,1,1) on a 2x2x2 volume: raster order, 3 faces OOB."""
        model = _SpyModel()
        gen, size_mm = _generator(model, _ConstVAE(), (1, 1, 1))
        gen.generate(volume_size_mm=size_mm, target_porosity=0.03,
                     autocast_dtype=torch.float32, window_batch=8)
        states = torch.cat([c["nb_avail"].reshape(-1) for c in model.calls])
        n_steps = len(gen.sampler.timesteps) - 1
        # 8 chunks x 1 window x 6 faces per timestep; every corner tile has 3
        # faces leaving the volume, and the other 3 split EXISTS/UNKNOWN by
        # raster order (12 of each over the grid).
        assert int((states == NB_OOB).sum()) == 8 * 3 * n_steps
        assert int((states == NB_EXISTS).sum()) == 12 * n_steps
        assert int((states == NB_UNKNOWN).sum()) == 12 * n_steps

    def test_one_chunk_over_the_volume_never_says_unknown(self):
        model = _SpyModel()
        gen, size_mm = _generator(model, _ConstVAE(), (2, 2, 2))
        gen.generate(volume_size_mm=size_mm, target_porosity=0.03,
                     autocast_dtype=torch.float32, window_batch=64)
        states = torch.cat([c["nb_avail"].reshape(-1) for c in model.calls])
        assert int((states == NB_UNKNOWN).sum()) == 0
        assert int((states == NB_EXISTS).sum()) > 0

    def test_nb_t_is_the_canvas_timestep_for_exists_and_zero_otherwise(self):
        model = _SpyModel()
        gen, size_mm = _generator(model, _ConstVAE(), (2, 2, 2))
        gen.generate(volume_size_mm=size_mm, target_porosity=0.03,
                     autocast_dtype=torch.float32, window_batch=64)
        for call in model.calls:
            ex = call["nb_avail"] == NB_EXISTS
            assert bool((call["nb_t"][ex] == call["t"]).all())
            assert bool((call["nb_t"][~ex] == 0).all())

    def test_an_in_chunk_neighbour_is_the_canvas_block_itself(self):
        """The -z face of the window at cell 16 IS the window at cell 0."""
        model = _SpyModel()
        gen, size_mm = _generator(model, _ConstVAE(), (3, 3, 3), tiles=(3, 3, 3))
        gen.generate(volume_size_mm=size_mm, target_porosity=0.03,
                     autocast_dtype=torch.float32, window_batch=64)
        first = [c for c in model.calls if c["t"] == model.calls[0]["t"]]
        # Windows are emitted in the order window_origins produces them:
        # z outer, then y, then x, at cell stride 8 over a 48-cell canvas.
        origins = window_origins((48, 48, 48), LAT, 8)
        xw = torch.cat([c["z_t"] for c in first])
        nb = torch.cat([c["nb_latents"] for c in first])
        idx = {o: i for i, o in enumerate(origins)}
        j = idx[(16, 0, 0)]
        k = idx[(0, 0, 0)]
        f = NEIGHBOUR_DIRS.index((-1, 0, 0))
        assert torch.allclose(nb[j, f], xw[k], atol=1e-6)

    def test_a_finished_chunk_is_re_noised_afresh_at_every_timestep(self):
        model = _SpyModel()
        gen, size_mm = _generator(model, _ConstVAE(), (1, 1, 1), n_steps=3)
        gen.generate(volume_size_mm=size_mm, target_porosity=0.03,
                     autocast_dtype=torch.float32, window_batch=8)
        # Second chunk (tile 0,0,1): its -x face is chunk 0, already finished.
        n_steps = len(gen.sampler.timesteps) - 1
        chunk1 = model.calls[n_steps:2 * n_steps]
        f = NEIGHBOUR_DIRS.index((0, 0, -1))
        assert all(int(c["nb_avail"][0, f]) == NB_EXISTS for c in chunk1)
        blocks = [c["nb_latents"][0, f] for c in chunk1]
        assert float(blocks[0].abs().max()) > 0.0
        for a, b in zip(blocks, blocks[1:]):
            assert not torch.allclose(a, b)      # fresh noise every step

    def test_oob_and_unknown_faces_arrive_zeroed(self):
        model = _SpyModel()
        gen, size_mm = _generator(model, _ConstVAE(), (1, 1, 1))
        gen.generate(volume_size_mm=size_mm, target_porosity=0.03,
                     autocast_dtype=torch.float32, window_batch=8)
        for call in model.calls:
            blank = call["nb_avail"] != NB_EXISTS
            assert float(call["nb_latents"][blank].abs().max()) == 0.0


# ── per-window conditioning ──────────────────────────────────────────────────

class TestWindowConditioning:

    def test_porosity_is_clamped_to_the_training_range(self):
        model = _SpyModel()
        gen, size_mm = _generator(model, _ConstVAE(), (2, 2, 2))
        gen.generate(volume_size_mm=size_mm, target_porosity=0.9,
                     autocast_dtype=torch.float32, window_batch=64)
        from poregen.diffusion.conditioning import POR_MAX, porosity_to_cond
        expected = float(porosity_to_cond(POR_MAX, (-3.0, 1.0)))
        assert all(float(c["cond_por"].max()) == pytest.approx(expected)
                   for c in model.calls)

    def test_dist6_is_zero_on_the_faces_where_the_volume_ends(self):
        model = _SpyModel()
        gen, size_mm = _generator(model, _ConstVAE(), (2, 2, 2))
        gen.generate(volume_size_mm=size_mm, target_porosity=0.03,
                     autocast_dtype=torch.float32, window_batch=64)
        d6 = torch.cat([c["cond_dist6"] for c in model.calls])
        # The specimen box is the volume, so the corner windows are flush.
        assert float(d6.min()) == pytest.approx(0.0)
        assert float(d6.max()) <= 1.0

    def test_the_default_material_map_is_all_material(self):
        model = _SpyModel()
        gen, size_mm = _generator(model, _ConstVAE(), (2, 2, 2))
        gen.generate(volume_size_mm=size_mm, target_porosity=0.03,
                     autocast_dtype=torch.float32, window_batch=64)
        assert all(float(c["cond_material"].min()) == 1.0 for c in model.calls)

    def test_a_painted_material_map_reaches_the_windows(self):
        model = _SpyModel()
        gen, size_mm = _generator(model, _ConstVAE(), (2, 2, 2))
        cells = (2 * P // DS,) * 3
        painted = np.zeros(cells, dtype=np.float32)
        painted[: cells[0] // 2] = 1.0          # material in the low-z half only
        gen.generate(volume_size_mm=size_mm, target_porosity=0.03,
                     material_map=painted, autocast_dtype=torch.float32,
                     window_batch=64)
        seen = torch.cat([c["cond_material"].reshape(-1) for c in model.calls])
        assert float(seen.min()) == 0.0 and float(seen.max()) == 1.0

    def test_a_material_map_of_the_wrong_shape_is_refused(self):
        gen, size_mm = _generator(_SpyModel(), _ConstVAE(), (2, 2, 2))
        with pytest.raises(ValueError, match="material_map has shape"):
            gen.generate(volume_size_mm=size_mm, material_map=np.ones((4, 4, 4)),
                         autocast_dtype=torch.float32)


# ── the requested porosity field over a window's footprint ───────────────────

class TestWindowPorosityFootprint:
    """A window steps by 32 voxels but the field is defined on 64-voxel tiles.

    A window therefore straddles up to eight tiles, and its request is the RAW
    tile field averaged over its own footprint, weighted by the volume each
    tile covers.  Taking the tile under the window CENTRE instead handed a
    window that spans 0.01 and 0.05 material one of the two extremes.
    """

    TILES = (2, 2, 2)
    CELLS = (TILES[0] * P // DS,) * 3

    def _cond_por(self, por_map, g_origin, por_default=0.03):
        gen, _ = _generator(_SpyModel(), _ConstVAE(), (2, 2, 2), tiles=self.TILES)
        cond = gen._window_conditioning(
            [g_origin], DS, por_default, por_map,
            np.ones(self.CELLS, dtype=np.float32),
            (0, 0, 0), (self.TILES[0] * P,) * 3,
        )
        return float(cond["por"][0])

    @staticmethod
    def _expected(phi):
        from poregen.diffusion.conditioning import porosity_to_cond
        return float(porosity_to_cond(phi, (-3.0, 1.0)))

    def test_a_window_straddling_two_tiles_takes_their_footprint_mean(self):
        """Half in a 0.01 tile and half in a 0.05 tile is a request for 0.03."""
        por_map = {(iz, iy, ix): (0.01 if ix == 0 else 0.05)
                   for iz in range(2) for iy in range(2) for ix in range(2)}
        # Voxel origin (0, 0, 32): x 32..63 is tile 0, x 64..95 is tile 1.
        got = self._cond_por(por_map, (0, 0, P // 2 // DS))
        assert got == pytest.approx(self._expected(0.03), rel=1e-6)

    def test_a_window_inside_one_tile_is_that_tile(self):
        """The aligned case must not move: it is one tile, so it is its value."""
        por_map = {(iz, iy, ix): (0.01 if ix == 0 else 0.05)
                   for iz in range(2) for iy in range(2) for ix in range(2)}
        assert self._cond_por(por_map, (0, 0, 0)) == pytest.approx(
            self._expected(0.01), rel=1e-6)
        assert self._cond_por(por_map, (0, 0, P // DS)) == pytest.approx(
            self._expected(0.05), rel=1e-6)

    def test_the_weights_are_volumes_not_lengths(self):
        """Straddling in y AND x covers four tiles at a quarter each."""
        vals = {(0, 0): 0.01, (0, 1): 0.03, (1, 0): 0.05, (1, 1): 0.07}
        por_map = {(iz, iy, ix): vals[(iy, ix)]
                   for iz in range(2) for iy in range(2) for ix in range(2)}
        got = self._cond_por(por_map, (0, P // 2 // DS, P // 2 // DS))
        assert got == pytest.approx(self._expected(0.04), rel=1e-6)

    def test_the_mean_is_taken_on_raw_porosity_not_on_the_transform(self):
        """`porosity_to_cond` is a log: averaging after it is a different number."""
        por_map = {(iz, iy, ix): (0.01 if ix == 0 else 0.05)
                   for iz in range(2) for iy in range(2) for ix in range(2)}
        got = self._cond_por(por_map, (0, 0, P // 2 // DS))
        mean_of_transform = 0.5 * (self._expected(0.01) + self._expected(0.05))
        assert got != pytest.approx(mean_of_transform, rel=1e-3)

    def test_a_footprint_mean_outside_the_training_range_is_still_clamped(self):
        from poregen.diffusion.conditioning import POR_MAX

        por_map = {(iz, iy, ix): 0.5
                   for iz in range(2) for iy in range(2) for ix in range(2)}
        got = self._cond_por(por_map, (0, 0, P // 2 // DS))
        assert got == pytest.approx(self._expected(POR_MAX), rel=1e-6)

    def test_a_tile_missing_from_the_field_falls_back_per_tile(self):
        por_map = {(0, 0, 0): 0.01}          # tile (0,0,1) is not requested
        got = self._cond_por(por_map, (0, 0, P // 2 // DS), por_default=0.05)
        assert got == pytest.approx(self._expected(0.03), rel=1e-6)

    def test_the_footprint_mean_helper_weights_partial_tiles(self):
        """Unaligned origins are weighted by overlap, not counted equally."""
        from poregen.diffusion.sampler import window_tile_mean

        field = {(0, 0, 0): 0.01, (0, 0, 1): 0.05}
        # x 16..79: 48 voxels in tile 0, 16 voxels in tile 1.
        got = window_tile_mean((0, 0, 16), P, field, 0.0)
        assert got == pytest.approx(0.75 * 0.01 + 0.25 * 0.05)
        assert window_tile_mean((0, 0, 0), P, field, 0.0) == pytest.approx(0.01)
        assert window_tile_mean((0, 0, P), P, field, 0.0) == pytest.approx(0.05)


# ── the request is MATERIAL porosity, the model was trained on FULL-patch phi ─

class TestMaterialPorosityRescaling:
    """A request of phi is pore/material; ``cond_por`` is pore/64³.

    Training conditions on ``phi = pore / 64**3`` — the whole patch, air
    included — while eval measures ``pore / material``.  A window that is half
    outside the specimen must therefore be asked for half the requested
    material porosity, or the delivered volume is scored against a target that
    was never requested.
    """

    TILES = (2, 2, 2)
    CELLS = (TILES[0] * P // DS,) * 3

    def _cond_por(self, material_map, g_origin=(0, 0, 0), por_default=0.03,
                  por_map=None):
        gen, _ = _generator(_SpyModel(), _ConstVAE(), (2, 2, 2), tiles=self.TILES)
        cond = gen._window_conditioning(
            [g_origin], DS, por_default, por_map,
            np.asarray(material_map, dtype=np.float32),
            (0, 0, 0), (self.TILES[0] * P,) * 3,
        )
        return float(cond["por"][0])

    @staticmethod
    def _expected(phi):
        from poregen.diffusion.conditioning import porosity_to_cond
        return float(porosity_to_cond(phi, (-3.0, 1.0)))

    def _half_material(self):
        m = np.zeros(self.CELLS, dtype=np.float32)
        m[: LAT // 2] = 1.0            # the low-z half of every window is solid
        return m

    def test_a_half_material_window_asks_for_half_the_requested_phi(self):
        """Material fraction 0.5, request 0.03 -> full-patch phi 0.015."""
        got = self._cond_por(self._half_material(), por_default=0.03)
        assert got == pytest.approx(self._expected(0.015), rel=1e-6)

    def test_a_full_material_window_is_unchanged(self):
        got = self._cond_por(np.ones(self.CELLS, dtype=np.float32),
                             por_default=0.03)
        assert got == pytest.approx(self._expected(0.03), rel=1e-6)

    def test_partial_cells_count_by_volume(self):
        """The envelope fraction per cell is a volume, so 0.25 everywhere is 0.25."""
        got = self._cond_por(np.full(self.CELLS, 0.25, dtype=np.float32),
                             por_default=0.04)
        assert got == pytest.approx(self._expected(0.01), rel=1e-6)

    def test_the_scaling_is_applied_to_the_footprint_mean_of_the_field(self):
        """It multiplies the tile-field mean, not the per-tile values."""
        por_map = {(iz, iy, ix): (0.01 if ix == 0 else 0.05)
                   for iz in range(2) for iy in range(2) for ix in range(2)}
        got = self._cond_por(self._half_material(), g_origin=(0, 0, P // 2 // DS),
                             por_map=por_map)
        assert got == pytest.approx(self._expected(0.5 * 0.03), rel=1e-6)

    def test_the_scaling_comes_before_the_clip(self):
        """0.2 * 0.5 = 0.1 is inside the training range; clipping first is not."""
        from poregen.diffusion.conditioning import POR_MAX

        got = self._cond_por(self._half_material(), por_default=0.2)
        assert got == pytest.approx(self._expected(0.1), rel=1e-6)
        assert got != pytest.approx(self._expected(0.5 * POR_MAX), rel=1e-3)

    def test_the_scaling_comes_before_the_log_transform(self):
        """`porosity_to_cond` is a log, so scaling after it is a different number."""
        got = self._cond_por(self._half_material(), por_default=0.03)
        assert got != pytest.approx(0.5 * self._expected(0.03), rel=1e-3)

    def test_an_all_air_window_falls_to_the_bottom_of_the_training_range(self):
        from poregen.diffusion.conditioning import POR_MIN

        got = self._cond_por(np.zeros(self.CELLS, dtype=np.float32),
                             por_default=0.03)
        assert got == pytest.approx(self._expected(POR_MIN), rel=1e-6)


# ── CFG ──────────────────────────────────────────────────────────────────────

class TestGuidance:

    def test_unguided_takes_one_pass_per_window(self):
        model = _SpyModel()
        gen, size_mm = _generator(model, _ConstVAE(), (2, 2, 2))
        gen.generate(volume_size_mm=size_mm, autocast_dtype=torch.float32,
                     window_batch=64)
        assert all(c["drop_por"] is None for c in model.calls)

    def test_guided_takes_the_three_nested_passes(self):
        model = _SpyModel()
        gen, size_mm = _generator(model, _ConstVAE(), (2, 2, 2),
                                  s_por=1.5, s_nb=2.0)
        gen.generate(volume_size_mm=size_mm, autocast_dtype=torch.float32,
                     window_batch=64)
        n_steps = len(gen.sampler.timesteps) - 1
        assert len(model.calls) == 3 * n_steps
        uncond, por, full = model.calls[0], model.calls[1], model.calls[2]
        assert uncond["drop_por"] is not None and bool(uncond["drop_por"].all())
        assert por["drop_por"] is None and full["drop_por"] is None
        # The two null arms carry no neighbour information at all.
        assert bool((uncond["nb_avail"] == NB_UNKNOWN).all())
        assert bool((por["nb_avail"] == NB_UNKNOWN).all())
        assert bool((uncond["nb_t"] == 0).all())
        assert not bool((full["nb_avail"] == NB_UNKNOWN).all())


# ── overlapped decode ────────────────────────────────────────────────────────

class TestBlendedDecode:

    def test_a_constant_decoder_gives_a_constant_volume(self):
        """Weight normalisation: every voxel divides by its own total weight."""
        vae = _ConstVAE(grey=0.4, logits=(2.0, 0.5, -1.0))
        gen, size_mm = _generator(_SpyModel(), vae, (2, 2, 2))
        xct, label, stats = gen.generate(volume_size_mm=size_mm,
                                         autocast_dtype=torch.float32,
                                         window_batch=64)
        assert xct.shape == (2 * P,) * 3 and xct.dtype == np.uint8
        assert np.all(xct == round(0.4 * 255))
        assert np.all(label == 0)                       # argmax of (2.0, .5, -1)

    def test_the_argmax_follows_the_blended_logits(self):
        vae = _ConstVAE(grey=0.4, logits=(0.0, 3.0, 1.0))
        gen, size_mm = _generator(_SpyModel(), vae, (2, 2, 2))
        _, label, _ = gen.generate(volume_size_mm=size_mm,
                                   autocast_dtype=torch.float32, window_batch=64)
        assert np.all(label == 1)

    def test_class_probabilities_are_a_simplex(self):
        vae = _ConstVAE(grey=0.4, logits=(1.0, 0.0, -2.0))
        gen, size_mm = _generator(_SpyModel(), vae, (2, 2, 2))
        out = gen.generate(volume_size_mm=size_mm, autocast_dtype=torch.float32,
                           window_batch=64, return_class_probs=True)
        assert len(out) == 4
        probs = out[3]
        assert probs.shape == (3, 2 * P, 2 * P, 2 * P)
        assert np.allclose(probs.sum(axis=0), 1.0, atol=1e-5)

    def test_decode_windows_overlap_by_the_requested_stride(self):
        """stride 64 would be plain tiling; stride 32 must halve the step."""
        gen, _ = _generator(_SpyModel(), _ConstVAE(), (2, 2, 2), decode_stride=32)
        assert len(window_origins((32, 32, 32), LAT, gen.decode_stride // DS)) == 27
        gen64, _ = _generator(_SpyModel(), _ConstVAE(), (2, 2, 2), decode_stride=64)
        assert len(window_origins((32, 32, 32), LAT, gen64.decode_stride // DS)) == 8

    def test_a_stride_that_is_not_a_cell_multiple_is_refused(self):
        with pytest.raises(ValueError, match="decode_stride"):
            _generator(_SpyModel(), _ConstVAE(), (2, 2, 2), decode_stride=6)
        with pytest.raises(ValueError, match="window_stride"):
            _generator(_SpyModel(), _ConstVAE(), (2, 2, 2), window_stride=48)


# ── seam diagnostics ─────────────────────────────────────────────────────────

class TestSeamMetrics:

    def test_a_coherent_volume_has_ratio_one(self):
        rng = np.random.default_rng(0)
        vol = rng.standard_normal((128, 128, 128)).astype(np.float32)
        m = seam_discontinuity(vol, 64, prefix="s")
        assert m["s_ratio"] == pytest.approx(1.0, abs=0.05)

    def test_independent_blocks_are_flagged(self):
        rng = np.random.default_rng(1)
        vol = rng.standard_normal((128, 128, 128)).astype(np.float32) * 0.01
        vol[64:] += 5.0                       # a step exactly on the seam plane
        m = seam_discontinuity(vol, 64, prefix="s")
        assert m["s_z_ratio"] > 50.0

    def test_window_and_chunk_periods_count_different_planes(self):
        vol = np.zeros((384, 128, 128), np.float32)
        win = seam_discontinuity(vol, 64, prefix="w")
        chunk = seam_discontinuity(vol, (192, 64, 64), prefix="c",
                                   interior_exclude=64)
        assert win["w_z_planes"] == 5          # 64,128,192,256,320
        assert chunk["c_z_planes"] == 1        # 192 only
        assert win["w_y_planes"] == chunk["c_y_planes"] == 1

    def test_the_chunk_metric_uses_the_window_interior_baseline(self):
        """Both ratios must be judged against the same slice-to-slice noise."""
        rng = np.random.default_rng(2)
        vol = rng.standard_normal((384, 128, 128)).astype(np.float32)
        win = seam_discontinuity(vol, 64, prefix="w")
        chunk = seam_discontinuity(vol, (192, 64, 64), prefix="c",
                                   interior_exclude=64)
        assert win["w_z_interior_mad"] == pytest.approx(chunk["c_z_interior_mad"])

    def test_generate_reports_both_periods_on_grey_and_pore_logit(self):
        gen, size_mm = _generator(_SpyModel(), _LatentVAE(), (1, 1, 1),
                                  tiles=(2, 2, 2))
        _, _, stats = gen.generate(volume_size_mm=size_mm,
                                   autocast_dtype=torch.float32, window_batch=8)
        for key in ("seam_xct_ratio", "seam_pore_ratio",
                    "seam_chunk_xct_ratio", "seam_chunk_pore_ratio"):
            assert key in stats
        assert stats["chunk_tiles"] == [1, 1, 1]
        assert stats["window_stride"] == 32
        assert stats["ddim_steps"] == len(gen.sampler.timesteps) - 1

    def test_a_3d_volume_is_required(self):
        with pytest.raises(ValueError, match="3-D volume"):
            seam_discontinuity(np.zeros((4, 4)), 2)


# ── stats and outputs ────────────────────────────────────────────────────────

def test_stats_carry_the_self_audit_and_the_geometry():
    vae = _ConstVAE(grey=0.4, logits=(0.0, 3.0, 1.0))
    gen, size_mm = _generator(_SpyModel(), vae, (2, 2, 2))
    _, label, stats = gen.generate(volume_size_mm=size_mm, target_porosity=0.04,
                                   autocast_dtype=torch.float32, window_batch=64)
    assert stats["target_porosity"] == pytest.approx(0.04)
    assert stats["conditioned_porosity"] == pytest.approx(0.04)
    assert stats["actual_label_porosity"] == pytest.approx(float((label == 1).mean()))
    assert stats["actual_label_air"] == pytest.approx(float((label == 2).mean()))
    assert stats["volume_shape"] == [2 * P] * 3


def test_a_volume_smaller_than_one_tile_is_refused():
    gen, _ = _generator(_SpyModel(), _ConstVAE(), (1, 1, 1))
    with pytest.raises(ValueError, match="at least one"):
        gen.generate(volume_size_mm=(0.5, 0.5, 0.5), autocast_dtype=torch.float32)


def test_a_vae_without_a_class_head_is_refused():
    class _MaskVAE(nn.Module):
        def decoder(self, z):
            return z

        def xct_head(self, dec):
            return torch.zeros(dec.shape[0], 1, P, P, P)

    gen, size_mm = _generator(_SpyModel(), _MaskVAE(), (2, 2, 2))
    with pytest.raises(TypeError, match="class_head"):
        gen.generate(volume_size_mm=size_mm, autocast_dtype=torch.float32,
                     window_batch=64)


# ── seeding ──────────────────────────────────────────────────────────────────

class _EchoModel(nn.Module):
    """Returns a scaled copy of its input, so the sample depends on the noise.

    A model that ignores ``z_t`` would make every seeding test pass by
    accident: the volume would be identical whatever noise the chain started
    from, and a seed that was silently dropped would look reproducible.
    """

    cfg = _Cfg()

    def forward(self, z_t, t, nb_latents, nb_avail, nb_t, cond_por, cond_depth,
                cond_dist6, cond_orient, cond_material, drop_por=None):
        return z_t * 0.5


class TestSeeding:
    """Every random draw of the reverse process comes from the local generator."""

    @staticmethod
    def _run(seed, tiles=(2, 2, 2), chunk_tiles=(1, 1, 1)):
        gen, size_mm = _generator(_EchoModel(), _LatentVAE(), chunk_tiles,
                                  n_steps=3, tiles=tiles)
        return gen.generate(volume_size_mm=size_mm, target_porosity=0.03,
                            autocast_dtype=torch.float32, window_batch=8,
                            return_class_probs=True, seed=seed)

    def test_the_same_seed_gives_a_bit_identical_volume(self):
        xct_a, label_a, stats_a, probs_a = self._run(101)
        xct_b, label_b, stats_b, probs_b = self._run(101)
        np.testing.assert_array_equal(xct_a, xct_b)
        np.testing.assert_array_equal(label_a, label_b)
        # Bit-identical, not merely close: the probabilities are the decoder's
        # own floats, so any divergence in the chain shows up here first.
        np.testing.assert_array_equal(probs_a, probs_b)
        assert stats_a["seed"] == stats_b["seed"] == 101

    def test_a_different_seed_gives_a_different_volume(self):
        """Guards the test above: it must be the seed that fixes the result."""
        _, _, _, probs_a = self._run(101)
        _, _, _, probs_b = self._run(202)
        assert not np.array_equal(probs_a, probs_b)

    def test_the_re_noising_draw_is_seeded_too(self):
        """Several chunks, so the finished-chunk re-noising actually happens.

        With one chunk per tile the canvas is re-noised at every timestep of
        every chunk after the first; a seed that only reached the initial noise
        would leave those draws on the global generator and this pair would
        differ.
        """
        _, _, _, probs_a = self._run(303, tiles=(2, 2, 2), chunk_tiles=(1, 1, 1))
        _, _, _, probs_b = self._run(303, tiles=(2, 2, 2), chunk_tiles=(1, 1, 1))
        np.testing.assert_array_equal(probs_a, probs_b)

    def test_generate_leaves_the_global_generator_untouched(self):
        """A seeded generation must not consume or reset the caller's stream."""
        torch.manual_seed(7)
        before_state = torch.get_rng_state()
        expected = torch.randn(4)

        torch.manual_seed(7)
        self._run(101)
        after_state = torch.get_rng_state()
        assert torch.equal(before_state, after_state)
        torch.testing.assert_close(torch.randn(4), expected)

    def test_an_unseeded_generation_says_so(self):
        gen, size_mm = _generator(_EchoModel(), _LatentVAE(), (2, 2, 2), n_steps=2)
        _, _, stats = gen.generate(volume_size_mm=size_mm, target_porosity=0.03,
                                   autocast_dtype=torch.float32, window_batch=8)
        assert stats["seed"] is None


def test_sample_batch_takes_the_same_generator():
    """The patch-level sampler's one draw is seedable the same way."""
    from poregen.diffusion.noise_schedule import DDPMSchedule

    dev = torch.device("cpu")
    sampler = DDIMSampler(_EchoModel(), DDPMSchedule(T=100, device=dev), dev, n_steps=3)
    args = (
        torch.zeros(2, 6, C, LAT, LAT, LAT),
        torch.full((2, 6), NB_OOB, dtype=torch.long),
        torch.zeros(2, 6, dtype=torch.long),
        torch.zeros(2), torch.zeros(2), torch.zeros(2, 6),
        torch.zeros(2, 2, LAT, LAT, LAT), torch.zeros(2, 1, LAT, LAT, LAT),
    )

    def draw(seed):
        g = torch.Generator(device=dev)
        g.manual_seed(seed)
        return sampler.sample_batch(*args, autocast_dtype=torch.float32, generator=g)

    torch.testing.assert_close(draw(11), draw(11), rtol=0, atol=0)
    assert not torch.equal(draw(11), draw(12))


# ── the region-relative noise frame ──────────────────────────────────────────

def _denoise_canvas(gen, offset_vox, seed, tiles):
    """Run the latent side only, with the request translated by ``offset_vox``."""
    shape = tuple(t * P for t in tiles)
    g = torch.Generator(device="cpu")
    g.manual_seed(seed)
    gen._generate_latents(
        shape,
        target_porosity=0.03,
        local_por_map=None,
        material_map=None,
        specimen_box=((0, 0, 0), shape),
        autocast_dtype=torch.float32,
        window_batch=8,
        offset_cells=tuple(o // DS for o in offset_vox),
        generator=g,
    )


def _initial_canvas_noise(offset_vox, seed, tiles=(2, 2, 2), n_steps=2):
    """The noise every chunk started from, reassembled onto the canvas.

    ``chunk_tiles`` is one tile, so a chunk holds exactly one window and the
    latent the model sees at the first timestep of chunk *k* IS that chunk's
    initial noise.  Chunks run in raster order.
    """
    model = _SpyModel()
    gen, _ = _generator(model, _ConstVAE(), (1, 1, 1), n_steps=n_steps, tiles=tiles)
    _denoise_canvas(gen, offset_vox, seed, tiles)
    cells = tuple(t * P // DS for t in tiles)
    canvas = torch.zeros(C, *cells)
    origins = [(z, y, x)
               for z in range(0, cells[0], LAT)
               for y in range(0, cells[1], LAT)
               for x in range(0, cells[2], LAT)]
    for k, o in enumerate(origins):
        canvas[:, o[0]:o[0] + LAT, o[1]:o[1] + LAT, o[2]:o[2] + LAT] = (
            model.calls[k * n_steps]["z_t"][0]
        )
    return canvas


def _renoise_draws(offset_vox, seed, tiles=(2, 2, 2), n_steps=2):
    """Every noise tensor the finished-chunk re-noising handed to `q_sample`."""
    model = _SpyModel()
    gen, _ = _generator(model, _ConstVAE(), (1, 1, 1), n_steps=n_steps, tiles=tiles)
    schedule = gen.sampler.schedule
    original = schedule.q_sample
    drawn: list[torch.Tensor] = []

    def spy(x0, t, noise=None):
        drawn.append(noise.detach().clone())
        return original(x0, t, noise=noise)

    schedule.q_sample = spy      # `.to()` returns self, so the patch survives
    _denoise_canvas(gen, offset_vox, seed, tiles)
    return drawn


class TestRegionRelativeNoiseFrame:
    """Translating the request must translate its noise with it.

    The assembly-offset assessment generates one region twice at two places in
    a bigger canvas and asks whether the answer moved.  With the noise anchored
    to the CANVAS the two runs differ in the assembly geometry AND in the noise
    realisation, so the pore Dice across the pair cannot attribute the
    difference to either — it is not measuring what it claims to.
    """

    def test_the_field_is_one_draw_translated(self):
        def field(off):
            g = torch.Generator(device="cpu")
            g.manual_seed(5)
            return region_noise_field(C, (12, 12, 12), off, torch.device("cpu"), g)

        base = field((0, 0, 0))
        for d in (2, 3):
            moved = field((d, d, d))
            assert torch.equal(moved[:, d:, d:, d:], base[:, :-d, :-d, :-d])
            assert not torch.equal(moved, base)
            # A roll is a permutation, so the field is still the same draw and
            # therefore still exactly iid standard normal.
            assert torch.equal(torch.sort(moved.flatten()).values,
                               torch.sort(base.flatten()).values)

    @pytest.mark.parametrize("offset", [16, 32])
    def test_two_offsets_start_the_region_from_identical_noise(self, offset):
        """A 64-voxel region placed at 0 and at ``offset`` in a 128-voxel canvas."""
        a = _initial_canvas_noise((0, 0, 0), 101)
        b = _initial_canvas_noise((offset,) * 3, 101)
        d, r = offset // DS, P // DS
        assert torch.equal(a[:, :r, :r, :r], b[:, d:d + r, d:d + r, d:d + r])
        # Guard: the canvas-anchored draw makes the two runs bit-identical, and
        # then the check above passes for the wrong reason.
        assert not torch.equal(a, b)

    def test_the_re_noising_draws_use_the_same_frame(self):
        """The fresh noise for finished chunks is in the request's frame too.

        Both runs re-noise the SAME context blocks in canvas coordinates, so a
        region-relative draw makes one recorded field the shifted copy of the
        other; a canvas-relative draw makes them equal.
        """
        a = _renoise_draws((0, 0, 0), 202)
        b = _renoise_draws((32, 32, 32), 202)
        d = 32 // DS
        assert a and len(a) == len(b)
        for na, nb in zip(a, b):
            assert na.shape == nb.shape
            assert torch.equal(nb[..., d:, d:, d:], na[..., :-d, :-d, :-d])
        assert not torch.equal(a[0], b[0])

    def test_the_request_offset_reaches_the_sampler(self):
        def run(offset):
            gen, size_mm = _generator(_EchoModel(), _LatentVAE(), (1, 1, 1),
                                      n_steps=2, tiles=(1, 1, 1))
            return gen.generate(
                volume_size_mm=size_mm, target_porosity=0.03,
                autocast_dtype=torch.float32, window_batch=8,
                return_class_probs=True, seed=99, request_offset=offset,
            )[3]

        assert not np.array_equal(run((0, 0, 0)), run((32, 32, 32)))
        np.testing.assert_array_equal(run((32, 32, 32)), run((32, 32, 32)))

    def test_an_offset_off_the_latent_grid_is_refused(self):
        gen, size_mm = _generator(_SpyModel(), _ConstVAE(), (2, 2, 2))
        with pytest.raises(ValueError, match="request_offset"):
            gen.generate(volume_size_mm=size_mm, autocast_dtype=torch.float32,
                         window_batch=64, request_offset=(2, 0, 0))


# ── the default specimen envelope ────────────────────────────────────────────

class TestDefaultMaterialMap:
    """``cond_material`` is the envelope FRACTION per latent cell.

    A cell the box crosses is partly specimen and partly air.  Rounding the box
    to whole cells would tell the model the surface cells are entirely one or
    the other, which is exactly the edge it was asked to render.  The fraction
    is the cell-box intersection volume over the cell volume, in closed form.
    """

    def _gen(self):
        return _generator(_SpyModel(), _ConstVAE(), (2, 2, 2))[0]

    def test_a_box_flush_with_the_cell_grid_is_binary(self):
        m = self._gen()._default_material_map((4, 4, 4), (0, 0, 0), (8, 16, 16))
        assert m[:2].min() == 1.0
        assert m[2:].max() == 0.0

    def test_a_box_starting_mid_cell_gives_the_exact_near_edge_fraction(self):
        m = self._gen()._default_material_map((4, 4, 4), (1, 0, 0), (16, 16, 16))
        assert m[0].min() == pytest.approx(0.75)     # 3 of the 4 voxels
        assert m[1:].min() == 1.0

    def test_a_box_ending_mid_cell_gives_the_exact_far_edge_fraction(self):
        m = self._gen()._default_material_map((4, 4, 4), (0, 0, 0), (16, 16, 15))
        assert m[..., 3].max() == pytest.approx(0.75)
        assert m[..., :3].min() == 1.0

    def test_a_box_narrower_than_one_cell_is_not_lost(self):
        m = self._gen()._default_material_map((4, 4, 4), (1, 0, 0), (3, 16, 16))
        assert m[0].min() == pytest.approx(0.5)      # voxels 1 and 2 of 4
        assert m[1:].max() == 0.0

    def test_the_fraction_is_the_product_over_the_three_axes(self):
        m = self._gen()._default_material_map((4, 4, 4), (1, 2, 0), (16, 16, 15))
        assert m[0, 0, 3] == pytest.approx(0.75 * 0.5 * 0.75)

    def test_it_matches_the_block_mean_of_the_voxel_envelope(self):
        """The same quantity ``latent_material_map`` pools from a voxel mask."""
        from poregen.eval_v4.generate import latent_material_map
        lo, hi = (1, 5, 2), (13, 16, 15)
        vox = np.zeros((16, 16, 16), dtype=np.float32)
        vox[lo[0]:hi[0], lo[1]:hi[1], lo[2]:hi[2]] = 1.0
        m = self._gen()._default_material_map((4, 4, 4), lo, hi)
        np.testing.assert_allclose(m, latent_material_map(vox), atol=1e-6)

    def test_every_value_stays_a_fraction(self):
        m = self._gen()._default_material_map((4, 4, 4), (3, 0, 7), (14, 16, 9))
        assert m.dtype == np.float32
        assert float(m.min()) >= 0.0 and float(m.max()) <= 1.0
