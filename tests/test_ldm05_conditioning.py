"""Tests for the ldm05 conditioning data side (D32 §1–§3, §5).

Covers the orientation pooling rules, the touching-neighbour geometry, the
eight-group availability schedule, and the scalar ranges — the four places
where a silent convention error would poison training.

The neighbour geometry tests carry the ldm05 leak rationale: at
``neighbour_offset = 32`` a face neighbour shared half its voxels with the
target and an opposite pair tiled it completely, so the model was handed the
answer.  ``neighbour_offset = 64`` makes neighbours touch instead.
"""

from __future__ import annotations

import itertools
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

from poregen.diffusion.conditioning import (
    NB_EXISTS,
    NB_OOB,
    NB_UNKNOWN,
    NEIGHBOUR_DIRS,
    PARITY_GROUP_ORDER,
    grid_index,
    group_rank,
    latent_shift_cells,
    neighbour_shared_voxels,
    overlap_slices,
    parity_group,
    shift_into_target_frame,
    validate_neighbour_geometry,
    validate_shift,
)
from poregen.diffusion.latents import LatentDataset
from poregen.diffusion.orientation import (
    OrientationField,
    encode_theta,
    orientation_tensor,
    pool_components,
)

REPO = Path(__file__).resolve().parents[1]
STORE = REPO / "data" / "split_v2" / "latents_r07z4"
ORIENT_FIELD = REPO / "data" / "split_v2" / "orientation_field.json"


# ── orientation encoding ─────────────────────────────────────────────────────

class TestOrientationEncoding:

    def test_four_ply_classes_map_to_unit_vectors(self):
        enc = encode_theta(np.array([0.0, 45.0, 90.0, 135.0]))
        assert np.allclose(enc[:, 0], [1, 0], atol=1e-6)
        assert np.allclose(enc[:, 1], [0, 1], atol=1e-6)
        assert np.allclose(enc[:, 2], [-1, 0], atol=1e-6)
        assert np.allclose(enc[:, 3], [0, -1], atol=1e-6)

    def test_axial_wrap_is_removed(self):
        """theta and theta+180 are the same orientation, so the same vector."""
        a = encode_theta(np.array([10.0, 170.0]))
        b = encode_theta(np.array([190.0, -10.0]))
        assert np.allclose(a, b, atol=1e-6)

    def test_unknown_becomes_the_zero_vector(self):
        enc = encode_theta(np.array([0.0, np.nan]))
        assert np.allclose(enc[:, 1], 0.0)
        assert np.linalg.norm(enc[:, 0]) == pytest.approx(1.0, abs=1e-6)

    def test_uniform_ply_keeps_unit_magnitude(self):
        theta = np.full(64, 45.0)
        pooled = pool_components(encode_theta(theta), 16)
        mag = np.hypot(pooled[0], pooled[1])
        assert np.allclose(mag, 1.0, atol=1e-6)

    def test_plane_straddling_an_interface_loses_magnitude(self):
        """The magnitude drop at a ply interface IS the interface signal."""
        theta = np.empty(64)
        theta[:34] = 0.0          # boundary inside pooling group 8 (voxels 32..35)
        theta[34:] = 45.0
        pooled = pool_components(encode_theta(theta), 16)
        mag = np.hypot(pooled[0], pooled[1])
        interior = np.r_[mag[:8], mag[9:]]
        assert np.allclose(interior, 1.0, atol=1e-6)
        assert mag[8] < 0.95                       # this plane straddles
        assert mag[8] > 0.0

    def test_pooled_vector_is_not_renormalised(self):
        theta = np.array([0.0] * 2 + [90.0] * 2)
        pooled = pool_components(encode_theta(theta), 1)
        assert np.hypot(*pooled[:, 0]) == pytest.approx(0.0, abs=1e-6)

    def test_pooling_components_differs_from_pooling_angles(self):
        """Averaging theta directly is wrong across the 180 deg wrap."""
        theta = np.array([10.0, 170.0, 10.0, 170.0])
        pooled = pool_components(encode_theta(theta), 1)[:, 0]
        angle_from_components = np.rad2deg(np.arctan2(pooled[1], pooled[0])) / 2 % 180
        naive = theta.mean()                       # 90 deg — the wrong answer
        assert angle_from_components == pytest.approx(0.0, abs=1e-4)
        assert abs(naive - angle_from_components) > 45.0
        # and the magnitude carries the spread, which a naive mean discards
        assert np.hypot(*pooled) < 1.0

    def test_tensor_shape_and_in_plane_broadcast(self):
        t = orientation_tensor(np.linspace(0, 179, 64), 16)
        assert t.shape == (2, 16, 16, 16)
        assert t.dtype == np.float32
        for k in range(16):
            assert np.allclose(t[:, k], t[:, k, :1, :1])   # constant in-plane


# ── neighbour geometry: touching, not overlapping ────────────────────────────

class TestNeighbourGeometry:

    def test_touching_neighbours_share_no_voxel(self):
        """The invariant the whole design rests on, checked patch by patch."""
        patch, offset = 64, 64
        assert neighbour_shared_voxels(offset, patch) == 0
        origin = np.array([256, 256, 256])
        for d in NEIGHBOUR_DIRS:
            nb = origin + np.array(d) * offset
            shared = 1
            for o_t, o_n in zip(origin, nb):
                shared *= max(0, min(o_t + patch, o_n + patch) - max(o_t, o_n))
            assert shared == 0, f"neighbour {d} shares {shared} voxels"

    def test_offset_32_would_share_voxels(self):
        """Why ldm05's first run was thrown away.

        Each face neighbour handed over 32x64x64 voxels of the target — half of
        it.  Opposite faces flip the same parity bit, so +z and -z are always
        both EXISTS or both UNKNOWN, and together they tile the target
        completely (-z gives latent z-cells 0-7, +z gives 8-15).  Seven of the
        eight parity groups therefore received the whole answer.  Measured:
        neighbour-vs-target MAE 0.067 against 0.744 for a random patch.
        """
        patch, offset = 64, 32
        assert neighbour_shared_voxels(offset, patch) == 32 * 64 * 64
        origin = np.array([256, 256, 256])
        for d in NEIGHBOUR_DIRS:
            nb = origin + np.array(d) * offset
            shared = 1
            for o_t, o_n in zip(origin, nb):
                shared *= max(0, min(o_t + patch, o_n + patch) - max(o_t, o_n))
            assert shared == 32 * 64 * 64
        # the opposite pair covers the target exactly, with nothing left over
        assert parity_group((1, 0, 0)) == parity_group((-1, 0, 0))
        assert offset * 2 == patch

    def test_guard_rejects_overlap_unless_explicitly_allowed(self):
        validate_neighbour_geometry(64, 64)
        with pytest.raises(ValueError, match="ablation only"):
            validate_neighbour_geometry(32, 64)
        validate_neighbour_geometry(32, 64, allow_neighbour_overlap=True)

    def test_grid_index_is_defined_by_the_neighbour_offset(self):
        """Patches stored every 32 voxels still index consistently.

        The store is eight interleaved copies of the stride-64 grid; each
        copy's own neighbours are ±1 grid step whatever its offset.
        """
        for base in (0, 32, 64, 96):
            gi = grid_index((base, base, base), 64)
            for d in NEIGHBOUR_DIRS:
                ngi = grid_index(tuple(base + k * 64 for k in d), 64)
                assert tuple(ngi[k] - gi[k] for k in range(3)) == d


class TestNeighbourShift:
    """The shift exists only for overlapping neighbours; it must not run here."""

    def test_shift_cells(self):
        assert latent_shift_cells(32, 64, 16) == 8
        assert latent_shift_cells(64, 64, 16) == 16       # the full latent extent
        with pytest.raises(ValueError):
            latent_shift_cells(2, 64, 16)

    def test_a_full_extent_shift_raises_instead_of_zeroing(self):
        """The silent-zero guard: 16 cells on a 16-cell axis has no overlap."""
        with pytest.raises(ValueError, match="all zeros"):
            validate_shift(16, 16)
        with pytest.raises(ValueError, match="all zeros"):
            shift_into_target_frame(torch.randn(2, 16, 16, 16), (1, 0, 0), 16)
        validate_shift(15, 16)

    @pytest.mark.parametrize("direction", NEIGHBOUR_DIRS)
    def test_shift_still_works_for_an_overlapping_neighbour(self, direction):
        """Kept correct for the leak ablation: the shared region lines up."""
        L, shift = 16, 8
        rng = np.random.default_rng(0)
        g = torch.from_numpy(rng.standard_normal((2, 40, 40, 40)).astype(np.float32))
        o = np.array([12, 12, 12])                 # target origin in the field
        n = o + np.array(direction) * shift        # neighbour origin

        target = g[:, o[0]:o[0] + L, o[1]:o[1] + L, o[2]:o[2] + L]
        nb = g[:, n[0]:n[0] + L, n[1]:n[1] + L, n[2]:n[2] + L]
        shifted = shift_into_target_frame(nb, direction, shift)

        dst, src = overlap_slices(direction, L, shift)
        assert torch.equal(shifted[:, dst[0], dst[1], dst[2]],
                           target[:, dst[0], dst[1], dst[2]])
        # everything outside the shared region is zero-padded
        mask = torch.zeros((L, L, L), dtype=torch.bool)
        mask[dst[0], dst[1], dst[2]] = True
        assert torch.all(shifted[:, ~mask] == 0)


# ── eight-group parity schedule ──────────────────────────────────────────────

class TestParitySchedule:

    def test_ordering_is_a_permutation_of_the_eight_groups(self):
        assert sorted(PARITY_GROUP_ORDER) == sorted(
            itertools.product((0, 1), repeat=3))
        assert len(set(PARITY_GROUP_ORDER)) == 8

    def test_rank_is_4pz_2py_px(self):
        for g in PARITY_GROUP_ORDER:
            assert group_rank(g) == 4 * g[0] + 2 * g[1] + g[2]

    def test_no_two_patches_in_a_group_overlap(self):
        """Same-group patches are >= 2 grid steps apart on some axis."""
        stride, patch = 64, 64
        grid = list(itertools.product(range(6), repeat=3))
        groups: dict[tuple, list] = {}
        for gi in grid:
            groups.setdefault(parity_group(gi), []).append(gi)
        assert len(groups) == 8
        for members in groups.values():
            for a, b in itertools.combinations(members, 2):
                sep = [abs(a[k] - b[k]) * stride for k in range(3)]
                assert max(sep) >= patch, (
                    f"patches at {a} and {b} share the same parity group but "
                    f"overlap (separation {sep} < patch {patch})")

    def test_eight_groups_beat_a_two_colour_checkerboard(self):
        """Why the eight-group ordering is kept now that nothing overlaps.

        A two-colour ``(iz+iy+ix) % 2`` schedule is also collision-free at
        stride 64, but it splits the volume into 'no context at all' and 'all
        six neighbours'.  The eight-group ordering grades the context, so only
        one patch in eight is generated blind.
        """
        eight = []
        for g in PARITY_GROUP_ORDER:
            rank = group_rank(g)
            eight.append(sum(
                1 for d in NEIGHBOUR_DIRS
                if group_rank(tuple(g[k] + d[k] for k in range(3))) < rank
            ))
        assert sorted(eight) == [0, 2, 2, 2, 4, 4, 4, 6]
        assert sum(1 for n in eight if n == 0) / len(eight) == 0.125
        # the two-colour alternative: half the patches see nothing, half see six
        two_colour = [0] * 4 + [6] * 4
        assert sum(1 for n in two_colour if n == 0) / len(two_colour) == 0.5

    def test_face_neighbour_flips_exactly_one_parity_bit(self):
        for gi in itertools.product(range(4), repeat=3):
            for d in NEIGHBOUR_DIRS:
                ngi = tuple(gi[k] + d[k] for k in range(3))
                diff = sum(int(parity_group(gi)[k] != parity_group(ngi)[k])
                           for k in range(3))
                assert diff == 1

    def test_schedule_is_consistent_and_acyclic(self):
        """Every EXISTS neighbour is generated strictly earlier, and vice versa."""
        for gi in itertools.product(range(4), repeat=3):
            rank = group_rank(gi)
            for d in NEIGHBOUR_DIRS:
                ngi = tuple(gi[k] + d[k] for k in range(3))
                nrank = group_rank(ngi)
                assert nrank != rank                     # never simultaneous
        # first group sees no EXISTS neighbour, last sees six
        first, last = PARITY_GROUP_ORDER[0], PARITY_GROUP_ORDER[-1]
        assert all(group_rank(tuple(first[k] + d[k] for k in range(3))) > 0
                   for d in NEIGHBOUR_DIRS)
        assert all(group_rank(tuple(last[k] + d[k] for k in range(3))) < 7
                   for d in NEIGHBOUR_DIRS)


# ── synthetic end-to-end store ───────────────────────────────────────────────

_SYN = dict(L=4, C=1, PATCH=16, SAMPLE_STRIDE=8, OFFSET=16, VOL=80)


@pytest.fixture(scope="module")
def synthetic_store(tmp_path_factory):
    """A miniature latent store whose patches are crops of one global field.

    Scaled-down copy of the production geometry: C=1, latent 4³, patch 16
    voxels (downsample 4), ``sample_stride`` 8 (half the patch, so the store is
    eight interleaved copies of the assembly grid) and ``neighbour_offset`` 16
    == the patch size, so neighbours TOUCH.  Because every patch is a crop of
    the same field, a neighbour must reproduce the field block immediately next
    to the target — and must share none of the target's own cells.
    """
    root = tmp_path_factory.mktemp("latents")
    L, C = _SYN["L"], _SYN["C"]
    patch, stride, vol = _SYN["PATCH"], _SYN["SAMPLE_STRIDE"], _SYN["VOL"]
    ds_factor = patch // L                                   # 4
    vol_shape = (vol, vol, vol)
    gsize = tuple(s // ds_factor for s in vol_shape)         # 20³ latent cells
    rng = np.random.default_rng(7)
    field = rng.standard_normal((C, *gsize)).astype(np.float32)

    coords = [c for c in itertools.product(range(0, vol - patch + 1, stride), repeat=3)]
    rows = []
    data = np.zeros((len(coords), 2 * C, L, L, L), np.float16)
    for i, (z0, y0, x0) in enumerate(coords):
        cz, cy, cx = z0 // ds_factor, y0 // ds_factor, x0 // ds_factor
        data[i, :C] = field[:, cz:cz + L, cy:cy + L, cx:cx + L]
        data[i, C:] = 0.25
        rows.append({"source_row": i, "volume_id": "vol_a", "z0": z0, "y0": y0,
                     "x0": x0, "phi": 0.01 + 0.0001 * i})
    df = pd.DataFrame(rows)

    split_dir = root / "train"
    split_dir.mkdir()
    data.tofile(split_dir / "latents.bin")
    df.to_parquet(split_dir / "index.parquet", index=False)
    por_raw = np.log(df["phi"].to_numpy() + 1e-3).astype(np.float32)
    pd.DataFrame({
        "source_row": df["source_row"].to_numpy(np.int64),
        "cond_depth": np.linspace(0, 1, len(df), dtype=np.float32),
        "cond_dist": np.linspace(0, 1, len(df), dtype=np.float32),
        "cond_por_raw": por_raw,
    }).to_parquet(split_dir / "cond.parquet", index=False)

    of = root / "orientation_field.json"
    theta = [0.0] * (vol // 2) + [90.0] * (vol - vol // 2)
    of.write_text(json.dumps({
        "voxel_size_um": 25.0,
        "volumes": {"vol_a": {"shape": list(vol_shape), "orientation_usable": True,
                              "confidence": "high", "theta_deg": theta}},
    }))

    (root / "metadata.json").write_text(json.dumps({
        "latent_shape": [C, L, L, L],
        "patch_size": patch,
        "voxel_size_um": 25.0,
        "storage": {"format": "memmap", "file": "latents.bin",
                    "dtype": "float16", "pack_scheme": "mu_then_std"},
        "normalization": {"computed_over": "train",
                          "per_channel_mean": [0.0] * C,
                          "per_channel_std": [1.0] * C},
        "assembly": {
            "parity_group_order": [list(g) for g in PARITY_GROUP_ORDER],
            "sample_stride": stride,
            "generation_stride": _SYN["OFFSET"],
            "neighbour_offset": _SYN["OFFSET"],
        },
        "conditioning": {
            "orientation_field": str(of),
            "por_standardisation": {"mean": float(por_raw.mean()),
                                    "std": float(por_raw.std())},
        },
    }))
    return root, field, ds_factor, stride


class TestLatentDataset:

    def _ds(self, store, **kw):
        root, *_ = store
        kw.setdefault("sample_stride", _SYN["SAMPLE_STRIDE"])
        kw.setdefault("generation_stride", _SYN["OFFSET"])
        kw.setdefault("neighbour_offset", _SYN["OFFSET"])
        return LatentDataset(root, "train", normalize=True, **kw)

    def test_batch_contract(self, synthetic_store):
        ds = self._ds(synthetic_store)
        b = ds[0]
        assert b["z"].shape == (1, 4, 4, 4) and b["z"].dtype == torch.float32
        assert b["std"].shape == (1, 4, 4, 4)
        for k in ("cond_por", "cond_depth", "cond_dist"):
            assert b[k].shape == () and b[k].dtype == torch.float32
        assert b["cond_orient"].shape == (2, 4, 4, 4)
        assert b["nb_latents"].shape == (6, 1, 4, 4, 4)
        assert b["nb_avail"].shape == (6,) and b["nb_avail"].dtype == torch.int64
        assert isinstance(b["volume_id"], str)
        assert b["coords"].shape == (3,)
        assert b["grid_index"].shape == (3,)
        assert isinstance(b["source_row"], int)

    def test_neighbour_is_the_whole_adjacent_block(self, synthetic_store):
        """The strongest check: an EXISTS neighbour is the field block next door.

        Every patch is a crop of one global field, so the neighbour latent must
        equal the field exactly at the neighbour's own position — whole, in its
        own frame, with no roll and no zero padding.
        """
        root, field, ds_factor, _ = synthetic_store
        ds = self._ds(synthetic_store)
        L = ds.latent_size
        checked = 0
        for idx in range(len(ds)):
            b = ds[idx]
            z0, y0, x0 = (int(v) for v in b["coords"])
            for i, d in enumerate(NEIGHBOUR_DIRS):
                if b["nb_avail"][i].item() != NB_EXISTS:
                    continue
                n0 = [o + k * ds.neighbour_offset for o, k in
                      zip((z0, y0, x0), d)]
                c = [v // ds_factor for v in n0]
                expected = torch.from_numpy(
                    field[:, c[0]:c[0] + L, c[1]:c[1] + L, c[2]:c[2] + L]
                )
                assert torch.allclose(b["nb_latents"][i], expected, atol=1e-3), (
                    f"row {idx} neighbour {d} is not the adjacent field block")
                checked += 1
        assert checked > 0

    def test_no_neighbour_cell_belongs_to_the_target(self, synthetic_store):
        """The leak test, on real dataset output.

        The target occupies latent cells [c, c+L) of the global field; every
        neighbour must occupy a disjoint block, so no cell of the target can be
        read off a neighbour tensor.
        """
        _, _, ds_factor, _ = synthetic_store
        ds = self._ds(synthetic_store)
        L = ds.latent_size
        checked = 0
        for idx in range(len(ds)):
            b = ds[idx]
            origin = [int(v) // ds_factor for v in b["coords"]]
            tgt = [(o, o + L) for o in origin]
            for i, d in enumerate(NEIGHBOUR_DIRS):
                if b["nb_avail"][i].item() != NB_EXISTS:
                    continue
                n_origin = [o + k * ds.neighbour_offset // ds_factor
                            for o, k in zip(origin, d)]
                shared = 1
                for (a0, a1), o_n in zip(tgt, n_origin):
                    shared *= max(0, min(a1, o_n + L) - max(a0, o_n))
                assert shared == 0, f"neighbour {d} shares {shared} latent cells"
                checked += 1
        assert checked > 0

    def test_a_shift_is_refused_at_this_offset(self, synthetic_store):
        with pytest.raises(ValueError, match="all zeros"):
            self._ds(synthetic_store, neighbour_shift=True)

    def test_overlapping_offset_is_refused(self, synthetic_store):
        """offset 8 < patch 16 would leak, so the dataset refuses to build."""
        with pytest.raises(ValueError, match="ablation only"):
            self._ds(synthetic_store, generation_stride=8, neighbour_offset=8)

    def test_store_metadata_must_match_the_config(self, synthetic_store):
        with pytest.raises(ValueError, match="assembly.generation_stride"):
            self._ds(synthetic_store, generation_stride=32, neighbour_offset=32)

    def test_stored_origins_must_sit_on_the_sample_stride(self, synthetic_store):
        """The store's own density is validated before anything else is trusted."""
        with pytest.raises(ValueError, match="not a\n?\\s*multiple of sample_stride"):
            self._ds(synthetic_store, sample_stride=5)

    def test_availability_follows_the_group_ordering(self, synthetic_store):
        """OOB / EXISTS / UNKNOWN come from the shared schedule, not a local copy.

        A grid index alone is ambiguous: the store holds eight interleaved
        copies of the assembly grid, so patches at origin 0 and origin 8 share
        the grid index 0.  Availability is therefore keyed on the exact origin
        — the two copies never see each other because a face neighbour is a
        whole neighbour_offset away.
        """
        ds = self._ds(synthetic_store)
        origins = {(int(z), int(y), int(x))
                   for z, y, x in zip(ds._z0, ds._y0, ds._x0)}
        seen = set()
        for idx in range(len(ds)):
            b = ds[idx]
            gi = tuple(b["grid_index"].tolist())
            rank = group_rank(gi, ds.group_order)
            for i, d in enumerate(NEIGHBOUR_DIRS):
                state = b["nb_avail"][i].item()
                seen.add(state)
                ngi = tuple(gi[k] + d[k] for k in range(3))
                n_origin = tuple(int(v) + k * ds.neighbour_offset
                                 for v, k in zip(b["coords"], d))
                if n_origin not in origins:
                    assert state == NB_OOB
                elif group_rank(ngi, ds.group_order) < rank:
                    assert state == NB_EXISTS
                else:
                    assert state == NB_UNKNOWN
                if state != NB_EXISTS:
                    assert torch.all(b["nb_latents"][i] == 0)
        assert seen == {NB_OOB, NB_EXISTS, NB_UNKNOWN}

    def test_neighbours_can_be_switched_off(self, synthetic_store):
        ds = self._ds(synthetic_store, neighbours=False)
        b = ds[len(ds) // 2]
        assert torch.all(b["nb_latents"] == 0)
        assert torch.all(b["nb_avail"] == NB_OOB)

    def test_orientation_matches_the_field(self, synthetic_store):
        ds = self._ds(synthetic_store)
        b = ds[0]                                   # z0 = 0, theta 0 deg over 0..15
        assert torch.allclose(b["cond_orient"][0], torch.ones_like(b["cond_orient"][0]),
                              atol=1e-5)
        assert torch.allclose(b["cond_orient"][1], torch.zeros_like(b["cond_orient"][1]),
                              atol=1e-5)

    def test_mismatched_sidecar_is_rejected(self, synthetic_store, tmp_path):
        import shutil
        root, *_ = synthetic_store
        bad = tmp_path / "bad"
        shutil.copytree(root, bad)
        cond = pd.read_parquet(bad / "train" / "cond.parquet")
        cond["source_row"] = cond["source_row"] + 1
        cond.to_parquet(bad / "train" / "cond.parquet", index=False)
        with pytest.raises(RuntimeError, match="row-aligned"):
            LatentDataset(bad, "train", sample_stride=_SYN["SAMPLE_STRIDE"],
                          generation_stride=_SYN["OFFSET"],
                          neighbour_offset=_SYN["OFFSET"])


# ── the real artefacts ───────────────────────────────────────────────────────

needs_data = pytest.mark.skipif(
    not (STORE / "train" / "cond.parquet").exists() or not ORIENT_FIELD.exists(),
    reason="ldm05 conditioning artefacts not built (run scripts/build_conditioning.py)",
)


@needs_data
class TestBuiltArtefacts:

    def test_scalar_ranges(self):
        for split in ("train", "val", "test"):
            c = pd.read_parquet(STORE / split / "cond.parquet")
            idx = pd.read_parquet(STORE / split / "index.parquet",
                                  columns=["source_row", "phi"])
            assert len(c) == len(idx)
            assert np.array_equal(c["source_row"], idx["source_row"])
            d = c["cond_depth"].to_numpy()
            s = c["cond_dist"].to_numpy()
            assert np.isfinite(d).all() and np.isfinite(s).all()
            assert d.min() >= 0.0 and d.max() <= 1.0
            assert s.min() >= 0.0 and s.max() <= 1.0
            # cond_por_raw is exactly log(phi + 1e-3) on the unchanged phi
            assert np.allclose(c["cond_por_raw"].to_numpy(),
                               np.log(idx["phi"].to_numpy() + 1e-3).astype(np.float32),
                               atol=1e-5)

    def test_standardisation_is_train_only_and_applied(self):
        meta = json.loads((STORE / "metadata.json").read_text())
        st = meta["conditioning"]["por_standardisation"]
        assert st["computed_over"] == "train"
        tr = pd.read_parquet(STORE / "train" / "cond.parquet")["cond_por_raw"].to_numpy()
        assert st["mean"] == pytest.approx(float(tr.mean()), abs=1e-4)
        assert st["std"] == pytest.approx(float(tr.std()), abs=1e-4)
        assert meta["voxel_size_um"] == 25.0

    def test_orientation_field_is_verified_and_complete(self):
        f = json.loads(ORIENT_FIELD.read_text())
        assert f["summary"]["n_verification_failures"] == 0
        assert f["voxel_size_um"] == 25.0
        of = OrientationField(ORIENT_FIELD)
        for split in ("train", "val", "test"):
            vols = pd.read_parquet(STORE / split / "index.parquet",
                                   columns=["volume_id"])["volume_id"].unique()
            for v in vols:
                assert v in of
                t = of.theta(v)
                known = t[np.isfinite(t)]
                assert ((known >= 0) & (known < 180)).all()

    def test_dataset_returns_the_contract(self):
        ds = LatentDataset(STORE, "val", normalize=True)
        b = ds[0]
        assert ds.voxel_size_um == 25.0
        assert b["z"].shape == (4, 16, 16, 16)
        assert b["cond_orient"].shape == (2, 16, 16, 16)
        assert b["nb_latents"].shape == (6, 4, 16, 16, 16)
        assert b["nb_avail"].shape == (6,)
        assert 0.0 <= float(b["cond_depth"]) <= 1.0
        assert 0.0 <= float(b["cond_dist"]) <= 1.0
