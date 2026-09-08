"""The ldm06 conditioning data side.

Covers the conventions both halves of the stack must agree on — orientation
encoding, the touching-neighbour geometry guard, the six per-face distances —
and the batch contract ``LatentDataset`` serves, against a synthetic store
whose patches are crops of one known field.
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

from _ldm06_store import DIST6_COLUMNS, SYN, build_store, dataset_kwargs, row_std
from poregen.diffusion.conditioning import (
    DIST6_DIRS,
    DIST6_NAMES,
    DIST_CAP,
    NB_EXISTS,
    NB_OOB,
    NEIGHBOUR_DIRS,
    N_DIST6,
    N_NEIGHBOURS,
    dist6_from_box,
    dist6_from_box_array,
    grid_index,
    neighbour_shared_voxels,
    porosity_to_cond,
    validate_neighbour_geometry,
)
from poregen.diffusion.latents import LatentDataset
from poregen.diffusion.orientation import (
    encode_theta,
    orientation_tensor,
    pool_components,
)

REPO = Path(__file__).resolve().parents[1]


# ── orientation encoding ─────────────────────────────────────────────────────

class TestOrientationEncoding:

    def test_four_ply_classes_map_to_unit_vectors(self):
        got = encode_theta(np.array([0.0, 45.0, 90.0, -45.0]))
        assert np.allclose(got[:, 0], [1, 0], atol=1e-6)
        assert np.allclose(got[:, 1], [0, 1], atol=1e-6)
        assert np.allclose(got[:, 2], [-1, 0], atol=1e-6)
        assert np.allclose(got[:, 3], [0, -1], atol=1e-6)

    def test_axial_wrap_is_removed(self):
        """theta and theta+180 are the same orientation."""
        a = encode_theta(np.array([30.0, 210.0, -150.0]))
        assert np.allclose(a[:, 0], a[:, 1], atol=1e-6)
        assert np.allclose(a[:, 0], a[:, 2], atol=1e-6)

    def test_unknown_becomes_the_zero_vector(self):
        got = encode_theta(np.array([np.nan, 0.0]))
        assert np.allclose(got[:, 0], [0.0, 0.0])
        assert not np.allclose(got[:, 1], [0.0, 0.0])

    def test_plane_straddling_an_interface_loses_magnitude(self):
        """The shrinkage IS the interface marker — never renormalise it."""
        uniform = pool_components(encode_theta(np.full(4, 0.0)), 1)
        straddle = pool_components(
            encode_theta(np.array([0.0, 0.0, 90.0, 90.0])), 1)
        assert np.isclose(np.linalg.norm(uniform[:, 0]), 1.0, atol=1e-6)
        assert np.linalg.norm(straddle[:, 0]) < 1e-6

    def test_pooling_components_differs_from_pooling_angles(self):
        """A mean of angles is meaningless across the 180 deg wrap."""
        theta = np.array([175.0, 5.0])
        comp = pool_components(encode_theta(theta), 1)[:, 0]
        naive = encode_theta(np.array([theta.mean()]))[:, 0]
        assert not np.allclose(comp / max(np.linalg.norm(comp), 1e-9), naive,
                               atol=1e-3)

    def test_tensor_shape_and_in_plane_broadcast(self):
        t = orientation_tensor(np.linspace(0.0, 90.0, 16), 4)
        assert t.shape == (2, 4, 4, 4)
        for k in range(4):
            assert np.allclose(t[:, k], t[:, k, :1, :1])


# ── neighbour geometry ───────────────────────────────────────────────────────

class TestNeighbourGeometry:

    def test_touching_neighbours_share_no_voxel(self):
        assert neighbour_shared_voxels(64, 64) == 0
        validate_neighbour_geometry(64, 64)

    def test_offset_32_would_share_voxels(self):
        """The ldm05 leak: half the target handed over as 'context'."""
        assert neighbour_shared_voxels(32, 64) == 32 * 64 * 64
        with pytest.raises(ValueError, match="handing the denoiser part of the answer"):
            validate_neighbour_geometry(32, 64)

    def test_there_is_no_escape_hatch(self):
        """The guard takes no flag — an overlapping neighbour is never valid."""
        with pytest.raises(ValueError):
            validate_neighbour_geometry(63, 64)

    def test_grid_index_is_defined_by_the_neighbour_offset(self):
        """A face neighbour is exactly one grid step, whatever the origin."""
        for origin in [(0, 0, 0), (32, 64, 96), (128, 32, 32)]:
            gi = grid_index(origin, 64)
            for d in NEIGHBOUR_DIRS:
                n_origin = tuple(o + dd * 64 for o, dd in zip(origin, d))
                ngi = grid_index(n_origin, 64)
                assert tuple(n - g for n, g in zip(ngi, gi)) == d


# ── six per-face distances ───────────────────────────────────────────────────

class TestDist6:

    def test_order_is_low_then_high_per_axis(self):
        assert DIST6_NAMES == ("zm", "zp", "ym", "yp", "xm", "xp")
        assert DIST6_DIRS == ((-1, 0, 0), (1, 0, 0), (0, -1, 0),
                              (0, 1, 0), (0, 0, -1), (0, 0, 1))
        assert N_DIST6 == 6

    def test_patch_flush_against_a_face_reads_zero_there_only(self):
        # 512-voxel box; the patch sits in the z=lo corner and far from the rest.
        d = dist6_from_box((0, 200, 200), 64, (0, 0, 0), (512, 512, 512))
        assert d[0] == pytest.approx(0.0)          # z- : flush with the box
        assert d[1] == pytest.approx(1.0)          # z+ : 448 voxels, capped
        assert np.all(d[2:] == pytest.approx(1.0))

    def test_a_gap_below_the_cap_is_reported_proportionally(self):
        d = dist6_from_box((32, 200, 200), 64, (0, 0, 0), (512, 512, 512))
        assert d[0] == pytest.approx(32.0 / DIST_CAP)

    def test_both_faces_of_a_thin_specimen_read_short(self):
        """One scalar could not say this: BOTH z faces are close at once."""
        d = dist6_from_box((0, 200, 200), 64, (0, 0, 0), (96, 512, 512))
        assert d[0] == pytest.approx(0.0)
        assert d[1] == pytest.approx(32.0 / DIST_CAP)

    def test_a_patch_outside_the_box_clamps_to_zero_not_negative(self):
        d = dist6_from_box((-20, 200, 200), 64, (0, 0, 0), (512, 512, 512))
        assert d[0] == pytest.approx(0.0)

    def test_array_form_matches_the_scalar_form(self):
        rng = np.random.default_rng(3)
        origins = rng.integers(0, 400, size=(20, 3))
        lo = np.zeros((20, 3))
        hi = np.full((20, 3), 448.0)
        batch = dist6_from_box_array(origins, 64, lo, hi)
        for i in range(20):
            one = dist6_from_box(origins[i], 64, lo[i], hi[i])
            assert np.allclose(batch[i], one)

    def test_values_stay_in_the_unit_interval(self):
        rng = np.random.default_rng(4)
        origins = rng.integers(-100, 600, size=(200, 3))
        d = dist6_from_box_array(origins, 64, np.zeros((200, 3)),
                                 np.full((200, 3), 512.0))
        assert d.min() >= 0.0 and d.max() <= 1.0
        assert d.dtype == np.float32


# ── the conditioning builder ─────────────────────────────────────────────────

def _load_builder():
    spec = importlib.util.spec_from_file_location(
        "build_conditioning", REPO / "scripts" / "build_conditioning.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


PHI = np.array([0.01, 0.02, 0.05], dtype=np.float32)


@pytest.fixture(scope="module")
def built(tmp_path_factory):
    """``build_scalars`` run against a synthetic specimen extent."""
    mod = _load_builder()
    store = tmp_path_factory.mktemp("cond_store")
    # A specimen box of [10, 90] x [0, 200] x [0, 200] INCLUSIVE, i.e. 81
    # voxels thick in z.  Patch size is the builder's own PATCH_SIZE.
    extent = {"z": [10, 90], "y": [0, 200], "x": [0, 200]}
    field = {"volumes": {"vol_a": {"extent_foreground": extent}}}
    ps = mod.PATCH_SIZE
    origins = [(10, 0, 0), (26, 100, 100), (90 + 1 - ps, 100, 100)]
    for split in mod.SPLITS:
        d = store / split
        d.mkdir(parents=True)
        pd.DataFrame({
            "source_row": np.arange(len(origins), dtype=np.int64),
            "volume_id": ["vol_a"] * len(origins),
            "z0": [o[0] for o in origins],
            "y0": [o[1] for o in origins],
            "x0": [o[2] for o in origins],
            "phi": PHI,
        }).to_parquet(d / "index.parquet", index=False)
    return mod, store, mod.build_scalars(store, field), origins


class TestConditioningBuilder:
    """cond_dist6 straight out of ``build_scalars`` on a synthetic extent."""

    def test_columns_are_the_six_named_faces(self, built):
        _, store, _, _ = built
        cond = pd.read_parquet(store / "train" / "cond.parquet")
        assert list(DIST6_COLUMNS) == [c for c in cond.columns
                                       if c.startswith("cond_dist6_")]
        assert "cond_dist" not in cond.columns

    def test_first_patch_is_flush_with_the_low_faces(self, built):
        _, store, _, _ = built
        cond = pd.read_parquet(store / "train" / "cond.parquet")
        row = cond.iloc[0]
        assert row["cond_dist6_zm"] == pytest.approx(0.0)
        assert row["cond_dist6_ym"] == pytest.approx(0.0)
        assert row["cond_dist6_xm"] == pytest.approx(0.0)

    def test_last_patch_is_flush_with_the_high_z_face(self, built):
        """The box is [lo, hi+1), so a patch ending on hi reads 0, never -1."""
        _, store, _, _ = built
        cond = pd.read_parquet(store / "train" / "cond.parquet")
        row = cond.iloc[-1]
        assert row["cond_dist6_zp"] == pytest.approx(0.0)
        assert row["cond_dist6_zm"] > 0.0

    def test_reproduces_the_shared_implementation(self, built):
        mod, store, _, origins = built
        cond = pd.read_parquet(store / "train" / "cond.parquet")
        expected = dist6_from_box_array(
            np.array(origins, float), mod.PATCH_SIZE,
            np.tile([10.0, 0.0, 0.0], (len(origins), 1)),
            np.tile([91.0, 201.0, 201.0], (len(origins), 1)),
        )
        got = cond[list(DIST6_COLUMNS)].to_numpy(np.float32)
        assert np.allclose(got, expected)

    def test_metadata_geometry_and_standardisation(self, built):
        mod, _, sc, _ = built
        geom = mod.geometry_metadata()
        assert geom["neighbour_offset"] == geom["generation_stride"] == 64
        assert geom["neighbour_shared_voxels"] == 0
        assert sc["por_std"] >= 0.0

    def test_porosity_transform_matches_the_builder(self, built):
        _, store, sc, _ = built
        cond = pd.read_parquet(store / "train" / "cond.parquet")
        raw = cond["cond_por_raw"].to_numpy()
        assert np.allclose(raw, np.log(PHI + 1e-3), atol=1e-6)
        assert sc["por_std"] > 0.0
        # The sampler's transform standardises with those very stats, so a
        # requested phi and a stored one land on the same cond_por.
        stats = (sc["por_mean"], sc["por_std"])
        for i, phi in enumerate(PHI):
            assert porosity_to_cond(float(phi), stats) == pytest.approx(
                (raw[i] - sc["por_mean"]) / sc["por_std"], abs=1e-6)


# ── LatentDataset: the batch contract ────────────────────────────────────────

@pytest.fixture(scope="module")
def synthetic_store(tmp_path_factory):
    return build_store(tmp_path_factory.mktemp("latents"))


def _ds(store, **kw):
    root, *_ = store
    return LatentDataset(root, "train", normalize=True, **dataset_kwargs(**kw))


class TestLatentDatasetContract:

    def test_shapes_and_dtypes(self, synthetic_store):
        ds = _ds(synthetic_store)
        L, C = SYN["L"], SYN["C"]
        b = ds[0]
        assert b["z"].shape == (C, L, L, L) and b["z"].dtype == torch.float32
        assert b["std"].shape == (C, L, L, L)
        assert b["cond_por"].shape == () and b["cond_depth"].shape == ()
        assert b["cond_dist6"].shape == (N_DIST6,)
        assert b["cond_orient"].shape == (2, L, L, L)
        assert b["cond_material"].shape == (1, L, L, L)
        assert b["nb_latents"].shape == (N_NEIGHBOURS, C, L, L, L)
        assert b["nb_std"].shape == (N_NEIGHBOURS, C, L, L, L)
        assert b["nb_std"].dtype == torch.float32
        assert b["nb_avail"].shape == (N_NEIGHBOURS,)
        assert b["nb_avail"].dtype == torch.int64
        assert b["air_fraction"].shape == ()
        assert b["phi"].shape == (1,)
        assert b["coords"].shape == (3,) and b["grid_index"].shape == (3,)
        assert isinstance(b["volume_id"], str) and isinstance(b["source_row"], int)

    def test_scalar_ranges(self, synthetic_store):
        ds = _ds(synthetic_store)
        for i in (0, len(ds) // 2, len(ds) - 1):
            b = ds[i]
            assert 0.0 <= float(b["cond_depth"]) <= 1.0
            assert bool(((b["cond_dist6"] >= 0) & (b["cond_dist6"] <= 1)).all())
            assert 0.0 <= float(b["air_fraction"]) <= 1.0
            assert bool(((b["cond_material"] >= 0) & (b["cond_material"] <= 1)).all())
            assert 0.0 <= float(b["phi"][0]) <= 1.0

    def test_material_is_the_stored_fraction(self, synthetic_store):
        root, *_ = synthetic_store
        ds = _ds(synthetic_store)
        n = len(ds)
        raw = np.memmap(root / "train" / "material.bin", dtype=np.uint8, mode="r",
                        shape=(n, SYN["L"], SYN["L"], SYN["L"]))
        got = ds[5]["cond_material"][0].numpy()
        assert np.allclose(got, raw[5].astype(np.float32) / 255.0)

    def test_air_is_one_minus_the_material_mean(self, synthetic_store):
        """The map is the specimen ENVELOPE, so the two agree by construction.

        If cond_material ever went back to the solid fraction this identity
        would break, because pores would pull the mean down without being air.
        """
        ds = _ds(synthetic_store)
        for i in (0, len(ds) // 3, len(ds) - 1):
            b = ds[i]
            assert float(b["air_fraction"]) == pytest.approx(
                1.0 - float(b["cond_material"].mean()), abs=2e-3)

    def test_the_material_map_is_saturated_inside_the_specimen(self, synthetic_store):
        """A patch clear of the surface must carry NO structure in the map.

        That is what makes it safe to feed: it cannot be a pore mask, because
        it is constant wherever the pores are.
        """
        ds = _ds(synthetic_store)
        interior = [i for i in range(len(ds))
                    if int(ds[i]["coords"][0]) >= 3 * SYN["PATCH"]]
        assert interior
        m = ds[interior[0]]["cond_material"]
        assert float(m.min()) == pytest.approx(1.0)
        assert float(ds[interior[0]]["air_fraction"]) == pytest.approx(0.0, abs=2e-3)

    def test_availability_is_only_exists_or_oob(self, synthetic_store):
        """The store never emits UNKNOWN — that state comes from training and
        the sampler, and conflating the two hides which one produced it."""
        ds = _ds(synthetic_store)
        seen = set()
        for i in range(0, len(ds), 7):
            seen |= set(int(v) for v in ds[i]["nb_avail"])
        assert seen <= {NB_OOB, NB_EXISTS}
        assert seen == {NB_OOB, NB_EXISTS}

    def test_neighbour_is_the_whole_adjacent_block(self, synthetic_store):
        root, field, ds_factor, _ = synthetic_store
        ds = _ds(synthetic_store)
        L = SYN["L"]
        idx = len(ds) // 2
        b = ds[idx]
        z0, y0, x0 = (int(v) for v in b["coords"])
        for i, d in enumerate(NEIGHBOUR_DIRS):
            if int(b["nb_avail"][i]) != NB_EXISTS:
                continue
            n0 = (z0 + d[0] * SYN["OFFSET"], y0 + d[1] * SYN["OFFSET"],
                  x0 + d[2] * SYN["OFFSET"])
            c = tuple(v // ds_factor for v in n0)
            expected = field[:, c[0]:c[0] + L, c[1]:c[1] + L, c[2]:c[2] + L]
            assert np.allclose(b["nb_latents"][i].numpy(), expected, atol=2e-3)

    def test_neighbour_std_comes_from_the_neighbour_row(self, synthetic_store):
        """The store serves mu AND std, and both come from the SAME row.

        The training step draws ``mu + sigma*eps`` per neighbour, so a std
        taken from the target's row — or left at zero — would silently make
        every neighbour a mean-valued latent again.
        """
        ds = _ds(synthetic_store)
        idx = len(ds) // 2
        b = ds[idx]
        rows = ds.neighbour_rows(idx)
        assert float(b["nb_std"].max()) > 0.0
        for i in range(N_NEIGHBOURS):
            if int(b["nb_avail"][i]) != NB_EXISTS:
                assert float(b["nb_std"][i].abs().max()) == 0.0
                continue
            # Row-dependent by construction, and the six faces sit on other
            # residues than the target — so this cannot pass on the target's.
            assert float(b["nb_std"][i].mean()) == pytest.approx(
                row_std(int(rows[i])), abs=1e-3)
            assert row_std(int(rows[i])) != pytest.approx(
                row_std(int(b["source_row"])), abs=1e-3)

    def test_no_neighbour_cell_belongs_to_the_target(self, synthetic_store):
        """The whole point of neighbour_offset == patch_size."""
        root, field, ds_factor, _ = synthetic_store
        ds = _ds(synthetic_store)
        L = SYN["L"]
        b = ds[len(ds) // 2]
        z0, y0, x0 = (int(v) for v in b["coords"])
        c = (z0 // ds_factor, y0 // ds_factor, x0 // ds_factor)
        target_cells = {(c[0] + i, c[1] + j, c[2] + k)
                        for i in range(L) for j in range(L) for k in range(L)}
        for i, d in enumerate(NEIGHBOUR_DIRS):
            n0 = tuple(o + dd * SYN["OFFSET"] for o, dd in zip((z0, y0, x0), d))
            nc = tuple(v // ds_factor for v in n0)
            nb_cells = {(nc[0] + a, nc[1] + b_, nc[2] + k)
                        for a in range(L) for b_ in range(L) for k in range(L)}
            assert not (nb_cells & target_cells)

    def test_orientation_matches_the_field(self, synthetic_store):
        ds = _ds(synthetic_store)
        b = ds[0]
        z0 = int(b["coords"][0])
        expected = ds.orientation.patch_tensor("vol_a", z0, SYN["PATCH"], SYN["L"])
        assert np.allclose(b["cond_orient"].numpy(), expected)

    def test_overlapping_offset_is_refused(self, synthetic_store):
        with pytest.raises(ValueError, match="handing the denoiser part of the answer"):
            _ds(synthetic_store, neighbour_offset=8, generation_stride=8)

    def test_store_geometry_must_match_the_config(self, synthetic_store):
        with pytest.raises(ValueError, match="conditioning.geometry"):
            _ds(synthetic_store, sample_stride=4)

    def test_stored_origins_must_sit_on_the_sample_stride(self, synthetic_store):
        with pytest.raises(ValueError, match="not a"):
            _ds(synthetic_store, sample_stride=3, generation_stride=SYN["OFFSET"],
                neighbour_offset=SYN["OFFSET"])

    def test_a_sidecar_without_dist6_is_rejected(self, synthetic_store, tmp_path):
        """An ldm05 sidecar must fail loudly, not silently lose five faces."""
        root, *_ = synthetic_store
        clone = tmp_path / "clone"
        clone.mkdir()
        (clone / "metadata.json").write_text((root / "metadata.json").read_text())
        (clone / "train").mkdir()
        for f in ("latents.bin", "index.parquet", "material.bin", "air.bin"):
            (clone / "train" / f).write_bytes((root / "train" / f).read_bytes())
        cond = pd.read_parquet(root / "train" / "cond.parquet")
        cond = cond.drop(columns=list(DIST6_COLUMNS))
        cond["cond_dist"] = np.float32(0.5)
        cond.to_parquet(clone / "train" / "cond.parquet", index=False)
        with pytest.raises(RuntimeError, match="six per-face distances"):
            LatentDataset(clone, "train", normalize=True, **dataset_kwargs())

    def test_missing_material_sidecar_is_rejected(self, synthetic_store, tmp_path):
        root, *_ = synthetic_store
        clone = tmp_path / "clone2"
        clone.mkdir()
        (clone / "metadata.json").write_text((root / "metadata.json").read_text())
        (clone / "train").mkdir()
        for f in ("latents.bin", "index.parquet", "cond.parquet", "air.bin"):
            (clone / "train" / f).write_bytes((root / "train" / f).read_bytes())
        with pytest.raises(RuntimeError, match="material.bin"):
            LatentDataset(clone, "train", normalize=True, **dataset_kwargs())

    def test_normalisation_round_trips(self, synthetic_store):
        root, *_ = synthetic_store
        raw = LatentDataset(root, "train", normalize=False, **dataset_kwargs())
        norm = LatentDataset(root, "train", normalize=True, **dataset_kwargs())
        back = norm.denormalize_latent(norm[3]["z"])
        assert torch.allclose(back, raw[3]["z"], atol=1e-5)


def test_store_metadata_is_json_serialisable(synthetic_store):
    root, *_ = synthetic_store
    meta = json.loads((root / "metadata.json").read_text())
    assert "assembly" not in meta          # the parity block is gone for good
    assert "geometry" in meta["conditioning"]
