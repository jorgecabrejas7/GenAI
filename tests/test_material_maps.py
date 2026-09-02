"""ldm06 material-map pipeline (D40 §1).

Covers: sample_mask downsampling correctness, the uint8 store round-trip via
scripts/build_material_maps.py, LatentDataset serving of cond_material /
air_fraction / nb_air_fraction behind the `material` flag, and the all-air
patch cap.
"""

from __future__ import annotations

import itertools
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch
import zarr

from poregen.dataset.material import (
    air_fraction,
    decode_material_u8,
    encode_material_u8,
    patch_material_cells,
    pool_material_fractions,
)
from poregen.diffusion.conditioning import NB_EXISTS, NEIGHBOUR_DIRS, PARITY_GROUP_ORDER
from poregen.diffusion.latents import LatentDataset

REPO = Path(__file__).resolve().parents[1]
SCRIPT = REPO / "scripts" / "build_material_maps.py"

# Scaled-down production geometry (same convention as test_ldm05_conditioning):
# C=1, latent 4³, patch 16 voxels (factor 4), stride 8, touching offset 16.
C, L, PATCH, STRIDE, OFFSET, VOL = 1, 4, 16, 8, 16, 48
FACTOR = PATCH // L
N_SIDE = (VOL - PATCH) // STRIDE + 1          # 5 origins per axis
DS_KW = dict(sample_stride=STRIDE, generation_stride=OFFSET, neighbour_offset=OFFSET)


# ---------------------------------------------------------------------------
# Pure downsampling / encoding units
# ---------------------------------------------------------------------------

class TestPooling:

    def test_block_mean_matches_brute_force(self):
        rng = np.random.default_rng(3)
        mask = rng.random((24, 20, 16)) < 0.4
        pooled = pool_material_fractions(mask, 4)
        assert pooled.shape == (6, 5, 4)
        for i, j, k in itertools.product(range(6), range(5), range(4)):
            block = mask[4 * i:4 * i + 4, 4 * j:4 * j + 4, 4 * k:4 * k + 4]
            assert pooled[i, j, k] == pytest.approx(block.mean(), abs=1e-7)

    def test_remainder_is_cropped(self):
        mask = np.ones((10, 9, 11), bool)
        assert pool_material_fractions(mask, 4).shape == (2, 2, 2)

    def test_patch_cells_slices_the_right_block(self):
        rng = np.random.default_rng(4)
        mask = rng.random((VOL, VOL, VOL)) < 0.5
        pooled = pool_material_fractions(mask, FACTOR)
        z0, y0, x0 = 16, 8, 24
        cells = patch_material_cells(pooled, z0, y0, x0, FACTOR, L)
        patch = mask[z0:z0 + PATCH, y0:y0 + PATCH, x0:x0 + PATCH]
        np.testing.assert_allclose(cells, pool_material_fractions(patch, FACTOR),
                                   atol=1e-7)

    def test_misaligned_origin_raises(self):
        pooled = np.zeros((12, 12, 12), np.float32)
        with pytest.raises(ValueError, match="aligned"):
            patch_material_cells(pooled, 2, 0, 0, FACTOR, L)

    def test_air_fraction_is_exact_from_cells(self):
        rng = np.random.default_rng(5)
        patch = rng.random((PATCH,) * 3) < 0.3
        cells = pool_material_fractions(patch, FACTOR)
        assert air_fraction(cells) == pytest.approx(1.0 - patch.mean(), abs=1e-12)


class TestEncoding:

    def test_u8_round_trip_error_bound(self):
        rng = np.random.default_rng(6)
        cells = rng.random((L, L, L)).astype(np.float32)
        back = decode_material_u8(encode_material_u8(cells))
        assert np.abs(back - cells).max() <= 0.5 / 255 + 1e-7

    def test_extremes_are_exact(self):
        assert encode_material_u8(np.array([0.0, 1.0])).tolist() == [0, 255]
        assert decode_material_u8(np.array([0, 255], np.uint8)).tolist() == [0.0, 1.0]


# ---------------------------------------------------------------------------
# compute_sample_mask (segmentation side)
# ---------------------------------------------------------------------------

def test_compute_sample_mask_fills_internal_voids():
    from poregen.dataset.segmentation import compute_sample_mask

    xct = np.zeros((40, 40, 40), np.uint8)
    xct[8:32, 8:32, 8:32] = 200          # bright specimen cube
    xct[18:22, 18:22, 18:22] = 10        # dark internal pore
    sm = compute_sample_mask(xct)
    assert sm is not None
    assert sm[20, 20, 20]                # internal pore is filled: material envelope
    assert sm[16:24, 16:24, 16:24].all()
    assert not sm[2, 2, 2]               # exterior air stays out
    assert not sm[:6].any()


# ---------------------------------------------------------------------------
# Store fixture + the builder script end-to-end
# ---------------------------------------------------------------------------

def _make_store(root: Path, sample_mask: np.ndarray) -> pd.DataFrame:
    """A miniature latent store + volumes.zarr for one volume ``vol_a``."""
    rng = np.random.default_rng(0)
    coords = [c for c in itertools.product(range(0, VOL - PATCH + 1, STRIDE), repeat=3)]
    n = len(coords)
    rows = []
    data = np.zeros((n, 2 * C, L, L, L), np.float16)
    for i, (z0, y0, x0) in enumerate(coords):
        data[i, :C] = rng.standard_normal((C, L, L, L))
        data[i, C:] = 0.25
        rows.append({"source_row": i, "volume_id": "vol_a",
                     "z0": z0, "y0": y0, "x0": x0, "phi": 0.01})
    df = pd.DataFrame(rows)

    split_dir = root / "train"
    split_dir.mkdir(parents=True)
    data.tofile(split_dir / "latents.bin")
    df.to_parquet(split_dir / "index.parquet", index=False)
    pd.DataFrame({
        "source_row": df["source_row"].to_numpy(np.int64),
        "cond_depth": np.linspace(0, 1, n, dtype=np.float32),
        "cond_dist": np.linspace(0, 1, n, dtype=np.float32),
        "cond_por_raw": np.full(n, -4.0, np.float32),
    }).to_parquet(split_dir / "cond.parquet", index=False)

    of = root / "orientation_field.json"
    of.write_text(json.dumps({
        "voxel_size_um": 25.0,
        "volumes": {"vol_a": {"shape": [VOL] * 3, "orientation_usable": True,
                              "confidence": "high", "theta_deg": [45.0] * VOL}},
    }))
    (root / "metadata.json").write_text(json.dumps({
        "latent_shape": [C, L, L, L],
        "patch_size": PATCH,
        "voxel_size_um": 25.0,
        "storage": {"format": "memmap", "file": "latents.bin",
                    "dtype": "float16", "pack_scheme": "mu_then_std"},
        "normalization": {"computed_over": "train",
                          "per_channel_mean": [0.0] * C,
                          "per_channel_std": [1.0] * C},
        "assembly": {"parity_group_order": [list(g) for g in PARITY_GROUP_ORDER],
                     "sample_stride": STRIDE, "generation_stride": OFFSET,
                     "neighbour_offset": OFFSET},
        "conditioning": {"orientation_field": str(of),
                         "por_standardisation": {"mean": -4.0, "std": 1.0}},
    }))

    zroot = zarr.open_group(str(root / "volumes.zarr"), mode="a")
    grp = zroot.require_group("vol_a")
    grp.create_array("xct", data=(sample_mask * 200).astype(np.uint8),
                     chunks=(16, 16, 16), overwrite=True)
    grp.create_array("sample_mask", data=sample_mask.astype(np.uint8),
                     chunks=(16, 16, 16), overwrite=True)
    return df


@pytest.fixture(scope="module")
def built_store(tmp_path_factory):
    root = tmp_path_factory.mktemp("mat_store")
    rng = np.random.default_rng(11)
    sample_mask = rng.random((VOL, VOL, VOL)) < 0.6
    sample_mask[:, :, 32:] = False        # x ≥ 32 is pure exterior air →
    df = _make_store(root, sample_mask)   # every x0=32 patch is ALL AIR
    latents_before = (root / "train" / "latents.bin").read_bytes()

    res = subprocess.run(
        [sys.executable, str(SCRIPT), "--store", str(root),
         "--data-root", str(root), "--splits", "train"],
        capture_output=True, text=True,
    )
    assert res.returncode == 0, res.stderr
    assert (root / "train" / "latents.bin").read_bytes() == latents_before, \
        "the builder must not touch latents.bin"
    return root, df, sample_mask


class TestBuilderRoundTrip:

    def test_material_matches_brute_force(self, built_store):
        root, df, mask = built_store
        mat = np.memmap(root / "train" / "material.bin", np.uint8, "r",
                        shape=(len(df), L, L, L))
        for i in (0, 17, 62, 88, len(df) - 1):
            z0, y0, x0 = int(df.z0[i]), int(df.y0[i]), int(df.x0[i])
            patch = mask[z0:z0 + PATCH, y0:y0 + PATCH, x0:x0 + PATCH]
            expected = encode_material_u8(pool_material_fractions(patch, FACTOR))
            np.testing.assert_array_equal(mat[i], expected)

    def test_air_matches_patch_mean(self, built_store):
        root, df, mask = built_store
        air = np.memmap(root / "train" / "air.bin", np.float32, "r", shape=(len(df),))
        for i in (3, 41, 90):
            z0, y0, x0 = int(df.z0[i]), int(df.y0[i]), int(df.x0[i])
            patch = mask[z0:z0 + PATCH, y0:y0 + PATCH, x0:x0 + PATCH]
            assert air[i] == pytest.approx(1.0 - patch.mean(), abs=1e-6)

    def test_metadata_block_finalised(self, built_store):
        root, df, _ = built_store
        meta = json.loads((root / "metadata.json").read_text())
        block = meta["material"]
        assert block["material_dtype"] == "uint8"
        assert block["files"] == {"material": "material.bin", "air": "air.bin"}
        assert block["per_split_stats"]["train"]["n"] == len(df)
        # the x0=32 plane: 25 of 125 patches are all air
        assert block["per_split_stats"]["train"]["frac_all_air"] == pytest.approx(
            (N_SIDE ** 2) / (N_SIDE ** 3))

    def test_rerun_is_a_noop(self, built_store):
        root, *_ = built_store
        before = (root / "train" / "material.bin").read_bytes()
        res = subprocess.run(
            [sys.executable, str(SCRIPT), "--store", str(root),
             "--data-root", str(root), "--splits", "train"],
            capture_output=True, text=True,
        )
        assert res.returncode == 0, res.stderr
        assert (root / "train" / "material.bin").read_bytes() == before


# ---------------------------------------------------------------------------
# LatentDataset serving
# ---------------------------------------------------------------------------

class TestLatentDatasetMaterial:

    def test_flag_off_keeps_the_ldm05_contract(self, built_store):
        root, *_ = built_store
        item = LatentDataset(root, "train", **DS_KW)[0]
        for key in ("cond_material", "air_fraction", "nb_air_fraction"):
            assert key not in item

    def test_shapes_dtypes_and_values(self, built_store):
        root, df, mask = built_store
        ds = LatentDataset(root, "train", material=True, **DS_KW)
        i = 62
        item = ds[i]
        assert item["cond_material"].shape == (1, L, L, L)
        assert item["cond_material"].dtype == torch.float32
        assert item["air_fraction"].shape == ()
        assert item["nb_air_fraction"].shape == (6,)
        z0, y0, x0 = int(df.z0[i]), int(df.y0[i]), int(df.x0[i])
        patch = mask[z0:z0 + PATCH, y0:y0 + PATCH, x0:x0 + PATCH]
        expected = decode_material_u8(
            encode_material_u8(pool_material_fractions(patch, FACTOR)))
        np.testing.assert_allclose(item["cond_material"][0].numpy(), expected)
        assert item["air_fraction"].item() == pytest.approx(1.0 - patch.mean(), abs=1e-6)
        assert 0.0 <= item["cond_material"].min() <= item["cond_material"].max() <= 1.0

    def test_neighbour_air_fractions(self, built_store):
        root, df, _ = built_store
        ds = LatentDataset(root, "train", material=True, **DS_KW)
        air = np.memmap(root / "train" / "air.bin", np.float32, "r", shape=(len(df),))
        lookup = {(int(r.z0), int(r.y0), int(r.x0)): i for i, r in df.iterrows()}

        # interior patch: all six neighbours are stored
        i = lookup[(16, 16, 16)]
        nb_air = ds[i]["nb_air_fraction"].numpy()
        for k, d in enumerate(NEIGHBOUR_DIRS):
            j = lookup[(16 + OFFSET * d[0], 16 + OFFSET * d[1], 16 + OFFSET * d[2])]
            assert nb_air[k] == pytest.approx(float(air[j]), abs=1e-7)

        # corner patch: negative-direction neighbours are absent → all air (1.0)
        i = lookup[(0, 0, 0)]
        item = ds[i]
        nb_air = item["nb_air_fraction"].numpy()
        for k, d in enumerate(NEIGHBOUR_DIRS):
            if min(d) < 0:
                assert nb_air[k] == 1.0
            else:
                j = lookup[(OFFSET * d[0], OFFSET * d[1], OFFSET * d[2])]
                assert nb_air[k] == pytest.approx(float(air[j]), abs=1e-7)

    def test_stored_neighbour_air_is_served_even_when_unknown(self, built_store):
        # D40 §4: the material map is paintable before generation, so air
        # fractions are conditioning for ALL stored neighbours, not just EXISTS.
        root, df, _ = built_store
        ds = LatentDataset(root, "train", material=True, **DS_KW)
        air = np.memmap(root / "train" / "air.bin", np.float32, "r", shape=(len(df),))
        lookup = {(int(r.z0), int(r.y0), int(r.x0)): i for i, r in df.iterrows()}
        # origin (16, 32, 16): grid parity (1, 0, 1), so the stored -y
        # neighbour at (16, 16, 16) is UNKNOWN in the schedule.
        i = lookup[(16, 32, 16)]
        item = ds[i]
        k = NEIGHBOUR_DIRS.index((0, -1, 0))
        assert item["nb_avail"][k].item() != NB_EXISTS
        j = lookup[(16, 16, 16)]
        assert item["nb_air_fraction"][k].item() == pytest.approx(float(air[j]), abs=1e-7)

    def test_material_without_built_files_raises(self, tmp_path):
        rng = np.random.default_rng(1)
        _make_store(tmp_path, rng.random((VOL, VOL, VOL)) < 0.5)
        with pytest.raises(FileNotFoundError, match="build_material_maps"):
            LatentDataset(tmp_path, "train", material=True, **DS_KW)

    def test_cap_requires_material(self, built_store):
        root, *_ = built_store
        with pytest.raises(ValueError, match="material=True"):
            LatentDataset(root, "train", air_patch_cap=0.05, **DS_KW)


class TestAirPatchCap:

    def test_default_off_serves_everything(self, built_store):
        root, df, _ = built_store
        ds = LatentDataset(root, "train", material=True, **DS_KW)
        assert len(ds) == len(df)

    def test_cap_limits_the_all_air_fraction(self, built_store):
        root, df, _ = built_store
        cap = 0.05
        ds = LatentDataset(root, "train", material=True, air_patch_cap=cap, **DS_KW)
        n_air_total = N_SIDE ** 2                       # the x0=32 plane
        assert len(ds) < len(df)
        served_air = sum(ds[i]["air_fraction"].item() >= 0.999 for i in range(len(ds)))
        assert served_air < n_air_total
        assert served_air / len(ds) <= cap + 1e-9

    def test_cap_is_deterministic(self, built_store):
        root, *_ = built_store
        a = LatentDataset(root, "train", material=True, air_patch_cap=0.05, **DS_KW)
        b = LatentDataset(root, "train", material=True, air_patch_cap=0.05, **DS_KW)
        np.testing.assert_array_equal(a._sel, b._sel)

    def test_capped_rows_still_serve_as_neighbours(self, built_store):
        # A kept patch next to the air plane must still see its (possibly
        # dropped) all-air neighbour as a stored row, not as OOB.
        root, df, _ = built_store
        ds = LatentDataset(root, "train", material=True, air_patch_cap=0.0, **DS_KW)
        # cap 0.0 drops every all-air row from sampling
        assert all(ds[i]["air_fraction"].item() < 0.999 for i in range(len(ds)))
        lookup = {(int(r.z0), int(r.y0), int(r.x0)): i for i, r in df.iterrows()}
        row = lookup[(16, 16, 16)]           # parity (1,1,1): all neighbours EXISTS
        pos = int(np.searchsorted(ds._sel, row))
        assert ds._sel[pos] == row
        item = ds[pos]
        k = NEIGHBOUR_DIRS.index((0, 0, 1))  # +x neighbour is the dropped all-air row
        assert item["nb_avail"][k].item() == NB_EXISTS
        assert item["nb_air_fraction"][k].item() == 1.0
        assert item["nb_latents"][k].abs().sum() > 0  # content still served
