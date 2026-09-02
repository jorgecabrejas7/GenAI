"""Shapes, ranges and the 3-class label contract of both patch loaders."""

import json

import numpy as np
import pandas as pd
import pytest
import zarr
from zarr.codecs import BloscCodec

from poregen.dataset.loader import (
    LABEL_AIR,
    LABEL_MATERIAL,
    LABEL_PORE,
    MemmapPatchDataset,
    PatchDataset,
    build_label,
)

PS = 8
SHAPE = (16, 16, 16)
VOL_ID = "test_vol"
SOURCE_GROUP = "MedidasDB"


@pytest.fixture()
def tiny_dataset(tmp_path):
    """A minimal Zarr volume + Parquet index, with all three label classes."""
    store_dir = tmp_path / "volumes.zarr"
    root = zarr.open_group(str(store_dir), mode="w")
    grp = root.create_group(VOL_ID)

    rng = np.random.default_rng(42)
    xct = rng.integers(0, 256, size=SHAPE, dtype=np.uint8)
    mask = rng.integers(0, 2, size=SHAPE, dtype=np.uint8)
    # sample_mask False in a corner block — the stand-in for a drilled hole,
    # so every patch of the fixture is not all-material.
    sample_mask = np.ones(SHAPE, dtype=np.uint8)
    sample_mask[:, :4, :4] = 0

    compressor = BloscCodec(cname="zstd", clevel=1)
    for name, arr in (("xct", xct), ("mask", mask), ("sample_mask", sample_mask)):
        grp.create_array(name, data=arr, chunks=(8, 8, 8), compressors=compressor)

    rows = []
    for z0 in range(0, SHAPE[0] - PS + 1, PS):
        for y0 in range(0, SHAPE[1] - PS + 1, PS):
            for x0 in range(0, SHAPE[2] - PS + 1, PS):
                sl = np.s_[z0:z0 + PS, y0:y0 + PS, x0:x0 + PS]
                rows.append({
                    "volume_id": VOL_ID,
                    "source_group": SOURCE_GROUP,
                    "split": "train",
                    "z0": z0, "y0": y0, "x0": x0,
                    "ps": PS, "stride": PS,
                    "porosity": float(mask[sl].mean()),
                    "air_fraction": float((sample_mask[sl] == 0).mean()),
                    "panel_id": "P1",
                })
    rows[0] = {**rows[0], "split": "val"}

    df = pd.DataFrame(rows)
    index_path = tmp_path / "patch_index.parquet"
    df.to_parquet(str(index_path), index=False)

    return tmp_path, index_path, xct, mask, sample_mask, df


@pytest.fixture()
def tiny_memmap(tiny_dataset):
    """Write the memmap backend for the same fixture."""
    root, index_path, xct, mask, sample_mask, df = tiny_dataset
    label_vol = build_label(mask, sample_mask)
    n = len(df)
    shape = (n, PS, PS, PS)

    mm_x = np.memmap(str(root / "patches_xct.bin"), dtype=np.uint8,
                     mode="w+", shape=shape)
    mm_l = np.memmap(str(root / "patches_label.bin"), dtype=np.uint8,
                     mode="w+", shape=shape)
    for i, r in df.iterrows():
        sl = np.s_[r.z0:r.z0 + PS, r.y0:r.y0 + PS, r.x0:r.x0 + PS]
        mm_x[i] = xct[sl]
        mm_l[i] = label_vol[sl]
    mm_x.flush()
    mm_l.flush()
    del mm_x, mm_l

    (root / "patches_meta.json").write_text(json.dumps({
        "N": n, "patch_size": PS, "stride": PS, "shape": list(shape),
    }))
    return root, index_path, label_vol


# ---------------------------------------------------------------------------
# The label rule
# ---------------------------------------------------------------------------

def test_build_label_precedence():
    mask = np.array([[[0, 1, 0, 1]]], dtype=np.uint8)
    sample = np.array([[[1, 1, 0, 0]]], dtype=np.uint8)
    got = build_label(mask, sample)
    # material, pore, air, air — air wins wherever sample_mask is 0.
    assert got.tolist() == [[[LABEL_MATERIAL, LABEL_PORE, LABEL_AIR, LABEL_AIR]]]


# ---------------------------------------------------------------------------
# Zarr backend
# ---------------------------------------------------------------------------

class TestPatchDataset:

    def test_shapes(self, tiny_dataset):
        root, idx, *_ = tiny_dataset
        s = PatchDataset(idx, root, split="train")[0]
        assert s["xct"].shape == (1, PS, PS, PS)
        assert s["mask"].shape == (1, PS, PS, PS)
        assert s["label"].shape == (PS, PS, PS)

    def test_xct_range(self, tiny_dataset):
        root, idx, *_ = tiny_dataset
        s = PatchDataset(idx, root, split="train")[0]
        assert s["xct"].min() >= 0.0
        assert s["xct"].max() <= 1.0

    def test_label_classes(self, tiny_dataset):
        root, idx, *_ = tiny_dataset
        ds = PatchDataset(idx, root, split="train")
        seen = set()
        for i in range(len(ds)):
            seen |= set(ds[i]["label"].unique().tolist())
        assert seen <= {LABEL_MATERIAL, LABEL_PORE, LABEL_AIR}
        assert LABEL_AIR in seen, "fixture must contain air voxels"

    def test_mask_is_pore_class(self, tiny_dataset):
        root, idx, *_ = tiny_dataset
        ds = PatchDataset(idx, root, split="train")
        for i in range(len(ds)):
            s = ds[i]
            assert s["mask"].squeeze(0).eq(1.0).equal(s["label"].eq(LABEL_PORE))

    def test_val_split(self, tiny_dataset):
        root, idx, *_ = tiny_dataset
        assert len(PatchDataset(idx, root, split="val")) == 1

    def test_dict_keys(self, tiny_dataset):
        root, idx, *_ = tiny_dataset
        s = PatchDataset(idx, root, split="train")[0]
        assert set(s.keys()) == {"xct", "label", "mask", "volume_id", "coords",
                                 "porosity", "source_group"}

    def test_porosity_non_negative(self, tiny_dataset):
        root, idx, *_ = tiny_dataset
        ds = PatchDataset(idx, root, split="train")
        assert all(ds[i]["porosity"] >= 0.0 for i in range(len(ds)))


# ---------------------------------------------------------------------------
# Memmap backend
# ---------------------------------------------------------------------------

class TestMemmapPatchDataset:

    def test_shapes_and_keys(self, tiny_memmap):
        root, idx, _ = tiny_memmap
        s = MemmapPatchDataset(idx, root, split="train")[0]
        assert s["xct"].shape == (1, PS, PS, PS)
        assert s["mask"].shape == (1, PS, PS, PS)
        assert s["label"].shape == (PS, PS, PS)
        assert set(s.keys()) == {"xct", "label", "mask", "volume_id", "coords",
                                 "porosity", "source_group"}

    def test_matches_zarr_backend(self, tiny_dataset, tiny_memmap):
        root, idx, *_ = tiny_dataset
        z = PatchDataset(idx, root, split="train")
        m = MemmapPatchDataset(idx, root, split="train")
        assert len(z) == len(m)
        for i in range(len(z)):
            assert z[i]["label"].equal(m[i]["label"])
            assert z[i]["xct"].equal(m[i]["xct"])

    def test_mask_is_pore_class(self, tiny_memmap):
        root, idx, _ = tiny_memmap
        ds = MemmapPatchDataset(idx, root, split="train")
        for i in range(len(ds)):
            s = ds[i]
            assert s["mask"].squeeze(0).eq(1.0).equal(s["label"].eq(LABEL_PORE))
