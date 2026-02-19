"""Test that the PatchDataset returns tensors with correct shapes and ranges."""

import numpy as np
import pandas as pd
import pytest
import zarr
from zarr.codecs import BloscCodec

from poregen.dataset.loader import PatchDataset


@pytest.fixture()
def tiny_dataset(tmp_path):
    """Create a minimal Zarr volume + Parquet index for testing."""
    ps = 8
    vol_id = "test_vol"
    source_group = "MedidasDB"
    shape = (16, 16, 16)

    # ------ Zarr volume ------
    store_dir = tmp_path / "volumes.zarr"
    root = zarr.open_group(str(store_dir), mode="w")
    grp = root.create_group(vol_id)

    rng = np.random.default_rng(42)
    xct = rng.integers(0, 256, size=shape, dtype=np.uint8)
    mask = rng.integers(0, 2, size=shape, dtype=np.uint8)

    compressor = BloscCodec(cname="zstd", clevel=1)
    grp.create_array("xct", data=xct, chunks=(8, 8, 8), compressors=compressor)
    grp.create_array("mask", data=mask, chunks=(8, 8, 8), compressors=compressor)

    # ------ Parquet index ------
    rows = []
    for z0 in range(0, shape[0] - ps + 1, ps):
        for y0 in range(0, shape[1] - ps + 1, ps):
            for x0 in range(0, shape[2] - ps + 1, ps):
                rows.append(
                    {
                        "volume_id": vol_id,
                        "source_group": source_group,
                        "split": "train",
                        "z0": z0,
                        "y0": y0,
                        "x0": x0,
                        "ps": ps,
                        "stride": ps,
                        "porosity": float(mask[z0 : z0 + ps, y0 : y0 + ps, x0 : x0 + ps].mean()),
                    }
                )
    # Add one val patch
    rows[0] = {**rows[0], "split": "val"}

    df = pd.DataFrame(rows)
    index_path = tmp_path / "patch_index.parquet"
    df.to_parquet(str(index_path), index=False)

    return tmp_path, index_path, ps, xct, mask


class TestDatasetLoaderShapes:

    def test_train_shape(self, tiny_dataset):
        root, idx_path, ps, _xct, _mask = tiny_dataset
        ds = PatchDataset(idx_path, root, split="train")
        assert len(ds) > 0
        sample = ds[0]
        assert sample["xct"].shape == (1, ps, ps, ps)
        assert sample["mask"].shape == (1, ps, ps, ps)

    def test_xct_range(self, tiny_dataset):
        root, idx_path, ps, _xct, _mask = tiny_dataset
        ds = PatchDataset(idx_path, root, split="train")
        sample = ds[0]
        assert sample["xct"].min() >= 0.0
        assert sample["xct"].max() <= 1.0

    def test_mask_values(self, tiny_dataset):
        root, idx_path, ps, _xct, _mask = tiny_dataset
        ds = PatchDataset(idx_path, root, split="train")
        sample = ds[0]
        unique_vals = set(sample["mask"].unique().tolist())
        assert unique_vals <= {0.0, 1.0}

    def test_val_split(self, tiny_dataset):
        root, idx_path, ps, _xct, _mask = tiny_dataset
        ds = PatchDataset(idx_path, root, split="val")
        assert len(ds) == 1

    def test_dict_keys(self, tiny_dataset):
        root, idx_path, ps, _xct, _mask = tiny_dataset
        ds = PatchDataset(idx_path, root, split="train")
        sample = ds[0]
        expected_keys = {"xct", "mask", "volume_id", "coords", "porosity", "source_group"}
        assert set(sample.keys()) == expected_keys

    def test_porosity_non_negative(self, tiny_dataset):
        root, idx_path, ps, _xct, _mask = tiny_dataset
        ds = PatchDataset(idx_path, root, split="train")
        for i in range(len(ds)):
            assert ds[i]["porosity"] >= 0.0
