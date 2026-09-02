"""Tests for poregen.diffusion.latents.LatentDataset (ldm04 latent store)."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest
import torch
import zarr

from poregen.diffusion.latents import LatentDataset

C, S = 4, 16
N = 12
MEAN = [0.5, -0.25, 1.0, 0.0]
STD = [2.0, 0.5, 1.5, 1.0]


@pytest.fixture(scope="module")
def store_root(tmp_path_factory):
    """Build a tiny latent store with the layout of build_latent_dataset.py."""
    root = tmp_path_factory.mktemp("latents_r07z4")
    rng = np.random.default_rng(0)

    (root / "metadata.json").write_text(json.dumps({
        "latent_shape": [C, S, S, S],
        "dtype": "float16",
        "arrays": ["mu", "std"],
        "normalization": {
            "computed_over": "train",
            "per_channel_mean": MEAN,
            "per_channel_std": STD,
        },
    }))

    for split in ("train", "val"):
        split_dir = root / split
        split_dir.mkdir()
        group = zarr.open_group(str(split_dir / "latents.zarr"), mode="w")
        mu = rng.normal(size=(N, C, S, S, S)).astype(np.float16)
        std = rng.uniform(0.1, 1.0, size=(N, C, S, S, S)).astype(np.float16)
        group.create_array("mu", shape=mu.shape, chunks=(1, C, S, S, S), dtype="float16")[:] = mu
        group.create_array("std", shape=std.shape, chunks=(1, C, S, S, S), dtype="float16")[:] = std

        pd.DataFrame({
            "source_row": np.arange(100, 100 + N, dtype=np.int64),
            "volume_id": [f"vol_{i % 3}" for i in range(N)],
            "source_group": ["g"] * N,
            "split": [split] * N,
            "z0": np.arange(N) * 64,
            "y0": np.zeros(N, dtype=np.int64),
            "x0": np.zeros(N, dtype=np.int64),
            "ps": np.full(N, 64),
            "stride": np.full(N, 32),
            "porosity": rng.uniform(0, 0.1, N).astype(np.float32),
            "phi": rng.uniform(0, 0.1, N).astype(np.float32),
        }).to_parquet(str(split_dir / "index.parquet"), index=False)

    return root


def test_shapes_and_metadata(store_root):
    ds = LatentDataset(store_root, "train")
    assert len(ds) == N
    item = ds[3]
    assert item["z"].shape == (C, S, S, S)
    assert item["z"].dtype == torch.float32
    assert item["std"].shape == (C, S, S, S)
    assert item["phi"].shape == (1,)
    assert item["coords"].tolist() == [3 * 64, 0, 0]
    assert item["source_row"] == 103
    assert item["volume_id"] == "vol_0"


def test_phi_matches_index(store_root):
    ds = LatentDataset(store_root, "val")
    df = pd.read_parquet(str(store_root / "val" / "index.parquet"))
    for i in (0, 5, N - 1):
        assert ds[i]["phi"].item() == pytest.approx(float(df["phi"].iloc[i]))


def test_normalization_and_roundtrip(store_root):
    raw = LatentDataset(store_root, "train")
    norm = LatentDataset(store_root, "train", normalize=True)

    z_raw, z_norm = raw[0]["z"], norm[0]["z"]

    mean = torch.tensor(MEAN).view(C, 1, 1, 1)
    std = torch.tensor(STD).view(C, 1, 1, 1)
    assert torch.allclose(z_norm, (z_raw - mean) / std)

    # posterior std scales by 1/std_c under the same affine map
    assert torch.allclose(norm[0]["std"], raw[0]["std"] / std)

    # de-normalization round-trips
    assert torch.allclose(norm.denormalize_latent(z_norm), z_raw, atol=1e-6)
    assert torch.allclose(raw.normalize_latent(raw.denormalize_latent(z_raw)), z_raw, atol=1e-6)


def test_row_count_mismatch_raises(store_root, tmp_path):
    import shutil as sh

    bad = tmp_path / "bad_store"
    sh.copytree(store_root, bad)
    df = pd.read_parquet(str(bad / "train" / "index.parquet"))
    df.iloc[:-1].to_parquet(str(bad / "train" / "index.parquet"), index=False)
    with pytest.raises(RuntimeError, match="rebuild"):
        LatentDataset(bad, "train")
