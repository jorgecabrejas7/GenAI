"""Regression tests for LatentPatchDataset and SampledLatentPatchDataset.

Tests that both memmap and zarr backends produce bit-identical reads, that
the fallback / error paths behave as documented, and that basic construction
invariants hold.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest
import torch
import zarr

from poregen.diffusion.latent_dataset import LatentPatchDataset
from poregen.diffusion.sampled_latent_dataset import SampledLatentPatchDataset

# ---------------------------------------------------------------------------
# Shared constants
# ---------------------------------------------------------------------------

_N       = 8
_Z_CH    = 4
_SPATIAL = (4, 4, 4)   # tiny spatial dims for fast tests

# 8 patches: 2 volumes × 4 patches each
_PATCH_GRID = [
    (0, 0, 0),
    (0, 0, 64),
    (0, 64, 0),
    (0, 64, 64),
]


# ---------------------------------------------------------------------------
# Helper: build the shared latents_index.parquet
# ---------------------------------------------------------------------------

def _build_index_df() -> pd.DataFrame:
    """Return the 8-row DataFrame that represents two volumes with 4 patches each."""
    rows: list[dict[str, Any]] = []
    for vol_id in ("vol_A", "vol_B"):
        for z0, y0, x0 in _PATCH_GRID:
            grid_iz = z0 // 64
            grid_iy = y0 // 64
            grid_ix = x0 // 64
            rows.append(
                {
                    "volume_id":    vol_id,
                    "z0":           z0,
                    "y0":           y0,
                    "x0":           x0,
                    "ps":           64,
                    "stride":       64,
                    "porosity":     0.1,
                    "vol_porosity": 0.1,
                    "split":        "train",
                    "source_group": "test",
                    "vol_depth":    192,
                    "vol_height":   192,
                    "vol_width":    192,
                    "grid_iz":      grid_iz,
                    "grid_iy":      grid_iy,
                    "grid_ix":      grid_ix,
                    "parity":       (grid_iz + grid_iy + grid_ix) % 2,
                }
            )
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Helper: build ldm01 fixture dirs (LatentPatchDataset)
# ---------------------------------------------------------------------------

def _build_ldm01_dirs(tmp_path: Path) -> tuple[Path, Path]:
    """Build zarr_dir and mmap_dir for LatentPatchDataset tests.

    Returns
    -------
    (zarr_dir, mmap_dir)
    """
    rng  = np.random.default_rng(42)
    data = rng.random((_N, _Z_CH, *_SPATIAL)).astype(np.float16)

    df = _build_index_df()

    stats_content = json.dumps({"std": 1.0})
    meta_content  = json.dumps(
        {
            "N":           _N,
            "n_channels":  _Z_CH,
            "spatial":     list(_SPATIAL),
            "dtype":       "float16",
            "pack_scheme": "none",
            "z_channels":  _Z_CH,
        }
    )

    # --- zarr_dir -----------------------------------------------------------
    zarr_dir = tmp_path / "zarr_dir"
    zarr_dir.mkdir()
    df.to_parquet(str(zarr_dir / "latents_index.parquet"), index=True)
    (zarr_dir / "latent_scale_stats.json").write_text(stats_content)

    z_arr = zarr.open_array(
        str(zarr_dir / "latents.zarr" / "latents"),
        mode="w",
        zarr_format=2,
        shape=(_N, _Z_CH, *_SPATIAL),
        chunks=(_N, _Z_CH, *_SPATIAL),
        compressor=None,
        dtype="float16",
    )
    z_arr[:] = data

    # --- mmap_dir -----------------------------------------------------------
    mmap_dir = tmp_path / "mmap_dir"
    mmap_dir.mkdir()
    df.to_parquet(str(mmap_dir / "latents_index.parquet"), index=True)
    (mmap_dir / "latent_scale_stats.json").write_text(stats_content)
    (mmap_dir / "latents_meta.json").write_text(meta_content)

    mmap = np.memmap(
        str(mmap_dir / "latents.bin"),
        dtype=np.float16,
        mode="w+",
        shape=(_N, _Z_CH, *_SPATIAL),
    )
    mmap[:] = data
    del mmap   # flush and close

    return zarr_dir, mmap_dir


# ---------------------------------------------------------------------------
# Helper: build ldm02 fixture dirs (SampledLatentPatchDataset)
# ---------------------------------------------------------------------------

def _build_ldm02_dirs(tmp_path: Path) -> tuple[Path, Path]:
    """Build zarr_s_dir and mmap_s_dir for SampledLatentPatchDataset tests.

    The packed array has shape (N, 2*z_ch, *spatial):
    channels 0..z_ch-1 = mu, channels z_ch..2*z_ch-1 = logvar.

    zarr_s_dir  — zarr store only (no memmap files)
    mmap_s_dir  — memmap backend (latents.bin + latents_meta.json with
                  pack_scheme="mu_then_logvar"); no zarr store

    Returns
    -------
    (zarr_s_dir, mmap_s_dir)
    """
    total_ch = _Z_CH * 2   # 8 packed channels
    rng      = np.random.default_rng(99)
    data     = rng.random((_N, total_ch, *_SPATIAL)).astype(np.float16)

    df = _build_index_df()

    stats_content = json.dumps({"z_channels": _Z_CH, "std": 1.0})
    meta_content  = json.dumps(
        {
            "N":           _N,
            "n_channels":  total_ch,
            "spatial":     list(_SPATIAL),
            "dtype":       "float16",
            "pack_scheme": "mu_then_logvar",
            "z_channels":  _Z_CH,
        }
    )

    # --- zarr_s_dir: only zarr store, no memmap files -----------------------
    zarr_s_dir = tmp_path / "zarr_s_dir"
    zarr_s_dir.mkdir()
    df.to_parquet(str(zarr_s_dir / "latents_index.parquet"), index=True)
    (zarr_s_dir / "latent_scale_stats.json").write_text(stats_content)

    z_arr = zarr.open_array(
        str(zarr_s_dir / "latents.zarr" / "latents"),
        mode="w",
        zarr_format=2,
        shape=(_N, total_ch, *_SPATIAL),
        chunks=(_N, total_ch, *_SPATIAL),
        compressor=None,
        dtype="float16",
    )
    z_arr[:] = data

    # --- mmap_s_dir: only memmap backend, no zarr store ---------------------
    mmap_s_dir = tmp_path / "mmap_s_dir"
    mmap_s_dir.mkdir()
    df.to_parquet(str(mmap_s_dir / "latents_index.parquet"), index=True)
    (mmap_s_dir / "latent_scale_stats.json").write_text(stats_content)
    (mmap_s_dir / "latents_meta.json").write_text(meta_content)

    mmap = np.memmap(
        str(mmap_s_dir / "latents.bin"),
        dtype=np.float16,
        mode="w+",
        shape=(_N, total_ch, *_SPATIAL),
    )
    mmap[:] = data
    del mmap   # flush and close

    return zarr_s_dir, mmap_s_dir


# ===========================================================================
# LatentPatchDataset tests
# ===========================================================================


def test_latent_zarr_opens(tmp_path: Path) -> None:
    """LatentPatchDataset opens the zarr backend and exposes correct metadata."""
    zarr_dir, _ = _build_ldm01_dirs(tmp_path)
    ds = LatentPatchDataset(str(zarr_dir), "train")

    assert ds.z_channels == _Z_CH
    assert len(ds) == _N

    item = ds[0]
    for key in ("z", "nb_latents", "nb_avail", "pos_frac"):
        assert key in item, f"key {key!r} missing from __getitem__ output"

    assert item["z"].shape == torch.Size([_Z_CH, *_SPATIAL])
    assert item["nb_latents"].shape == torch.Size([6, _Z_CH, *_SPATIAL])
    assert item["nb_avail"].shape == torch.Size([6])


def test_latent_mmap_opens(tmp_path: Path) -> None:
    """LatentPatchDataset opens the memmap backend and exposes correct metadata."""
    _, mmap_dir = _build_ldm01_dirs(tmp_path)
    ds = LatentPatchDataset(str(mmap_dir), "train")

    assert ds.z_channels == _Z_CH
    assert len(ds) == _N

    item = ds[0]
    for key in ("z", "nb_latents", "nb_avail", "pos_frac"):
        assert key in item, f"key {key!r} missing from __getitem__ output"

    assert item["z"].shape == torch.Size([_Z_CH, *_SPATIAL])


def test_latent_backends_bit_identical(tmp_path: Path) -> None:
    """Memmap and zarr backends return bit-identical latents for every index.

    With latent_std=1.0 the normalisation is a no-op, so both backends read
    the same float16 data and cast to float32 via np.array(..., dtype=float32).
    The nb_avail availability masks are also checked for equality.
    """
    zarr_dir, mmap_dir = _build_ldm01_dirs(tmp_path)

    # zarr_dir has no latents.bin → zarr fallback path
    # mmap_dir has latents.bin + latents_meta.json → memmap backend (takes priority)
    ds_zarr = LatentPatchDataset(str(zarr_dir), "train")
    ds_mmap = LatentPatchDataset(str(mmap_dir), "train")

    for idx in range(_N):
        item_z = ds_zarr[idx]
        item_m = ds_mmap[idx]
        assert torch.equal(item_z["z"], item_m["z"]), (
            f"z mismatch at idx={idx}: "
            f"max_diff={(item_z['z'] - item_m['z']).abs().max().item()}"
        )
        assert torch.equal(item_z["nb_avail"], item_m["nb_avail"]), (
            f"nb_avail mismatch at idx={idx}"
        )


def test_latent_fallback_no_bin(tmp_path: Path) -> None:
    """LatentPatchDataset falls back to zarr when no memmap files are present.

    The zarr_dir fixture has no latents.bin, so the zarr fallback path is
    exercised.  Confirm the dataset opens without error and z_channels is set.
    """
    zarr_dir, _ = _build_ldm01_dirs(tmp_path)
    # Sanity: no bin in zarr_dir
    assert not (zarr_dir / "latents.bin").exists()
    assert not (zarr_dir / "latents_meta.json").exists()

    ds = LatentPatchDataset(str(zarr_dir), "train")
    # Dataset must still be usable
    assert ds.z_channels == _Z_CH
    assert len(ds) == _N


def test_latent_error_no_store(tmp_path: Path) -> None:
    """LatentPatchDataset raises FileNotFoundError when no store is present."""
    df = _build_index_df()
    empty_dir = tmp_path / "empty"
    empty_dir.mkdir()
    df.to_parquet(str(empty_dir / "latents_index.parquet"), index=True)
    # No zarr, no bin → must raise
    with pytest.raises(FileNotFoundError):
        LatentPatchDataset(str(empty_dir), "train")


def test_latent_parquet_bin_drift(tmp_path: Path) -> None:
    """Mismatch between parquet row count and latents_meta.json N raises AssertionError."""
    _, mmap_dir = _build_ldm01_dirs(tmp_path)

    # Copy to a new dir and corrupt the meta
    drift_dir = tmp_path / "drift_dir"
    shutil.copytree(str(mmap_dir), str(drift_dir))

    meta_path = drift_dir / "latents_meta.json"
    with open(meta_path) as f:
        meta = json.load(f)
    meta["N"] = 999   # deliberate mismatch — parquet has 8 rows
    with open(meta_path, "w") as f:
        json.dump(meta, f)

    with pytest.raises(AssertionError):
        LatentPatchDataset(str(drift_dir), "train")


# ===========================================================================
# SampledLatentPatchDataset tests
# ===========================================================================


def test_sampled_backends_bit_identical(tmp_path: Path) -> None:
    """Both SampledLatentPatchDataset instances return bit-identical z given the same seed.

    zarr_s_dir uses the zarr backend; mmap_s_dir uses the memmap backend.
    Both stores are written from the same numpy array (rng seed 99), so with
    an identical manual_seed the reparameterisation z = mu + sigma*eps produces
    identical results.
    """
    zarr_s_dir, mmap_s_dir = _build_ldm02_dirs(tmp_path)

    ds_zarr = SampledLatentPatchDataset(str(zarr_s_dir), "train")
    ds_mmap = SampledLatentPatchDataset(str(mmap_s_dir), "train")

    assert ds_zarr.z_channels == _Z_CH
    assert ds_mmap.z_channels == _Z_CH
    assert len(ds_zarr) == _N
    assert len(ds_mmap) == _N

    for idx in range(_N):
        torch.manual_seed(7)
        item_z = ds_zarr[idx]

        torch.manual_seed(7)
        item_m = ds_mmap[idx]

        assert torch.equal(item_z["z"], item_m["z"]), (
            f"z mismatch at idx={idx}: "
            f"max_diff={(item_z['z'] - item_m['z']).abs().max().item()}"
        )
        assert torch.equal(item_z["nb_latents"], item_m["nb_latents"]), (
            f"nb_latents mismatch at idx={idx}"
        )


def test_sampled_wrong_pack_scheme(tmp_path: Path) -> None:
    """SampledLatentPatchDataset raises ValueError when pack_scheme is not 'mu_then_logvar'.

    A memmap meta with pack_scheme='none' causes the dataset to raise ValueError
    immediately upon construction, before any data is accessed.
    """
    bad_dir = tmp_path / "bad_dir"
    bad_dir.mkdir()

    df = _build_index_df()
    df.to_parquet(str(bad_dir / "latents_index.parquet"), index=True)
    (bad_dir / "latent_scale_stats.json").write_text(json.dumps({"z_channels": _Z_CH, "std": 1.0}))

    # pack_scheme "none" is invalid for SampledLatentPatchDataset
    (bad_dir / "latents_meta.json").write_text(
        json.dumps(
            {
                "N":           _N,
                "n_channels":  _Z_CH,
                "spatial":     list(_SPATIAL),
                "dtype":       "float16",
                "pack_scheme": "none",
                "z_channels":  _Z_CH,
            }
        )
    )

    # Write a valid-sized memmap so the memmap backend branch is entered
    rng  = np.random.default_rng(0)
    data = rng.random((_N, _Z_CH, *_SPATIAL)).astype(np.float16)
    mmap = np.memmap(
        str(bad_dir / "latents.bin"),
        dtype=np.float16,
        mode="w+",
        shape=(_N, _Z_CH, *_SPATIAL),
    )
    mmap[:] = data
    del mmap

    with pytest.raises(ValueError):
        SampledLatentPatchDataset(str(bad_dir), "train")
