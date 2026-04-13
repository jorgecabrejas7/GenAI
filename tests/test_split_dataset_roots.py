"""Tests for stratified split helpers and lightweight split dataset roots."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from poregen.dataset.splits import (
    assign_stratified_volume_splits,
    write_split_dataset_root,
)


def test_assign_stratified_volume_splits_covers_each_bin():
    volume_stats = pd.DataFrame(
        {
            "volume_id": [
                "lo_0", "lo_1", "lo_2",
                "mid_0", "mid_1", "mid_2",
                "hi_0", "hi_1", "hi_2",
                "vhi_0", "vhi_1", "vhi_2",
                "ultra_0", "ultra_1", "ultra_2",
                "excluded",
            ],
            "phi_median": [
                0.001, 0.002, 0.004,
                0.006, 0.008, 0.009,
                0.011, 0.015, 0.018,
                0.022, 0.035, 0.055,
                0.070, 0.090, 0.110,
                0.013,
            ],
        }
    )
    targets = {
        "<0.5%": {"n_train": 1, "n_val": 1, "n_test": 1},
        "0.5-1%": {"n_train": 1, "n_val": 1, "n_test": 1},
        "1-2%": {"n_train": 1, "n_val": 1, "n_test": 1},
        "2-6%": {"n_train": 1, "n_val": 1, "n_test": 1},
        "6-12%": {"n_train": 1, "n_val": 1, "n_test": 1},
    }

    assignments, annotated, _summary = assign_stratified_volume_splits(
        volume_stats,
        bin_edges=[0.0, 0.005, 0.01, 0.02, 0.06, 0.12],
        bin_labels=["<0.5%", "0.5-1%", "1-2%", "2-6%", "6-12%"],
        target_counts=targets,
        seed=42,
        excluded_volume_ids=["excluded"],
    )

    assert assignments["excluded"] == "excluded"
    split_table = pd.crosstab(annotated["bin"], annotated["split_v2"])
    assert (split_table["train"] == 1).all()
    assert (split_table["val"] == 1).all()
    assert (split_table["test"] == 1).all()


def test_write_split_dataset_root_filters_index_and_shares_volumes(tmp_path: Path):
    source_root = tmp_path / "split_v1"
    source_root.mkdir()
    (source_root / "volumes.zarr").mkdir()

    index_df = pd.DataFrame(
        [
            {
                "volume_id": "vol_a",
                "source_group": "g",
                "split": "train",
                "z0": 0,
                "y0": 0,
                "x0": 0,
                "ps": 8,
                "stride": 8,
                "porosity": 0.01,
            },
            {
                "volume_id": "vol_b",
                "source_group": "g",
                "split": "train",
                "z0": 0,
                "y0": 0,
                "x0": 0,
                "ps": 8,
                "stride": 8,
                "porosity": 0.02,
            },
        ]
    )
    index_df.to_parquet(source_root / "patch_index.parquet", index=False)
    (source_root / "splits.json").write_text(
        json.dumps(
            {
                "seed": 123,
                "counts": {"train": 2, "val": 0, "test": 0},
                "volumes": {"vol_a": "train", "vol_b": "train"},
            },
            indent=2,
        )
    )
    (source_root / "volume_stats.json").write_text(
        json.dumps({"vol_a": {"mean": 1.0}, "vol_b": {"mean": 2.0}}, indent=2)
    )

    target_root = tmp_path / "split_v2"
    write_split_dataset_root(
        source_root,
        target_root,
        splits={"vol_a": "val"},
        seed=42,
    )

    target_index = pd.read_parquet(target_root / "patch_index.parquet")
    target_splits = json.loads((target_root / "splits.json").read_text())
    target_stats = json.loads((target_root / "volume_stats.json").read_text())

    assert set(target_index["volume_id"]) == {"vol_a"}
    assert set(target_index["split"]) == {"val"}
    assert target_splits["counts"] == {"train": 0, "val": 1, "test": 0}
    assert target_splits["volumes"] == {"vol_a": "val"}
    assert target_stats == {"vol_a": {"mean": 1.0}}
    assert (target_root / "volumes.zarr").is_symlink()
    assert (target_root / "volumes.zarr").resolve() == (source_root / "volumes.zarr").resolve()
