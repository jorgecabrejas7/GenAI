"""Volume-level split helpers.

Supports the original deterministic ``v1`` split plus the stratified
``v2`` split used for ConvVAE training and evaluation.
"""

from __future__ import annotations

import json
import logging
import shutil
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

DEFAULT_SPLIT_VERSION = "v1"
SPLIT_V2_VERSION = "v2"
SPLIT_V2_COLUMN = "split_v2"
SPLIT_V2_SEED = 42  # Fixed seed requested for the stratified split_v2.
SPLIT_V2_EXCLUDED_VOLUME_IDS = (
    "MedidasDB__Juan_Ignacio_probetas_11_volume_eq_aligned",
)
SPLIT_V2_BIN_EDGES = (0.0, 0.005, 0.01, 0.02, 0.06, 0.12)
SPLIT_V2_BIN_LABELS = ("<0.5%", "0.5-1%", "1-2%", "2-6%", "6-12%")
SPLIT_V2_TARGET_COUNTS: dict[str, dict[str, int]] = {
    "<0.5%": {"n_train": 25, "n_val": 3, "n_test": 3},
    "0.5-1%": {"n_train": 16, "n_val": 2, "n_test": 2},
    "1-2%": {"n_train": 12, "n_val": 1, "n_test": 1},
    "2-6%": {"n_train": 8, "n_val": 1, "n_test": 1},
    "6-12%": {"n_train": 3, "n_val": 1, "n_test": 1},
}


def split_column_for_version(split_version: str = DEFAULT_SPLIT_VERSION) -> str:
    """Return the patch-index column name for *split_version*."""
    version = split_version.strip().lower()
    if version == DEFAULT_SPLIT_VERSION:
        return "split"
    if version == SPLIT_V2_VERSION:
        return SPLIT_V2_COLUMN
    raise ValueError(
        f"Unsupported split_version '{split_version}'. Expected 'v1' or 'v2'."
    )


def _count_splits(splits: Mapping[str, str]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for split_name in splits.values():
        counts[split_name] = counts.get(split_name, 0) + 1
    return counts


def assign_volume_splits(
    volume_ids: list[str],
    n_train: int,
    n_val: int,
    n_test: int,
    seed: int = 123,
) -> dict[str, str]:
    """Deterministically assign volumes to ``train / val / test``.

    Volume IDs are sorted then shuffled with *seed*.  The first *n_val* go
    to ``"val"``, the next *n_test* to ``"test"``, then up to *n_train* to
    ``"train"``.  Any remaining volumes are **not assigned**.

    Raises
    ------
    ValueError
        If ``n_train + n_val + n_test > len(volume_ids)``.
    """
    total_needed = n_train + n_val + n_test
    if total_needed > len(volume_ids):
        raise ValueError(
            f"Requested {total_needed} volumes (train={n_train}, val={n_val}, "
            f"test={n_test}) but only {len(volume_ids)} available."
        )

    ids = sorted(volume_ids)
    rng = np.random.default_rng(seed)
    rng.shuffle(ids)

    splits: dict[str, str] = {}
    idx = 0
    for _ in range(n_val):
        splits[ids[idx]] = "val"
        idx += 1
    for _ in range(n_test):
        splits[ids[idx]] = "test"
        idx += 1
    for _ in range(n_train):
        splits[ids[idx]] = "train"
        idx += 1

    logger.info(
        "Split %d volumes: train=%d, val=%d, test=%d, unused=%d",
        len(volume_ids),
        n_train,
        n_val,
        n_test,
        len(volume_ids) - total_needed,
    )
    return splits


def save_splits(
    splits: dict[str, str],
    out_path: str | Path,
    seed: int,
) -> Path:
    """Serialise *splits* to JSON with per-split counts."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    counts = {}
    for split_name in ("train", "val", "test"):
        counts[split_name] = sum(1 for v in splits.values() if v == split_name)

    payload: dict[str, Any]
    if out_path.exists():
        payload = json.loads(out_path.read_text())
    else:
        payload = {}

    payload.update({
        "seed": seed,
        "counts": counts,
        "volumes": splits,
    })
    out_path.write_text(json.dumps(payload, indent=2))
    logger.info("Wrote splits to %s", out_path)
    return out_path


def load_splits(
    path: str | Path,
    split_version: str = DEFAULT_SPLIT_VERSION,
) -> dict[str, str]:
    """Load a splits JSON written by :func:`save_splits`.

    Returns
    -------
    dict[str, str]
        Mapping of ``volume_id -> split_name`` (same format as
        :func:`assign_volume_splits`).
    """
    path = Path(path)
    payload = json.loads(path.read_text())
    version = split_version.strip().lower()

    if version == DEFAULT_SPLIT_VERSION:
        splits: dict[str, str] = payload["volumes"]
        counts = payload.get("counts", {})
    elif version == SPLIT_V2_VERSION:
        raw = payload[SPLIT_V2_COLUMN]
        splits = {
            volume_id: meta["split"] if isinstance(meta, dict) else meta
            for volume_id, meta in raw.items()
        }
        counts = _count_splits(splits)
    else:
        raise ValueError(
            f"Unsupported split_version '{split_version}'. Expected 'v1' or 'v2'."
        )

    logger.info("Loaded %s splits from %s (%s)", version, path, counts)
    return splits


def compute_volume_porosity_medians(index_df: pd.DataFrame) -> pd.DataFrame:
    """Return one row per volume with the median patch porosity."""
    required = {"volume_id", "porosity"}
    missing = required - set(index_df.columns)
    if missing:
        missing_s = ", ".join(sorted(missing))
        raise KeyError(f"patch_index is missing required column(s): {missing_s}")

    return (
        index_df.groupby("volume_id", as_index=False)
        .agg(phi_median=("porosity", "median"))
        .sort_values("volume_id")
        .reset_index(drop=True)
    )


def assign_stratified_volume_splits(
    volume_stats: pd.DataFrame,
    *,
    bin_edges: Sequence[float],
    bin_labels: Sequence[str],
    target_counts: Mapping[str, Mapping[str, int]],
    seed: int,
    excluded_volume_ids: Sequence[str] = (),
) -> tuple[dict[str, str], pd.DataFrame, pd.DataFrame]:
    """Assign volume-level splits within porosity bins.

    ``n_val`` and ``n_test`` from ``target_counts`` are treated as hard
    requirements. Any remaining volumes in a bin are assigned to train.
    If the observed bin size differs from the table encoded in
    ``target_counts``, a warning is logged and the actual remainder is kept
    in train so every usable volume remains assigned.
    """
    required = {"volume_id", "phi_median"}
    missing = required - set(volume_stats.columns)
    if missing:
        missing_s = ", ".join(sorted(missing))
        raise KeyError(f"volume_stats is missing required column(s): {missing_s}")

    if set(bin_labels) != set(target_counts):
        raise ValueError("bin_labels and target_counts keys must describe the same bins.")

    stats = (
        volume_stats.loc[:, ["volume_id", "phi_median"]]
        .drop_duplicates(subset="volume_id", keep="first")
        .copy()
    )
    if stats["volume_id"].duplicated().any():
        raise ValueError("volume_stats contains duplicate volume_id entries.")

    stats["bin"] = pd.cut(
        stats["phi_median"],
        bins=list(bin_edges),
        labels=list(bin_labels),
        right=False,
        include_lowest=True,
    )
    if stats["bin"].isna().any():
        bad = stats.loc[stats["bin"].isna(), ["volume_id", "phi_median"]]
        raise ValueError(
            "Some volumes fall outside the configured split_v2 bins:\n"
            f"{bad.to_string(index=False)}"
        )

    excluded = sorted(set(excluded_volume_ids) & set(stats["volume_id"]))
    assignments: dict[str, str] = {volume_id: "excluded" for volume_id in excluded}

    usable = stats.loc[~stats["volume_id"].isin(excluded)].copy()
    rng = np.random.default_rng(seed)
    summary_rows: list[dict[str, int | str]] = []

    for bin_label in bin_labels:
        target = target_counts[bin_label]
        n_val = int(target["n_val"])
        n_test = int(target["n_test"])
        requested_total = int(target["n_train"]) + n_val + n_test

        bin_df = usable.loc[usable["bin"] == bin_label].sort_values("volume_id")
        volume_ids = bin_df["volume_id"].tolist()
        if len(volume_ids) < n_val + n_test:
            raise ValueError(
                f"Bin {bin_label} only has {len(volume_ids)} volume(s), but "
                f"needs at least {n_val + n_test} to satisfy val/test coverage."
            )

        rng.shuffle(volume_ids)
        actual_n_train = len(volume_ids) - n_val - n_test
        if len(volume_ids) != requested_total:
            logger.warning(
                "split_v2 bin %s has %d volume(s) on disk, expected %d from the "
                "requested table; keeping val=%d and test=%d, assigning the "
                "remaining %d volume(s) to train.",
                bin_label,
                len(volume_ids),
                requested_total,
                n_val,
                n_test,
                actual_n_train,
            )

        for volume_id in volume_ids[:n_val]:
            assignments[volume_id] = "val"
        for volume_id in volume_ids[n_val : n_val + n_test]:
            assignments[volume_id] = "test"
        for volume_id in volume_ids[n_val + n_test :]:
            assignments[volume_id] = "train"

        summary_rows.append({
            "bin": bin_label,
            "n_volumes": len(volume_ids),
            "requested_total": requested_total,
            "train": actual_n_train,
            "val": n_val,
            "test": n_test,
        })

    if len(assignments) != len(stats):
        missing_ids = sorted(set(stats["volume_id"]) - set(assignments))
        raise ValueError(
            "Not every volume received a split assignment. Missing: "
            + ", ".join(missing_ids)
        )

    usable["split_v2"] = usable["volume_id"].map(assignments)
    summary = pd.DataFrame(summary_rows)
    return assignments, usable, summary


def summarise_volume_distribution(
    annotated_stats: pd.DataFrame,
    split_column: str = SPLIT_V2_COLUMN,
) -> pd.DataFrame:
    """Summarise ``phi_median`` by split for quick visual inspection."""
    usable = annotated_stats.loc[annotated_stats[split_column].isin(["train", "val", "test"])]
    grouped = usable.groupby(split_column)["phi_median"]
    summary = grouped.agg(["count", "min", "median", "mean", "max"]).rename(
        columns={"count": "n_volumes"}
    )
    summary["q25"] = grouped.quantile(0.25)
    summary["q75"] = grouped.quantile(0.75)
    summary = summary.loc[["train", "val", "test"], ["n_volumes", "min", "q25", "median", "mean", "q75", "max"]]
    return summary.reset_index().rename(columns={split_column: "split"})


def _build_splits_payload(splits: Mapping[str, str], seed: int) -> dict[str, Any]:
    counts = {
        split_name: sum(1 for value in splits.values() if value == split_name)
        for split_name in ("train", "val", "test")
    }
    return {
        "seed": seed,
        "counts": counts,
        "volumes": dict(sorted(splits.items())),
    }


def _ensure_shared_volumes(source_root: Path, target_root: Path) -> None:
    source_volumes = (source_root / "volumes.zarr").resolve()
    target_volumes = target_root / "volumes.zarr"

    if target_volumes.is_symlink():
        if target_volumes.resolve() != source_volumes:
            raise FileExistsError(
                f"{target_volumes} already points to {target_volumes.resolve()}, "
                f"expected {source_volumes}."
            )
        return

    if target_volumes.exists():
        raise FileExistsError(
            f"{target_volumes} already exists and is not the expected shared symlink."
        )

    target_volumes.symlink_to(source_volumes)


def _write_volume_stats_copy(
    source_root: Path,
    target_root: Path,
    keep_volume_ids: set[str] | None = None,
) -> None:
    source_stats = source_root / "volume_stats.json"
    if not source_stats.exists():
        return

    target_stats = target_root / "volume_stats.json"
    stats = json.loads(source_stats.read_text())
    if keep_volume_ids is not None:
        stats = {
            volume_id: payload
            for volume_id, payload in stats.items()
            if volume_id in keep_volume_ids
        }
    target_stats.write_text(json.dumps(stats, indent=2))


def write_split_dataset_root(
    source_root: str | Path,
    target_root: str | Path,
    *,
    splits: Mapping[str, str] | None = None,
    seed: int | None = None,
) -> Path:
    """Create a dataset root with the standard ``split`` column/schema.

    ``volumes.zarr`` is shared via symlink to avoid duplicating the heavy 3-D
    data. If ``splits`` is provided, the target patch index and ``splits.json``
    are rewritten to reflect that assignment; otherwise the v1 files are copied
    as-is.
    """
    source_root = Path(source_root)
    target_root = Path(target_root)
    target_root.mkdir(parents=True, exist_ok=True)
    _ensure_shared_volumes(source_root, target_root)

    source_index = source_root / "patch_index.parquet"
    target_index = target_root / "patch_index.parquet"
    source_splits = source_root / "splits.json"
    target_splits = target_root / "splits.json"

    if splits is None:
        shutil.copy2(source_index, target_index)
        shutil.copy2(source_splits, target_splits)
        _write_volume_stats_copy(source_root, target_root)
        return target_root

    if seed is None:
        raise ValueError("seed must be provided when writing a derived split dataset.")

    keep_volume_ids = set(splits)
    index_df = pd.read_parquet(str(source_index))
    filtered = index_df.loc[index_df["volume_id"].isin(keep_volume_ids)].copy()
    if filtered.empty:
        raise ValueError("Derived split dataset would contain no patches.")

    filtered["split"] = filtered["volume_id"].map(splits)
    if filtered["split"].isna().any():
        missing_ids = sorted(filtered.loc[filtered["split"].isna(), "volume_id"].unique())
        raise ValueError(
            "Some kept volumes did not receive a split assignment: "
            + ", ".join(missing_ids)
        )

    filtered.to_parquet(str(target_index), index=False, engine="pyarrow")
    target_splits.write_text(json.dumps(_build_splits_payload(splits, seed), indent=2))
    _write_volume_stats_copy(source_root, target_root, keep_volume_ids=keep_volume_ids)
    return target_root


def materialize_split_v2(
    source_root: str | Path,
    target_root: str | Path,
    *,
    seed: int = SPLIT_V2_SEED,
) -> dict[str, pd.DataFrame | dict[str, str]]:
    """Create a ``split_v2`` dataset root with the classic ``split`` schema."""
    source_root = Path(source_root)
    target_root = Path(target_root)
    index_path = source_root / "patch_index.parquet"

    index_df = pd.read_parquet(str(index_path))
    volume_stats = compute_volume_porosity_medians(index_df)
    assignments, annotated_stats, bin_summary = assign_stratified_volume_splits(
        volume_stats,
        bin_edges=SPLIT_V2_BIN_EDGES,
        bin_labels=SPLIT_V2_BIN_LABELS,
        target_counts=SPLIT_V2_TARGET_COUNTS,
        seed=seed,
        excluded_volume_ids=SPLIT_V2_EXCLUDED_VOLUME_IDS,
    )
    usable_assignments = {
        volume_id: split_name
        for volume_id, split_name in assignments.items()
        if split_name != "excluded"
    }
    annotated_stats = annotated_stats.rename(columns={SPLIT_V2_COLUMN: "split"})

    write_split_dataset_root(
        source_root,
        target_root,
        splits=usable_assignments,
        seed=seed,
    )

    split_table = pd.crosstab(
        annotated_stats["bin"],
        annotated_stats["split"],
        dropna=False,
    ).reindex(index=list(SPLIT_V2_BIN_LABELS), columns=["train", "val", "test"], fill_value=0)
    for split_name in ("val", "test"):
        missing_bins = split_table.index[split_table[split_name] < 1].tolist()
        if missing_bins:
            raise ValueError(
                f"split_v2 {split_name} is missing coverage for bin(s): "
                + ", ".join(missing_bins)
            )

    distribution_summary = summarise_volume_distribution(
        annotated_stats.rename(columns={"split": SPLIT_V2_COLUMN})
    )
    logger.info("split_v2 counts: %s", _count_splits(usable_assignments))
    logger.info("split_v2 bin coverage:\n%s", split_table.to_string())
    logger.info(
        "split_v2 phi_median summary:\n%s",
        distribution_summary.to_string(index=False, float_format=lambda x: f"{x:.4f}"),
    )

    return {
        "assignments": assignments,
        "bin_summary": bin_summary,
        "split_table": split_table,
        "distribution_summary": distribution_summary,
    }


def materialize_split_roots(
    data_dir: str | Path,
    *,
    seed: int = SPLIT_V2_SEED,
) -> dict[str, Any]:
    """Reshape ``data/processed`` into ``data/split_v1`` and ``data/split_v2``."""
    data_dir = Path(data_dir)
    processed_root = data_dir / "processed"
    split_v1_root = data_dir / "split_v1"
    split_v2_root = data_dir / "split_v2"

    if processed_root.exists():
        if split_v1_root.exists():
            raise FileExistsError(
                f"Both {processed_root} and {split_v1_root} exist. "
                "Please keep only one v1 source root before materializing splits."
            )
        processed_root.rename(split_v1_root)
        logger.info("Renamed %s -> %s", processed_root, split_v1_root)
    elif not split_v1_root.exists():
        raise FileNotFoundError(
            f"Expected either {processed_root} or {split_v1_root} to exist."
        )

    if not split_v2_root.exists():
        split_v2_root.mkdir(parents=True, exist_ok=True)

    split_v2_summary = materialize_split_v2(
        split_v1_root,
        split_v2_root,
        seed=seed,
    )
    return {
        "split_v1_root": split_v1_root,
        "split_v2_root": split_v2_root,
        **split_v2_summary,
    }
