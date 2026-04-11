"""Volume-level deterministic splits.

Assigns each volume to train / val / test based on explicit counts and a
reproducible random seed.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)


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

    payload = {
        "seed": seed,
        "counts": counts,
        "volumes": splits,
    }
    out_path.write_text(json.dumps(payload, indent=2))
    logger.info("Wrote splits to %s", out_path)
    return out_path


def load_splits(path: str | Path) -> dict[str, str]:
    """Load a splits JSON written by :func:`save_splits`.

    Returns
    -------
    dict[str, str]
        Mapping of ``volume_id -> split_name`` (same format as
        :func:`assign_volume_splits`).
    """
    path = Path(path)
    payload = json.loads(path.read_text())
    splits: dict[str, str] = payload["volumes"]
    logger.info("Loaded splits from %s (%s)", path, payload.get("counts", {}))
    return splits
