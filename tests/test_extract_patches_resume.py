"""Resumed patch extraction must describe the WHOLE extraction.

``scripts/extract_patches_memmap.py`` counts label voxels while it writes
patches.  A resumed run skips the volumes an earlier run already finished, so
the running total only ever saw the tail of the extraction — and
``patches_meta.json`` then reported label voxel fractions for that tail alone.
Those fractions are what the 3-class weights are derived from, so a resumed
extraction silently produced wrong weights.

The test extracts the same synthetic split twice: once straight through, once
interrupted after the first volume and resumed.  The metadata must be identical.
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import zarr

REPO = Path(__file__).resolve().parents[1]

PS = 4
SHAPE = (8, 8, 8)
VOLUME_IDS = ("vol_a", "vol_b", "vol_c")


def _load_extractor():
    path = REPO / "scripts" / "extract_patches_memmap.py"
    spec = importlib.util.spec_from_file_location("extract_patches_memmap", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


extract = _load_extractor()


def _build_split(root: Path) -> None:
    """A three-volume split whose volumes have very different class mixes.

    The mixes differ on purpose: if the counts came from only part of the
    volumes, the fractions would not just be noisy, they would be wrong.
    """
    root.mkdir(parents=True, exist_ok=True)
    zroot = zarr.open_group(str(root / "volumes.zarr"), mode="w")

    rows = []
    for vi, vid in enumerate(VOLUME_IDS):
        rng = np.random.default_rng(100 + vi)
        xct = rng.integers(0, 256, size=SHAPE, dtype=np.uint8)

        # vol_a mostly material, vol_b pore-rich, vol_c air-rich.
        mask = np.zeros(SHAPE, dtype=np.uint8)
        mask[: 2 * vi + 1] = 1
        sample_mask = np.ones(SHAPE, dtype=np.uint8)
        sample_mask[:, :, : vi + 1] = 0

        grp = zroot.create_group(vid)
        for name, arr in (("xct", xct), ("mask", mask),
                          ("sample_mask", sample_mask)):
            grp.create_array(name, data=arr, chunks=(4, 4, 4))

        for z0 in range(0, SHAPE[0] - PS + 1, PS):
            for y0 in range(0, SHAPE[1] - PS + 1, PS):
                for x0 in range(0, SHAPE[2] - PS + 1, PS):
                    rows.append({
                        "volume_id": vid,
                        "source_group": "synthetic",
                        "split": "train" if vi else "val",
                        "z0": z0, "y0": y0, "x0": x0,
                        "ps": PS, "stride": PS,
                        "porosity": 0.0,
                        "air_fraction": 0.0,
                        "panel_id": f"P{vi}",
                    })

    pd.DataFrame(rows).to_parquet(str(root / "patch_index.parquet"), index=False)
    (root / "index_report.json").write_text(json.dumps({
        "voxel_size_um": 25.0,
        "hole_rule": "synthetic fixture",
        "split_rule": "synthetic fixture",
    }))


def _run(root: Path, monkeypatch, argv_extra: tuple[str, ...] = ()) -> None:
    monkeypatch.setattr(
        "sys.argv",
        ["extract_patches_memmap.py", "--data-root", str(root),
         "--chunk-size", "2", *argv_extra],
    )
    extract.main()


def _interrupt_after_first_volume(monkeypatch) -> None:
    """Make the extractor die while loading the SECOND volume's label."""
    real = extract.build_label_volume
    state = {"calls": 0}

    def flaky(mask, sample_mask):
        state["calls"] += 1
        if state["calls"] > 1:
            raise KeyboardInterrupt("simulated interruption")
        return real(mask, sample_mask)

    monkeypatch.setattr(extract, "build_label_volume", flaky)


def test_resumed_extraction_counts_the_whole_extraction(tmp_path, monkeypatch):
    straight = tmp_path / "straight"
    resumed = tmp_path / "resumed"
    _build_split(straight)
    _build_split(resumed)

    _run(straight, monkeypatch)
    meta_straight = json.loads((straight / "patches_meta.json").read_text())

    _interrupt_after_first_volume(monkeypatch)
    with pytest.raises(KeyboardInterrupt):
        _run(resumed, monkeypatch)
    assert not (resumed / "patches_meta.json").exists()
    progress = json.loads((resumed / "patches_progress.json").read_text())
    assert sum(progress.values()) == 1, "fixture must stop after one volume"

    monkeypatch.undo()          # restore build_label_volume, then resume
    _run(resumed, monkeypatch)

    meta_resumed = json.loads((resumed / "patches_meta.json").read_text())
    assert meta_resumed["label_voxel_fraction"] == meta_straight["label_voxel_fraction"]

    # And the fractions really are the whole split, not a subset of it.
    label = np.memmap(str(resumed / "patches_label.bin"), dtype=np.uint8, mode="r")
    counts = np.bincount(np.asarray(label).ravel(), minlength=3)
    expected = {name: float(counts[i]) / counts.sum()
                for i, name in extract.LABEL_NAMES.items()}
    assert meta_resumed["label_voxel_fraction"] == pytest.approx(expected)


def test_resumed_extraction_writes_the_same_patches(tmp_path, monkeypatch):
    """Sanity guard: the resume itself must not lose or reorder patches."""
    straight = tmp_path / "straight"
    resumed = tmp_path / "resumed"
    _build_split(straight)
    _build_split(resumed)

    _run(straight, monkeypatch)

    _interrupt_after_first_volume(monkeypatch)
    with pytest.raises(KeyboardInterrupt):
        _run(resumed, monkeypatch)
    monkeypatch.undo()
    _run(resumed, monkeypatch)

    for name in ("patches_xct.bin", "patches_label.bin"):
        a = np.fromfile(straight / name, dtype=np.uint8)
        b = np.fromfile(resumed / name, dtype=np.uint8)
        assert np.array_equal(a, b), name
