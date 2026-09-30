"""``--link-xct-from`` shares the grey patches of another root and writes only the labels.

Two builds of the same scans differ only in their labels (split_v3 -> split_v4), so the
565 GB grey-level memmap is hard-linked, not re-extracted. The link is refused when the
patch rows differ, because row i of the linked file would then be another patch.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from test_extract_patches_resume import _build_split, _run, extract  # noqa: F401


def test_link_shares_xct_and_writes_labels(tmp_path, monkeypatch):
    a, b = tmp_path / "a", tmp_path / "b"
    _build_split(a)
    _build_split(b)
    _run(a, monkeypatch)
    _run(b, monkeypatch, ("--link-xct-from", str(a), "--verify"))
    assert (b / "patches_xct.bin").stat().st_ino == (a / "patches_xct.bin").stat().st_ino
    la = np.memmap(str(a / "patches_label.bin"), dtype=np.uint8, mode="r")
    lb = np.memmap(str(b / "patches_label.bin"), dtype=np.uint8, mode="r")
    assert np.array_equal(np.asarray(la), np.asarray(lb))
    meta = json.loads((b / "patches_meta.json").read_text())
    assert meta["xct_linked_from"] == str((a / "patches_xct.bin").resolve())


def test_link_refused_when_rows_differ(tmp_path, monkeypatch):
    a, b = tmp_path / "a", tmp_path / "b"
    _build_split(a)
    _build_split(b)
    _run(a, monkeypatch)
    df = pd.read_parquet(str(b / "patch_index.parquet"))
    df = df.iloc[:-1]
    df.to_parquet(str(b / "patch_index.parquet"), index=False)
    with pytest.raises(SystemExit):
        _run(b, monkeypatch, ("--link-xct-from", str(a)))
    assert not (b / "patches_xct.bin").exists()
