"""Export and migrate saved 3-D patch samples as ImageJ-readable TIFF stacks."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import tifffile

PATCH_SAMPLE_KEYS = ("xct_gt", "mask_gt", "xct_recon", "mask_recon")


def _as_zyx_volume(volume: np.ndarray) -> np.ndarray:
    """Normalise a stored sample volume to ``(Z, Y, X)`` float32."""
    arr = np.asarray(volume)
    if arr.ndim == 4 and arr.shape[0] == 1:
        arr = arr[0]
    if arr.ndim != 3:
        raise ValueError(f"Expected a single 3-D patch volume, got shape {arr.shape}.")
    return arr.astype(np.float32, copy=False)


def write_imagej_volume(path: str | Path, volume: np.ndarray) -> Path:
    """Write one 3-D patch volume as an ImageJ-readable TIFF stack."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tifffile.imwrite(
        str(path),
        _as_zyx_volume(volume),
        imagej=True,
        metadata={"axes": "ZYX"},
    )
    return path


def export_patch_sample_split(
    split_dir: str | Path,
    arrays: Mapping[str, np.ndarray],
    metas: Sequence[dict[str, Any]] | None = None,
) -> Path:
    """Export all saved samples for one split into per-patch TIFF stacks."""
    split_dir = Path(split_dir)
    split_dir.mkdir(parents=True, exist_ok=True)

    missing = [key for key in PATCH_SAMPLE_KEYS if key not in arrays]
    if missing:
        raise KeyError(f"Missing required sample array(s): {', '.join(missing)}")

    n_samples = int(arrays["xct_gt"].shape[0])
    if metas is None:
        metas = [{} for _ in range(n_samples)]
    if len(metas) != n_samples:
        raise ValueError(
            f"Expected {n_samples} metadata entries, got {len(metas)}."
        )

    manifest: list[dict[str, Any]] = []
    for sample_idx in range(n_samples):
        sample_dir = split_dir / f"sample_{sample_idx:03d}"
        sample_dir.mkdir(parents=True, exist_ok=True)

        for key in PATCH_SAMPLE_KEYS:
            write_imagej_volume(sample_dir / f"{key}.tiff", arrays[key][sample_idx])

        meta = dict(metas[sample_idx])
        meta["sample_id"] = f"sample_{sample_idx:03d}"
        (sample_dir / "meta.json").write_text(json.dumps(meta, indent=2))
        manifest.append(meta)

    (split_dir / "index.json").write_text(json.dumps(manifest, indent=2))
    return split_dir


def convert_patch_sample_archive(
    npz_path: str | Path,
    *,
    delete_source: bool = True,
) -> Path:
    """Convert one legacy ``{split}.npz`` archive into TIFF stacks."""
    npz_path = Path(npz_path)
    split_name = npz_path.stem
    split_dir = npz_path.with_suffix("")
    meta_path = npz_path.with_name(f"{split_name}_meta.json")

    with np.load(npz_path) as archive:
        arrays = {key: archive[key] for key in PATCH_SAMPLE_KEYS}

    metas = json.loads(meta_path.read_text()) if meta_path.exists() else None
    export_patch_sample_split(split_dir, arrays, metas)

    if delete_source:
        npz_path.unlink()
        if meta_path.exists():
            meta_path.unlink()

    return split_dir


def convert_patch_sample_archives_under(
    root: str | Path,
    *,
    delete_source: bool = True,
) -> list[Path]:
    """Convert every legacy sample archive below ``root``."""
    root = Path(root)
    converted: list[Path] = []

    for npz_path in sorted(root.rglob("*.npz")):
        converted.append(
            convert_patch_sample_archive(npz_path, delete_source=delete_source)
        )

    return converted
