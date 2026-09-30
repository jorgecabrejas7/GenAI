"""Volume I/O: discovery, TIFF loading, Zarr storage, and per-volume stats."""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import zarr

logger = logging.getLogger(__name__)

TIFF_EXTENSIONS = {".tif", ".tiff"}
#: Where the reference notebooks write their masks, beside each raw volume.
REFERENCE_OUTPUT_DIR = "onlypores files"


@dataclass
class VolumeInfo:
    """Metadata for a discovered raw volume."""

    volume_id: str
    path: Path
    source_group: str
    shape: tuple[int, int, int] = field(default=(0, 0, 0))


def discover_volumes(raw_root: str | Path) -> list[VolumeInfo]:
    """Recursively find TIFF volumes under *raw_root*.

    Source-group logic:
      - ``MedidasDB`` → ``"MedidasDB"``
      - any other top-level directory → ``"others/<dirname>"``

    ``volume_id`` is the relative path (without extension) with path
    separators replaced by ``__`` and spaces by ``_``, making it a
    valid Zarr group name.
    """
    raw_root = Path(raw_root)
    if not raw_root.is_dir():
        raise FileNotFoundError(f"raw_root does not exist: {raw_root}")

    volumes: list[VolumeInfo] = []

    for tif_path in sorted(raw_root.rglob("*")):
        if tif_path.suffix.lower() not in TIFF_EXTENSIONS:
            continue
        if not tif_path.is_file():
            continue
        if REFERENCE_OUTPUT_DIR in tif_path.parts:
            continue  # reference masks beside the volumes, not volumes

        rel = tif_path.relative_to(raw_root)
        parts = rel.parts

        # Source group from top-level directory name
        top_dir = parts[0] if len(parts) > 1 else "ungrouped"
        if top_dir == "MedidasDB":
            source_group = "MedidasDB"
        else:
            source_group = f"others/{top_dir}"

        # Stable volume_id from relative path
        vol_id = (
            str(rel.with_suffix(""))
            .replace(os.sep, "__")
            .replace(" ", "_")
        )

        volumes.append(
            VolumeInfo(
                volume_id=vol_id,
                path=tif_path,
                source_group=source_group,
            )
        )

    logger.info("Discovered %d volumes under %s", len(volumes), raw_root)
    return volumes


#: The two settings the reference notebooks run ``onlypores`` with
#: (UTvsXCT-preprocessing, ``produccion/onlypores/``).  Each builds its own
#: dataset root.  The notebooks disagree with each other; both are kept.
SEGMENTATION: dict[str, dict] = {
    # onlypores.ipynb, cell 6
    "ipynb": {"sauvola_radius": 30, "sauvola_k": 0.125, "min_size_filtering": 8},
    # onlypores_batch.ipynb, cell 5
    "batch": {"sauvola_radius": 15, "sauvola_k": 0.2, "min_size_filtering": 8},
}


def find_walls(volume: np.ndarray) -> tuple[int, int]:
    """Front and back wall slices of *volume*, as the reference notebooks find them.

    The notebooks rotate and reslice the volume only to find the walls
    (``reslicer.rotate_90(volume, clockwise=False)``, then
    ``reslicer.reslice(., 'Right')``, then ``aligner.crop_walls``); ``onlypores``
    then runs on the original volume.  Both are views, so no copy is made.
    """
    from preprocess_tools import aligner, reslicer

    resliced = reslicer.reslice(reslicer.rotate_90(volume, clockwise=False), "Right")
    _, frontwall, backwall = aligner.crop_walls(resliced)
    return int(frontwall), int(backwall)


def reference_outputs(volume_path: str | Path, segmentation: str) -> dict[str, Path]:
    """The reference notebook outputs for a raw volume and a ``SEGMENTATION`` name.

    ``scripts/build_reference_onlypores.py`` writes them where the notebooks do,
    ``<volume dir>/onlypores files/``, with the notebook suffixes and the
    parameters appended so both settings coexist:
    ``<stem>_{onlypores,samplemask,binary}_r30_k0.125_min8.tif`` and
    ``<stem>_report_r30_k0.125_min8.txt``.
    """
    p = SEGMENTATION[segmentation]
    tag = f"r{p['sauvola_radius']}_k{p['sauvola_k']}_min{p['min_size_filtering']}"
    volume_path = Path(volume_path)
    d = volume_path.parent / REFERENCE_OUTPUT_DIR
    return {name: d / f"{volume_path.stem}_{name}_{tag}.{'txt' if name == 'report' else 'tif'}"
            for name in ("onlypores", "samplemask", "binary", "report")}


def read_reference_report(path: str | Path) -> dict:
    """Walls and parameters from a reference notebook report (``*_report_*.txt``)."""
    import re

    text = Path(path).read_text()
    def grab(pattern: str) -> str:
        m = re.search(pattern, text)
        if m is None:
            raise ValueError(f"{path}: no match for {pattern!r}")
        return m.group(1)
    return {
        "frontwall": int(grab(r"Front wall slice: (-?\d+)")),
        "backwall": int(grab(r"Back wall slice: (-?\d+)")),
        "sauvola_radius": int(grab(r"window_size \(sauvola_radius\): (\d+)")),
        "sauvola_k": float(grab(r"- k: ([0-9.]+)")),
        "min_size_filtering": int(grab(r"- min_size: (\d+) voxels")),
        "reference_commit": grab(r'"commit_id": "([0-9a-f]+)"'),
    }


def compute_mask(volume: np.ndarray, segmentation: str) -> tuple[np.ndarray, np.ndarray, dict]:
    """Pore mask and sample mask of a raw volume, with the reference pipeline.

    Runs the reference notebook procedure: :func:`find_walls`, then
    ``preprocess_tools.onlypores.onlypores(volume, frontwall, backwall, ...)``
    with the settings ``SEGMENTATION[segmentation]``.  *volume* is the raw
    TIFF as ``preprocess_tools.io.load_tif`` returns it, axes ``(z, y, x)`` with
    z through the thickness.

    Returns
    -------
    pore_mask : uint8 array, values in {0, 1}
    sample_mask : bool array, True inside the specimen (internal pores filled)
    record : dict with the walls and the settings, for the build record
    """
    from preprocess_tools import onlypores

    params = SEGMENTATION[segmentation]
    frontwall, backwall = find_walls(volume)
    pores, sample_mask, _binary = onlypores.onlypores(volume, frontwall, backwall, **params)
    if pores is None:
        raise ValueError("onlypores found no non-zero voxel in the volume")
    del _binary
    record = {"segmentation": segmentation, **params,
              "frontwall": frontwall, "backwall": backwall}
    return pores.astype(np.uint8), sample_mask.astype(bool), record


def compute_volume_stats(xct: np.ndarray, sample_mask: np.ndarray) -> dict:
    """Compute foreground intensity statistics using the sample mask from ``onlypores``.

    Parameters
    ----------
    xct : uint8 volume array
    sample_mask : bool array, True where material (output of ``onlypores``)

    Returns
    -------
    dict with keys: mean, std, n_foreground
    """
    fg_vals = xct[sample_mask.astype(bool)].astype(np.float64)
    if len(fg_vals) == 0:
        logger.warning("compute_volume_stats: no foreground voxels — returning fallback stats")
        return {"mean": 128.0, "std": 50.0, "n_foreground": 0}
    return {
        "mean": float(fg_vals.mean()),
        "std": float(fg_vals.std()),
        "n_foreground": int(len(fg_vals)),
    }


def compute_volume_stats_from_zarr(zarr_xct: zarr.Array, chunk_z: int = 64) -> dict:
    """Stream through a zarr XCT array and compute foreground intensity stats.

    Uses Otsu thresholding (computed from a full histogram pass) to define
    foreground, then accumulates mean and std in a second streaming pass.
    Memory usage is O(chunk_z × H × W) — safe for multi-GB volumes.

    Parameters
    ----------
    zarr_xct : zarr.Array, shape (D, H, W), dtype uint8
    chunk_z  : number of Z-slices to read per iteration

    Returns
    -------
    dict with keys: mean, std, n_foreground, otsu_threshold
    """
    D = zarr_xct.shape[0]

    # ── Pass 1: build 256-bin histogram for Otsu threshold ──
    hist = np.zeros(256, dtype=np.int64)
    for z in range(0, D, chunk_z):
        chunk = np.array(zarr_xct[z : min(z + chunk_z, D)])
        h, _ = np.histogram(chunk.ravel(), bins=256, range=(0, 256))
        hist += h

    thresh = _otsu_from_hist(hist)
    logger.info("compute_volume_stats_from_zarr: Otsu threshold = %d", thresh)

    # ── Pass 2: streaming mean/std over foreground voxels ──
    n: int = 0
    s: float = 0.0
    sq: float = 0.0
    for z in range(0, D, chunk_z):
        chunk = np.array(zarr_xct[z : min(z + chunk_z, D)]).astype(np.float64)
        fg = chunk > thresh
        if fg.any():
            v = chunk[fg]
            n += len(v)
            s += float(v.sum())
            sq += float((v ** 2).sum())

    if n == 0:
        logger.warning("compute_volume_stats_from_zarr: no foreground voxels found")
        return {"mean": 128.0, "std": 50.0, "n_foreground": 0, "otsu_threshold": thresh}

    mean = s / n
    variance = max(sq / n - mean ** 2, 0.0)
    return {
        "mean": float(mean),
        "std": float(np.sqrt(variance)),
        "n_foreground": int(n),
        "otsu_threshold": int(thresh),
    }


def _otsu_from_hist(hist: np.ndarray) -> int:
    """Compute Otsu threshold from a 256-bin intensity histogram."""
    total = int(hist.sum())
    if total == 0:
        return 128
    p = hist.astype(np.float64) / total
    levels = np.arange(len(hist), dtype=np.float64)
    omega = np.cumsum(p)
    mu = np.cumsum(levels * p)
    mu_T = mu[-1]
    with np.errstate(divide="ignore", invalid="ignore"):
        sigma_b = np.where(
            (omega > 0) & (omega < 1),
            (mu_T * omega - mu) ** 2 / (omega * (1.0 - omega)),
            0.0,
        )
    return int(np.argmax(sigma_b))


# ── Stats file I/O ────────────────────────────────────────────────────────────

def load_volume_stats(out_root: str | Path) -> dict[str, dict]:
    """Load per-volume intensity statistics from ``volume_stats.json``.

    Returns an empty dict if the file does not exist yet.
    """
    path = Path(out_root) / "volume_stats.json"
    if not path.exists():
        return {}
    with open(path) as f:
        return json.load(f)


def save_volume_stats(stats: dict[str, dict], out_root: str | Path) -> None:
    """Persist per-volume intensity statistics to ``volume_stats.json``."""
    path = Path(out_root) / "volume_stats.json"
    with open(path, "w") as f:
        json.dump(stats, f, indent=2)
    logger.info("Saved volume stats for %d volumes → %s", len(stats), path)


# ── Zarr storage ──────────────────────────────────────────────────────────────

def save_volume_zarr(
    xct: np.ndarray,
    mask: np.ndarray,
    out_root: str | Path,
    volume_id: str,
    chunk_size: tuple[int, int, int] = (64, 64, 64),
    sample_mask: np.ndarray | None = None,
) -> None:
    """Write *xct* and *mask* into ``<out_root>/volumes.zarr/<volume_id>/``.

    Chunks are aligned to the default patch size (64³) so each patch read
    hits exactly one chunk per array.  No compression is applied — on fast
    NVMe the decompression overhead exceeds the I/O savings.

    When *sample_mask* is given it is written as a third array beside
    ``xct``/``mask``.  volumes.zarr is the natural home: it is where the
    per-volume mask arrays already live, and every offline builder that needs
    the specimen envelope reads it back per volume.  Unlike xct/mask it IS
    compressed (default codec): the mask is mostly-constant regions, so it
    compresses ~100x, and it is only read by offline builders where
    decompression cost is irrelevant.
    """
    store_path = Path(out_root) / "volumes.zarr"

    root = zarr.open_group(str(store_path), mode="a")
    grp = root.require_group(volume_id)

    for name, arr in [("xct", xct), ("mask", mask)]:
        grp.create_array(
            name,
            data=arr.astype(np.uint8, copy=False),
            chunks=chunk_size,
            compressors=None,
            overwrite=True,
        )
    if sample_mask is not None:
        save_sample_mask_zarr(sample_mask, out_root, volume_id, chunk_size=chunk_size)

    logger.info(
        "Saved %s  xct=%s  mask=%s  chunks=%s  compression=none",
        volume_id,
        xct.shape,
        mask.shape,
        chunk_size,
    )


def save_labels_zarr(
    mask: np.ndarray,
    sample_mask: np.ndarray,
    out_root: str | Path,
    volume_id: str,
    xct_array: str | Path,
    chunk_size: tuple[int, int, int] = (64, 64, 64),
) -> None:
    """Write new labels for a volume whose ``xct`` already exists in another store.

    ``mask`` is written like :func:`save_volume_zarr` writes it (uncompressed,
    64³ chunks) and ``sample_mask`` like :func:`save_sample_mask_zarr`.
    ``<volume_id>/xct`` becomes a symlink to the existing array directory
    *xct_array*, so the grey data is not copied.  The caller checks that the
    raw volume equals that array.
    """
    store_path = Path(out_root) / "volumes.zarr"
    root = zarr.open_group(str(store_path), mode="a")
    grp = root.require_group(volume_id)
    grp.create_array("mask", data=mask.astype(np.uint8, copy=False),
                     chunks=chunk_size, compressors=None, overwrite=True)
    save_sample_mask_zarr(sample_mask, out_root, volume_id, chunk_size=chunk_size)
    link = store_path / volume_id / "xct"
    if link.is_symlink():
        link.unlink()
    link.symlink_to(Path(xct_array).resolve(), target_is_directory=True)
    logger.info("Saved %s  mask=%s  xct -> %s", volume_id, mask.shape, link.resolve())


def save_sample_mask_zarr(
    sample_mask: np.ndarray,
    out_root: str | Path,
    volume_id: str,
    chunk_size: tuple[int, int, int] = (64, 64, 64),
) -> None:
    """Persist the onlypores ``sample_mask`` as ``<volume_id>/sample_mask``.

    Additive: only the ``sample_mask`` array is (over)written; ``xct`` and
    ``mask`` are never touched.  Compressed — see ``save_volume_zarr``.
    """
    store_path = Path(out_root) / "volumes.zarr"
    root = zarr.open_group(str(store_path), mode="a")
    grp = root.require_group(volume_id)
    grp.create_array(
        "sample_mask",
        data=sample_mask.astype(np.uint8, copy=False),
        chunks=chunk_size,
        overwrite=True,
    )
    logger.info("Saved %s/sample_mask  shape=%s (compressed)", volume_id, sample_mask.shape)
