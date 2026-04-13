"""Pore detection and material segmentation for 3-D X-ray CT volumes.

Provides adaptive thresholding (Sauvola + Otsu), material mask generation,
and pore extraction.  Moved from the repo-root ``onlypores.py`` into the
package so it can be imported as ``from poregen.dataset.segmentation import onlypores``.

Public API
----------
onlypores(xct, ...)
    Main entry point: returns (pore_mask, sample_mask, binary).
"""

from __future__ import annotations

import logging

import numpy as np
import psutil
import fill_voids
from joblib import Parallel, delayed
from skimage import filters, measure
from skimage.filters import threshold_sauvola
from skimage.measure import label, regionprops
from skimage.morphology import remove_small_objects
from tqdm import tqdm

logger = logging.getLogger(__name__)


# ── Sauvola thresholding ──────────────────────────────────────────────────────

def sauvola_thresholding_concurrent(volume: np.ndarray, window_size: int, k: float) -> np.ndarray:
    """Apply Sauvola thresholding in parallel along the Y-axis."""
    if window_size % 2 == 0:
        window_size += 1
        logger.info("Window size adjusted to %d (must be odd)", window_size)

    def _slice(s):
        thresh = threshold_sauvola(s, window_size=window_size, k=k, r=128)
        return s > thresh

    logger.info("Applying Sauvola thresholding (parallel)...")
    binary = np.array(Parallel(n_jobs=-1)(
        delayed(_slice)(volume[:, i, :]) for i in range(volume.shape[1])
    ))
    return np.transpose(binary, (1, 0, 2))


def sauvola_thresholding_nonconcurrent(volume: np.ndarray, window_size: int, k: float) -> np.ndarray:
    """Apply Sauvola thresholding sequentially (low-memory path)."""
    if window_size % 2 == 0:
        window_size += 1
        logger.info("Window size adjusted to %d (must be odd)", window_size)

    binary = np.zeros_like(volume, dtype=bool)
    logger.info("Applying Sauvola thresholding (sequential)...")
    for i in range(volume.shape[1]):
        thresh = threshold_sauvola(volume[:, i, :], window_size=window_size, k=k, r=128)
        binary[:, i, :] = volume[:, i, :] > thresh
        if (i + 1) % 50 == 0:
            logger.info("Processed %d/%d slices", i + 1, volume.shape[1])
    return binary


def sauvola_thresholding(volume: np.ndarray, window_size: int, k: float) -> np.ndarray:
    """Sauvola thresholding with automatic parallel/sequential selection based on RAM."""
    required = volume.nbytes * 2
    available = psutil.virtual_memory().available
    logger.info(
        "Volume %.2f GB | Available %.2f GB | Required %.2f GB",
        volume.nbytes / 1024**3, available / 1024**3, required / 1024**3,
    )
    if available > required:
        return sauvola_thresholding_concurrent(volume, window_size, k)
    logger.info("Insufficient memory for parallel processing; falling back to sequential.")
    return sauvola_thresholding_nonconcurrent(volume, window_size, k)


# ── Otsu thresholding ─────────────────────────────────────────────────────────

def otsu_thresholding(volume: np.ndarray) -> np.ndarray:
    """Apply global Otsu thresholding to a 3-D volume."""
    threshold_value = filters.threshold_otsu(volume)
    logger.info("Otsu threshold: %s", threshold_value)
    return volume > threshold_value


# ── Slice cleaning ────────────────────────────────────────────────────────────

def slice_cleaning(img: np.ndarray, min_size: int = 2) -> np.ndarray:
    """Remove small objects from a 2-D binary image."""
    return remove_small_objects(img > 0, min_size=min_size, connectivity=1) > 0


# ── Material mask ─────────────────────────────────────────────────────────────

def material_mask(xct: np.ndarray) -> np.ndarray:
    """Generate a binary material mask via Otsu + max-projection + void-filling."""
    logger.info("Computing material mask...")
    threshold_value = filters.threshold_otsu(xct)
    binary = xct > threshold_value

    max_proj = np.max(binary, axis=0)
    labels = measure.label(max_proj)
    props = regionprops(labels)

    if props:
        minr, minc, maxr, maxc = props[0].bbox
        binary_cropped = binary[:, minr:maxr, minc:maxc]
        logger.info("Filling internal voids...")
        filled = fill_voids.fill(binary_cropped, in_place=False)
        sample_mask = np.zeros_like(binary)
        sample_mask[:, minr:maxr, minc:maxc] = filled
    else:
        logger.warning("No connected components found; using raw Otsu mask.")
        sample_mask = binary

    logger.info("Material mask complete.")
    return sample_mask


# ── Pore cleaning ─────────────────────────────────────────────────────────────

def clean_pores(onlypores_mask: np.ndarray, min_size: int = 8) -> np.ndarray:
    """Remove small and dimensionally degenerate pore components.

    Keeps only connected components that have at least *min_size* voxels
    **and** a bounding-box extent ≥ 2 voxels in every spatial dimension.
    """
    logger.info("Cleaning pores (min_size=%d)...", min_size)
    labeled = label(onlypores_mask, connectivity=3)
    labeled = remove_small_objects(labeled, min_size=min_size, connectivity=3)

    props = regionprops(labeled)
    valid_labels = []
    for prop in props:
        z0, y0, x0, z1, y1, x1 = prop.bbox
        if (z1 - z0) >= 2 and (y1 - y0) >= 2 and (x1 - x0) >= 2:
            valid_labels.append(prop.label)

    cleaned = np.zeros_like(onlypores_mask, dtype=bool)
    if valid_labels:
        cleaned[np.isin(labeled, valid_labels)] = True

    logger.info("Clean pores: %d voxels retained.", int(cleaned.sum()))
    return cleaned


# ── Main entry point ──────────────────────────────────────────────────────────

def onlypores(
    xct: np.ndarray,
    frontwall: int = 0,
    backwall: int = 0,
    sauvola_radius: int = 30,
    sauvola_k: float = 0.125,
    min_size_filtering: int = -1,
) -> tuple[np.ndarray | None, np.ndarray | None, np.ndarray | None]:
    """Extract pores from a 3-D X-ray CT volume.

    Parameters
    ----------
    xct : (Z, Y, X) uint8 array
        Raw CT volume.
    frontwall, backwall : int
        Z-slice indices to exclude (set to 0 to disable).
    sauvola_radius : int
        Sauvola window radius.
    sauvola_k : float
        Sauvola sensitivity.
    min_size_filtering : int
        Minimum pore size in voxels; ≤ 0 disables filtering.

    Returns
    -------
    pore_mask : bool array  —  True where pores were detected.
    sample_mask : bool array  —  True where material exists.
    binary : bool array  —  Raw Sauvola output (material = True).
    All three are ``None`` if the input volume is empty.
    """
    logger.info("Starting pore detection...")

    # ── bounding box (Z) ──
    min_z = next((i for i in range(xct.shape[0]) if np.any(xct[i] > 0)), -1)
    if min_z == -1:
        logger.error("Volume contains no non-zero voxels.")
        return None, None, None
    max_z = next((i for i in range(xct.shape[0] - 1, -1, -1) if np.any(xct[i] > 0)), min_z)

    # ── bounding box (Y, X via projection) ──
    proj = np.zeros(xct.shape[1:], dtype=bool)
    for i in tqdm(range(min_z, max_z + 1), desc="Projecting slices", leave=False):
        proj |= xct[i] > 0
    y_inds, x_inds = np.nonzero(proj)
    min_y, max_y = int(y_inds.min()), int(y_inds.max())
    min_x, max_x = int(x_inds.min()), int(x_inds.max())

    # ── add margin ──
    margin = 2
    min_z = max(0, min_z - margin);  max_z = min(xct.shape[0] - 1, max_z + margin)
    min_y = max(0, min_y - margin);  max_y = min(xct.shape[1] - 1, max_y + margin)
    min_x = max(0, min_x - margin);  max_x = min(xct.shape[2] - 1, max_x + margin)
    logger.info("Cropped region: Z[%d:%d] Y[%d:%d] X[%d:%d]", min_z, max_z, min_y, max_y, min_x, max_x)

    cropped = xct[min_z:max_z + 1, min_y:max_y + 1, min_x:max_x + 1]

    # ── Sauvola ──
    binary_cropped = sauvola_thresholding(cropped, window_size=sauvola_radius, k=sauvola_k)

    if frontwall > 0:
        binary_cropped[:frontwall] = True
    if backwall > 0:
        binary_cropped[backwall:] = True

    binary = np.zeros(xct.shape, dtype=bool)
    binary[min_z:max_z + 1, min_y:max_y + 1, min_x:max_x + 1] = binary_cropped

    # ── material mask ──
    sm_cropped = material_mask(cropped)
    sample_mask = np.zeros_like(binary)
    sample_mask[min_z:max_z + 1, min_y:max_y + 1, min_x:max_x + 1] = sm_cropped

    # ── pore extraction ──
    pore_mask = np.logical_and(~binary, sample_mask)
    logger.info("Initial pores: %d voxels.", int(pore_mask.sum()))

    if min_size_filtering > 0:
        pore_mask = clean_pores(pore_mask, min_size=min_size_filtering)

    logger.info("Pore detection complete.")
    return pore_mask, sample_mask, binary
