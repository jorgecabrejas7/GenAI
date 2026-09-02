"""Depth-resolved ply-orientation conditioning (D32 §1, signal 4).

The orientation field itself is a dataset artefact,
``data/split_v2/orientation_field.json``, built by
``scripts/build_conditioning.py`` from the NOMINAL ply sequence aligned to the
scan.  This module turns it into the ``(2, L, L, L)`` conditioning tensor and
is the single implementation of that encoding, shared by the dataset (training)
and the sampler (generation).

Encoding rules — every one of them matters:

* Orientation is axial (mod 180°), so it is encoded as ``(cos 2θ, sin 2θ)``.
  0° → (1, 0), 45° → (0, 1), 90° → (−1, 0), −45° → (0, −1).
* **Pool the components, never the angle.**  A mean of angles is meaningless
  across the 180° wrap.
* **Never renormalise the pooled vector.**  Where one latent depth plane
  straddles a ply interface the two directions partially cancel and the
  magnitude drops — that shrinkage *is* the interface marker.
* Slices with no known orientation (outside the laminate, or a volume with no
  expert stacking sequence) contribute the ZERO vector, i.e. "unknown", which
  is distinguishable from every real angle.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np


def encode_theta(theta_deg: np.ndarray) -> np.ndarray:
    """``(cos 2θ, sin 2θ)`` for an array of angles in degrees.

    ``NaN`` entries (unknown orientation) become the zero vector.

    Parameters
    ----------
    theta_deg : (..., N) float array, degrees, ``NaN`` where unknown.

    Returns
    -------
    (2, ..., N) float32 — component 0 is ``cos 2θ``, component 1 ``sin 2θ``.
    """
    t = np.asarray(theta_deg, dtype=np.float64)
    known = np.isfinite(t)
    a = 2.0 * np.deg2rad(np.where(known, t, 0.0))
    return np.stack([np.cos(a) * known, np.sin(a) * known]).astype(np.float32)


def pool_components(comp: np.ndarray, n_planes: int) -> np.ndarray:
    """Mean-pool ``(2, N)`` orientation components down to ``(2, n_planes)``.

    ``N`` must be a whole multiple of ``n_planes``.  The pooled vector is
    returned as-is — deliberately NOT renormalised (see the module docstring).
    """
    comp = np.asarray(comp, dtype=np.float32)
    if comp.ndim != 2 or comp.shape[0] != 2:
        raise ValueError(f"pool_components expects (2, N), got {comp.shape}.")
    n = comp.shape[1]
    if n % n_planes != 0:
        raise ValueError(
            f"cannot pool {n} depths into {n_planes} planes — not a whole multiple."
        )
    return comp.reshape(2, n_planes, n // n_planes).mean(axis=2)


def orientation_tensor(theta_deg: np.ndarray, latent_size: int) -> np.ndarray:
    """Full ``(2, L, L, L)`` orientation conditioning for one patch.

    Parameters
    ----------
    theta_deg   : (patch_size,) angles in degrees through the patch depth,
                  ``NaN`` where unknown.
    latent_size : L, the latent grid size (16 for the r07z4 store).

    Returns
    -------
    (2, L, L, L) float32 — pooled depth profile broadcast over the two
    in-plane latent axes.
    """
    pooled = pool_components(encode_theta(theta_deg), latent_size)   # (2, L)
    out = np.broadcast_to(pooled[:, :, None, None],
                          (2, latent_size, latent_size, latent_size))
    return np.ascontiguousarray(out, dtype=np.float32)


class OrientationField:
    """Per-volume θ(z) in image coordinates, read from the dataset artefact.

    Parameters
    ----------
    path : path to ``orientation_field.json``.
    """

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        with open(self.path) as fh:
            raw: dict[str, Any] = json.load(fh)
        self.metadata = {k: v for k, v in raw.items() if k != "volumes"}
        self.voxel_size_um: float | None = raw.get("voxel_size_um")

        self._theta: dict[str, np.ndarray] = {}
        self.confidence: dict[str, str] = {}
        self.usable: dict[str, bool] = {}
        for vid, v in raw["volumes"].items():
            self.confidence[vid] = v.get("confidence", "none")
            self.usable[vid] = bool(v.get("orientation_usable", False))
            depth = int(v["shape"][0]) if v.get("shape") else 0
            th = v.get("theta_deg")
            if th is None:
                arr = np.full(depth, np.nan, dtype=np.float32)
            else:
                arr = np.array([np.nan if a is None else a for a in th],
                               dtype=np.float32)
            self._theta[vid] = arr

    def __contains__(self, volume_id: str) -> bool:
        return volume_id in self._theta

    def theta(self, volume_id: str) -> np.ndarray:
        """θ(z) for a whole volume, degrees, ``NaN`` outside the laminate."""
        return self._theta[volume_id]

    def patch_theta(self, volume_id: str, z0: int, patch_size: int) -> np.ndarray:
        """θ(z) over ``[z0, z0 + patch_size)``, zero-padded with ``NaN``."""
        full = self._theta[volume_id]
        out = np.full(patch_size, np.nan, dtype=np.float32)
        lo = max(0, z0)
        hi = min(len(full), z0 + patch_size)
        if hi > lo:
            out[lo - z0: hi - z0] = full[lo:hi]
        return out

    def patch_tensor(self, volume_id: str, z0: int, patch_size: int,
                     latent_size: int) -> np.ndarray:
        """``(2, L, L, L)`` orientation conditioning for one patch."""
        return orientation_tensor(
            self.patch_theta(volume_id, z0, patch_size), latent_size
        )
