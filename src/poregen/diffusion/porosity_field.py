"""Coherent per-patch porosity field for volume generation (D32 §4).

Given a requested GLOBAL porosity target ``G``, every patch in a generated
volume must be conditioned on a LOCAL porosity read from one spatially
coherent field over the whole volume.  Independent per-patch draws would let
adjacent patches contradict each other, and painting ``G`` uniformly is
off-manifold (only ~12.9% of real patches sit within 0.001 of their volume
mean).

Recipe (exactly as prescribed):

1. Marginal — the T-E conditional sampler
   (``runs/campaigns/01-conditioning-design/T-E/results.json`` → ``"sampler"``):
   pick the two nearest ``bin_centres_global_phi`` to ``G``, linearly
   interpolate ``ratio_quantiles`` between them, draw ``u ~ U(0, 1)`` per
   grid cell, read the interpolated quantile at ``u``, and set
   ``local_phi = G * ratio``.  This reproduces the marginal
   ``p(local | global)`` only.
2. Spatial coherence — smooth the i.i.d. field with a per-axis Gaussian
   whose sigma is the T-D correlation length (volume-mean-removed,
   patch-level, in voxels: z 79.45, y 413.58, x 900.95) divided by the
   generation stride (64 voxels).  Smoothing shrinks the variance; that is
   accepted — the spec is draw → smooth → rescale-mean, not
   variance-matching.
3. Rescale the smoothed field so its MEAN equals ``G``, clamp to the
   training phi range [0.002, 0.107] (D39), rescale the mean once more,
   then clamp again.  If the target sits near a range edge the final clamp
   can leave a small residual mean error; ``build_porosity_field`` does not
   iterate further.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from scipy.ndimage import gaussian_filter

# Training phi range (D39).
PHI_MIN = 0.002
PHI_MAX = 0.107

# Default artefact locations, relative to the repo root.
DEFAULT_TE_RESULTS = Path("runs/campaigns/01-conditioning-design/T-E/results.json")
DEFAULT_TD_RESULTS = Path("runs/campaigns/01-conditioning-design/T-D/results.json")


def load_sampler(path: str | Path) -> dict[str, np.ndarray]:
    """Load the T-E conditional sampler tables as float64 arrays.

    Returns a dict with ``quantile_levels`` (Q,), ``bin_centres_global_phi``
    (B,) and ``ratio_quantiles`` (B, Q).
    """
    raw = json.loads(Path(path).read_text())["sampler"]
    return {
        "quantile_levels": np.asarray(raw["quantile_levels"], dtype=np.float64),
        "bin_centres_global_phi": np.asarray(
            raw["bin_centres_global_phi"], dtype=np.float64
        ),
        "ratio_quantiles": np.asarray(raw["ratio_quantiles"], dtype=np.float64),
    }


def load_corr_lengths_voxels(path: str | Path) -> tuple[float, float, float]:
    """Load the (z, y, x) 1/e correlation lengths in voxels from T-D.

    Uses the patch-level, volume-mean-removed variant — the raw variant is
    inflated by the volume-to-volume mean differences, which are irrelevant
    inside one generated volume.
    """
    patch_level = json.loads(Path(path).read_text())["patch_level"]
    return tuple(
        float(patch_level[f"{axis}_volume_mean_removed"]["corr_length_1_over_e_voxels"])
        for axis in ("z", "y", "x")
    )


def _draw_marginal(
    target: float, sampler: dict[str, np.ndarray], u: np.ndarray
) -> np.ndarray:
    """Draw local phi values from the T-E marginal p(local | global=target).

    ``u`` holds uniform(0, 1) variates; one local phi is returned per entry.
    Targets outside the binned range use the nearest edge bin's quantiles.
    """
    centres = sampler["bin_centres_global_phi"]
    rq = sampler["ratio_quantiles"]
    levels = sampler["quantile_levels"]

    hi = int(np.searchsorted(centres, target))
    if hi <= 0:
        row = rq[0]
    elif hi >= len(centres):
        row = rq[-1]
    else:
        lo = hi - 1
        w = (target - centres[lo]) / (centres[hi] - centres[lo])
        row = (1.0 - w) * rq[lo] + w * rq[hi]

    ratio = np.interp(u, levels, row)
    return target * ratio


def build_porosity_field(
    grid_shape: tuple[int, int, int],
    target: float,
    sampler: dict[str, np.ndarray],
    corr_lengths_voxels: tuple[float, float, float],
    stride_voxels: int = 64,
    seed: int = 0,
) -> np.ndarray:
    """Build a coherent (gz, gy, gx) field of local porosities with mean ~target.

    Deterministic per ``seed``.  Values lie in [PHI_MIN, PHI_MAX]; the mean
    equals ``target`` exactly unless the final clamp bites (target near a
    range edge), in which case a small residual error remains.
    """
    rng = np.random.default_rng(seed)
    u = rng.uniform(0.0, 1.0, size=grid_shape)
    field = _draw_marginal(float(target), sampler, u)

    # T-D lengths are in voxels; the field lives on the patch grid, so one
    # grid step is `stride_voxels`.  mode="nearest": the volume is not
    # periodic ("wrap" would correlate opposite faces), and "nearest"
    # extends the boundary value, keeping the local porosity level at the
    # faces instead of mirroring interior fluctuations back in.
    sigma = tuple(cl / stride_voxels for cl in corr_lengths_voxels)
    field = gaussian_filter(field, sigma=sigma, mode="nearest")

    # Rescale mean → clamp → rescale mean once more → final clamp (D32 §4).
    field *= target / field.mean()
    field = np.clip(field, PHI_MIN, PHI_MAX)
    field *= target / field.mean()
    return np.clip(field, PHI_MIN, PHI_MAX)
