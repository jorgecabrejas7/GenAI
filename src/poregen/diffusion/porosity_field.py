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
import logging
from pathlib import Path

import numpy as np
from scipy.ndimage import gaussian_filter

# Training phi range (D39).
logger = logging.getLogger(__name__)

PHI_MIN = 0.002
PHI_MAX = 0.107

# Default artefact locations, relative to the repo root.
# THE _v3 FITS, AND WHY THE ORIGINALS ARE NOT USED.
#
# T-D and T-E are read at GENERATION time, so whatever they were fitted on is
# data the generator has seen. The originals were fitted on all 80 volumes of
# the split_v2 index — before the current split existed, with hole-touching
# patches still in — so the priors every published field was drawn from had
# seen the val and test panels. That is leakage into the generator, not into
# the model, and it is exactly as disqualifying.
#
# The _v3 fits are the same two scripts run with `--split train --patch-index
# data/split_v3/patch_index.parquet`: 58 volumes, 1 598 000 patches, the split
# every current model was actually trained on. The originals are kept, with a
# banner, because the published numbers were built from them.
DEFAULT_TE_RESULTS = Path("runs/campaigns/01-conditioning-design/T-E_v3/results.json")
DEFAULT_TD_RESULTS = Path("runs/campaigns/01-conditioning-design/T-D_v3/results.json")


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


def corr_length_is_well_defined(block: dict) -> bool:
    """Is this axis's 1/e length a LENGTH, or where a plateau happened to dip?

    A 1/e correlation length means something only if the curve DECAYS to 1/e.
    Where it instead flattens out above 1/e and wanders, the "crossing" is
    wherever noise takes it briefly under the line, and that position moves
    with the sample rather than with the material.

    This is not hypothetical. Refitting T-D on the split_v3 TRAIN panels made
    the in-plane y curve plateau near 0.45 with a single excursion to 0.359
    against a 1/e of 0.3679 — so its reported length moved from 416 voxels to
    2528, a factor of six, on a curve that never really decays. Smoothing with
    that number would make the generated field constant along y.

    The test is monotonicity up to the crossing: a curve that only ever falls
    has a length; one that rises again before crossing does not.
    """
    r = np.asarray(block["correlation"], dtype=np.float64)
    below = np.flatnonzero(r < np.exp(-1.0))
    if not below.size:
        return False
    return bool(np.all(np.diff(r[: below[0] + 1]) <= 1e-9))


def load_corr_lengths_voxels(path: str | Path) -> tuple[float, float, float]:
    """Load the (z, y, x) 1/e correlation lengths in voxels from T-D.

    Uses the patch-level, volume-mean-removed variant — the raw variant is
    inflated by the volume-to-volume mean differences, which are irrelevant
    inside one generated volume.

    WARNS, per axis, when the curve does not actually decay to 1/e. The number
    is still returned, because refusing it would take down generation over a
    property of the dataset, but a length that is not a length must not pass
    silently into a smoothing kernel.
    """
    patch_level = json.loads(Path(path).read_text())["patch_level"]
    out = []
    for axis in ("z", "y", "x"):
        block = patch_level[f"{axis}_volume_mean_removed"]
        length = float(block["corr_length_1_over_e_voxels"])
        if not corr_length_is_well_defined(block):
            logger.warning(
                "%s: the %s correlation curve does not decay monotonically to "
                "1/e, so its reported length of %.0f voxels is where a plateau "
                "happens to dip and not a correlation length. Smoothing with it "
                "will make the field nearly constant along %s.",
                Path(path).parent.name, axis, length, axis)
        out.append(length)
    return tuple(out)


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

    TWO PROPERTIES OF THE DELIVERED FIELD, BOTH MEASURED, NEITHER OBVIOUS.

    1. The requested correlation lengths are CAPPED AT THE CANVAS EXTENT.
       Real in-plane porosity correlation is longer than any coupon-scale
       canvas — campaign 21 found the in-plane porosity of all 80 volumes flat
       to within 6 % across y and x — so the fitted in-plane length is not a
       length this canvas can express. On the split_v3 train refit the y curve
       does not decay to 1/e at all, and its nominal 2521 voxels is 20 grid
       steps of a 16-step production grid: smoothing with it would make the
       field constant along y while claiming to have applied a measurement.
       Capping says the honest thing instead — at these sizes A SAMPLED FIELD
       IS EFFECTIVELY A THROUGH-THICKNESS PROFILE, and the project's
       controllable local-field claims rest on the PAINTED fields
       (`field_two_halves`, `field_checkerboard`), not on this one.

    2. The FITTED MARGINAL IS NOT PRESERVED. Gaussian smoothing reduces
       variance and the rescale-and-clip below restores the MEAN but not the
       SPREAD: the delivered field carries about 5 % of the fitted T-E spread
       on a large grid and 7-12 % on a production one (campaign 26). A field
       cannot have both a long correlation length and the unsmoothed marginal;
       this one buys the length. Do not describe the output as drawn from the
       T-E marginal — it is drawn from it and then smoothed, which is a
       different distribution.
    """
    rng = np.random.default_rng(seed)
    u = rng.uniform(0.0, 1.0, size=grid_shape)
    field = _draw_marginal(float(target), sampler, u)

    # T-D lengths are in voxels; the field lives on the patch grid, so one
    # grid step is `stride_voxels`.  mode="nearest": the volume is not
    # periodic ("wrap" would correlate opposite faces), and "nearest"
    # extends the boundary value, keeping the local porosity level at the
    # faces instead of mirroring interior fluctuations back in.
    #
    # THE FACTOR OF TWO IS NOT COSMETIC. Smoothing white noise with a Gaussian
    # kernel of standard deviation s gives an autocorrelation that is itself
    # Gaussian with standard deviation s*sqrt(2):
    #
    #     rho(r) = exp(-r^2 / (4 s^2))
    #
    # so rho falls to 1/e at r = 2s, NOT at r = s. Setting sigma = cl/stride
    # therefore delivered a 1/e correlation length of 2*cl — every coherent
    # field this project generated was twice as smooth as the T-D length it
    # was asked for. sigma = cl/(2*stride) delivers cl.
    # A length longer than the canvas cannot be delivered by it, and asking for
    # one produces a constant field rather than a correlated one. Capped per
    # axis at the canvas extent; in practice only the in-plane axes are ever
    # affected, because the through-thickness length is tens of voxels and the
    # in-plane ones are hundreds to thousands.
    extents = tuple(float(g) * stride_voxels for g in grid_shape)
    capped = tuple(min(float(cl), e) for cl, e in zip(corr_lengths_voxels, extents))
    for axis, (cl, cap) in enumerate(zip(corr_lengths_voxels, capped)):
        if cap < cl:
            logger.info(
                "porosity field: %s correlation length capped from %.0f to the "
                "%.0f-voxel canvas extent — a longer length cannot be expressed "
                "on this canvas, and at this size the field is effectively a "
                "through-thickness profile",
                "zyx"[axis], cl, cap)
    sigma = tuple(cl / (2.0 * stride_voxels) for cl in capped)
    field = gaussian_filter(field, sigma=sigma, mode="nearest")

    # Rescale mean → clamp → rescale mean once more → final clamp (D32 §4).
    field *= target / field.mean()
    field = np.clip(field, PHI_MIN, PHI_MAX)
    field *= target / field.mean()
    return np.clip(field, PHI_MIN, PHI_MAX)
