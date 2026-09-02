"""Shared infrastructure for the eval v3 campaign (correct-decode re-run).

Why v3 exists
-------------
Every volume under ``runs/eval_v2/volumes/`` and
``runs/analysis/ldm06_probe/volumes/`` was produced while
``poregen.diffusion.sampler`` applied ``expit()`` to the VAE's XCT head before
the u8 cast.  That head regresses ``xct / 255`` directly, so the sigmoid was
spurious: it squashed every generated volume into grey levels [133, 187]
(float [0.522, 0.733]) and destroyed contrast.  The sampler now clamps and
scales (``poregen.models.vae.base.decode_xct`` / ``decode_xct_u8``), and
``scripts/analysis/_eval_v2.py:save_volume`` writes ``volume.tif`` as NATIVE
uint8.  v3 regenerates the volumes needed to restate the porosity/air results
and redoes the analyses on them.  ``runs/eval_v2`` is left untouched as the
record of the buggy run.

Tree (all under ``runs/eval_v3/``)::

    volumes/dose_response/<arm>/target_<t>_seed_<s>/   63 volumes, 192^3
    volumes/ddim_probe/steps_<n>_seed_<s>/              8 volumes, 192^3
    volumes/layup/<arm>/<layup>_seed_101/               9 volumes, 1024x1024x192
    dose_response/  ddim/  air_audit/  onlypores/  comparison/   run.log

Intensity scale
---------------
``volume.tif`` is uint8 on the raw-scan grey scale, the SAME scale as the real
scans in ``data/split_v2/volumes.zarr``.  No logit inversion, no ``*255``
guessing: :func:`load_u8` verifies the dtype it actually finds and refuses to
silently reinterpret anything.  The real-calibrated absolute air threshold
(``T_abs = 182``, Dice 0.842 on real volumes) is therefore valid on generated
volumes for the first time.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import tifffile

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import REPO  # noqa: E402

# Methodology reused verbatim from the v2 air audit (same definitions, same
# constants) so v2 and v3 numbers are directly comparable.
from air_audit_v2 import (  # noqa: E402
    EDGE_VOX, MIN_CC, PATCH, VOXEL_MM3, analyse_modes, cc_filter, cellwise,
    dice_prec_rec, edge_shell, hist_u8, intensity_stats, largest_components,
    otsu,
)

ROOT = REPO / "runs" / "eval_v3"
VOL_ROOT = ROOT / "volumes"
DOSE_VOL_ROOT = VOL_ROOT / "dose_response"
DDIM_VOL_ROOT = VOL_ROOT / "ddim_probe"
LAYUP_VOL_ROOT = VOL_ROOT / "layup"

# v2 artefacts read (never written) for the old-vs-new comparison and for the
# real-volume calibration that the decode fix does not change.
V2_ROOT = REPO / "runs" / "eval_v2"
V2_AIR_AUDIT = REPO / "runs" / "analysis" / "air_audit_v2"
V2_ONLYPORES = REPO / "runs" / "analysis" / "onlypores_generated"
V2_LDM06_PROBE = REPO / "runs" / "analysis" / "ldm06_probe"

ARMS = ["seq", "joint_legacy", "joint_oob"]
ARM_SPOR = {"seq": 1.0, "joint_legacy": 1.5, "joint_oob": 1.5}
# Categorical hues checked against the dataviz palette rules (lightness band,
# chroma floor, CVD separation); arms also carry distinct markers.
ARM_COLORS = {"seq": "#1b6ca8", "joint_legacy": "#c2571a",
              "joint_oob": "#6a3d9a", "probe": "#2e7d32"}
ARM_MARKERS = {"seq": "o", "joint_legacy": "s", "joint_oob": "^", "probe": "D"}

TARGETS = [0.005, 0.01, 0.02, 0.03, 0.05, 0.07, 0.10]
SEEDS = [101, 202, 303]
GATE = 0.005                       # D39: |delivered - requested| < 0.005

# DDIM probe protocol (mirror of scripts/analysis/ldm06_probe.py part B).
DDIM_STEP_COUNTS = [50, 100, 200, 300]
DDIM_SEEDS = [101, 202]
DDIM_TARGET = 0.03
DDIM_S_POR = 1.5
DDIM_ARM = "joint_oob"

LAYUP_SEED = 101
LAYUP_TARGET = 0.03


# ---------------------------------------------------------------------------
# Calibration — reused from the v2 air audit, which calibrated on REAL volumes
# ---------------------------------------------------------------------------

def load_calibration() -> dict:
    """Real-volume air-detector calibration (unchanged by the decode fix).

    ``T_abs`` is the Dice-optimal absolute u8 threshold inside ``sample_mask``
    on 8 real val/test volumes.  It was already valid on real data; the decode
    fix is what makes it valid on GENERATED data too, because generated
    volumes now carry the same grey scale.
    """
    res = json.loads((V2_AIR_AUDIT / "results.json").read_text())
    cal = res["calibration"]
    return {
        "t_abs": int(cal["t_abs"]),
        "dice": float(cal["dice"]),
        "precision": float(cal["precision"]),
        "recall": float(cal["recall"]),
        "n_calibration_volumes": int(cal["n_calibration_volumes"]),
        "calibration_volumes": list(cal["calibration_volumes"]),
        "offset_from_material_mode": float(cal["offset_from_material_mode"]),
        "real_material_mode_mean": float(cal["real_material_mode_mean"]),
        "min_cc_voxels": MIN_CC,
        "edge_shell_vox": EDGE_VOX,
        "real_false_positive_baseline": res["real_false_positive_baseline"],
        "source": str(V2_AIR_AUDIT / "results.json"),
    }


# ---------------------------------------------------------------------------
# THE loader.  Verify the dtype; never guess a transform.
# ---------------------------------------------------------------------------

def load_u8(vol_path: Path) -> tuple[np.ndarray, str]:
    """Read ``volume.tif`` as uint8 grey levels, verifying the stored dtype.

    Returns ``(u8, provenance)``.  Three cases, decided by what is on disk:

    * ``uint8``            — the v3 native format; returned unchanged.
    * float in [0, 1]      — a v2-era file (``u8 / 255`` written as float32);
      rescaled by 255 so v2 volumes can still be read for the comparison.
    * float in [0, 255]    — rounded.

    Anything else raises.  Reading is chunked because a 1024x1024x192 volume is
    201 M voxels.
    """
    mm = tifffile.memmap(str(vol_path), mode="r")
    dtype = mm.dtype
    try:
        if dtype == np.uint8:
            out = np.array(mm)
            return out, "uint8 (native, v3)"
        if not np.issubdtype(dtype, np.floating):
            raise TypeError(f"{vol_path}: unsupported dtype {dtype}")
        vmax = float(np.asarray(mm[0]).max())
        for z0 in range(0, mm.shape[0], 32):
            vmax = max(vmax, float(np.asarray(mm[z0:z0 + 32]).max()))
        if vmax <= 1.5:
            scale, prov = 255.0, f"float{dtype.itemsize * 8} in [0,1] x255 (v2 format)"
        elif vmax <= 255.5:
            scale, prov = 1.0, f"float{dtype.itemsize * 8} in [0,255] rounded"
        else:
            raise ValueError(f"{vol_path}: float volume with max {vmax}")
        out = np.empty(mm.shape, np.uint8)
        for z0 in range(0, mm.shape[0], 32):
            chunk = np.asarray(mm[z0:z0 + 32], np.float32) * scale
            out[z0:z0 + 32] = np.clip(chunk, 0.0, 255.0).astype(np.uint8)
        return out, prov
    finally:
        del mm


def load_mask(vol_dir: Path) -> np.ndarray:
    return tifffile.imread(str(vol_dir / "mask.tif")) > 0


# ---------------------------------------------------------------------------
# Volume enumeration
# ---------------------------------------------------------------------------

def iter_volumes(root: Path = VOL_ROOT):
    """Yield ``(experiment, arm, dir)`` for every completed volume under root."""
    for exp, arm_dirs in (("dose_response", True), ("layup", True),
                          ("ddim_probe", False)):
        base = root / exp
        if not base.exists():
            continue
        if arm_dirs:
            for arm in ARMS:
                sub = base / arm
                if not sub.exists():
                    continue
                for d in sorted(sub.iterdir()):
                    if _complete(d):
                        yield exp, arm, d
        else:
            for d in sorted(base.iterdir()):
                if _complete(d):
                    yield exp, "probe", d


def _complete(d: Path) -> bool:
    return (d.is_dir() and (d / "volume.tif").exists()
            and (d / "mask.tif").exists() and (d / "stats.json").exists())


def read_stats(d: Path) -> dict:
    return json.loads((d / "stats.json").read_text())


# ---------------------------------------------------------------------------
# Small shared statistics
# ---------------------------------------------------------------------------

def fit_ols(x: np.ndarray, y: np.ndarray) -> dict:
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    ok = np.isfinite(x) & np.isfinite(y)
    x, y = x[ok], y[ok]
    if len(x) < 3 or np.allclose(x, x[0]):
        return {"slope": None, "intercept": None, "r2": None, "n": int(len(x))}
    slope, intercept = np.polyfit(x, y, 1)
    pred = slope * x + intercept
    ss_res = float(np.sum((y - pred) ** 2))
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    return {"slope": float(slope), "intercept": float(intercept),
            "r2": (1.0 - ss_res / ss_tot) if ss_tot > 0 else None,
            "n": int(len(x))}


def err_stats(err: np.ndarray, gate: float = GATE) -> dict:
    a = np.abs(np.asarray(err, float))
    a = a[np.isfinite(a)]
    if not len(a):
        return {"n": 0}
    return {"abs_error_mean": float(a.mean()),
            "abs_error_median": float(np.median(a)),
            "abs_error_p95": float(np.percentile(a, 95)),
            "abs_error_max": float(a.max()),
            "frac_within_gate": float((a < gate).mean()),
            "n": int(len(a))}


def block_sums(a: np.ndarray, patch: int = PATCH) -> np.ndarray:
    """Voxel count of a boolean array per ``patch``^3 tile cell."""
    gz, gy, gx = (s // patch for s in a.shape)
    return (a[:gz * patch, :gy * patch, :gx * patch]
            .reshape(gz, patch, gy, patch, gx, patch)
            .sum(axis=(1, 3, 5), dtype=np.int64))


__all__ = [
    "ROOT", "VOL_ROOT", "DOSE_VOL_ROOT", "DDIM_VOL_ROOT", "LAYUP_VOL_ROOT",
    "V2_ROOT", "V2_AIR_AUDIT", "V2_ONLYPORES", "V2_LDM06_PROBE",
    "ARMS", "ARM_SPOR", "ARM_COLORS", "ARM_MARKERS", "TARGETS", "SEEDS",
    "GATE", "DDIM_STEP_COUNTS", "DDIM_SEEDS", "DDIM_TARGET", "DDIM_S_POR",
    "DDIM_ARM", "LAYUP_SEED", "LAYUP_TARGET",
    "PATCH", "MIN_CC", "EDGE_VOX", "VOXEL_MM3",
    "load_calibration", "load_u8", "load_mask", "iter_volumes", "read_stats",
    "fit_ols", "err_stats", "block_sums",
    "analyse_modes", "cc_filter", "cellwise", "dice_prec_rec", "edge_shell",
    "hist_u8", "intensity_stats", "largest_components", "otsu",
]
