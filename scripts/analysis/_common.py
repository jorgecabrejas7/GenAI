"""Shared helpers for the ldm05 conditioning-design analysis suite.

All scripts under ``scripts/analysis/`` write to
``runs/analysis/conditioning_design/<test_id>/`` and share the plotting style
and JSON-serialisation helpers defined here.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

REPO = Path(__file__).resolve().parents[2]
DATA_ROOT = REPO / "data" / "split_v2"
PATCH_INDEX = DATA_ROOT / "patch_index.parquet"
ZARR_ROOT = DATA_ROOT / "volumes.zarr"
LATENT_INDEX_DIR = DATA_ROOT / "latents_r07z4"
OUT_ROOT = REPO / "runs" / "analysis" / "conditioning_design"

PATCH_SIZE = 64
STRIDE = 32

# Voxel size is NOT recorded anywhere in the dataset metadata (patches_meta.json,
# volume_stats.json, splits.json, the zarr attrs, or the build_dataset code).
# Everything below is therefore reported in voxels.
VOXEL_SIZE_UM: float | None = None


# ---------------------------------------------------------------------------
# Plot style — publication quality
# ---------------------------------------------------------------------------

def set_style() -> None:
    plt.rcParams.update({
        "figure.dpi": 120,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "font.size": 10,
        "axes.titlesize": 11,
        "axes.labelsize": 10,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "legend.fontsize": 8.5,
        "axes.grid": True,
        "grid.alpha": 0.25,
        "grid.linewidth": 0.6,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "lines.linewidth": 1.3,
        "figure.autolayout": False,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })


AXIS_COLORS = {"z": "#1b6ca8", "y": "#c2571a", "x": "#2e7d32"}


def savefig(fig, out_dir: Path, name: str) -> list[str]:
    """Save a figure as both .pdf and .png at 300 dpi. Returns the paths."""
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for ext in ("pdf", "png"):
        p = out_dir / f"{name}.{ext}"
        fig.savefig(p, dpi=300)
        paths.append(str(p))
    plt.close(fig)
    return paths


# ---------------------------------------------------------------------------
# JSON
# ---------------------------------------------------------------------------

def _jsonable(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {str(k): _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return _jsonable(obj.tolist())
    if isinstance(obj, (np.floating, float)):
        f = float(obj)
        return None if (np.isnan(f) or np.isinf(f)) else f
    if isinstance(obj, (np.integer, int)):
        return int(obj)
    if isinstance(obj, (np.bool_, bool)):
        return bool(obj)
    if obj is None or isinstance(obj, str):
        return obj
    return str(obj)


def write_json(results: dict, out_dir: Path, name: str = "results.json") -> str:
    out_dir.mkdir(parents=True, exist_ok=True)
    p = out_dir / name
    with open(p, "w") as fh:
        json.dump(_jsonable(results), fh, indent=2)
    return str(p)


def write_findings(text: str, out_dir: Path) -> str:
    out_dir.mkdir(parents=True, exist_ok=True)
    p = out_dir / "findings.md"
    p.write_text(text.strip() + "\n")
    return str(p)


# ---------------------------------------------------------------------------
# Signal analysis
# ---------------------------------------------------------------------------

def detrend_profile(y: np.ndarray, poly_order: int = 3) -> np.ndarray:
    """Remove a low-order polynomial trend (slow drift / edge effects)."""
    n = len(y)
    x = np.linspace(-1.0, 1.0, n)
    good = np.isfinite(y)
    if good.sum() < poly_order + 2:
        return np.zeros_like(y)
    coef = np.polyfit(x[good], y[good], poly_order)
    resid = y - np.polyval(coef, x)
    resid[~good] = 0.0
    return resid


def autocorrelation(y: np.ndarray, max_lag: int | None = None) -> np.ndarray:
    """Unbiased-ish normalised autocorrelation of a 1-D signal (lag 0 -> 1)."""
    y = np.asarray(y, dtype=np.float64)
    y = y - y.mean()
    n = len(y)
    if max_lag is None:
        max_lag = n // 2
    nfft = 1 << int(np.ceil(np.log2(2 * n)))
    f = np.fft.rfft(y, nfft)
    ac = np.fft.irfft(f * np.conj(f), nfft)[: max_lag + 1].real
    denom = ac[0] if ac[0] > 0 else 1.0
    return ac / denom


def periodogram(y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Hann-windowed periodogram. Returns (period_in_samples, power)."""
    y = np.asarray(y, dtype=np.float64)
    y = y - y.mean()
    n = len(y)
    w = np.hanning(n)
    yw = y * w
    f = np.fft.rfft(yw)
    power = (np.abs(f) ** 2) / (np.sum(w ** 2) + 1e-30)
    freq = np.fft.rfftfreq(n, d=1.0)
    with np.errstate(divide="ignore"):
        period = np.where(freq > 0, 1.0 / np.maximum(freq, 1e-30), np.inf)
    return period, power


def ar1_coefficient(y: np.ndarray) -> float:
    y = np.asarray(y, dtype=np.float64)
    y = y - y.mean()
    if len(y) < 3 or np.allclose(y, 0):
        return 0.0
    num = float(np.sum(y[:-1] * y[1:]))
    den = float(np.sum(y[:-1] ** 2))
    if den <= 0:
        return 0.0
    return float(np.clip(num / den, -0.99, 0.99))


def dominant_period(
    y: np.ndarray,
    period_min: float,
    period_max: float,
    n_surrogate: int = 300,
    rng: np.random.Generator | None = None,
) -> dict:
    """Find the dominant period in a band and test it against an AR(1) null.

    The null keeps the AR(1) coefficient and variance of the detrended signal,
    so smooth non-periodic drift does not produce a false detection.
    """
    rng = rng or np.random.default_rng(0)
    y = np.asarray(y, dtype=np.float64)
    n = len(y)
    if n < 16:
        return {"n": n, "detected": False, "reason": "profile too short"}

    period, power = periodogram(y)
    band = (period >= period_min) & (period <= period_max) & np.isfinite(period)
    if band.sum() < 3:
        return {"n": n, "detected": False, "reason": "no frequency bins in band"}

    k = int(np.argmax(np.where(band, power, -np.inf)))
    peak_period = float(period[k])
    peak_power = float(power[k])

    band_power = power[band]
    snr = float(peak_power / (np.median(band_power) + 1e-30))

    # AR(1) surrogate null
    phi = ar1_coefficient(y)
    sd = float(np.std(y))
    from scipy.signal import lfilter

    e = rng.standard_normal((n_surrogate, n))
    s = lfilter([1.0], [1.0, -phi], e, axis=1)
    s *= sd / (s.std(axis=1, keepdims=True) + 1e-30)
    w = np.hanning(n)
    sw = (s - s.mean(axis=1, keepdims=True)) * w
    f = np.fft.rfft(sw, axis=1)
    p_null = (np.abs(f) ** 2) / (np.sum(w ** 2) + 1e-30)
    peaks_null = p_null[:, band].max(axis=1)
    exceed = int(np.sum(peaks_null >= peak_power))
    p_value = (exceed + 1) / (n_surrogate + 1)

    return {
        "n": n,
        "peak_period_voxels": peak_period,
        "peak_power": peak_power,
        "band_median_power": float(np.median(band_power)),
        "spectral_snr": snr,
        "ar1_phi": phi,
        "p_value_vs_ar1": float(p_value),
        "null_peak_p95": float(np.percentile(peaks_null, 95)),
        "detected": bool(p_value < 0.05 and snr > 3.0),
    }


def correlation_length(lags: np.ndarray, corr: np.ndarray) -> float:
    """First lag where the correlation drops below 1/e, by linear interpolation."""
    thr = float(np.exp(-1.0))
    below = np.where(corr < thr)[0]
    if len(below) == 0:
        return float("nan")
    i = int(below[0])
    if i == 0:
        return 0.0
    c0, c1 = corr[i - 1], corr[i]
    l0, l1 = lags[i - 1], lags[i]
    if c0 == c1:
        return float(l1)
    frac = (c0 - thr) / (c0 - c1)
    return float(l0 + frac * (l1 - l0))
