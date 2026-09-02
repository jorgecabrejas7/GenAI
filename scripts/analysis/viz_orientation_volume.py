"""Colour-coded map of local in-plane tow orientation for one CFRP XCT volume.

The output is a visual check of the layup: an RGB TIFF stack in which HUE is the
local in-plane texture angle (mod 180 degrees), SATURATION is the anisotropy
strength, and VALUE is the original XCT grey level.  A second TIFF paints every
z-slice with its single dominant orientation, so the ply stack is visible as
coloured bands in an orthogonal view.

Method, per z-slice
-------------------
* An in-plane crop of ``--xy-window`` voxels is cut from the slice and placed
  automatically on the most solid material -- these specimens are open-hole
  coupons.  A window of ``--window`` voxels slides over the crop at ``--stride``.
* Each window is plane-detrended, multiplied by a **radially symmetric** Tukey
  taper and Fourier transformed with 4x zero padding.  The taper must be
  circular: a separable Hann window puts its leakage on the kx and ky axes and
  fakes a strong 0/90-degree texture.
* The power in the 16-64-voxel wavelength annulus is kept, the kx = 0 and ky = 0
  lines are dropped (the discrete FFT collapses each of them into one angular
  direction and produces a spurious spike at exactly 0 and 90 degrees -- the
  same artifact ``t_h_angular_shape.py`` excises), and the remaining samples are
  divided by an **isotropic flat field** measured by Monte Carlo (below).
* The angular power density h(theta) is fitted by least squares with the modes
  m = 2, 4, 6.  A least-squares fit is used, not a moment sum, because the
  spectrum samples are not uniformly spread in angle at these small radii.
* The **dominant angle is the argmax of h**, not the doubled-angle first moment.
  This matters: T-G collapsed two tow directions 90 degrees apart into one
  averaged angle and wrongly called the Nacho family unidirectional (see
  ``runs/analysis/conditioning_design/T-H/findings.md``).  Two lobes cancel in
  the 2-theta moment; they do not cancel in the density itself.
* Spectral angle is converted to real-space texture angle by +90 degrees, the
  same convention as T-G and T-H.

Noise floor -- why the map is smoothed
--------------------------------------
A single 64-voxel window holds only about 30 independent Fourier samples in the
16-64-voxel band, so its orientation estimate is almost pure noise.  The script
measures this directly: it synthesises isotropic slices with the volume's own
radial power spectrum, runs the identical pipeline on them, and reports the
resulting a2 amplitude.  The same synthetic pass gives the flat field that
removes the pipeline's own angular response.

Two things are done with that measurement.  The coefficient maps are smoothed
over ``--smooth`` window-grid steps before the angle is read, which trades
spatial resolution for signal.  The mode amplitudes then have the residual noise
floor subtracted in quadrature, so a window with no resolvable texture gets
saturation 0 and comes out grey instead of a confident wrong colour.

Outputs (``runs/analysis/orientation_viz/<volume_id>/``)
-------------------------------------------------------
* ``orientation_rgb.tif``  -- RGB stack, local orientation colour over the XCT.
* ``ply_map_rgb.tif``      -- RGB stack, one flat colour per slice.
* ``legend.png``           -- hue/angle wheel, hue bar and saturation ramp.
* ``orthogonal_cut.png``   -- y-z cut through the colour volume, the ply-map
                              strip, the per-slice angle profile and the
                              periodogram of the orientation signal.
* ``results.json``         -- every number quoted in the README.
* ``README.md``            -- what the colours mean and what the run found.
"""

from __future__ import annotations

import argparse
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import tifffile
import zarr
from matplotlib.colors import hsv_to_rgb
from numpy.lib.stride_tricks import sliding_window_view
from scipy import ndimage
from scipy.signal import find_peaks

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import ZARR_ROOT, set_style, write_json, plt  # noqa: E402

REPO = Path(__file__).resolve().parents[2]
OUT_ROOT = REPO / "runs" / "analysis" / "orientation_viz"

DEFAULT_VOLUME = (
    "MedidasDB__Fabricacion_Nacho_05_Probetas_Nacho_2025_probetas_Na_07_2"
    "_volume_eq_aligned"
)

BAND = (16.0, 64.0)      # tow-scale wavelength annulus, voxels (T-G/T-H primary band)
PAD = 256                # zero-padded FFT size; keeps the axis gap under ~15 deg
MODES = (2, 4, 6)        # angular Fourier modes of h(theta); resolves ~15 deg
TAPER_FRAC = 0.25        # radial Tukey taper width, as a fraction of the radius
NEVAL = 180              # 1-degree evaluation grid for h(theta)
PLY_PITCH = 19.6         # voxels, from T-A
N_NULL = 8               # synthetic isotropic slices, per Monte-Carlo round
PLY_TILE = 256           # in-plane size of one ply-map page (colour is constant)
EPS = 1e-20

FAMILIES = {
    "Fabricacion_Nacho_05": "Nacho -- 0/90 cross-ply, 50 volumes",
    "Airbus_Panel_Pegaso": "Pegaso -- ~45-degree steps, 24 volumes",
    "Juan_Ignacio": "Juan_Ignacio -- ~45-degree steps, 6 volumes",
}


# ---------------------------------------------------------------------------
# Geometry: taper, spectrum selection, least-squares fit, evaluation grid
# ---------------------------------------------------------------------------

class Geometry:
    """Everything that depends only on (window, stride) -- built once per run."""

    def __init__(self, window: int, stride: int):
        self.window = int(window)
        self.stride = int(stride)
        w = self.window

        yy, xx = np.mgrid[0:w, 0:w]
        r = np.hypot(yy - (w - 1) / 2, xx - (w - 1) / 2) / (w / 2)
        t = (r - (1.0 - TAPER_FRAC)) / TAPER_FRAC
        self.taper = np.where(
            r >= 1.0, 0.0,
            np.where(r > 1.0 - TAPER_FRAC, 0.5 * (1 + np.cos(np.pi * np.clip(t, 0, 1))), 1.0),
        ).astype(np.float32)

        # plane detrend basis (constant + two ramps), applied before the taper
        basis = np.stack([np.ones(w * w),
                          (yy.ravel() - (w - 1) / 2) / w,
                          (xx.ravel() - (w - 1) / 2) / w], 1).astype(np.float32)
        self.detrend_basis = basis
        self.detrend_pinv = np.linalg.pinv(basis).astype(np.float32)

        fy = np.fft.fftfreq(PAD)[:, None]
        fx = np.fft.fftfreq(PAD)[None, :]
        fy2 = np.broadcast_to(fy, (PAD, PAD))
        fx2 = np.broadcast_to(fx, (PAD, PAD))
        rad = np.hypot(fy2, fx2)
        on_axis = (np.abs(fx2) < 0.5 / PAD) | (np.abs(fy2) < 0.5 / PAD)
        self.sel = (rad >= 1.0 / BAND[1]) & (rad <= 1.0 / BAND[0]) & ~on_axis

        ang = np.arctan2(fy2[self.sel], fx2[self.sel])
        cols = [np.ones(ang.size)]
        for m in MODES:
            cols += [np.cos(m * ang), np.sin(m * ang)]
        design = np.stack(cols, 1)
        self.pinv = np.linalg.pinv(design).astype(np.float32)
        self.condition = float(np.linalg.cond(design))
        self.n_spectrum = int(self.sel.sum())

        # h(theta) on a texture-angle grid; spectral angle = texture angle - 90 deg
        tex = np.deg2rad(np.arange(NEVAL) + 0.5)
        spec = tex - np.pi / 2
        rows = [np.ones(NEVAL)]
        for m in MODES:
            rows += [np.cos(m * spec), np.sin(m * spec)]
        self.evalmat = np.stack(rows, 0).astype(np.float32)

    def grid_shape(self, n: int) -> tuple[int, int]:
        k = (n - self.window) // self.stride + 1
        return k, k

    def window_power(self, img: np.ndarray) -> np.ndarray:
        """Annulus power of every sliding window of one slice -> (n_win, n_spectrum)."""
        w = self.window
        view = sliding_window_view(img, (w, w))[::self.stride, ::self.stride]
        flat = view.reshape(-1, w * w).astype(np.float32)
        flat = flat - (flat @ self.detrend_pinv.T) @ self.detrend_basis.T
        f = np.fft.fft2(flat.reshape(-1, w, w) * self.taper, s=(PAD, PAD))
        return (f.real ** 2 + f.imag ** 2).astype(np.float32)[:, self.sel]

    def coefs(self, img: np.ndarray, iso: np.ndarray) -> np.ndarray:
        """Least-squares angular-mode coefficients per window -> (ny, nx, 1+2*len(MODES))."""
        p = self.window_power(img) / iso
        ny, nx = self.grid_shape(img.shape[0])
        return (p @ self.pinv.T).reshape(ny, nx, -1)


def mode_amplitudes(c: np.ndarray) -> np.ndarray:
    """a_m = 2|C_m| for every mode, from fit coefficients -> (..., len(MODES))."""
    c0 = np.maximum(c[..., 0], EPS)
    return np.stack([2.0 * np.hypot(c[..., 1 + 2 * i], c[..., 2 + 2 * i]) / c0
                     for i in range(len(MODES))], -1)


def subtract_noise(c: np.ndarray, floor: np.ndarray) -> np.ndarray:
    """Shrink each mode amplitude by its noise floor, in quadrature.

    A mode weaker than the floor is set to zero, so a window with no resolvable
    texture yields a flat h(theta), hence saturation 0, hence grey.
    """
    out = np.array(c, dtype=np.float32, copy=True)
    a = mode_amplitudes(c)
    for i in range(len(MODES)):
        keep = np.sqrt(np.clip(1.0 - (floor[i] / np.maximum(a[..., i], EPS)) ** 2, 0.0, 1.0))
        out[..., 1 + 2 * i] *= keep
        out[..., 2 + 2 * i] *= keep
    return out


def angle_strength(c: np.ndarray, geo: Geometry) -> tuple[np.ndarray, np.ndarray]:
    """Dominant texture angle (deg, mod 180) and anisotropy strength in [0, 1]."""
    dens = c @ geo.evalmat
    c0 = np.maximum(c[..., 0], EPS)
    k = np.argmax(dens, axis=-1)
    take = lambda off: np.take_along_axis(  # noqa: E731
        dens, ((k + off) % NEVAL)[..., None], axis=-1)[..., 0]
    lo, mid, hi = take(-1), take(0), take(1)
    denom = lo - 2.0 * mid + hi
    shift = np.where(np.abs(denom) > EPS, 0.5 * (lo - hi) / np.where(denom == 0, 1.0, denom), 0.0)
    shift = np.clip(shift, -0.5, 0.5)
    angle = ((k + 0.5 + shift) * (180.0 / NEVAL)) % 180.0
    strength = np.clip((dens.max(-1) - dens.min(-1)) / (2.0 * c0), 0.0, 1.0)
    return angle.astype(np.float32), strength.astype(np.float32)


# ---------------------------------------------------------------------------
# Worker process
# ---------------------------------------------------------------------------

_W: dict = {}


def _init(volume_id: str, window: int, stride: int, crop: tuple[int, int, int],
          amp_iso: np.ndarray) -> None:
    _W["geo"] = Geometry(window, stride)
    _W["arr"] = zarr.open_group(str(ZARR_ROOT), mode="r")[volume_id]["xct"]
    _W["crop"] = crop            # (n, y0, x0)
    _W["amp"] = amp_iso          # isotropic amplitude field on the crop grid


def _synthetic(seed: int) -> np.ndarray:
    n = _W["crop"][0]
    rng = np.random.default_rng(seed)
    white = rng.standard_normal((n, n))
    return np.fft.ifft2(np.fft.fft2(white) * _W["amp"]).real.astype(np.float32)


def _task_null_power(seed: int) -> np.ndarray:
    return _W["geo"].window_power(_synthetic(seed)).mean(0)


def _task_null_coefs(args: tuple[int, np.ndarray]) -> np.ndarray:
    seed, iso = args
    return _W["geo"].coefs(_synthetic(seed), iso)


def _read(z0: int, z1: int) -> np.ndarray:
    n, y0, x0 = _W["crop"]
    return np.asarray(_W["arr"][z0:z1, y0:y0 + n, x0:x0 + n])


def _task_real(args: tuple[int, int, np.ndarray]) -> tuple[int, np.ndarray]:
    z0, z1, iso = args
    blk = _read(z0, z1)
    geo = _W["geo"]
    out = np.stack([geo.coefs(blk[i].astype(np.float32), iso) for i in range(blk.shape[0])])
    return z0, out.astype(np.float32)


# ---------------------------------------------------------------------------
# Volume set-up
# ---------------------------------------------------------------------------

def place_crop(arr, n: int) -> tuple[int, int]:
    """In-plane offset of the crop with the most solid material.

    These specimens are open-hole coupons: each slice carries drilled holes
    several hundred voxels across.  A hole rim has a strong, real, but
    non-tow orientation, so the crop is put where there is least of it.
    """
    depth, height, width = arr.shape
    ds = 4
    zs = (np.array([0.3, 0.5, 0.7]) * depth).astype(int)
    sub = np.stack([np.asarray(arr[int(z), ::ds, ::ds]).astype(np.float32) for z in zs])
    lo, hi = np.percentile(sub, [1.0, 99.0])
    solid = (sub > 0.5 * (lo + hi)).all(0).astype(np.float32)
    k = max(1, n // ds)
    score = ndimage.uniform_filter(solid, size=k, mode="constant", cval=0.0)
    half = k // 2
    h, w = solid.shape
    # tiny pull towards the middle of the specimen, to break ties sensibly
    gy, gx = np.mgrid[0:h, 0:w]
    score = score - 1e-3 * np.hypot((gy - h / 2) / h, (gx - w / 2) / w).astype(np.float32)
    inside = np.full(score.shape, -np.inf, np.float32)
    inside[half:h - k + half + 1, half:w - k + half + 1] = \
        score[half:h - k + half + 1, half:w - k + half + 1]
    i, j = np.unravel_index(int(np.argmax(inside)), inside.shape)
    y0 = int(np.clip((i - half) * ds, 0, height - n))
    x0 = int(np.clip((j - half) * ds, 0, width - n))
    return y0, x0


def interior_z_range(arr, y0: int, x0: int, n: int) -> tuple[int, int]:
    """First and last z-slice whose crop is fully inside the specimen.

    The `_aligned` volumes are padded with a constant grey value in z, so the
    first and last tens of slices carry no material at all.
    """
    sub = np.asarray(arr[:, y0:y0 + n:16, x0:x0 + n:16]).astype(np.float32)
    lo, hi = np.percentile(sub, [1.0, 99.0])
    thr = 0.5 * (lo + hi)
    fg = (sub > thr).mean(axis=(1, 2))
    ok = np.flatnonzero(fg >= 0.9 * np.nanmax(fg))
    if ok.size < 8:
        return 0, int(arr.shape[0])
    return int(ok.min()), int(ok.max()) + 1


def isotropic_amplitude(arr, z0: int, z1: int, y0: int, x0: int, n: int) -> np.ndarray:
    """sqrt of the radially averaged power spectrum of the crop.

    Filtering white noise with this gives a synthetic slice that has the real
    radial spectrum and no angular preference -- the null this script needs.
    """
    acc = np.zeros((n, n))
    zs = np.linspace(z0, z1 - 1, 5).astype(int)
    for z in zs:
        im = np.asarray(arr[int(z), y0:y0 + n, x0:x0 + n]).astype(np.float32)
        acc += np.abs(np.fft.fft2(im - im.mean())) ** 2
    rbin = np.round(np.hypot(np.fft.fftfreq(n)[:, None], np.fft.fftfreq(n)[None, :]) * n).astype(int)
    prof = (np.bincount(rbin.ravel(), weights=acc.ravel())
            / np.maximum(np.bincount(rbin.ravel()), 1))
    return np.sqrt(prof[rbin]).astype(np.float32)


def grey_limits(arr, z0: int, z1: int, y0: int, x0: int, n: int) -> tuple[float, float]:
    zs = np.linspace(z0, z1 - 1, 9).astype(int)
    vals = np.concatenate([np.asarray(arr[int(z), y0:y0 + n:4, x0:x0 + n:4]).ravel() for z in zs])
    lo, hi = np.percentile(vals.astype(np.float32), [1.0, 99.0])
    return float(lo), float(max(hi, lo + 1.0))


def family_of(volume_id: str) -> tuple[str, str]:
    for key, label in FAMILIES.items():
        if key in volume_id:
            return key, label
    return "unknown", "family not recognised from the volume id"


# ---------------------------------------------------------------------------
# Colour
# ---------------------------------------------------------------------------

def to_rgb(angle_deg: np.ndarray, strength: np.ndarray, value: np.ndarray) -> np.ndarray:
    """HSV -> uint8 RGB.  Hue wraps over 0-180 deg, so 0 and 180 share a colour."""
    hsv = np.stack([np.mod(angle_deg, 180.0) / 180.0,
                    np.clip(strength, 0.0, 1.0),
                    np.clip(value, 0.0, 1.0)], -1).astype(np.float32)
    return (hsv_to_rgb(hsv) * 255.0 + 0.5).astype(np.uint8)


def upsample_coords(geo: Geometry, n: int) -> np.ndarray:
    idx = (np.arange(n) - (geo.window - 1) / 2.0) / geo.stride
    return np.stack(np.meshgrid(idx, idx, indexing="ij")).astype(np.float32)


def upsample_axial(angle_deg: np.ndarray, strength: np.ndarray,
                   coords: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Window grid -> full slice resolution, interpolating in the doubled angle.

    Orientation is an axial quantity: it must be interpolated as
    u = strength * exp(2 i theta), never as a raw angle.  Where neighbouring
    windows disagree the phasors cancel and the strength drops, which is the
    honest result -- that region comes out grey.
    """
    u = strength * np.exp(2j * np.deg2rad(angle_deg))
    re = ndimage.map_coordinates(u.real, coords, order=1, mode="nearest")
    im = ndimage.map_coordinates(u.imag, coords, order=1, mode="nearest")
    return np.degrees(np.arctan2(im, re)) / 2.0 % 180.0, np.hypot(re, im)


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------

def write_legend(path: Path) -> None:
    set_style()
    fig = plt.figure(figsize=(11.0, 4.0))
    gs = fig.add_gridspec(2, 3, width_ratios=[1.0, 1.25, 1.25], height_ratios=[1.0, 0.16],
                          wspace=0.32, hspace=0.45)

    # --- angle wheel, drawn in image coordinates (x right, y down)
    ax = fig.add_subplot(gs[:, 0])
    m = 401
    yy, xx = np.mgrid[0:m, 0:m]
    dy, dx = yy - (m - 1) / 2, xx - (m - 1) / 2
    rad = np.hypot(dy, dx) / ((m - 1) / 2)
    ang = np.degrees(np.arctan2(dy, dx)) % 180.0
    disc = to_rgb(ang, np.ones_like(ang), np.ones_like(ang)).astype(float) / 255.0
    alpha = ((rad <= 1.0) & (rad >= 0.34)).astype(float)
    ax.imshow(np.dstack([disc, alpha]), extent=[-1, 1, 1, -1])
    for deg in (0, 45, 90, 135):
        t = np.deg2rad(deg)
        ax.plot([-0.30 * np.cos(t), 0.30 * np.cos(t)], [-0.30 * np.sin(t), 0.30 * np.sin(t)],
                color="0.15", lw=1.4)
        ax.text(1.17 * np.cos(t), 1.17 * np.sin(t), f"{deg}°", ha="center", va="center",
                fontsize=9.5, color="0.15")
        ax.text(-1.17 * np.cos(t), -1.17 * np.sin(t), f"{deg}°", ha="center", va="center",
                fontsize=9.5, color="0.6")
    ax.annotate("", xy=(1.05, -1.30), xytext=(0.60, -1.30),
                arrowprops=dict(arrowstyle="->", color="0.15", lw=1.1))
    ax.text(0.55, -1.30, "x", ha="right", va="center", fontsize=9.5)
    ax.annotate("", xy=(-1.30, 1.05), xytext=(-1.30, 0.60),
                arrowprops=dict(arrowstyle="->", color="0.15", lw=1.1))
    ax.text(-1.30, 0.55, "y", ha="center", va="bottom", fontsize=9.5)
    ax.set_xlim(-1.45, 1.45)
    ax.set_ylim(1.45, -1.45)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_title("tow direction → hue\n(image axes: x right, y down)", fontsize=10)

    # --- hue bar
    axb = fig.add_subplot(gs[0, 1])
    bar_ang = np.linspace(0, 180, 721)
    strip = to_rgb(np.tile(bar_ang, (48, 1)), np.ones((48, 721)), np.ones((48, 721)))
    axb.imshow(strip, extent=[0, 180, 0, 1], aspect="auto")
    axb.set_xticks([0, 45, 90, 135, 180])
    axb.set_yticks([])
    axb.set_xlabel("in-plane texture angle (degrees)")
    axb.set_title("hue = orientation, modulo 180°\n(0° and 180° are the same colour)",
                  fontsize=10)
    axb.grid(False)

    axn = fig.add_subplot(gs[1, 1])
    axn.axis("off")
    axn.text(0.0, 0.5,
             "value = XCT grey level, so the\nmicrostructure stays visible",
             fontsize=9, va="center", color="0.25")

    # --- angle x anisotropy chart
    axs = fig.add_subplot(gs[0, 2])
    ga, gs_ = np.meshgrid(np.linspace(0, 180, 361), np.linspace(0, 1, 181))
    axs.imshow(to_rgb(ga, gs_, np.ones_like(ga)), extent=[0, 180, 0, 1], aspect="auto",
               origin="lower")
    axs.set_xticks([0, 45, 90, 135, 180])
    axs.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
    axs.set_xlabel("in-plane texture angle (degrees)")
    axs.set_ylabel("anisotropy strength")
    axs.set_title("saturation = anisotropy\n(isotropic → grey, never a strong colour)",
                  fontsize=10)
    axs.grid(False)

    axn2 = fig.add_subplot(gs[1, 2])
    axn2.axis("off")
    axn2.text(0.0, 0.5,
              "the noise floor is subtracted, so an\nunresolved window reads 0, not a\nconfident wrong colour",
              fontsize=9, va="center", color="0.25")

    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def write_orthogonal_cut(path: Path, cut_rgb: np.ndarray, ply_rgb: np.ndarray,
                         z: np.ndarray, slice_angle: np.ndarray, slice_strength: np.ndarray,
                         period: np.ndarray, power: np.ndarray, cut_x: int,
                         volume_id: str, family_label: str, steps: list[float]) -> None:
    set_style()
    fig = plt.figure(figsize=(13.0, 10.0))
    gs = fig.add_gridspec(4, 1, height_ratios=[2.6, 0.34, 1.0, 1.0], hspace=0.62)

    z0, z1 = int(z[0]), int(z[-1]) + 1
    ny = cut_rgb.shape[0]

    ax0 = fig.add_subplot(gs[0])
    ax0.imshow(cut_rgb, extent=[z0, z1, ny, 0], aspect="auto", interpolation="nearest")
    ax0.set_ylabel("y (voxels)")
    ax0.set_title(f"y-z cut through the orientation colour volume, at x = {cut_x} "
                  f"(slab mean)\n{volume_id}\n{family_label}\n"
                  f"dashed lines: orientation turns detected from the slice-level angle",
                  fontsize=10)
    ax0.grid(False)
    for k in steps:
        ax0.axvline(k, color="k", lw=0.9, ls="--", alpha=0.7)

    ax1 = fig.add_subplot(gs[1], sharex=ax0)
    ax1.imshow(ply_rgb[None, :, :], extent=[z0, z1, 0, 1], aspect="auto",
               interpolation="nearest")
    ax1.set_yticks([])
    ax1.set_ylabel("ply\nmap", rotation=0, ha="right", va="center", fontsize=9)
    ax1.grid(False)

    ax2 = fig.add_subplot(gs[2], sharex=ax0)
    colour = to_rgb(slice_angle, np.ones_like(slice_angle), np.ones_like(slice_angle)) / 255.0
    # ghost copies at +-180 deg, so a track that wraps still reads as continuous
    for off, alpha in ((-180.0, 0.25), (0.0, 1.0), (180.0, 0.25)):
        ax2.scatter(z, slice_angle + off, c=colour, s=9, alpha=alpha, zorder=3)
    for k in np.arange(z0, z1, PLY_PITCH):
        ax2.axvline(k, color="0.7", lw=0.5, alpha=0.6)
    for k in steps:
        ax2.axvline(k, color="k", lw=0.9, ls="--", alpha=0.7)
    ax2.set_ylim(-40, 220)
    ax2.set_yticks([0, 45, 90, 135, 180])
    ax2.set_ylabel("dominant angle (deg)")
    ax2b = ax2.twinx()
    ax2b.plot(z, slice_strength, color="0.25", lw=1.0)
    ax2b.set_ylabel("anisotropy", color="0.25")
    ax2b.grid(False)
    ax2.set_xlabel("z (voxels)   -- vertical lines are the 19.6-voxel ply pitch")

    ax3 = fig.add_subplot(gs[3])
    ax3.plot(period, power / max(power.max(), EPS), color="#1b6ca8", lw=1.3)
    ax3.axvline(PLY_PITCH, color="#c2571a", ls="--", lw=1.2, label=f"ply pitch {PLY_PITCH}")
    ax3.axvline(2 * PLY_PITCH, color="#2e7d32", ls=":", lw=1.2,
                label=f"2x ply pitch {2 * PLY_PITCH:.1f}")
    ax3.set_xlabel("period along z (voxels)")
    ax3.set_ylabel("normalised power")
    ax3.set_title("periodogram of the slice-level orientation phasor "
                  "u(z) = A exp(2 i theta)", fontsize=10)
    ax3.legend()

    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Depth analysis of the orientation signal
# ---------------------------------------------------------------------------

def orientation_periodogram(angle: np.ndarray, strength: np.ndarray
                            ) -> tuple[np.ndarray, np.ndarray]:
    """Periodogram of u(z) = A exp(2 i theta), the T-G orientation phasor.

    u is complex, so +f and -f are different rotation senses of the same period.
    They are summed, because a layup is periodic regardless of the turn sense.
    """
    u = strength * np.exp(2j * np.deg2rad(angle))
    u = u - u.mean()
    n = u.size
    f = np.fft.fft(u * np.hanning(n))
    k = np.arange(1, n // 2 + 1)
    power = np.abs(f[k]) ** 2 + np.abs(f[-k]) ** 2
    period = n / k
    return period[::-1], power[::-1]


def ply_step_analysis(angle: np.ndarray, strength: np.ndarray, z: np.ndarray,
                      half: int = 5, jump_deg: float = 20.0) -> dict:
    """Where the dominant orientation turns, and how that spacing sits against the pitch.

    A laminate holds one orientation through a ply and turns at the interface,
    so the spacing between turns should be the ply pitch or a small multiple of
    it.  A single-slice difference is far too noisy to find those interfaces, so
    each position is scored by the angle between the mean phasor of the ``half``
    slices before it and of the ``half`` slices after it.  Peaks of that score
    are the interfaces, with at most one per 0.6 ply pitch.
    """
    u = strength * np.exp(2j * np.deg2rad(angle))
    n = u.size
    cs = np.concatenate([[0.0 + 0j], np.cumsum(u)])
    i = np.arange(n)
    a0, b1 = np.maximum(i - half, 0), np.minimum(i + half, n)
    pre = (cs[i] - cs[a0]) / np.maximum(i - a0, 1)
    post = (cs[b1] - cs[i]) / np.maximum(b1 - i, 1)
    turn = 0.5 * np.abs(np.degrees(np.angle(post * np.conj(pre))))
    turn[:half] = 0.0
    turn[n - half:] = 0.0
    pk, _ = find_peaks(turn, height=jump_deg, distance=max(3, int(0.6 * PLY_PITCH)))
    pos = z[pk].astype(float)
    if pos.size < 3:
        return {"n_steps": int(pos.size), "step_positions": pos.tolist(), "detected": False}
    gaps = np.diff(pos)
    ratio = gaps / PLY_PITCH
    resid = np.abs(ratio - np.round(np.maximum(ratio, 1.0)))
    near = float((resid <= 0.25).mean())
    return {
        "n_steps": int(pos.size),
        "step_positions": pos.tolist(),
        "gaps_voxels": gaps.tolist(),
        "median_gap_voxels": float(np.median(gaps)),
        "gaps_in_ply_multiples": np.round(ratio, 2).tolist(),
        "fraction_within_0p25_ply_of_a_multiple": near,
        "median_turn_deg": float(np.median(turn[pk])),
        "detected": bool(near >= 0.6),
    }


def autocorr_at(angle: np.ndarray, strength: np.ndarray, lags: list[float]) -> dict:
    u = strength * np.exp(2j * np.deg2rad(angle))
    u = u - u.mean()
    denom = float(np.vdot(u, u).real)
    out = {}
    for lag in lags:
        k = int(round(lag))
        if k <= 0 or k >= u.size or denom <= 0:
            out[f"lag_{lag:g}"] = float("nan")
            continue
        out[f"lag_{lag:g}"] = float(np.vdot(u[:-k], u[k:]).real / denom)
    return out


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--volume-id", default=DEFAULT_VOLUME)
    ap.add_argument("--out-dir", default=str(OUT_ROOT))
    ap.add_argument("--window", type=int, default=64)
    ap.add_argument("--stride", type=int, default=32)
    ap.add_argument("--z-range", default=None,
                    help="z0:z1, half-open. Default: the interior slices of the volume.")
    ap.add_argument("--xy-window", type=int, default=1024,
                    help="centred in-plane crop; also bounds the TIFF size")
    ap.add_argument("--smooth", type=float, default=2.0,
                    help="Gaussian sigma on the window grid, in grid steps")
    ap.add_argument("--cut-slab", type=int, default=32,
                    help="x-thickness averaged into the orthogonal cut")
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()

    t_start = time.time()
    group = zarr.open_group(str(ZARR_ROOT), mode="r")
    if args.volume_id not in group:
        raise SystemExit(f"volume id not in {ZARR_ROOT}: {args.volume_id}")
    arr = group[args.volume_id]["xct"]
    depth, height, width = arr.shape

    n = min(args.xy_window, height, width)
    n -= (n - args.window) % args.stride           # exact number of window steps
    y0, x0 = place_crop(arr, n)
    geo = Geometry(args.window, args.stride)
    ny, nx = geo.grid_shape(n)

    if args.z_range:
        a, b = args.z_range.split(":")
        z0, z1 = int(a), int(b)
    else:
        z0, z1 = interior_z_range(arr, y0, x0, n)
    z0, z1 = max(0, z0), min(depth, z1)
    nz = z1 - z0
    if nz < 8:
        raise SystemExit(f"z range too thin: {z0}:{z1}")

    family_key, family_label = family_of(args.volume_id)
    out_dir = Path(args.out_dir) / args.volume_id
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"volume   {args.volume_id}")
    print(f"family   {family_label}")
    print(f"shape    {arr.shape}, crop {n}x{n} at y={y0} x={x0}, z {z0}:{z1} ({nz} slices)")
    print(f"windows  {args.window} voxels, stride {args.stride} -> {ny}x{nx} grid per slice")
    print(f"spectrum {geo.n_spectrum} samples in the {BAND[0]:.0f}-{BAND[1]:.0f} voxel band, "
          f"design condition {geo.condition:.2f}")

    amp_iso = isotropic_amplitude(arr, z0, z1, y0, x0, n)
    grey_lo, grey_hi = grey_limits(arr, z0, z1, y0, x0, n)

    pool = ProcessPoolExecutor(max_workers=args.workers, initializer=_init,
                               initargs=(args.volume_id, args.window, args.stride,
                                         (n, y0, x0), amp_iso))
    with pool:
        # --- round 1: isotropic flat field (the pipeline's own angular response)
        t0 = time.time()
        iso = np.mean(list(pool.map(_task_null_power, range(1000, 1000 + N_NULL))), axis=0)
        iso = np.maximum(iso.astype(np.float32), EPS)

        # --- round 2: noise floor of the same pipeline under isotropy
        null = np.stack(list(pool.map(_task_null_coefs,
                                      [(s, iso) for s in range(2000, 2000 + N_NULL)])))
        print(f"null pass {time.time() - t0:.1f} s")

        raw_floor = np.sqrt((mode_amplitudes(null) ** 2).mean(axis=(0, 1, 2)))
        smooth_null = ndimage.gaussian_filter(null, (0, args.smooth, args.smooth, 0),
                                              mode="nearest")
        floor = np.sqrt((mode_amplitudes(smooth_null) ** 2).mean(axis=(0, 1, 2)))
        slice_floor = np.sqrt((mode_amplitudes(null.sum(axis=(1, 2))) ** 2).mean(axis=0))
        print(f"noise floor a_m per window   raw {np.round(raw_floor, 3)}")
        print(f"noise floor a_m per window   smoothed sigma={args.smooth} {np.round(floor, 3)}")
        print(f"noise floor a_m per slice    {np.round(slice_floor, 4)}")

        # --- orientation pass over the volume
        t0 = time.time()
        chunk = 8
        jobs = [(z, min(z + chunk, z1), iso) for z in range(z0, z1, chunk)]
        coefs = np.zeros((nz, ny, nx, 1 + 2 * len(MODES)), np.float32)
        for zs, block in pool.map(_task_real, jobs):
            coefs[zs - z0: zs - z0 + block.shape[0]] = block
        print(f"orientation pass {time.time() - t0:.1f} s for {nz} slices")

    # --- slice-level dominant orientation (the ply map)
    slice_c = subtract_noise(coefs.sum(axis=(1, 2)), slice_floor)
    slice_angle, slice_strength = angle_strength(slice_c, geo)

    # --- local maps: smooth the coefficients, then remove the noise floor
    local_c = subtract_noise(
        ndimage.gaussian_filter(coefs, (0, args.smooth, args.smooth, 0), mode="nearest"), floor)
    local_angle, local_strength = angle_strength(local_c, geo)

    # --- colour volume, streamed to TIFF; the orthogonal cut is accumulated on the way
    cut_x = n // 2
    half = max(1, args.cut_slab // 2)
    xa, xb = max(0, cut_x - half), min(n, cut_x + half)
    cut_u = np.zeros((nz, n), np.complex128)
    cut_v = np.zeros((nz, n), np.float64)
    zs = np.arange(z0, z1)
    coords = upsample_coords(geo, n)

    def pages():
        for i, z in enumerate(zs):
            grey = np.asarray(arr[int(z), y0:y0 + n, x0:x0 + n]).astype(np.float32)
            value = np.clip((grey - grey_lo) / (grey_hi - grey_lo), 0.0, 1.0)
            ang, stg = upsample_axial(local_angle[i], local_strength[i], coords)
            cut_u[i] = (stg[:, xa:xb] * np.exp(2j * np.deg2rad(ang[:, xa:xb]))).mean(1)
            cut_v[i] = value[:, xa:xb].mean(1)
            yield to_rgb(ang, stg, value)

    t0 = time.time()
    tif_local = out_dir / "orientation_rgb.tif"
    with tifffile.TiffWriter(tif_local, imagej=True) as tw:
        tw.write(pages(), shape=(nz, n, n, 3), dtype=np.uint8, photometric="rgb",
                 metadata={"axes": "ZYXS",
                           "Info": f"{args.volume_id} | hue=orientation mod 180 deg, "
                                   f"sat=anisotropy, value=XCT grey | z0={z0} y0={y0} x0={x0}"})
    print(f"colour pass {time.time() - t0:.1f} s -> {tif_local.name} "
          f"({tif_local.stat().st_size / 1e6:.0f} MB)")

    tif_ply = out_dir / "ply_map_rgb.tif"
    ply_rgb = to_rgb(slice_angle, slice_strength, np.ones_like(slice_angle))
    with tifffile.TiffWriter(tif_ply, imagej=True) as tw:
        tw.write(np.broadcast_to(ply_rgb[:, None, None, :], (nz, PLY_TILE, PLY_TILE, 3)).copy(),
                 photometric="rgb",
                 metadata={"axes": "ZYXS",
                           "Info": f"{args.volume_id} | one flat colour per z-slice"})

    # --- verification numbers
    period, power = orientation_periodogram(slice_angle, slice_strength)
    band = (period >= 8.0) & (period <= 80.0)
    peak_period = float(period[band][np.argmax(power[band])]) if band.any() else float("nan")
    ac = autocorr_at(slice_angle, slice_strength, [PLY_PITCH, 2 * PLY_PITCH, 4 * PLY_PITCH])
    steps = ply_step_analysis(slice_angle, slice_strength, zs)
    step = np.abs(np.diff(slice_angle))
    step = np.minimum(step, 180.0 - step)

    # --- figures
    write_legend(out_dir / "legend.png")
    cut_rgb = to_rgb(np.degrees(np.angle(cut_u.T)) / 2.0 % 180.0, np.abs(cut_u.T), cut_v.T)
    write_orthogonal_cut(out_dir / "orthogonal_cut.png", cut_rgb, ply_rgb, zs,
                         slice_angle, slice_strength, period, power,
                         x0 + cut_x, args.volume_id, family_label,
                         steps.get("step_positions", []))

    results = {
        "volume_id": args.volume_id,
        "family": family_key,
        "family_label": family_label,
        "shape": [int(v) for v in arr.shape],
        "crop": {"n": int(n), "y0": int(y0), "x0": int(x0), "z0": int(z0), "z1": int(z1)},
        "settings": {"window": args.window, "stride": args.stride, "pad": PAD,
                     "band_voxels": list(BAND), "modes": list(MODES),
                     "taper_fraction": TAPER_FRAC, "smooth_grid_sigma": args.smooth,
                     "window_grid": [int(ny), int(nx)],
                     "spectrum_samples": geo.n_spectrum,
                     "design_condition": geo.condition,
                     "grey_limits": [grey_lo, grey_hi]},
        "noise_floor_a_m": {"per_window_raw": raw_floor.tolist(),
                            "per_window_smoothed": floor.tolist(),
                            "per_slice": slice_floor.tolist()},
        "slice_level": {
            "z": zs.tolist(),
            "dominant_angle_deg": slice_angle.tolist(),
            "anisotropy": slice_strength.tolist(),
            "a_m": mode_amplitudes(coefs.sum(axis=(1, 2))).tolist(),
        },
        "local_level": {
            "anisotropy_p10_p50_p90": np.percentile(local_strength, [10, 50, 90]).tolist(),
            "fraction_below_0p05": float((local_strength < 0.05).mean()),
        },
        "depth_structure": {
            "phasor_peak_period_voxels": peak_period,
            "phasor_autocorrelation": ac,
            "ply_pitch_voxels": PLY_PITCH,
            "orientation_steps": steps,
            "slice_to_slice_angle_step_deg": {
                "median": float(np.median(step)),
                "p90": float(np.percentile(step, 90)),
            },
        },
        "runtime_seconds": time.time() - t_start,
    }
    write_json(results, out_dir)
    write_readme(out_dir, args, results)
    print(f"\nwrote {out_dir}")
    print(f"peak period of the orientation phasor: {peak_period:.1f} voxels "
          f"(ply pitch {PLY_PITCH}, 2x = {2 * PLY_PITCH:.1f})")
    print(f"phasor autocorrelation {ac}")
    if steps.get("gaps_voxels"):
        print(f"orientation turns at z = {np.round(steps['step_positions'], 0).tolist()}")
        print(f"gaps between turns (voxels) {np.round(steps['gaps_voxels'], 1).tolist()}"
              f" -> {np.round(steps['gaps_in_ply_multiples'], 2).tolist()} ply pitches; "
              f"{steps['fraction_within_0p25_ply_of_a_multiple'] * 100:.0f} % land on a "
              f"multiple, median turn {steps['median_turn_deg']:.0f} deg")
    print(f"local anisotropy p10/p50/p90 "
          f"{np.round(results['local_level']['anisotropy_p10_p50_p90'], 3)}")
    print(f"total {time.time() - t_start:.1f} s")


def write_readme(out_dir: Path, args, r: dict) -> None:
    nf = r["noise_floor_a_m"]
    ds = r["depth_structure"]
    st = ds["orientation_steps"]
    lv = r["local_level"]
    text = f"""# Orientation colour volume -- {r['volume_id']}

Family: **{r['family_label']}**
Script: `scripts/analysis/viz_orientation_volume.py`
Volume: `data/split_v2/volumes.zarr`, shape {r['shape']} (z, y, x).
Crop: {r['crop']['n']}x{r['crop']['n']} in-plane at y={r['crop']['y0']}, x={r['crop']['x0']} (placed on the most solid material); z {r['crop']['z0']}:{r['crop']['z1']}.
All lengths are in voxels -- the dataset records no voxel size.

## What the colours mean

| channel | quantity |
|---|---|
| hue | local in-plane texture (tow) angle, **modulo 180 degrees**: 0 and 180 get the same colour |
| saturation | anisotropy strength in [0, 1]; an isotropic window is grey, never a strong colour |
| value | the original XCT grey level, stretched between the 1st and 99th percentile |

Angles are real-space texture angles in image coordinates: **x to the right is
0 degrees, y downwards is 90 degrees**, the same convention as T-G and T-H.
`legend.png` shows the wheel, the hue bar and the saturation ramp.

## Files

| file | what it is |
|---|---|
| `orientation_rgb.tif` | RGB stack, one page per z-slice, local orientation colour over the XCT |
| `ply_map_rgb.tif` | RGB stack, each page one flat colour: that slice's dominant orientation |
| `legend.png` | colour key |
| `orthogonal_cut.png` | y-z cut through the colour volume, ply-map strip, angle profile, periodogram |
| `results.json` | every number below, per slice |

Both TIFFs are plain ImageJ RGB stacks. Open `orientation_rgb.tif` in Fiji and
use `Image > Stacks > Reslice` to get the same y-z view as the figure.

## Method, in short

Each window is plane-detrended, tapered with a **radially symmetric** Tukey
window and Fourier transformed with {int(r['settings']['pad'] / args.window)}x zero
padding. A separable Hann window is not usable here: its leakage lies on the kx
and ky axes and fakes a strong 0/90-degree texture. The power in the
{BAND[0]:.0f}-{BAND[1]:.0f}-voxel band is kept, the kx = 0 and ky = 0 lines are dropped
(the discrete FFT collapses each into a single angular direction and spikes at
exactly 0 and 90 degrees), and the rest is divided by an isotropic flat field
measured by Monte Carlo. The angular density h(theta) is then fitted by least
squares with the modes m = {', '.join(str(m) for m in MODES)}.

**The dominant angle is the argmax of h(theta), not the doubled-angle moment.**
Two tow directions 90 degrees apart cancel in the 2-theta moment; that is how
T-G wrongly concluded that the Nacho family is unidirectional. They do not
cancel in the density.

## Noise floor -- read this before trusting a local colour

A single {args.window}-voxel window holds only about 30 independent Fourier samples in
the {BAND[0]:.0f}-{BAND[1]:.0f}-voxel band. Measured on synthetic isotropic slices built
from this volume's own radial spectrum, the pipeline returns

* a_m = {np.round(nf['per_window_raw'], 3).tolist()} for a **single raw window**,
* a_m = {np.round(nf['per_window_smoothed'], 3).tolist()} after smoothing the coefficient map
  by sigma = {args.smooth} window steps ({args.smooth * args.stride:.0f} voxels),
* a_m = {np.round(nf['per_slice'], 4).tolist()} for a **whole slice**.

The real slice-level a2 is around {np.round(np.median([a[0] for a in r['slice_level']['a_m']]), 3)}.
So a single raw window carries essentially no orientation information, a
smoothed window carries some, and a whole slice is well measured. The map is
therefore smoothed before the angle is read, and the residual noise floor is
subtracted from every mode amplitude in quadrature. The effective in-plane
resolution of `orientation_rgb.tif` is about {args.window + 4 * args.smooth * args.stride:.0f} voxels, not {args.window}.

Local anisotropy after that correction: p10/p50/p90 = {np.round(lv['anisotropy_p10_p50_p90'], 3).tolist()}; {lv['fraction_below_0p05'] * 100:.0f} % of
the volume is below 0.05 and is drawn grey.

## Depth structure -- is the ply stack visible?

The dominant orientation is flat inside a ply and turns at the interface, so
the direct test is **where it turns**. This run finds {st['n_steps']} turns bigger than
20 degrees, at z = {np.round(st.get('step_positions', []), 0).tolist()}.
The gaps between them are {np.round(st.get('gaps_voxels', []), 1).tolist()} voxels,
that is {np.round(st.get('gaps_in_ply_multiples', []), 2).tolist()} ply pitches.
**{st.get('fraction_within_0p25_ply_of_a_multiple', float('nan')) * 100:.0f} %** of the gaps land within a quarter of a ply of an integer
multiple of {PLY_PITCH} voxels. Median turn: {st.get('median_turn_deg', float('nan')):.0f} degrees.

Supporting numbers: phasor autocorrelation at multiples of the ply pitch,
{ {k: round(v, 3) for k, v in ds['phasor_autocorrelation'].items()} }; periodogram peak in the
8-80-voxel band, **{ds['phasor_peak_period_voxels']:.1f} voxels**; median slice-to-slice change of
the dominant angle, {ds['slice_to_slice_angle_step_deg']['median']:.1f} degrees.

A [0/90]n stack turns by 90 degrees at every interface, so its phasor
anti-correlates at one ply pitch and repeats at two. Read `orthogonal_cut.png`
together with these numbers, and treat a periodogram peak far from the pitch as
what it is: the orientation pattern repeating over several plies, not the plies
themselves.

## Caveats

* These specimens are open-hole coupons. The crop is placed automatically on the
  most solid material, but a 1024-voxel window cannot always miss every hole. A
  hole appears black (value 0) with a coloured rim, and the rim orientation is
  real but is not tow structure.
* The angular density is fitted with modes up to m = {max(MODES)}, so features narrower
  than about {180 / (2 * max(MODES)):.0f} degrees are not resolved; two lobes closer than that merge into one.
* Where two tow directions are equally strong the argmax is genuinely ambiguous
  and can jump between them from slice to slice. The phasor also cancels there,
  so those slices show low anisotropy -- check the anisotropy trace before
  reading a turn as a ply interface.
* No voxel size is recorded in the dataset, so every length here is in voxels.
"""
    (out_dir / "README.md").write_text(text)


if __name__ == "__main__":
    main()
