"""Ply pitch measured FROM a volume, not told to it. (CPU, read-only.)

WHY. The T-I layup readers in `scripts/analysis/layup_roundtrip.py` are GIVEN
the requested pitch — `ply_edges(depth, pitch)` imposes the ply segmentation
before a single angle is estimated. So they cannot say whether the model built
the pitch it was asked for, and a poor class accuracy at a thin pitch cannot be
attributed between the model and the reader's own resolution. This estimator
finds the boundaries instead.

HOW. Per z-slice, the T-I angular map over a centred 1024-window, summed over a
wide wavelength band, gives a 180-bin orientation signature. Adjacent slices
inside one ply look alike; across a ply boundary they do not. The
slice-to-slice cosine dissimilarity therefore spikes at every boundary, and the
first autocorrelation peak of that spike train is the pitch.

The band is FIXED and wide (6 to 64 voxels) rather than scaled by the pitch as
`measure_volume` does, because scaling it by the pitch would require knowing
the answer first.

VALIDATION, in two parts, both reported in the campaign 20 README.

1. Synthetic plies at a known pitch — oriented gratings, one angle per block.
   Recovered/true = **1.00 at every pitch tested (8, 10, 16, 20, 32)**, with
   autocorrelation peak r between 0.80 and 0.96. The method is unbiased across
   the whole range of interest, thin plies included.

2. Real sequence-A material, ALL 10 large crops from the split_v3 test panels,
   known nominal pitch 20.32 voxels (0.508 mm) and a campaign-08 working value
   of 19.6. Recovered **median 18.0, sd 16.0**.

   That sd is the finding. The first six crops agree closely (17-19) and the
   remaining four return 56, 37, 6 and 18 — so a six-crop sample looked like
   "18.0 +/- 0.6" and the full ten are not that at all. The median survives;
   the per-volume estimate does not.

THE LIMITATION, WHICH IS THE POINT AND WHICH THIS METHOD DOES NOT CLEAR. On
real laminate the autocorrelation peak is r ~= 0.2-0.47, against 0.80-0.96 on
clean synthetic plies: real ply boundaries are far less distinct than a grating
change, and at that signal level a single volume's pitch is not reportable.
`R_SINGLE_VOLUME` guards against quoting one, and every real crop fails it.

Generated volumes measure the SAME, matched for depth and window: r ~= 0.20 and
a dissimilarity variation of 0.338 against real material's 0.307. **So the
thin-ply question is still open.** The generated material is not worse than
real by this measure — the estimator is simply not sharp enough on either to
attribute a pitch per volume, and pooling three seeds is nowhere near the ten
that gave real material a usable median. Separating the reader's floor from the
model would need either many more seeds per condition or a boundary detector
that does not rely on this contrast.

Usage:
    python scripts/analysis/ply_pitch_estimator.py --volume path/to/volume.tif
    python scripts/analysis/ply_pitch_estimator.py --glob 'runs/campaigns/18-*/layup/volumes/A_seed*'
"""

from __future__ import annotations

import argparse
import glob as globmod
import sys
from pathlib import Path

import numpy as np
import tifffile

REPO = Path(__file__).resolve().parents[2]
if str(REPO / "scripts" / "analysis") not in sys.path:
    sys.path.insert(0, str(REPO / "scripts" / "analysis"))
import t_i_layup_validation as ti  # noqa: E402

#: Wavelength band, voxels. Wide and FIXED: scaling it by the pitch, as the
#: angle readers do, would need the answer in advance.
BAND_VOX = (6.0, 64.0)
#: Pitches considered, voxels. 4 is below anything physical; 60 is above the
#: thickest request in the project (32).
LAG_RANGE = range(4, 60)
#: Below this autocorrelation peak the pitch from a SINGLE volume is not
#: reportable — real laminate sits at about 0.23, so single-volume estimates
#: must be pooled. See the module docstring.
R_SINGLE_VOLUME = 0.5


def signatures(xct: np.ndarray, band=BAND_VOX) -> np.ndarray:
    """(D, 180) orientation signature per z-slice, unit-mean normalised."""
    n = ti.WINDOW
    d, h, w = xct.shape
    if min(h, w) < n:
        raise ValueError(
            f"the T-I window is {n}x{n} and this volume is {h}x{w} in plane.")
    win, sel, flat, counts = ti._geometry(n)
    y0, x0 = max(0, h // 2 - n // 2), max(0, w // 2 - n // 2)
    s = np.asarray([
        ti.band_hist(ti.angular_map(
            xct[z, y0:y0 + n, x0:x0 + n].astype(np.float32),
            win, sel, flat, counts)[None], band[0], band[1])[0]
        for z in range(d)], np.float64)
    return s / np.maximum(s.mean(axis=1, keepdims=True), 1e-30)


def boundary_signal(sig: np.ndarray) -> np.ndarray:
    """Slice-to-slice cosine dissimilarity: one spike per ply boundary."""
    a = sig[:-1] / np.linalg.norm(sig[:-1], axis=1, keepdims=True)
    b = sig[1:] / np.linalg.norm(sig[1:], axis=1, keepdims=True)
    return 1.0 - (a * b).sum(axis=1)


def estimate_pitch(diss: np.ndarray, lags=LAG_RANGE) -> tuple[int, float]:
    """(pitch, peak r) from the autocorrelation of the boundary spike train."""
    v = diss - diss.mean()
    ac = np.correlate(v, v, mode="full")[len(v) - 1:]
    if ac[0] <= 0:
        return 0, 0.0
    ac = ac / ac[0]
    usable = [lag for lag in lags if lag < len(ac)]
    if not usable:
        return 0, 0.0
    lag = max(usable, key=lambda k: ac[k])
    return int(lag), float(ac[lag])


def boundaries(diss: np.ndarray, pitch: int) -> list[int]:
    """Boundary slice indices: the largest dissimilarity in each pitch window.

    Reported for inspection. The pitch itself comes from the autocorrelation,
    which uses every plane at once and is far steadier than picking peaks.
    """
    if pitch <= 0:
        return []
    out = []
    for start in range(0, len(diss), pitch):
        seg = diss[start:start + pitch]
        if seg.size:
            out.append(int(start + int(np.argmax(seg)) + 1))
    return out


def measure(path: Path, band=BAND_VOX) -> dict:
    xct = tifffile.imread(path)
    try:
        diss = boundary_signal(signatures(xct, band))
    finally:
        del xct
    pitch, r = estimate_pitch(diss)
    return {
        "path": str(path),
        "pitch_vox": pitch,
        "autocorr_r": r,
        "single_volume_reportable": bool(r >= R_SINGLE_VOLUME),
        "dissimilarity_mean": float(diss.mean()),
        "dissimilarity_sd": float(diss.std()),
        "n_planes": int(diss.size),
        "boundaries": boundaries(diss, pitch),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--volume", type=Path, help="a single volume.tif")
    ap.add_argument("--glob", help="glob of case directories or volume.tif files")
    ap.add_argument("--expected", type=float, default=None,
                    help="the requested pitch, printed beside the recovered one")
    args = ap.parse_args()

    paths: list[Path] = []
    if args.volume:
        paths.append(args.volume)
    if args.glob:
        for m in sorted(globmod.glob(args.glob)):
            p = Path(m)
            paths.append(p if p.is_file() else p / "volume.tif")
    paths = [p for p in paths if p.exists()]
    if not paths:
        raise SystemExit("no volumes matched")

    rows = []
    hdr = f"{'volume':44s} {'pitch':>6} {'r':>7}"
    print(hdr + (f" {'expected':>9} {'ratio':>7}" if args.expected else ""))
    for p in paths:
        m = measure(p)
        rows.append(m)
        line = f"{p.parent.name[-44:]:44s} {m['pitch_vox']:6d} {m['autocorr_r']:7.3f}"
        if args.expected:
            line += f" {args.expected:9.2f} {m['pitch_vox']/args.expected:7.2f}"
        print(line)

    pit = [r["pitch_vox"] for r in rows]
    print(f"\nmedian pitch {np.median(pit):.1f}  sd {np.std(pit):.1f}  n={len(pit)}")
    weak = sum(1 for r in rows if not r["single_volume_reportable"])
    if weak:
        print(f"{weak} of {len(rows)} volumes are below r={R_SINGLE_VOLUME}: at that "
              f"signal level a SINGLE volume's pitch is not reportable and only the "
              f"pooled median is. Real laminate sits at about r=0.23.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
