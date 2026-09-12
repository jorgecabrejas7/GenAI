"""Score the chunk-band trial arms against the success criterion. (CPU)

The criterion, as set by the supervisor:

* the **-8 slab phi within +/-20 % of the volume mean** — and, since the
  depletion sits on BOTH single-covered strips, the +0 slab too: a fix that
  clears the trailing strip and leaves the leading one at 0.60 has not cleared
  the band,
* the **grey chunk seam still at the real floor** (0.906 +/- 0.016 at 1024 on
  real material — a ratio near 1 is the target, and the floor says real
  laminate structure sits just below it),
* **delivered phi within the gate** (0.005 of the request).

Each arm is reported beside the campaign-12 hybrid row it is meant to replace,
because "better" here only means "better than what we have".

Usage:
    python scripts/analysis/chunk_band_trial_report.py \
        --trial runs/campaigns/17-chunk-band-trial \
        --baseline runs/campaigns/12-eval-v4
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from poregen.eval_v4 import metrics as M

REPO = Path(__file__).resolve().parents[2]

TILE = 64
LABEL_PORE, LABEL_AIR = 1, 2
SLAB = 8
#: The real-volume floor for the grey chunk seam at 1024, from campaign 12's
#: real_floor block.  Generated volumes are read against this, not against 1.
GREY_CHUNK_FLOOR = 0.906
#: eval-v4's porosity gate.
POROSITY_GATE = 0.005
#: The supervisor's band criterion.
BAND_TOL = 0.20


#: ``phi_of``, ``chunk_bounds`` and the band reading itself now live in
#: ``poregen.eval_v4.metrics``: assessment 12 (stress_geometry) reads the same
#: band, and two copies of a measurement are two measurements.
def band_slabs(label: np.ndarray, axes, chunk_vox: int, overlap_vox: int) -> dict:
    """The trial report's view of :func:`metrics.chunk_band_profile`.

    ``axes`` is accepted and ignored — the metric drops an axis that holds one
    chunk by itself, which is what the caller computed ``axes`` for.
    """
    b = M.chunk_band_profile(label, chunk_vox, overlap_vox=overlap_vox, slab=SLAB)
    return {"-8": b["phi_-8"], "0": b["phi_+0"],
            "-8_worst": b["phi_-8_worst"], "0_worst": b["phi_+0_worst"],
            "-8_terminal": b["phi_-8_terminal"], "0_terminal": b["phi_+0_terminal"],
            "n_trailing": b["n_trailing_planes"], "n_leading": b["n_leading_planes"],
            "per_plane": b["per_plane"]}


def phi_of(block: np.ndarray) -> float | None:
    pore = int((block == 1).sum())
    solid = int((block != 2).sum())
    return pore / solid if solid else None



def seams(xct: np.ndarray, pore_logit: np.ndarray | None, period: int) -> dict:
    """Grey and pore-logit discontinuity at the chunk planes, via the sampler's
    own estimator, judged against the interior baseline with the WINDOW period
    excluded so both are read against the same material."""
    from poregen.diffusion.sampler import seam_discontinuity

    out = {}
    g = seam_discontinuity(xct.astype(np.float32) / 255.0, period,
                           prefix="grey", interior_exclude=TILE)
    out["grey_chunk_ratio"] = g.get("grey_ratio")
    # NOTE: seam_discontinuity takes a PERIOD, so for an overlap arm (whose
    # boundaries are not evenly spaced) it scores the unoverlapped grid. The
    # phi slabs above are the ones that follow the real boundaries.
    if pore_logit is not None:
        p = seam_discontinuity(pore_logit, period, prefix="pore",
                               interior_exclude=TILE)
        out["pore_chunk_ratio"] = p.get("pore_ratio")
    return out


def measure(case_dir: Path) -> dict:
    import tifffile

    m = json.loads((case_dir / "manifest.json").read_text())
    label = tifffile.imread(case_dir / "label.tif")
    notes = m.get("notes") or {}
    tiles = m.get("chunk_tiles") or [3, 3, 3]
    period = int(tiles[0]) * TILE
    overlap = int(notes.get("chunk_overlap", 0) or 0)
    # z carries no chunk plane at 1024 (192 deep = one chunk), so the axes with
    # planes are read per volume rather than assumed.
    axes = [a for a in range(3) if label.shape[a] > period]

    xct = tifffile.imread(case_dir / "volume.tif")
    pl = None
    p = case_dir / "probs.npz"
    if p.exists():
        pl = np.load(p)["pore_logit"]

    phi_vol = phi_of(label)
    band = band_slabs(label, axes, period, overlap)
    row = {
        "case": case_dir.name,
        "shape": list(label.shape),
        "s_nb": m.get("s_nb"),
        "chunk_overlap": notes.get("chunk_overlap", 0),
        "write": notes.get("chunk_overlap_write", "blend"),
        "pinned": notes.get("chunk_overlap_pinned"),
        "checkpoint_step": m.get("checkpoint_step"),
        "requested_phi": m.get("requested_global_phi"),
        "phi_volume": phi_vol,
        "phi_-8": band["-8"],
        "phi_+0": band["0"],
        "phi_-8_worst": band["-8_worst"],
        "phi_+0_worst": band["0_worst"],
        "phi_-8_terminal": band["-8_terminal"],
        "phi_+0_terminal": band["0_terminal"],
        "n_trailing_planes": band["n_trailing"],
        "n_leading_planes": band["n_leading"],
        "per_plane": band["per_plane"],
        "wall_time_s": m.get("wall_time_s"),
        **seams(xct, pl, period),
    }
    for k in ("-8", "+0", "-8_worst", "+0_worst", "-8_terminal", "+0_terminal"):
        v = row[f"phi_{k}"]
        row[f"ratio_{k}"] = (v / phi_vol) if v is not None and phi_vol else None
    row["phi_error"] = (abs(phi_vol - row["requested_phi"])
                        if phi_vol is not None and row["requested_phi"] else None)
    return row


def verdict(rows: list[dict]) -> dict:
    """Pass only if EVERY criterion holds, on both strips."""
    def mean(k):
        v = [r[k] for r in rows if r.get(k) is not None]
        return float(np.mean(v)) if v else None

    def n_with(k):
        return sum(1 for r in rows if r.get(k) is not None)

    r8, r0 = mean("ratio_-8"), mean("ratio_+0")      # NON-TERMINAL planes
    grey, phi_err = mean("grey_chunk_ratio"), mean("phi_error")
    band_ok = (r8 is not None and r0 is not None
               and abs(r8 - 1) <= BAND_TOL and abs(r0 - 1) <= BAND_TOL)
    grey_ok = grey is not None and grey >= GREY_CHUNK_FLOOR - 0.05
    phi_ok = phi_err is not None and phi_err <= POROSITY_GATE
    return {"ratio_-8": r8, "ratio_+0": r0,
            "worst_-8": mean("ratio_-8_worst"), "worst_+0": mean("ratio_+0_worst"),
            "terminal_-8": mean("ratio_-8_terminal"),
            "terminal_+0": mean("ratio_+0_terminal"),
            "grey_chunk_ratio": grey,
            "pore_chunk_ratio": mean("pore_chunk_ratio"),
            "phi_volume": mean("phi_volume"), "phi_error": phi_err,
            # A volume with ONE chunk end per axis has no non-terminal plane at
            # all: 384 without overlap is such a case. Those cases contribute
            # nothing to `ratio_-8`, so the case count beside it would otherwise
            # claim more evidence than there is.
            "n_cases_with_nonterminal": n_with("ratio_-8"),
            "wall_time_s": mean("wall_time_s"),
            "band_ok": band_ok, "grey_ok": grey_ok, "phi_ok": phi_ok,
            "PASS": bool(band_ok and grey_ok and phi_ok)}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--trial", type=Path, required=True)
    ap.add_argument("--baseline", type=Path, default=None)
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    groups: dict[str, list[dict]] = {}
    for arm_dir in sorted(args.trial.glob("arm_*")):
        vols = arm_dir / "volumes"
        if not vols.is_dir():
            continue
        rows = [measure(c) for c in sorted(vols.iterdir())
                if (c / "manifest.json").exists()]
        if rows:
            groups[arm_dir.name] = rows
    if args.baseline:
        base = args.baseline / "assembly_modes" / "volumes"
        rows = [measure(c) for c in sorted(base.glob("hybrid_*"))
                if (c / "manifest.json").exists()]
        if rows:
            groups["baseline hybrid (campaign 12)"] = rows

    out = {name: {"per_case": rows, "summary": verdict(rows)}
           for name, rows in groups.items()}
    dest = args.out or (args.trial / "trial_report.json")
    dest.write_text(json.dumps(out, indent=2) + "\n")

    print(f"\n{'arm':<24}{'n':>3}{'phi':>8}{'err':>8}{'-8':>8}{'worst':>8}"
          f"{'+0':>8}{'term-8':>8}{'grey':>7}{'pore':>7}{'min':>7}  verdict")
    for name, blk in out.items():
        s = blk["summary"]
        def f(k, w, p=3):
            v = s.get(k)
            return f"{v:>{w}.{p}f}" if v is not None else f"{'-':>{w}}"
        flags = ("band " if not s["band_ok"] else "") + \
                ("grey " if not s["grey_ok"] else "") + \
                ("phi" if not s["phi_ok"] else "")
        print(f"{name[:24]:<24}{s['n_cases_with_nonterminal']:>3}{f('phi_volume', 8, 4)}"
              f"{f('phi_error', 8, 4)}{f('ratio_-8', 8)}{f('worst_-8', 8)}"
              f"{f('ratio_+0', 8)}{f('terminal_-8', 8)}"
              f"{f('grey_chunk_ratio', 7)}{f('pore_chunk_ratio', 7)}"
              f"{(s['wall_time_s'] or 0) / 60:>7.1f}  "
              f"{'PASS' if s['PASS'] else 'fail: ' + flags.strip()}")
    print(f"\ncriterion: -8 AND +0 within +/-{BAND_TOL:.0%} of the volume mean, over "
          f"NON-TERMINAL planes; grey chunk seam >= {GREY_CHUNK_FLOOR - 0.05:.3f} "
          f"(real floor {GREY_CHUNK_FLOOR}); |phi - request| <= {POROSITY_GATE}.")
    print("n = cases that HAVE a non-terminal plane: an axis with one chunk end "
          "has none,\nso 384 without overlap contributes nothing to the -8/+0 "
          "columns.")
    print("'worst' is the single worst non-terminal plane. 'term-8' is the LAST "
          "plane on each axis,\nwhose successor's far face is the volume edge "
          "(OOB) — the trained state, and so the in-distribution anchor.")
    print(f"Wrote {dest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
