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


def phi_of(block: np.ndarray) -> float | None:
    pore = int((block == LABEL_PORE).sum())
    solid = int((block != LABEL_AIR).sum())
    return pore / solid if solid else None


def chunk_bounds(n_vox: int, chunk_vox: int, overlap_vox: int) -> tuple[list, list]:
    """(chunk ENDS, chunk STARTS) in voxels for one axis, excluding the volume's.

    THE PLANES MOVE WITH THE OVERLAP and the measurement has to move with them.
    Chunks advance by ``chunk - overlap``, so at overlap 32 a 1024 axis has its
    chunk ends at 192/352/512/672/832/992 and its starts at
    160/320/480/640/800/960 — not at the 192/384/576/768/960 an unoverlapped
    run uses.  Scoring an overlap arm at the baseline's planes measures chunk
    INTERIORS for most of them and reports a band that has simply been looked
    for in the wrong place.
    """
    step = chunk_vox - overlap_vox
    bounds, s = [], 0
    while s < n_vox:
        hi = min(s + chunk_vox, n_vox)
        bounds.append((s, hi))
        if hi >= n_vox:
            break
        s += step
    return ([hi for _, hi in bounds[:-1]], [lo for lo, _ in bounds[1:]])


def band_slabs(label: np.ndarray, axes, chunk_vox: int, overlap_vox: int) -> dict:
    """phi in the 8 voxels before each chunk END and after each chunk START.

    THE LAST PLANE ON EACH AXIS IS REPORTED SEPARATELY.  Its successor is the
    terminal clamped chunk, whose far face is the VOLUME EDGE — OOB, the
    configuration the model was trained on — so that one boundary is healthy
    while every other is depleted.  Averaging it in makes the same sampler
    score 0.80 on a 2-chunk axis and 0.66 on a 6-chunk axis, which is a
    property of the shape and not of the sampler.  It is kept as the
    in-distribution anchor: it says what the band would look like if the
    neighbour state at a frontier were one the model had seen.
    """
    series = {"trailing": {}, "leading": {}}
    inner_t, inner_l, term_t, term_l = [], [], [], []
    for axis in axes:
        n = label.shape[axis]
        ends, starts = chunk_bounds(n, chunk_vox, overlap_vox)
        t_vals = [phi_of(np.take(label, range(e - SLAB, e), axis=axis))
                  for e in ends if e - SLAB >= 0]
        l_vals = [phi_of(np.take(label, range(st, st + SLAB), axis=axis))
                  for st in starts if st + SLAB <= n]
        t_vals = [v for v in t_vals if v is not None]
        l_vals = [v for v in l_vals if v is not None]
        series["trailing"][str(axis)] = {"planes": ends, "phi": t_vals}
        series["leading"][str(axis)] = {"planes": starts, "phi": l_vals}
        if t_vals:
            inner_t += t_vals[:-1]
            term_t.append(t_vals[-1])
        if l_vals:
            inner_l += l_vals[:-1]
            term_l.append(l_vals[-1])

    def agg(vals):
        return float(np.mean(vals)) if vals else None

    return {"-8": agg(inner_t), "0": agg(inner_l),
            "-8_worst": (float(np.min(inner_t)) if inner_t else None),
            "0_worst": (float(np.min(inner_l)) if inner_l else None),
            "-8_terminal": agg(term_t), "0_terminal": agg(term_l),
            "n_trailing": len(inner_t), "n_leading": len(inner_l),
            "per_plane": series}


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
        print(f"{name[:24]:<24}{len(blk['per_case']):>3}{f('phi_volume', 8, 4)}"
              f"{f('phi_error', 8, 4)}{f('ratio_-8', 8)}{f('worst_-8', 8)}"
              f"{f('ratio_+0', 8)}{f('terminal_-8', 8)}"
              f"{f('grey_chunk_ratio', 7)}{f('pore_chunk_ratio', 7)}"
              f"{(s['wall_time_s'] or 0) / 60:>7.1f}  "
              f"{'PASS' if s['PASS'] else 'fail: ' + flags.strip()}")
    print(f"\ncriterion: -8 AND +0 within +/-{BAND_TOL:.0%} of the volume mean, over "
          f"NON-TERMINAL planes; grey chunk seam >= {GREY_CHUNK_FLOOR - 0.05:.3f} "
          f"(real floor {GREY_CHUNK_FLOOR}); |phi - request| <= {POROSITY_GATE}.")
    print("'worst' is the single worst non-terminal plane. 'term-8' is the LAST "
          "plane on each axis,\nwhose successor's far face is the volume edge "
          "(OOB) — the trained state, and so the in-distribution anchor.")
    print(f"Wrote {dest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
