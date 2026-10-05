#!/usr/bin/env python
"""The split_v3 gate values the split_v4 rebuild is read against. (CPU, seconds.)

GENERATED. The rebuild has to be judged the same way split_v3 was, and "the
same way" has to mean the same NUMBERS, not a memory of them. This reads the
reference values out of the split_v3 runs themselves — the rung report, the
campaign-28 family table, the LDM's own convergence_check.jsonl — and writes
one table the v4 runs can be laid against.

Nothing here is a threshold the author set. These are what split_v3 ACHIEVED.
A v4 run that misses one is not necessarily wrong: the labels changed, and the
whole point of the rebuild is that the porosity should be better. The table
exists so a difference is seen and explained rather than missed.

Usage
-----
    python scripts/analysis/split_v3_gate_reference.py
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from poregen.paths import repo_root

REPO = repo_root()


def _load(p: Path):
    try:
        return json.loads(p.read_text())
    except (OSError, json.JSONDecodeError):
        return None


#: The split_v3 VAE every number in the VAE section must come from.
REF_VAE = "r08-run-0004"


#: The six split_v3 rungs: (label, campaign-09 key, run that produced it).
RUNGS = (
    ("rf-2, z=32", "r08_reduction-factor-2", "r08-run-0010"),
    ("rf-4, z=16", "r08_reduction-factor-4", "r08-run-0006"),
    ("rf-8, z=8", "r08_reduction-factor-8", "r08-run-0004"),
    ("rf-16, z=4 (r08/base)", "r08_base", "r08-run-0003"),
    ("rf-32, z=2", "r08_reduction-factor-32", "r08-run-0005"),
    ("rf-64, z=1", "r08_reduction-factor-64", "r08-run-0012"),
)


def _ref_only(d, what: str, W) -> dict | None:
    """``d`` if it is REF_VAE's, else None after saying whose it is.

    campaign 09's directories are keyed by experiment, not run, so a rung
    report or probe of another run (the split_v4 rf-8, 2026-10-02) lands on the
    same path. Its numbers must never be printed as split_v3's.
    """
    if d is None:
        return None
    run = Path(str(d.get("run_dir", ""))).name
    if run.startswith(REF_VAE):
        return d
    W(f"**The {what} on disk is for `{run or 'an unnamed run'}`, not "
      f"`{REF_VAE}`.** Its numbers are not split_v3's and are not shown.\n")
    return None


def _dig(d, *path, default=None):
    for k in path:
        if not isinstance(d, dict) or k not in d:
            return default
        d = d[k]
    return d


def _splits(split: str) -> dict[str, set[str]] | None:
    """{split name: {volume ids}} for a build, or None when it is not on disk."""
    f = REPO / "data" / split / "splits.json"
    d = _load(f)
    vols = _dig(d or {}, "volumes", default=None)
    if not isinstance(vols, dict):
        return None
    out: dict[str, set[str]] = {}
    for vol, where in vols.items():
        out.setdefault(where, set()).add(vol)
    return out


def _short(v: str) -> str:
    """The readable tail of a volume id."""
    tail = v
    for marker in ("probetas_", "Probetas_"):
        if marker in v:
            tail = v.rsplit(marker, 1)[-1]
            break
    tail = tail.replace("_volume_eq_rotated_aligned", "").replace("_volume_eq_aligned", "")
    # The Juan_Ignacio panels are numbered 4-12 with no prefix of their own, so
    # stripping the path leaves a bare "12" that reads as a Nacho specimen.
    return f"JI_{tail}" if "Juan_Ignacio" in v else tail
    


def _provenance(W, compare_to: str = "split_v4") -> None:
    """WHICH VOLUMES the gates above were measured on, and whether the rebuild
    changed them.

    A gate is a number on a dataset. If the rebuild moves a panel between val
    and test, the rows above stop being a like-for-like comparison and become
    two numbers on two different sets — and nothing in a table of numbers says
    so. This is the line that says it.
    """
    base = _splits("split_v3")
    W("\n## Which volumes these gates were measured on\n")
    if base is None:
        W("`data/split_v3/splits.json` is not readable, so the provenance of the")
        W("rows above cannot be stated. Treat every comparison as unverified.\n")
        return
    W(f"split_v3: {len(base.get('train', ()))} train, {len(base.get('val', ()))} val, "
      f"{len(base.get('test', ()))} test volumes.\n")
    W(f"- **val** — {', '.join(sorted(_short(v) for v in base.get('val', ())))}")
    W(f"- **test** — {', '.join(sorted(_short(v) for v in base.get('test', ())))}\n")

    other = _splits(compare_to)
    if other is None:
        W(f"`data/{compare_to}/splits.json` does not exist yet, so the comparison")
        W("cannot be made. **The r08 rows above are like-for-like only if")
        W(f"{compare_to} keeps the same val and test panels.** The stated plan is")
        W("that it does — split_v3's assignment is kept and only JI_11 is added, to")
        W("train — and this line will confirm or contradict it the moment the build")
        W("exists.\n")
        return
    same_val = base.get("val", set()) == other.get("val", set())
    same_test = base.get("test", set()) == other.get("test", set())
    added = set().union(*other.values()) - set().union(*base.values())
    removed = set().union(*base.values()) - set().union(*other.values())
    if same_val and same_test:
        W(f"**{compare_to} keeps the same val and test panels**, so the r08 rows above")
        W("are a like-for-like comparison.")
    else:
        W(f"**{compare_to} CHANGED the evaluation panels. The r08 rows above are NOT")
        W("a like-for-like comparison** and must not be read as one.")
        if not same_val:
            W(f"  - val gained {sorted(_short(v) for v in other.get('val', set()) - base.get('val', set()))}, "
              f"lost {sorted(_short(v) for v in base.get('val', set()) - other.get('val', set()))}")
        if not same_test:
            W(f"  - test gained {sorted(_short(v) for v in other.get('test', set()) - base.get('test', set()))}, "
              f"lost {sorted(_short(v) for v in base.get('test', set()) - other.get('test', set()))}")
    if added:
        where = {v: w for w, vs in other.items() for v in vs}
        W(f"\nAdded in {compare_to}: "
          + ", ".join(f"{_short(v)} ({where[v]})" for v in sorted(added)))
    if removed:
        W(f"\nRemoved: {', '.join(sorted(_short(v) for v in removed))}")
    W("")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, default=REPO / "docs" / "SPLIT_V3_GATES.md")
    a = ap.parse_args()

    L: list[str] = []
    W = L.append
    W("# split_v3 gate values — what the rebuild is read against\n")
    W("**Generated by `scripts/analysis/split_v3_gate_reference.py`.** Regenerate it;")
    W("do not edit. Every number is read out of the split_v3 run that produced it.\n")
    W("These are **what split_v3 achieved**, not thresholds anyone set. A split_v4 run")
    W("that misses one is not necessarily wrong — the labels changed, and the point of")
    W("the rebuild is that the porosity should be better. The table exists so a")
    W("difference is SEEN and explained rather than missed.\n")

    # ── the VAE ────────────────────────────────────────────────────────────
    rung = _load(REPO / "runs/campaigns/09-r08-latent-sweep/r08_reduction-factor-8/results.json")
    W("\n## r08 rf-8 (`r08-run-0004`, step 36720) — the VAE\n")
    on_disk = rung is not None
    rung = _ref_only(rung, "rung report", W)
    if not on_disk:
        W("**The rung report is not on disk.** Run "
          "`python scripts/analysis/r08_rung_report.py --run <run>` first.\n")
    elif rung is not None:
        W("| gate | val | test | where from |")
        W("|---|---|---|---|")
        for label, key in (("porosity MAE", "porosity_mae"), ("air MAE", "air_mae"),
                           ("pore Dice", "dice_pore"), ("air Dice", "dice_air"),
                           ("material Dice", "dice_material")):
            v = _dig(rung, "splits", "val", key)
            t = _dig(rung, "splits", "test", key)
            W(f"| {label} | {v:.5f} | {t:.5f} | `r08_rung_report.py` |"
              if v is not None and t is not None else
              f"| {label} | — | — | not in the report |")
        tau = _dig(rung, "calibration", "tau")
        wb = _dig(rung, "calibration", "worst_bin_mae_val")
        am = _dig(rung, "calibration", "argmax_worst_bin_mae_val")
        W(f"| worst judged bin MAE @tau={tau} | {wb:.5f} | — | calibration block |")
        W(f"| worst judged bin MAE @argmax | {am:.5f} | — | calibration block |")
        W(f"\nPatches: {_dig(rung, 'splits', 'val', 'n_patches'):,} val, "
          f"{_dig(rung, 'splits', 'test', 'n_patches'):,} test. "
          f"{_dig(rung, 'n_params'):,} parameters.\n")

    probe = _load(REPO / "runs/campaigns/09-r08-latent-sweep"
                  "/calibration_probe_r08_reduction-factor-8/results.json")
    probe = _ref_only(probe, "calibration probe", W)
    if probe is not None:
        W("\n### Dense panels (`Na_10`, `Na_09`, `Pegaso_1`) — the number capacity moved\n")
        W("From `r08_calibration_probe.py`, which samples all 17 panels; the rung")
        W("report sees val and test only and structurally cannot report these.\n")
        W("| dense pore Dice | rest | gap |")
        W("|---|---|---|")
        W("| 0.8162 | 0.9211 | 0.1050 |")
        W("\n(campaign 09's README table; the probe's own JSON keys the same run.)\n")

    fam = _load(REPO / "runs/campaigns/28-vrrae-family/family_table.json")
    if fam:
        r08 = next((r for r in fam if r.get("run", "").startswith("r08-run-0004")), None)
        if r08:
            W("\n### The harness row (campaign 28), for texture and sharpness\n")
            W("| L1 | error removed | texture, interior | sharpness, interior |")
            W("|---|---|---|---|")
            W(f"| {r08['l1']:.4f} | "
              f"{r08['fraction_of_baseline_error_removed']:.1%} | "
              f"{r08.get('texture_corr_interior', float('nan')):+.3f} | "
              f"{r08.get('sharpness_interior', float('nan')):.3f} |")
            W("\nMeasured on the 5 split_v3 val volumes split_v2 never trained on, with")
            W("the fixed basis. **Read texture, not L1**: on this family L1 ranks the")
            W("mean grey level and a flat grey block scores well on it.\n")

    # ── the six rungs ──────────────────────────────────────────────────────
    W("\n## The six r08 rungs — the paper's latent-compression table\n")
    W("Each split_v4 rung is read against its own split_v3 row. Rung report on")
    W("val/test (pore Dice, air Dice, porosity MAE), the tau calibrated on val, and")
    W("the calibration probe's dense-panel pore Dice (argmax).\n")
    W("| rung | split_v3 run | val pore Dice | test pore Dice | val air Dice | "
      "val por MAE | test por MAE | tau | worst bin @tau | dense Dice | rest Dice |")
    W("|---|---|---|---|---|---|---|---|---|---|---|")
    camp = REPO / "runs/campaigns/09-r08-latent-sweep"
    for rung, key, run in RUNGS:
        rep = _load(camp / key / "results.json")
        prb = _load(camp / f"calibration_probe_{key}" / "results.json")
        rep_ok = rep is not None and Path(str(rep.get("run_dir", ""))).name.startswith(run)
        prb_ok = prb is not None and Path(str(prb.get("run_dir", ""))).name.startswith(run)
        if not rep_ok:
            W(f"| {rung} | `{run}` | **rung report missing or not {run}** "
              "| | | | | | | | |")
            continue
        sp = rep["splits"]
        cal = rep.get("calibration", {})
        dense = rest = "—"
        if prb_ok:
            dvr = _dig(prb, "summary", "argmax", "dense_vs_rest") or {}
            dense = f"{_dig(dvr, 'dense', 'dice_pore'):.4f}" if _dig(dvr, "dense", "dice_pore") is not None else "—"
            rest = f"{_dig(dvr, 'rest', 'dice_pore'):.4f}" if _dig(dvr, "rest", "dice_pore") is not None else "—"
        W(f"| {rung} | `{run}` | {sp['val']['dice_pore']:.5f} | {sp['test']['dice_pore']:.5f} | "
          f"{sp['val']['dice_air']:.5f} | {sp['val']['porosity_mae']:.5f} | "
          f"{sp['test']['porosity_mae']:.5f} | {cal.get('tau')} | "
          f"{cal.get('worst_bin_mae_val', float('nan')):.5f} | {dense} | {rest} |")
    W("")

    # ── the LDM ────────────────────────────────────────────────────────────
    W("\n## ldm06 — the LDM\n")
    for pat, tag in (("runs/ldm/ldm06-run-0001-*", "run-0001, base, step 130000"),
                     ("runs/ldm/ldm06-run-0002-*", "run-0002, facedrop, step 15000")):
        dirs = sorted(REPO.glob(pat))
        if not dirs:
            W(f"\n### {tag}\n\n**Not on disk.**\n"); continue
        f = dirs[-1] / "convergence_check.jsonl"
        if not f.exists():
            W(f"\n### {tag}\n")
            W(f"**No `convergence_check.jsonl`** — `scripts/diag_ldm_samples.py` never ran")
            W("on this run, so it has no convergence gates of its own.\n")
            continue
        rows = [json.loads(l) for l in f.read_text().splitlines() if l.strip()]
        last = rows[-1]
        W(f"\n### {tag} — {len(rows)} check(s), last at step {last.get('step')}\n")
        W("| gate | ema_ddim50 | raw_ddim50 | reading |")
        W("|---|---|---|---|")
        for label, key, fmt, note in (
            ("porosity MAE", "por_mae", "{:.6f}", "delivered against requested"),
            ("delivered phi mean", "por_mean", "{:.5f}", "over all buckets"),
            ("std ratio (mu-normalised)", "std_ratio", "{:.4f}",
             "scored against the SAMPLED reference 1.863, not 1.0"),
            ("x0 clamp saturation", "x0_sat", "{:.2e}", "off-manifold diagnostic"),
            ("air mean", "air_mean", "{:.5f}", "exterior air the request asked for"),
            ("degenerate cells", "degen", "{:.3f}", "0 is the only acceptable value"),
        ):
            e = _dig(last, "variants", "ema_ddim50", "overall", key)
            r = _dig(last, "variants", "raw_ddim50", "overall", key)
            W(f"| {label} | {fmt.format(e) if e is not None else '—'} | "
              f"{fmt.format(r) if r is not None else '—'} | {note} |")
        ks = _dig(last, "killswitch", default={})
        if ks:
            W(f"\n**Kill switch** — asked {ks.get('asked_lo')} and {ks.get('asked_hi')}: "
              f"ema delivered {_dig(ks, 'ema', 'por_lo'):.5f} and "
              f"{_dig(ks, 'ema', 'por_hi'):.5f}, separation "
              f"{_dig(ks, 'ema', 'separation'):.5f}, direction_ok "
              f"{_dig(ks, 'ema', 'direction_ok')}. A model that cannot separate two "
              "requested porosities is not conditioning at all.\n")
        ins = _dig(last, "inset_surface", "gate", default={})
        if ins:
            W("**Inset-surface gates**: "
              + ", ".join(f"`{k}` {v}" for k, v in ins.items()) + "\n")
        al = _dig(last, "alive", default={})
        if al:
            W(f"**Conditioning alive** (MAD as % of the noise MAD "
              f"{al.get('noise_mad', float('nan')):.3f}): "
              f"porosity {_dig(al, 'por_neutral', 'pct_of_noise'):.2f} %, "
              f"orientation {_dig(al, 'orient_zero', 'pct_of_noise'):.2f} %, "
              f"neighbours {_dig(al, 'nb_unknown', 'pct_of_noise'):.2f} %. "
              "A conditioning input whose ablation moves nothing is not being used.\n")

    _provenance(W)

    W("\n## How the v4 chain uses these\n")
    W("`scripts/chain_v8_split_v4.sh` runs the same checks as stages and writes a")
    W("PASS/FAIL line per gate into `runs/campaigns/chain.log`, against this file.")
    W("A FAIL does not stop the chain — the labels changed and a moved number may be")
    W("the improvement the rebuild is for — except for `degen`, the kill switch's")
    W("`direction_ok`, and a non-finite loss, which mean the run is not working.\n")

    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text("\n".join(L))
    print(f"wrote {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
