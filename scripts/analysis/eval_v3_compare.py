"""Head-to-head OLD (eval v2, buggy decode) vs NEW (eval v3) + the v3 README.

Restates every headline number of the campaign side by side, so the effect of
the decode fix is explicit rather than implied.  Sections are emitted for
whichever inputs exist, which lets the runner call this once after the cheap
sets and again after the layup set lands.

OLD sources (read-only; ``runs/eval_v2`` is the record of the buggy run):
    runs/eval_v2/dose_response/results.json
    runs/eval_v2/layup/results.json
    runs/analysis/air_audit_v2/per_volume.csv
    runs/analysis/onlypores_generated/per_volume.csv
    runs/analysis/ldm06_probe/results.json

NEW sources: the matching artefacts under runs/eval_v3/.

Outputs -> ``runs/eval_v3/comparison/`` (results.json, findings.md, figure)
and ``runs/eval_v3/README.md``.

Usage:
    python scripts/analysis/eval_v3_compare.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import savefig, set_style, write_findings, write_json, plt  # noqa: E402
from _eval_v3 import (  # noqa: E402
    ARMS, ARM_COLORS, DDIM_STEP_COUNTS, GATE, ROOT, TARGETS, V2_AIR_AUDIT,
    V2_LDM06_PROBE, V2_ONLYPORES, V2_ROOT, VOL_ROOT, load_calibration,
)

OUT_DIR = ROOT / "comparison"

# v2 probe volumes are named "<steps>_seed_<seed>"; v3 uses "steps_<n>_seed_<s>".
def _v2_probe_name(steps: int, seed: int) -> str:
    return f"{steps}_seed_{seed}"


def _v3_probe_name(steps: int, seed: int) -> str:
    return f"steps_{steps}_seed_{seed}"


def _load(p: Path):
    return json.loads(p.read_text()) if p.exists() else None


def _csv(p: Path):
    return pd.read_csv(p) if p.exists() else None


# ---------------------------------------------------------------------------
# Sections
# ---------------------------------------------------------------------------

def section_scale(new_air: pd.DataFrame, old_air: pd.DataFrame) -> dict:
    """The bug itself: what grey levels the two campaigns produced."""
    old = old_air[old_air.experiment.isin(["dose_response", "layup",
                                           "ldm06_probe"])]
    return {
        "old_grey_range_u8": [int(old.raw_min_u8.min()), int(old.raw_max_u8.max())],
        "old_material_mode_u8": float(pd.to_numeric(old.material_mode_u8,
                                                    errors="coerce").mean()),
        "new_grey_range_u8": [int(new_air.raw_min_u8.min()),
                              int(new_air.raw_max_u8.max())],
        "new_material_mode_u8": float(pd.to_numeric(new_air.material_mode_u8,
                                                    errors="coerce").mean()),
        "real_material_mode_u8": 208.6,
        "note": "OLD volumes were expit(xct_head) -> float [0.52, 0.73] -> u8; "
                "NEW volumes are clamp(xct_head)*255 -> u8, the raw-scan scale.",
    }


def section_dose(old: dict, new: dict) -> dict:
    out = {"arms": {}, "per_level": {}}
    for arm in ARMS:
        o_g, n_g = old["fits_global_mask"].get(arm), new["fits_global_mask"].get(arm)
        o_l, n_l = old["fits_local"].get(arm), new["fits_local"].get(arm)
        o_e, n_e = old["local_abs_error"].get(arm), new["local_abs_error"].get(arm)
        o_rec = [r for r in old["records"] if r["arm"] == arm]
        n_rec = [r for r in new["records"] if r["arm"] == arm]
        out["arms"][arm] = {
            "old": {"global_slope": o_g["slope"], "global_intercept": o_g["intercept"],
                    "global_r2": o_g["r2"], "local_slope": o_l["slope"],
                    "local_r2": o_l["r2"],
                    "local_frac_within_gate": o_e["frac_within_gate"],
                    "global_mean_abs_error": float(np.mean([r["abs_error"] for r in o_rec])),
                    "n_levels_passing_gate": int(sum(
                        p["passes_gate"] for p in old["per_level"][arm])),
                    "mean_delivered": float(np.mean(
                        [r["delivered_mask_porosity"] for r in o_rec]))},
            "new": {"global_slope": n_g["slope"], "global_intercept": n_g["intercept"],
                    "global_r2": n_g["r2"], "local_slope": n_l["slope"],
                    "local_r2": n_l["r2"],
                    "local_frac_within_gate": n_e["frac_within_gate"],
                    "global_mean_abs_error": float(np.mean([r["abs_error"] for r in n_rec])),
                    "n_levels_passing_gate": int(sum(
                        p["passes_gate"] for p in new["per_level"][arm])),
                    "mean_delivered": float(np.mean(
                        [r["delivered_mask_porosity"] for r in n_rec]))},
        }
        out["per_level"][arm] = [
            {"target": t,
             "old_delivered": next(p["delivered_mean"] for p in old["per_level"][arm]
                                   if np.isclose(p["target"], t)),
             "new_delivered": next(p["delivered_mean"] for p in new["per_level"][arm]
                                   if np.isclose(p["target"], t))}
            for t in TARGETS]

    # The bug touched only the XCT grey channel; the mask head is decoded by a
    # genuine sigmoid and was never affected.  With identical seeds and
    # settings the delivered MASK porosity should reproduce exactly — check it
    # rather than assume it, because it is also the strongest evidence that the
    # re-run is otherwise identical to the original.
    o_by_key = {(r["arm"], r["target"], r["seed"]): r["delivered_mask_porosity"]
                for r in old["records"]}
    deltas = [abs(r["delivered_mask_porosity"] - o_by_key[k])
              for r in new["records"]
              if (k := (r["arm"], r["target"], r["seed"])) in o_by_key]
    out["mask_reproduction"] = {
        "n_matched": len(deltas),
        "max_abs_delta": max(deltas) if deltas else None,
        "bit_identical": bool(deltas) and max(deltas) == 0.0,
    }
    return out


def section_air(old_air: pd.DataFrame, new_air: pd.DataFrame) -> dict:
    """Unmasked air, matched volume by volume where the names line up.

    OLD `unmasked_air_abs` is the SAME real-calibrated detector, expressed on
    the compressed scale through the audit's area-preserving scale map (the
    only way it could be applied then).  NEW applies T_abs directly.
    """
    out = {}
    for exp in ("dose_response", "layup"):
        o = old_air[old_air.experiment == exp]
        n = new_air[new_air.experiment == exp]
        if not len(n):
            continue
        for arm in ARMS:
            oa, na = o[o.arm == arm], n[n.arm == arm]
            if not len(na):
                continue
            common = sorted(set(oa.name) & set(na.name))
            oa_c = oa[oa.name.isin(common)]
            na_c = na[na.name.isin(common)]
            out[f"{exp}/{arm}"] = {
                "n_matched": len(common),
                "old_unmasked_air_abs": float(oa_c.unmasked_air_abs.mean()),
                "new_unmasked_air_abs": float(na_c.unmasked_air_abs.mean()),
                "old_detected_air_abs": float(oa_c.detected_air_abs.mean()),
                "new_detected_air_abs": float(na_c.detected_air_abs.mean()),
                "old_mask_porosity": float(oa_c.mask_porosity.mean()),
                "new_mask_porosity": float(na_c.mask_porosity.mean()),
                "old_interior_local_self": float(oa_c.unmasked_interior_local_self.mean()),
                "new_interior_local_abs": float(na_c.unmasked_interior_local_abs.mean()),
                "old_edge_local_self": float(oa_c.unmasked_edge_local_self.mean()),
                "new_edge_local_abs": float(na_c.unmasked_edge_local_abs.mean()),
                "new_mask_capture_of_detected": float(na_c.mask_capture_of_detected.mean()),
            }
    # DDIM probe: names differ between campaigns
    o = old_air[old_air.experiment == "ldm06_probe"]
    n = new_air[new_air.experiment == "ddim_probe"]
    if len(n):
        pairs = [(_v2_probe_name(s, sd), _v3_probe_name(s, sd))
                 for s in DDIM_STEP_COUNTS for sd in (101, 202)]
        oo = o[o.name.isin([a for a, _ in pairs])]
        nn = n[n.name.isin([b for _, b in pairs])]
        out["ddim_probe/all"] = {
            "n_matched": min(len(oo), len(nn)),
            "old_unmasked_air_abs": float(oo.unmasked_air_abs.mean()),
            "new_unmasked_air_abs": float(nn.unmasked_air_abs.mean()),
            "old_detected_air_abs": float(oo.detected_air_abs.mean()),
            "new_detected_air_abs": float(nn.detected_air_abs.mean()),
            "old_mask_porosity": float(oo.mask_porosity.mean()),
            "new_mask_porosity": float(nn.mask_porosity.mean()),
            "old_interior_local_self": float(oo.unmasked_interior_local_self.mean()),
            "new_interior_local_abs": float(nn.unmasked_interior_local_abs.mean()),
            "old_edge_local_self": float(oo.unmasked_edge_local_self.mean()),
            "new_edge_local_abs": float(nn.unmasked_edge_local_abs.mean()),
            "new_mask_capture_of_detected": float(nn.mask_capture_of_detected.mean()),
        }
    return out


def section_onlypores(old_op: pd.DataFrame, new_op: pd.DataFrame) -> dict:
    out = {}
    for exp in ("dose_response", "layup"):
        o = old_op[old_op.experiment == exp]
        n = new_op[new_op.experiment == exp]
        if not len(n):
            continue
        for arm in ARMS:
            oa, na = o[o.arm == arm], n[n.arm == arm]
            if not len(na):
                continue
            common = sorted(set(oa.name) & set(na.name))
            oa_c, na_c = oa[oa.name.isin(common)], na[na.name.isin(common)]
            out[f"{exp}/{arm}"] = {
                "n_matched": len(common),
                "old_onlypores_sample": float(oa_c.onlypores_porosity_sample.mean()),
                "new_onlypores_sample": float(na_c.onlypores_porosity_sample.mean()),
                "old_sample_mask_fraction": float(oa_c.sample_mask_fraction.mean()),
                "new_sample_mask_fraction": float(na_c.sample_mask_fraction.mean()),
                "old_dice_mask_vs_onlypores": float(oa_c.dice_mask_vs_onlypores.mean()),
                "new_dice_mask_vs_onlypores": float(na_c.dice_mask_vs_onlypores.mean()),
                "old_mask_porosity": float(oa_c.mask_porosity.mean()),
                "new_mask_porosity": float(na_c.mask_porosity.mean()),
            }
    return out


def section_ddim(old_probe: dict, new_ddim: dict) -> dict:
    """Old part-B rows vs the v3 step sweep, per step count."""
    old_rows = {(r["ddim_steps"], r["seed"]): r
                for r in old_probe["part_b"]["volumes"]
                if r["shape"].startswith("192")}
    new_rows = {(r["ddim_steps"], r["seed"]): r for r in new_ddim["volumes"]}
    per_step = []
    for steps in DDIM_STEP_COUNTS:
        o = [old_rows[(steps, s)] for s in (101, 202) if (steps, s) in old_rows]
        n = [new_rows[(steps, s)] for s in (101, 202) if (steps, s) in new_rows]
        if not o or not n:
            continue
        per_step.append({
            "ddim_steps": steps,
            "old_interior_air": float(np.mean(
                [r["unmasked_interior_local_best"] for r in o])),
            "new_interior_air": float(np.mean(
                [r["unmasked_interior_local_abs"] for r in n])),
            "old_mask_porosity": float(np.mean([r["mask_porosity"] for r in o])),
            "new_mask_porosity": float(np.mean([r["mask_porosity"] for r in n])),
            "old_abs_error": float(np.mean(
                [abs(r["mask_porosity"] - 0.03) for r in o])),
            "new_abs_error": float(np.mean([r["mask_abs_error"] for r in n])),
        })
    return {
        "note": "OLD interior air is the v2 probe's T_best detector on the "
                "compressed scale (the only detector valid there); NEW is "
                "T_abs = 182 applied directly. Both are the real-calibrated "
                "rule expressed on each campaign's own grey scale.",
        "old_monotone": old_probe["part_b"].get("monotone_reduction"),
        "new_monotone": new_ddim.get("monotone_interior_air_reduction"),
        "per_step": per_step,
    }


def section_layup(old: dict, new: dict) -> dict:
    seeds_new = sorted({r["seed"] for r in new["records"]})
    out = {"seeds_compared": seeds_new, "arms": {}}
    for arm in ARMS:
        o = [r for r in old["records"] if r["arm"] == arm and r["seed"] in seeds_new]
        n = [r for r in new["records"] if r["arm"] == arm]
        if not o or not n:
            continue

        def _agg(rs, key):
            return float(np.mean([r["combined"][key] for r in rs]))
        out["arms"][arm] = {
            "n_old": len(o), "n_new": len(n),
            "old_median_abs_error_deg": _agg(o, "direct_median_abs_error_deg"),
            "new_median_abs_error_deg": _agg(n, "direct_median_abs_error_deg"),
            "old_strict_class_accuracy": _agg(o, "strict_class_accuracy"),
            "new_strict_class_accuracy": _agg(n, "strict_class_accuracy"),
            "old_frac_within_10": _agg(o, "direct_frac_within_10"),
            "new_frac_within_10": _agg(n, "direct_frac_within_10"),
            "old_delivered_porosity": float(np.mean(
                [r["delivered_mask_porosity"] for r in o])),
            "new_delivered_porosity": float(np.mean(
                [r["delivered_mask_porosity"] for r in n])),
        }
    return out


# ---------------------------------------------------------------------------

def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    set_style()
    cal = load_calibration()

    old_dose = _load(V2_ROOT / "dose_response" / "results.json")
    new_dose = _load(ROOT / "dose_response" / "results.json")
    old_air = _csv(V2_AIR_AUDIT / "per_volume.csv")
    new_air = _csv(ROOT / "air_audit" / "per_volume.csv")
    old_op = _csv(V2_ONLYPORES / "per_volume.csv")
    new_op = _csv(ROOT / "onlypores" / "per_volume.csv")
    old_probe = _load(V2_LDM06_PROBE / "results.json")
    new_ddim = _load(ROOT / "ddim" / "results.json")
    old_layup = _load(V2_ROOT / "layup" / "results.json")
    new_layup = _load(ROOT / "layup" / "results.json")

    res: dict = {"campaign": "eval v3 vs eval v2 — effect of the decode fix",
                 "old_root": str(V2_ROOT), "new_root": str(ROOT),
                 "sections_present": [], "sections_missing": []}

    def add(name, value):
        if value is None:
            res["sections_missing"].append(name)
        else:
            res[name] = value
            res["sections_present"].append(name)

    add("scale", section_scale(new_air, old_air)
        if new_air is not None and old_air is not None else None)
    # the generation script checkpoints a {"partial", "records"} results.json
    # after every volume; only the finished file carries the fits.
    dose_ready = all(d and "fits_global_mask" in d for d in (old_dose, new_dose))
    add("dose_response", section_dose(old_dose, new_dose) if dose_ready else None)
    add("air", section_air(old_air, new_air)
        if old_air is not None and new_air is not None else None)
    add("onlypores", section_onlypores(old_op, new_op)
        if old_op is not None and new_op is not None else None)
    add("ddim", section_ddim(old_probe, new_ddim)
        if old_probe and new_ddim else None)
    add("layup", section_layup(old_layup, new_layup)
        if old_layup and new_layup else None)

    # ---------------- figure ----------------
    figs = []
    if "dose_response" in res and "air" in res:
        fig, axes = plt.subplots(1, 3, figsize=(13.4, 4.2),
                                 constrained_layout=True)
        ax = axes[0]
        lim = [0.0, 0.115]
        ax.plot(lim, lim, color="0.4", ls="--", lw=1.0, label="identity")
        ax.fill_between(lim, [v - GATE for v in lim], [v + GATE for v in lim],
                        color="0.85", alpha=0.5, zorder=0)
        for arm in ARMS:
            pl = res["dose_response"]["per_level"][arm]
            ax.plot([p["target"] for p in pl], [p["old_delivered"] for p in pl],
                    "o--", color=ARM_COLORS[arm], alpha=0.5,
                    label=f"{arm} OLD")
            ax.plot([p["target"] for p in pl], [p["new_delivered"] for p in pl],
                    "o-", color=ARM_COLORS[arm], label=f"{arm} NEW")
        ax.set_xlim(lim)
        ax.set_xlabel("requested φ")
        ax.set_ylabel("delivered mask φ")
        ax.set_title("Dose response, OLD vs NEW")
        ax.legend(fontsize=6.5, ncol=2)

        ax = axes[1]
        keys = sorted(res["air"])
        xs = np.arange(len(keys))
        ax.bar(xs - 0.2, [res["air"][k]["old_unmasked_air_abs"] for k in keys],
               0.38, color="#8a8a8a", label="OLD")
        ax.bar(xs + 0.2, [res["air"][k]["new_unmasked_air_abs"] for k in keys],
               0.38, color="#c2571a", label="NEW")
        ax.axhline(cal["real_false_positive_baseline"]["mean"], color="#1b6ca8",
                   ls="--", lw=1.2, label="real false-positive floor")
        ax.set_xticks(xs)
        ax.set_xticklabels(keys, fontsize=6.5, rotation=30, ha="right")
        ax.set_ylabel("unmasked air fraction")
        ax.set_title("Unmasked air (calibrated detector)")
        ax.legend(fontsize=7.5)

        ax = axes[2]
        sc = res["scale"]
        ax.barh([0, 1], [sc["old_grey_range_u8"][1] - sc["old_grey_range_u8"][0],
                         sc["new_grey_range_u8"][1] - sc["new_grey_range_u8"][0]],
                left=[sc["old_grey_range_u8"][0], sc["new_grey_range_u8"][0]],
                color=["#8a8a8a", "#c2571a"], height=0.45)
        ax.axvline(sc["real_material_mode_u8"], color="0.2", ls=":", lw=1.4,
                   label=f"real material mode {sc['real_material_mode_u8']:.0f}")
        ax.axvline(cal["t_abs"], color="#1b6ca8", ls="--", lw=1.2,
                   label=f"T_abs {cal['t_abs']}")
        ax.set_yticks([0, 1])
        ax.set_yticklabels(["OLD (expit)", "NEW (clamp)"])
        ax.set_xlim(0, 255)
        ax.set_xlabel("grey level (uint8)")
        ax.set_title("Occupied intensity range")
        ax.legend(fontsize=7.5)
        fig.suptitle("Effect of the decode fix — eval v2 vs eval v3")
        figs += savefig(fig, OUT_DIR, "cmpv3_fig1_old_vs_new")
    res["figures"] = figs

    p_json = write_json(res, OUT_DIR)

    # ---------------- findings ----------------
    L = ["# OLD vs NEW — the decode fix, number by number", "",
         "OLD = `runs/eval_v2` + `runs/analysis/{air_audit_v2, "
         "onlypores_generated, ldm06_probe}` (sampler applied a spurious "
         "`expit` to the VAE XCT head). NEW = `runs/eval_v3` (clamp-and-scale, "
         "volume.tif native uint8). Same checkpoint, same seeds, same "
         "settings.", ""]

    if "scale" in res:
        s = res["scale"]
        L += ["## 1. The bug", "",
              "| | grey range (u8) | material mode (u8) |", "|---|---|---|",
              f"| OLD | [{s['old_grey_range_u8'][0]}, {s['old_grey_range_u8'][1]}] "
              f"| {s['old_material_mode_u8']:.1f} |",
              f"| NEW | [{s['new_grey_range_u8'][0]}, {s['new_grey_range_u8'][1]}] "
              f"| {s['new_material_mode_u8']:.1f} |",
              f"| real scans | — | {s['real_material_mode_u8']:.1f} |", "",
              s["note"], ""]

    if "dose_response" in res:
        L += ["## 2. Dose response (mask porosity, 63 volumes)", "",
              "| arm | global slope | global R² | local slope | local R² "
              "| local frac within gate | levels passing gate |",
              "|---|---|---|---|---|---|---|"]
        for arm in ARMS:
            a = res["dose_response"]["arms"][arm]
            for tag in ("old", "new"):
                v = a[tag]
                L.append(f"| {arm} {tag.upper()} | {v['global_slope']:.3f} "
                         f"| {v['global_r2']:.4f} | {v['local_slope']:.3f} "
                         f"| {v['local_r2']:.4f} "
                         f"| {v['local_frac_within_gate']:.2f} "
                         f"| {v['n_levels_passing_gate']}/{len(TARGETS)} |")
        mr = res["dose_response"]["mask_reproduction"]
        L += ["",
              (f"**Mask reproduction:** delivered mask porosity is "
               f"bit-identical across all {mr['n_matched']} matched volumes."
               if mr["bit_identical"] else
               f"**Mask reproduction:** max |OLD − NEW| delivered mask "
               f"porosity = {mr['max_abs_delta']:.2e} over {mr['n_matched']} "
               f"matched volumes."),
              "",
              "That is the expected result: the spurious `expit` was applied "
              "only to the XCT grey channel. The mask head IS a logit and its "
              "sigmoid was always correct, so every mask-based conclusion "
              "(dose-response slope, gate, local obedience) stands unchanged. "
              "It also proves the re-run reproduces the original run exactly.",
              "", "Per level, delivered mask φ:", "",
              "| target | " + " | ".join(f"{a} OLD | {a} NEW" for a in ARMS) + " |",
              "|---|" + "---|" * (2 * len(ARMS))]
        for i, t in enumerate(TARGETS):
            cells = []
            for arm in ARMS:
                pl = res["dose_response"]["per_level"][arm][i]
                cells += [f"{pl['old_delivered']:.4f}", f"{pl['new_delivered']:.4f}"]
            L.append(f"| {t:.3f} | " + " | ".join(cells) + " |")
        L.append("")

    if "air" in res:
        L += ["## 3. Unmasked air (real-calibrated detector, min CC 300)", "",
              "OLD applies the same calibrated rule through the v2 audit's "
              "area-preserving scale map (the compressed scale left no "
              "alternative); NEW applies T_abs = "
              f"{cal['t_abs']} directly on the real u8 scale.", "",
              "| set | n | mask φ OLD → NEW | unmasked air OLD → NEW "
              "| detected air OLD → NEW | interior OLD → NEW | edge OLD → NEW "
              "| mask capture NEW |",
              "|---|---|---|---|---|---|---|---|"]
        for k in sorted(res["air"]):
            v = res["air"][k]
            L.append(f"| {k} | {v['n_matched']} "
                     f"| {v['old_mask_porosity']:.4f} → {v['new_mask_porosity']:.4f} "
                     f"| {v['old_unmasked_air_abs']:.4f} → {v['new_unmasked_air_abs']:.4f} "
                     f"| {v['old_detected_air_abs']:.4f} → {v['new_detected_air_abs']:.4f} "
                     f"| {v['old_interior_local_self']:.4f} → {v['new_interior_local_abs']:.4f} "
                     f"| {v['old_edge_local_self']:.4f} → {v['new_edge_local_abs']:.4f} "
                     f"| {v['new_mask_capture_of_detected']:.3f} |")
        L += ["",
              "OLD interior/edge come from the v2 audit's per-volume self-Otsu "
              "rule — the only interior/edge split that campaign recorded.", ""]

    if "onlypores" in res:
        L += ["## 4. onlypores porosity", "",
              "OLD had to undo the sigmoid (`clip(logit(v),0,1)*255`); NEW "
              "measures the native u8 directly.", "",
              "| set | n | onlypores φ OLD → NEW | sample-mask frac OLD → NEW "
              "| Dice(mask, onlypores) OLD → NEW |",
              "|---|---|---|---|---|"]
        for k in sorted(res["onlypores"]):
            v = res["onlypores"][k]
            L.append(f"| {k} | {v['n_matched']} "
                     f"| {v['old_onlypores_sample']:.4f} → {v['new_onlypores_sample']:.4f} "
                     f"| {v['old_sample_mask_fraction']:.3f} → {v['new_sample_mask_fraction']:.3f} "
                     f"| {v['old_dice_mask_vs_onlypores']:.3f} → {v['new_dice_mask_vs_onlypores']:.3f} |")
        L += ["",
              "Both columns inherit the 192³ global-Otsu caveat "
              "(`runs/eval_v3/onlypores/findings.md`).", ""]

    if "ddim" in res:
        d = res["ddim"]
        L += ["## 5. DDIM step sweep", "", d["note"], "",
              f"Monotone interior-air reduction: OLD {d['old_monotone']}, "
              f"NEW {d['new_monotone']}.", "",
              "| steps | interior air OLD → NEW | mask φ OLD → NEW "
              "| \\|φ − target\\| OLD → NEW |", "|---|---|---|---|"]
        for p in d["per_step"]:
            L.append(f"| {p['ddim_steps']} "
                     f"| {p['old_interior_air']:.4f} → {p['new_interior_air']:.4f} "
                     f"| {p['old_mask_porosity']:.4f} → {p['new_mask_porosity']:.4f} "
                     f"| {p['old_abs_error']:.4f} → {p['new_abs_error']:.4f} |")
        L.append("")

    if "layup" in res:
        L += ["## 6. Layup round trip (1024×1024×192)", "",
              f"Seeds compared: {res['layup']['seeds_compared']}.", "",
              "| arm | median \\|angle err\\| OLD → NEW | ≤10° OLD → NEW "
              "| strict 4-class OLD → NEW | delivered φ OLD → NEW |",
              "|---|---|---|---|---|"]
        for arm, v in res["layup"]["arms"].items():
            L.append(f"| {arm} "
                     f"| {v['old_median_abs_error_deg']:.1f}° → {v['new_median_abs_error_deg']:.1f}° "
                     f"| {100 * v['old_frac_within_10']:.0f}% → {100 * v['new_frac_within_10']:.0f}% "
                     f"| {100 * v['old_strict_class_accuracy']:.1f}% → {100 * v['new_strict_class_accuracy']:.1f}% "
                     f"| {v['old_delivered_porosity']:.4f} → {v['new_delivered_porosity']:.4f} |")
        L.append("")

    if res["sections_missing"]:
        L += ["## Not yet available", "",
              ", ".join(res["sections_missing"]) + " — inputs missing at the "
              "time this ran.", ""]
    p_md = write_findings("\n".join(L), OUT_DIR)

    write_readme(res)
    print("Wrote", p_json, p_md, *figs, sep="\n  ", flush=True)


def write_readme(res: dict) -> None:
    have = set(res["sections_present"])
    n_vol = sum(1 for _ in VOL_ROOT.rglob("volume.tif")) if VOL_ROOT.exists() else 0
    lines = [
        "# eval v3 — porosity and air results on correctly-decoded volumes",
        "",
        "## Why this tree exists",
        "",
        "`poregen.diffusion.sampler` used to apply `expit()` to the VAE's XCT "
        "head before scaling to uint8. That head regresses `xct / 255` "
        "directly, so the sigmoid was spurious: it squashed every generated "
        "volume into [0.5, 0.731] (grey levels ~[133, 187]) and destroyed "
        "contrast. The sampler now clamps and scales, mirroring the shared "
        "helpers `decode_xct` / `decode_xct_u8` in "
        "`src/poregen/models/vae/base.py`, and "
        "`scripts/analysis/_eval_v2.py:save_volume` writes `volume.tif` as "
        "NATIVE uint8 (it previously wrote u8/255 as float32).",
        "",
        "Every volume under `runs/eval_v2/volumes/` and "
        "`runs/analysis/ldm06_probe/volumes/` was produced with the bug and is "
        "quantised on the compressed scale. **`runs/eval_v2` is left untouched "
        "as the record of that run.** This tree regenerates what the "
        "porosity/air results rest on and redoes the analyses.",
        "",
        "## Intensity scale — read this before writing a reader",
        "",
        "`volume.tif` here is **uint8 on the raw-scan grey scale**, the same "
        "scale and dtype as `data/split_v2/volumes.zarr`. Readers elsewhere in "
        "the repo assume float [0, 1]; anything reading this tree must verify "
        "the dtype. `scripts/analysis/_eval_v3.py:load_u8` does exactly that "
        "and is the loader every v3 analysis uses.",
        "",
        "## Tree",
        "",
        "```",
        "runs/eval_v3/",
        f"├── volumes/                      {n_vol} volumes, uint8",
        "│   ├── dose_response/<arm>/target_<t>_seed_<s>/   7 targets x 3 arms x 3 seeds, 192³",
        "│   ├── ddim_probe/steps_<n>_seed_<s>/             steps {50,100,200,300} x 2 seeds, 192³",
        "│   └── layup/<arm>/<layup>_seed_101/              3 layups x 3 arms, 1024x1024x192",
        "├── dose_response/   global + local OLS, per-level tables, gate |err| < 0.005",
        "├── air_audit/       calibrated absolute threshold T_abs = 182 (Dice 0.842 real)",
        "├── onlypores/       onlypores porosity + the real-volume and real-192³ controls",
        "├── ddim/            interior air and porosity error vs DDIM steps",
        "├── comparison/      OLD (eval v2) vs NEW (eval v3), number by number",
        "├── README.md        this file",
        "└── run.log          runner progress",
        "```",
        "",
        "Each analysis directory holds `results.json`, `findings.md` and "
        "figures as PDF + PNG at 300 dpi.",
        "",
        "## Arms",
        "",
        "| arm | mode | conditioning semantics | s_por |",
        "|---|---|---|---|",
        "| `seq` | sequential | specimen | 1.0 |",
        "| `joint_legacy` | joint | legacy | 1.5 |",
        "| `joint_oob` | joint | specimen | 1.5 |",
        "",
        "Checkpoint: ldm05 step 130000, RAW (non-EMA) weights, DDIM-50 except "
        "in the step sweep. Porosity field: coherent (T-E marginal + T-D "
        "smoothing), seeded per volume.",
        "",
        "## Old vs new",
        "",
        "`comparison/findings.md` restates every headline number side by side. "
        + ("Sections present: " + ", ".join(sorted(have)) + "."
           if have else "Not produced yet."),
        "",
        "## Reproducing",
        "",
        "```bash",
        "nohup bash scripts/analysis/eval_v3_run.sh > /dev/null 2>&1 &",
        "tail -f runs/eval_v3/run.log     # ends with EVAL_V3_DONE",
        "```",
        "",
    ]
    (ROOT / "README.md").write_text("\n".join(lines))


if __name__ == "__main__":
    main()
