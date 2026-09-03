"""Running the metrics over a campaign, one assessment at a time.

``measure`` reads the case directories an assessment wrote, applies the metrics
that assessment is about, and aggregates over seeds.  It never re-reads the
model: a measurement must be repeatable from the volumes alone, and a metric
that needed the model back would not be.

Every per-case row carries its own manifest fields, so a results file says what
it measured and not only what it found.  Aggregation is always mean and sample
standard deviation over the three seeds of one cell.
"""

from __future__ import annotations

import logging
from collections import defaultdict
from pathlib import Path

import numpy as np

from poregen.eval_v4 import metrics as M
from poregen.eval_v4.cases import (
    ASSEMBLY_OFFSETS,
    ASSEMBLY_REGION,
    OFF_MANIFOLD_TARGET,
    build_cases,
)
from poregen.eval_v4.io import Case, load_cases, repo_root, write_results
from poregen.eval_v4.manifest import Manifest

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Per-case blocks reused by several assessments
# ---------------------------------------------------------------------------

def case_identity(case: Case) -> dict:
    m = case.manifest
    return {
        "case": m.case,
        "model_run": m.model_run,
        "seed": m.seed,
        "volume_shape": list(m.volume_shape),
        "ddim_steps": m.ddim_steps,
        "chunk_tiles": list(m.chunk_tiles) if m.chunk_tiles else None,
        "window_stride": m.window_stride,
        "decode": m.decode,
        "decode_overlap": m.decode_overlap,
        "s_por": m.s_por,
        "s_nb": m.s_nb,
        "objective": m.objective,
        "cfg_rescale": m.cfg_rescale,
        "weights": m.weights,
        "checkpoint_step": m.checkpoint_step,
        "requested_global_phi": m.requested_global_phi,
        "wall_time_s": m.wall_time_s,
        "peak_gpu_memory_bytes": m.peak_gpu_memory_bytes,
        "git_commit": m.git_commit,
        "notes": m.notes,
    }


def measure_core(case: Case) -> dict:
    """The block every generated volume gets: phases, failure, seams.

    Runs on the whole generated canvas, in the frame the assembly grid is
    defined in.  A case that is ABOUT a sub-region still reports these on the
    canvas, because a seam plane only means anything in canvas coordinates.
    """
    label = case.label
    material = case.material_voxels()
    m = case.manifest
    row = {
        **case_identity(case),
        **M.phase_fractions(label, material, manifest=m),
        "failure": M.failure_flags(label, material, manifest=m),
        "degenerate": M.degenerate_cells(label, material, manifest=m),
        "seams": M.seam_metrics(case.xct, manifest=m, pore_logit=case.pore_logit),
    }
    if m.requested_global_phi is not None:
        row["porosity"] = M.porosity_error(label, material, manifest=m)
    return row


def _agg(rows, path, *, key=None) -> dict:
    """Mean and sd over seeds of one nested value."""
    vals = []
    for r in rows:
        v = r
        for p in path:
            v = (v or {}).get(p) if isinstance(v, dict) else None
        vals.append(v)
    return M.mean_sd(vals) if key is None else {key: M.mean_sd(vals)}


def _group(rows, keyfn) -> dict:
    out = defaultdict(list)
    for r in rows:
        out[keyfn(r)].append(r)
    return dict(out)


def _seam_summary(rows) -> dict:
    """The four seam ratios every assessment reports, averaged over seeds."""
    return {
        "seam_xct_ratio": _agg(rows, ("seams", "seam_xct_ratio")),
        "seam_pore_ratio": _agg(rows, ("seams", "seam_pore_ratio")),
        "seam_chunk_xct_ratio": _agg(rows, ("seams", "seam_chunk_xct_ratio")),
        "seam_chunk_pore_ratio": _agg(rows, ("seams", "seam_chunk_pore_ratio")),
    }


def _failure_rate(rows) -> dict:
    flags = [r["failure"] for r in rows]
    return {
        "failure_rate": float(np.mean([f["failed"] for f in flags])),
        "n": len(flags),
        "phi_collapsed": int(sum(f["phi_collapsed"] for f in flags)),
        "phi_saturated": int(sum(f["phi_saturated"] for f in flags)),
        "air_in_material": int(sum(f["air_in_material"] for f in flags)),
    }


# ---------------------------------------------------------------------------
# 1 - sampler
# ---------------------------------------------------------------------------

def measure_sampler(root, repo) -> dict:
    rows = [measure_core(c) for c in load_cases(root, "sampler")]
    if not rows:
        raise FileNotFoundError(f"no sampler volumes under {root}")
    cells = {}
    for key, grp in _group(
        rows, lambda r: (tuple(r["volume_shape"]), r["ddim_steps"])
    ).items():
        shape, steps = key
        cells[f"{shape[1]}x{shape[2]}x{shape[0]}_ddim{steps}"] = {
            "volume_shape": list(shape),
            "ddim_steps": steps,
            "n_seeds": len(grp),
            "porosity_error": _agg(grp, ("porosity", "error")),
            "porosity_abs_error": _agg(grp, ("porosity", "abs_error")),
            "delivered_phi": _agg(grp, ("porosity", "delivered_phi")),
            "air_fraction_interior": _agg(grp, ("air_fraction_interior",)),
            **_seam_summary(grp),
            "wall_time_s": _agg(grp, ("wall_time_s",)),
            "peak_gpu_memory_bytes": _agg(grp, ("peak_gpu_memory_bytes",)),
            **_failure_rate(grp),
        }
    return {
        "assessment": "sampler",
        "question": "What does the DDIM step count buy, and does it hold at scale?",
        "per_case": rows,
        "cells": cells,
    }


# ---------------------------------------------------------------------------
# 2 - global porosity
# ---------------------------------------------------------------------------

def measure_porosity_global(root, repo) -> dict:
    rows = [measure_core(c) for c in load_cases(root, "porosity_global")]
    if not rows:
        raise FileNotFoundError(f"no porosity_global volumes under {root}")

    in_range = [r for r in rows if r["requested_global_phi"] != OFF_MANIFOLD_TARGET]
    off = [r for r in rows if r["requested_global_phi"] == OFF_MANIFOLD_TARGET]

    levels = {}
    for target, grp in sorted(_group(in_range, lambda r: r["requested_global_phi"]).items()):
        abs_err = [r["porosity"]["abs_error"] for r in grp]
        levels[f"{target:g}"] = {
            "requested": target,
            "delivered": _agg(grp, ("porosity", "delivered_phi")),
            "abs_error": M.mean_sd(abs_err),
            "within_gate": float(np.mean([r["porosity"]["within_gate"] for r in grp])),
            "n_seeds": len(grp),
            **_failure_rate(grp),
        }
    fit = M.fit_ols(
        [r["porosity"]["requested_phi"] for r in in_range],
        [r["porosity"]["delivered_phi"] for r in in_range],
    )
    return {
        "assessment": "porosity_global",
        "question": "Does the delivered porosity follow the requested one, one to one?",
        "per_case": rows,
        "dose_response": {
            **fit,
            "gate": M.POROSITY_GATE,
            "frac_within_gate": float(
                np.mean([r["porosity"]["within_gate"] for r in in_range])
            ),
        },
        "levels": levels,
        "off_manifold": {
            "requested": OFF_MANIFOLD_TARGET,
            "note": (
                "Above the training range: cond_por clamps at phi = 0.107, so this "
                "row is a failure mode and is excluded from the fit."
            ),
            "conditioned_phi_after_clamp": (
                off[0]["notes"].get("conditioned_phi_after_clamp") if off else None
            ),
            "delivered": _agg(off, ("porosity", "delivered_phi")),
            "abs_error": _agg(off, ("porosity", "abs_error")),
            "air_fraction_interior": _agg(off, ("air_fraction_interior",)),
            "n_seeds": len(off),
            **(_failure_rate(off) if off else {}),
        },
    }


# ---------------------------------------------------------------------------
# 3 - local porosity
# ---------------------------------------------------------------------------

def measure_porosity_local(root, repo) -> dict:
    cases = load_cases(root, "porosity_local")
    if not cases:
        raise FileNotFoundError(f"no porosity_local volumes under {root}")
    rows = []
    for case in cases:
        row = measure_core(case)
        row["local"] = M.local_obedience(
            case.label, case.material_voxels(),
            manifest=case.manifest,
            requested_tiles=case.requested_phi_per_tile(),
        )
        row["field"] = (case.manifest.notes or {}).get("field")
        rows.append(row)

    fields = {}
    for name, grp in _group(rows, lambda r: r["field"]).items():
        fields[name] = {
            "n_seeds": len(grp),
            "within_volume_slope": _agg(grp, ("local", "within_volume_slope")),
            "within_volume_r2": _agg(grp, ("local", "within_volume_r2")),
            "per_cell_abs_error": _agg(grp, ("local", "per_cell", "abs_error_mean")),
            "per_cell_frac_within_gate": _agg(grp, ("local", "per_cell", "frac_within_gate")),
            "delivered_cell_sd": _agg(grp, ("local", "delivered_cell_sd")),
            "requested_cell_sd": _agg(grp, ("local", "requested_cell_sd")),
            "pooled_over_seeds": M.pooled_dose_fit(
                [np.asarray(r["local"]["cells_requested"]) for r in grp],
                [np.asarray(r["local"]["cells_delivered"]) for r in grp],
            ),
        }
    return {
        "assessment": "porosity_local",
        "question": "Do the pores land in the cells the field asked for them?",
        "note": (
            "The obedience score is the WITHIN-volume slope and R^2, after each "
            "volume's own mean is removed from both sides. The pooled fit is "
            "reported as context and is not obedience: most of its variance is "
            "the global dose response that assessment 2 already measures."
        ),
        "per_case": rows,
        "fields": fields,
    }


# ---------------------------------------------------------------------------
# 4 - guidance
# ---------------------------------------------------------------------------

def measure_cfg(root, repo) -> dict:
    cases = load_cases(root, "cfg")
    if not cases:
        raise FileNotFoundError(f"no cfg volumes under {root}")
    by_name = {c.manifest.case: c for c in cases}
    rows = []
    for case in cases:
        row = measure_core(case)
        row["arm"] = (case.manifest.notes or {}).get("arm")
        rows.append(row)

    por_rows = [r for r in rows if r["arm"] == "s_por"]
    nb_rows = [r for r in rows if r["arm"] == "s_nb"]

    por_cells = {}
    for (s_por, target), grp in sorted(
        _group(por_rows, lambda r: (r["s_por"], r["requested_global_phi"])).items()
    ):
        por_cells[f"spor{s_por:g}_target{target:g}"] = {
            "s_por": s_por,
            "requested": target,
            "n_seeds": len(grp),
            "delivered_phi": _agg(grp, ("porosity", "delivered_phi")),
            "abs_error": _agg(grp, ("porosity", "abs_error")),
            "air_fraction_interior": _agg(grp, ("air_fraction_interior",)),
            "degenerate_cell_fraction": _agg(grp, ("degenerate", "degenerate_cell_fraction")),
            **_seam_summary(grp),
            **_failure_rate(grp),
        }

    # The neighbour arm: the same seed at s_nb 0 and 1, compared in the slab
    # around the chunk plane, which is the only place neighbour conditioning
    # can first act.
    pairs = []
    for seed in sorted({r["seed"] for r in nb_rows}):
        a, b = by_name.get(f"snb0_seed{seed}"), by_name.get(f"snb1_seed{seed}")
        if a is None or b is None:
            continue
        period = M.chunk_period(a.manifest)
        slab = M.chunk_plane_slab(tuple(a.manifest.volume_shape), period)
        if not slab.any():
            raise ValueError(
                f"cfg seed {seed}: the s_nb volumes have no chunk plane "
                f"(shape {a.manifest.volume_shape}, chunk period {period}); the "
                "neighbour arm cannot be measured on them."
            )
        pairs.append({
            "seed": seed,
            "chunk_period": list(period),
            "slab_voxels": int(slab.sum()),
            "pore_dice_chunk_plane": M.pore_dice(a.label, b.label, slab),
            "pore_dice_whole_volume": M.pore_dice(a.label, b.label),
        })

    nb_cells = {}
    for s_nb, grp in sorted(_group(nb_rows, lambda r: r["s_nb"]).items()):
        nb_cells[f"snb{s_nb:g}"] = {
            "s_nb": s_nb,
            "n_seeds": len(grp),
            "delivered_phi": _agg(grp, ("porosity", "delivered_phi")),
            "air_fraction_interior": _agg(grp, ("air_fraction_interior",)),
            **_seam_summary(grp),
            **_failure_rate(grp),
        }
    return {
        "assessment": "cfg",
        "question": "What do the two guidance scales do, and do the neighbours act?",
        "per_case": rows,
        "s_por_cells": por_cells,
        "s_nb_cells": nb_cells,
        "s_nb_pairs": pairs,
        "s_nb_summary": {
            "pore_dice_chunk_plane": M.mean_sd([p["pore_dice_chunk_plane"] for p in pairs]),
            "pore_dice_whole_volume": M.mean_sd([p["pore_dice_whole_volume"] for p in pairs]),
            "reading": (
                "A Dice near 1 across the chunk plane means turning the neighbour "
                "arm off changed nothing there, so the arm is inert."
            ),
        },
    }


# ---------------------------------------------------------------------------
# 5 - layup
# ---------------------------------------------------------------------------

def measure_layup(root, repo) -> dict:
    cases = load_cases(root, "layup")
    if not cases:
        raise FileNotFoundError(f"no layup volumes under {root}")
    rows = []
    for case in cases:
        row = measure_core(case)
        row["layup"] = (case.manifest.notes or {}).get("layup")
        row["recovery"] = M.layup_recovery(case.xct, case.label, manifest=case.manifest, repo=repo)
        rows.append(row)

    floor = M.layup_floor(repo)
    layups = {}
    for name, grp in _group(rows, lambda r: r["layup"]).items():
        entry = {
            "n_seeds": len(grp),
            "requested_deg": grp[0]["recovery"]["requested_deg"],
            "requested_ply_count": grp[0]["recovery"]["requested_ply_count"],
            "ply_thickness_vox": grp[0]["recovery"]["ply_thickness_vox"],
            "delivered_phi": _agg(grp, ("porosity", "delivered_phi")),
            "readers": {},
        }
        for reader in ("fft_slice", "pore_axes"):
            available = [r for r in grp if r["recovery"]["readers"].get(reader, {}).get("available")]
            if not available:
                entry["readers"][reader] = {"available": False}
                continue
            per_ply = np.array(
                [r["recovery"]["readers"][reader]["per_ply_hit"] for r in available], float
            )
            entry["readers"][reader] = {
                "available": True,
                "median_abs_error_deg": _agg(available, ("recovery", "readers", reader, "median_abs_error_deg")),
                "strict_class_accuracy": _agg(available, ("recovery", "readers", reader, "strict_class_accuracy")),
                "frac_within_10": _agg(available, ("recovery", "readers", reader, "frac_within_10")),
                "recovered_ply_count": _agg(available, ("recovery", "readers", reader, "recovered_ply_count")),
                "per_ply_hit_rate": per_ply.mean(axis=0).tolist(),
                "real_floor": floor.get(reader),
            }
        layups[name] = entry
    return {
        "assessment": "layup",
        "question": "Can the requested stacking sequence be read back out of the volume?",
        "note": (
            "Scored DIRECTLY: no offset, no sign flip, no face reversal. The real "
            "floor is campaign 08's reader floor on real scans, where pore_axes read "
            "the STORED mask - a ceiling no predicted-mask reader can beat."
        ),
        "real_floor": floor,
        "per_case": rows,
        "layups": layups,
    }


# ---------------------------------------------------------------------------
# 6 - assembly
# ---------------------------------------------------------------------------

def measure_assembly(root, repo) -> dict:
    """Seams and cross-head disagreement on the SAMPLER volumes, plus the
    window-phase pair this assessment generates itself."""
    sampler_cases_ = load_cases(root, "sampler")
    if not sampler_cases_:
        raise FileNotFoundError(
            f"assembly reads the sampler volumes; none found under {root}/sampler"
        )
    detector = M.grey_air_detector(repo)
    rows = []
    for case in sampler_cases_:
        row = measure_core(case)
        row["cross_head"] = M.cross_head_disagreement(
            case.xct, case.label, case.material_voxels(),
            manifest=case.manifest, detector=detector,
        )
        rows.append(row)

    cells = {}
    for (shape, steps), grp in _group(
        rows, lambda r: (tuple(r["volume_shape"]), r["ddim_steps"])
    ).items():
        cells[f"{shape[1]}x{shape[2]}x{shape[0]}_ddim{steps}"] = {
            "volume_shape": list(shape),
            "ddim_steps": steps,
            "n_seeds": len(grp),
            **_seam_summary(grp),
            "cross_head_disagreement": _agg(grp, ("cross_head", "disagreement_fraction")),
            "cross_head_disagreement_interior": _agg(
                grp, ("cross_head", "disagreement_fraction_interior")),
            "grey_air_claimed_by_label": _agg(grp, ("cross_head", "grey_air_claimed_by_label")),
        }

    phase = _measure_window_phase(root)
    return {
        "assessment": "assembly",
        "question": "Is the volume one object, or a grid of independently drawn blocks?",
        "detector": {
            "t_abs": detector.t_abs, "min_cc": detector.min_cc,
            "edge_vox": detector.edge_vox, "source": detector.source,
        },
        "per_case": rows,
        "cells": cells,
        "window_phase": phase,
        "vae_tile_decode_control": M.vae_tile_decode_control(repo),
    }


def _measure_window_phase(root) -> dict:
    """Pore Dice for the same region assembled on two window grids."""
    cases = {c.manifest.case: c for c in load_cases(root, "assembly")}
    if not cases:
        return {"available": False,
                "note": "run `eval_v4 generate assembly` for the window-phase pair"}
    lo, hi = ASSEMBLY_OFFSETS
    pairs = []
    for seed in sorted({c.manifest.seed for c in cases.values()}):
        a, b = cases.get(f"offset{lo}_seed{seed}"), cases.get(f"offset{hi}_seed{seed}")
        if a is None or b is None:
            continue
        ra = M.crop_region(a.label, a.manifest)
        rb = M.crop_region(b.label, b.manifest)
        if ra.shape != rb.shape:
            raise ValueError(
                f"window-phase seed {seed}: the two regions are {ra.shape} and "
                f"{rb.shape}. The pair must be the SAME region assembled two ways, "
                "or the Dice compares two different pieces of material."
            )
        pairs.append({
            "seed": seed,
            "offsets": [lo, hi],
            "region_shape": list(ra.shape),
            "pore_dice": M.pore_dice(ra, rb),
            "phi_offset0": float((ra == 1).mean()),
            "phi_offset32": float((rb == 1).mean()),
            "phi_difference": float((ra == 1).mean() - (rb == 1).mean()),
        })
    return {
        "available": bool(pairs),
        "region_shape": pairs[0]["region_shape"] if pairs else list(ASSEMBLY_REGION),
        "offsets": list(ASSEMBLY_OFFSETS),
        "note": (
            "The sampler anchors window origins at the chunk origin, so the shift "
            "is realised by translating the REQUEST inside a 256-cubed canvas: the "
            "specimen box, the orientation profile and the uniform porosity request "
            "all move with the region, and only the assembly grid stays put. At "
            "offset 32 the chunk plane at canvas voxel 192 runs through the region "
            "at region coordinate 160."
        ),
        "pairs": pairs,
        "pore_dice": M.mean_sd([p["pore_dice"] for p in pairs]),
        "phi_difference": M.mean_sd([p["phi_difference"] for p in pairs]),
    }


# ---------------------------------------------------------------------------
# 7 - geometry
# ---------------------------------------------------------------------------

def measure_geometry(root, repo) -> dict:
    cases = load_cases(root, "geometry")
    if not cases:
        raise FileNotFoundError(f"no geometry volumes under {root}")
    rows = []
    for case in cases:
        material = case.material_voxels()
        row = measure_core(case)
        row["geometry"] = M.geometry_agreement(case.label, material, manifest=case.manifest)
        rows.append(row)
    return {
        "assessment": "geometry",
        "question": "Does the model carve air where the material map asks for it?",
        "per_case": rows,
        "summary": {
            "n_seeds": len(rows),
            "dice_air": _agg(rows, ("geometry", "dice_air")),
            "precision_air": _agg(rows, ("geometry", "precision_air")),
            "recall_air": _agg(rows, ("geometry", "recall_air")),
            "air_fraction_inside_material": _agg(rows, ("geometry", "air_fraction_inside_material")),
            "air_fraction_outside_material": _agg(rows, ("geometry", "air_fraction_outside_material")),
            "phi_pore_inside_material": _agg(rows, ("geometry", "phi_pore_inside_material")),
            **_failure_rate(rows),
        },
    }


# ---------------------------------------------------------------------------
# 8 - microstructure statistics
# ---------------------------------------------------------------------------

def _micro_reference(root) -> dict[float, dict[str, list]]:
    """The matched real crops, as ``{level: {"a": [...], "b": [...]}}``.

    Written by ``eval_v4 real-floor --shapes micro`` and read back here, so the
    measure stage still needs only the volumes: the dataset was consulted once,
    at floor time, and what it produced is on disk beside everything else.
    """
    out: dict[float, dict[str, list]] = {}
    for case in load_cases(root, "real_floor"):
        notes = case.manifest.notes or {}
        level = notes.get("micro_level")
        if level is None:
            continue
        out.setdefault(float(level), {}).setdefault(notes["micro_pair"], []).append(case)
    return out


def measure_microstructure(root, repo) -> dict:
    """Generated microstructure against real, and real against real.

    Every distance here is read three ways: what the generated set scores
    against real material, what two disjoint crops of one real panel score
    against each other, and the ratio.  The floor is not a formality — a
    Wasserstein distance between two finite samples of the SAME material is not
    zero, so without it a generated number cannot be called large or small.
    """
    from poregen.eval_v4 import microstructure as MS  # noqa: PLC0415
    from poregen.eval_v4.generate import DEFAULT_LATENTS_ROOT  # noqa: PLC0415

    cases = load_cases(root, "microstructure")
    if not cases:
        raise FileNotFoundError(f"no microstructure volumes under {root}")
    reference = _micro_reference(root)
    if not reference:
        raise FileNotFoundError(
            f"{root}/real_floor holds no matched-porosity reference crops. Run "
            f"`eval_v4 real-floor --root {root} --shapes micro` first: every "
            "statistic in this assessment is a distance, and a distance with no "
            "real-vs-real floor cannot be read."
        )

    rows = [measure_core(c) for c in cases]
    for row, case in zip(rows, cases):
        row["micro_level"] = (case.manifest.notes or {}).get(
            "micro_level", case.manifest.requested_global_phi)

    by_level = _group(zip(rows, cases), lambda rc: float(rc[0]["micro_level"]))
    latents_root = Path(
        (cases[0].manifest.notes or {}).get("latents_root")
        or Path(repo) / DEFAULT_LATENTS_ROOT
    )

    levels: dict[str, dict] = {}
    for level in sorted(by_level):
        gen = [c for _, c in by_level[level]]
        ref = reference.get(level, {})
        real_a, real_b = ref.get("a", []), ref.get("b", [])
        if not real_a or not real_b:
            levels[f"{level:g}"] = {
                "requested": level, "n_generated": len(gen),
                "available": False,
                "reason": "the real reference for this level has no disjoint pair",
            }
            continue

        gen_p = [MS.profile_volume(c, group="generated") for c in gen]
        a_p = [MS.profile_volume(c, group="real_a") for c in real_a]
        b_p = [MS.profile_volume(c, group="real_b") for c in real_b]

        against = MS.compare_sets(gen_p, a_p + b_p)
        floor = MS.compare_sets(a_p, b_p)
        fid_gen = MS.fid_between(gen, real_a + real_b, seed=int(level * 1e6))
        fid_floor = MS.fid_between(real_a, real_b, seed=int(level * 1e6) + 1)
        memo_gen = MS.memorisation(gen, latents_root, repo=repo)
        memo_floor = MS.memorisation(real_a + real_b, latents_root, repo=repo)

        misses = [(c.manifest.notes or {}).get("phi_miss") for c in real_a + real_b]
        levels[f"{level:g}"] = {
            "available": True,
            "requested": level,
            "n_generated": len(gen),
            "n_real_a": len(real_a), "n_real_b": len(real_b),
            "generated_phi": _agg(
                [r for r, _ in by_level[level]], ("porosity", "delivered_phi")),
            "real_phi": M.mean_sd([p.phi for p in a_p + b_p]),
            "real_phi_miss_max": max((m for m in misses if m is not None), default=None),
            "real_panels": sorted({(c.manifest.notes or {}).get("panel_id")
                                   for c in real_a + real_b}),
            "generated_vs_real": against,
            "real_vs_real": floor,
            "ratio": {
                "s2_w1": MS.ratio(against["s2_w1"], floor["s2_w1"]),
                "psd_w1": MS.ratio(against["psd_w1"], floor["psd_w1"]),
                "ripley_log_ratio": MS.ratio(
                    against["ripley_log_ratio"], floor["ripley_log_ratio"]),
                "fid": MS.ratio(fid_gen.get("mean"), fid_floor.get("mean")),
                # A memorisation ratio is the other way round: the generated
                # patches should be at least as FAR from the training set as
                # held-out real material is, so 1 or more is the healthy side.
                "memorisation_nn_distance": MS.ratio(
                    memo_gen.get("nn_distance_mean"),
                    memo_floor.get("nn_distance_mean")),
            },
            "fid_generated_vs_real": fid_gen,
            "fid_real_vs_real": fid_floor,
            "memorisation_generated": memo_gen,
            "memorisation_real": memo_floor,
            "profiles": [p.summary() for p in gen_p + a_p + b_p],
        }

    return {
        "assessment": "microstructure",
        "question": "Does the generated microstructure have the statistics of real material?",
        "note": (
            "Five distribution distances, each reported three ways: generated "
            "against real, real against real (two disjoint crops of ONE panel), "
            "and the ratio. A ratio of 1 means the generated set is as close to "
            "real material as real material is to itself, which is as close as "
            "this measurement can tell. The memorisation ratio reads the other "
            "way: 1 or more means the generated patches are no nearer the "
            "training latents than held-out real patches are. Every level is "
            "scored against real crops matched to ITS porosity, because a "
            "porosity gap would otherwise be read as a texture gap."
        ),
        "geometry": {
            "s2_window": MS.S2_WINDOW, "s2_r_max": MS.S2_R_MAX,
            "ripley_r_max": MS.RIPLEY_R_MAX,
            "fid_crop": MS.FID_CROP, "fid_extractor": MS.FID_EXTRACTOR,
            "connectivity": "6 (face-adjacent) for every connected-component count",
        },
        "per_case": rows,
        "levels": levels,
    }


MEASURERS = {
    "sampler": measure_sampler,
    "porosity_global": measure_porosity_global,
    "porosity_local": measure_porosity_local,
    "cfg": measure_cfg,
    "layup": measure_layup,
    "assembly": measure_assembly,
    "geometry": measure_geometry,
    "microstructure": measure_microstructure,
}


def measure(root: str | Path, assessment: str, repo: str | Path | None = None) -> dict:
    """Measure one assessment and write ``<root>/<assessment>/results.json``."""
    if assessment not in MEASURERS:
        raise KeyError(f"unknown assessment {assessment!r}; choose from {sorted(MEASURERS)}")
    repo = Path(repo) if repo else repo_root()
    results = MEASURERS[assessment](Path(root), repo)
    results["n_cases_expected"] = len(build_cases(assessment, repo))
    results["n_cases_measured"] = len(results["per_case"])
    write_results(root, assessment, results)
    logger.info(
        "%s: measured %d of %d cases",
        assessment, results["n_cases_measured"], results["n_cases_expected"],
    )
    return results


def manifest_check(root: str | Path, repo: str | Path | None = None) -> dict:
    """Validate every manifest in a campaign against the schema and the cases.

    Three failure modes are reported separately, because they mean different
    things: a manifest that does not parse, a volume whose array contradicts its
    manifest, and a case the assessment defines but the campaign does not hold.
    """
    from poregen.eval_v4.cases import ASSESSMENTS  # noqa: PLC0415
    from poregen.eval_v4.io import iter_case_dirs, load_u8  # noqa: PLC0415

    repo = Path(repo) if repo else repo_root()
    root = Path(root)
    report: dict = {"root": str(root), "assessments": {}, "ok": True}
    for name in [*ASSESSMENTS, "real_floor"]:
        dirs = list(iter_case_dirs(root, name))
        if not dirs and name == "real_floor":
            continue
        entry = {"n_cases": len(dirs), "invalid": [], "mismatched": [], "missing": []}
        found = set()
        for d in dirs:
            try:
                man = Manifest.read(d)
            except Exception as exc:  # noqa: BLE001 - the report is the product
                entry["invalid"].append({"case": d.name, "error": str(exc)})
                continue
            found.add(man.case)
            for fname, kind in (("volume.tif", "volume"), ("label.tif", "label")):
                path = d / fname
                if not path.exists():
                    entry["mismatched"].append({"case": man.case, "error": f"no {fname}"})
                    continue
                try:
                    man.check_array(load_u8(path), "manifest_check", kind)
                except Exception as exc:  # noqa: BLE001
                    entry["mismatched"].append({"case": man.case, "error": str(exc)})
        if name in ASSESSMENTS:
            expected = {c.name for c in build_cases(name, repo)}
            entry["missing"] = sorted(expected - found)
            entry["unexpected"] = sorted(found - expected)
        entry["ok"] = not (entry["invalid"] or entry["mismatched"] or entry["missing"])
        report["assessments"][name] = entry
        report["ok"] = report["ok"] and entry["ok"]
    return report
