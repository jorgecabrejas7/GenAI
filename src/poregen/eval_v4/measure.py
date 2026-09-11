"""Running the metrics over a campaign, one assessment at a time.

``measure`` reads the case directories an assessment wrote, applies the metrics
that assessment is about, and aggregates over seeds.  It never re-reads the
DENOISER: a measurement must be repeatable from the volumes alone, and a metric
that needed the sampler back would not be.

The one thing it does reload is the frozen VAE, and only for assessment 8's
memorisation check, which has to put a generated volume back into the latent
space the training store lives in.  That check is a GPU pass over the whole
272 GB store, so ``measure microstructure`` is the only measure stage that is
not cheap and not CPU-only.

Every per-case row carries its own manifest fields, so a results file says what
it measured and not only what it found.  Aggregation is always mean and sample
standard deviation over the three seeds of one cell.
"""

from __future__ import annotations

import json
import logging
from collections import defaultdict
from pathlib import Path

import numpy as np

from poregen.eval_v4 import field_stats as FS
from poregen.eval_v4 import metrics as M
from poregen.eval_v4.cases import (
    ASSEMBLY_OFFSETS,
    ASSEMBLY_REGION,
    ASSESSMENTS,
    OFF_MANIFOLD_TARGET,
    build_cases,
)
from poregen.eval_v4.io import TILE, Case, load_cases, repo_root, write_results
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
    """Pore Dice for the same region assembled on the offset grids.

    Offset 0 is the reference and every other offset is read against it, so a
    row is one (seed, offset) pair.  The offsets isolate different things -
    32 is a whole window stride and moves only the chunk alignment, 16 is half
    a stride and moves the window phase as well - and mixing them into one mean
    would hide exactly the separation they were added for.
    """
    cases = {c.manifest.case: c for c in load_cases(root, "assembly")}
    if not cases:
        return {"available": False,
                "note": "run `eval_v4 generate assembly` for the offset triple"}
    ref, others = ASSEMBLY_OFFSETS[0], ASSEMBLY_OFFSETS[1:]
    pairs = []
    for seed in sorted({c.manifest.seed for c in cases.values()}):
        a = cases.get(f"offset{ref}_seed{seed}")
        if a is None:
            continue
        ra = M.crop_region(a.label, a.manifest)
        for off in others:
            b = cases.get(f"offset{off}_seed{seed}")
            if b is None:
                continue
            rb = M.crop_region(b.label, b.manifest)
            if ra.shape != rb.shape:
                raise ValueError(
                    f"window-phase seed {seed}, offset {off}: the two regions are "
                    f"{ra.shape} and {rb.shape}. The pair must be the SAME region "
                    "assembled two ways, or the Dice compares two different pieces "
                    "of material."
                )
            pairs.append({
                "seed": seed,
                "offset": off,
                "offsets": [ref, off],
                "region_shape": list(ra.shape),
                "pore_dice": M.pore_dice(ra, rb),
                "phi_reference": float((ra == 1).mean()),
                "phi_offset": float((rb == 1).mean()),
                "phi_difference": float((ra == 1).mean() - (rb == 1).mean()),
            })
    by_offset = {}
    for off in others:
        rows = [p for p in pairs if p["offset"] == off]
        if rows:
            by_offset[str(off)] = {
                "n_seeds": len(rows),
                "pore_dice": M.mean_sd([p["pore_dice"] for p in rows]),
                "phi_difference": M.mean_sd([p["phi_difference"] for p in rows]),
            }
    return {
        "available": bool(pairs),
        "region_shape": pairs[0]["region_shape"] if pairs else list(ASSEMBLY_REGION),
        "reference_offset": ref,
        "offsets": list(ASSEMBLY_OFFSETS),
        "note": (
            "The sampler anchors window origins at the chunk origin, so the shift "
            "is realised by translating the REQUEST inside a 256-cubed canvas: the "
            "specimen box, the orientation profile, the uniform porosity request "
            "and the frame every noise draw is taken in all move with the region, "
            "and only the assembly grid stays put. Offset 32 is a whole window "
            "stride, so it keeps the window phase and moves only the chunk "
            "alignment - the chunk plane at canvas voxel 192 runs through the "
            "region at region coordinate 160. Offset 16 is half a stride, so it "
            "moves the window phase too (region-relative window origins 16, 48, "
            "80 ...) with a chunk plane at region coordinate 176."
        ),
        "pairs": pairs,
        "by_offset": by_offset,
    }


# ---------------------------------------------------------------------------
# 7 - geometry
# ---------------------------------------------------------------------------

def measure_geometry(root, repo) -> dict:
    cases = load_cases(root, "geometry")
    if not cases:
        raise FileNotFoundError(f"no geometry volumes under {root}")
    rows, sphere_rows = [], []
    for case in cases:
        notes = case.manifest.notes or {}
        material = case.material_voxels()
        row = measure_core(case)
        row["request"] = notes.get("request", "notch_hole")
        if row["request"] == "sphere":
            # Exploratory: descriptive numbers only, kept apart from the gated
            # notch/hole rows so a reader cannot mistake one for the other.
            row["sphere"] = M.sphere_agreement(
                case.label, material,
                radius=float(notes.get("radius_vox", 80)),
                requested_phi=case.manifest.requested_global_phi)
            row["ddim_steps"] = case.manifest.ddim_steps
            row["scale"] = notes.get("scale")
            sphere_rows.append(row)
            continue
        row["geometry"] = M.geometry_agreement(case.label, material, manifest=case.manifest)
        rows.append(row)
    sphere_block = None
    if sphere_rows:
        sphere_block = {
            "exploratory": True,
            "no_gate_because": (
                "there is no real spherical coupon, so there is no floor to read "
                "these against and any threshold would be invented"
            ),
            "cond_dist6_note": (
                "the six face distances are computed from the sphere's BOUNDING "
                "BOX, as for every case, so they describe a cube rather than the "
                "curved surface — part of what this probes"
            ),
            "n_cases": len(sphere_rows),
            "dice_air": _agg(sphere_rows, ("sphere", "dice_air")),
            "air_fraction_inside_sphere": _agg(sphere_rows, ("sphere", "air_fraction_inside_sphere")),
            "air_fraction_outside_sphere": _agg(sphere_rows, ("sphere", "air_fraction_outside_sphere")),
            "phi_pore_inside_sphere": _agg(sphere_rows, ("sphere", "phi_pore_inside_sphere")),
            "phi_error": _agg(sphere_rows, ("sphere", "phi_error")),
            "radial_surface_error_vox": _agg(sphere_rows, ("sphere", "radial_surface", "error_vox")),
            "per_case": sphere_rows,
        }
    if not rows:
        raise FileNotFoundError(f"no gated geometry volumes under {root}")
    return {
        "assessment": "geometry",
        "question": "Does the model carve air where the material map asks for it?",
        "sphere_exploratory": sphere_block,
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
# 9 - the specimen surface
# ---------------------------------------------------------------------------

def _surface_floor(root) -> dict | None:
    """Roughness of the REAL top/bottom surfaces, the floor the generated ones are read against.

    Written by ``eval_v4 real-floor --shapes surface``; absent until that has
    run, in which case the generated roughness is reported with no floor beside
    it and the summary says so. A roughness number with no floor says only that
    a surface is not perfectly flat, which no real surface is either.
    """
    from poregen.eval_v4.real_floor import SURFACE_FLOOR_FILE  # noqa: PLC0415

    path = Path(root) / "real_floor" / SURFACE_FLOOR_FILE
    if not path.exists():
        return None
    try:
        d = json.loads(path.read_text())
    except Exception as exc:                            # noqa: BLE001
        logger.warning("could not read %s: %s", path, exc)
        return None
    if d.get("sa_mean") is None:
        return None
    keys = ("n_volumes", "n_faces", "sa_mean", "sa_sd", "sq_mean", "sq_sd",
            "detrended_sa_mean", "correlation_length_vox_mean")
    return {k: d[k] for k in keys if k in d}


def measure_surface(root, repo) -> dict:
    cases = load_cases(root, "surface")
    if not cases:
        raise FileNotFoundError(f"no surface volumes under {root}")
    floor = _surface_floor(root)
    floor_sa = (floor or {}).get("detrended_sa_mean") or (floor or {}).get("sa_mean")

    rows = []
    for case in cases:
        notes = case.manifest.notes or {}
        material = case.material_voxels()
        # The requested interface is read back FROM THE REQUEST ITSELF rather
        # than rebuilt from the generator's parameters: for a rough case the
        # request is a height field, and re-deriving it would risk measuring
        # against a field the model was never shown.
        lo_req = M.height_map(material, face="lower")
        hi_req = M.height_map(material, face="upper")
        row = measure_core(case)
        row["ddim_steps"] = case.manifest.ddim_steps
        row["scale"] = notes.get("scale")
        row["request"] = notes.get("request", "flat")
        row["surface"] = M.surface_agreement(
            case.label, material, z_lo=lo_req, z_hi=hi_req, xct=case.xct)
        for face in ("lower", "upper"):
            gen_sa = row["surface"][face]["roughness_sa"]
            row["surface"][face]["roughness_ratio_to_real_floor"] = (
                (gen_sa / floor_sa) if gen_sa is not None and floor_sa else None)
        rows.append(row)

    def block(sel, kind):
        if not sel:
            return None
        out = {
            "n_cases": len(sel),
            "air_fraction_outside_box": _agg(sel, ("surface", "air_fraction_outside_box")),
            "air_fraction_inside_box": _agg(sel, ("surface", "air_fraction_inside_box")),
            "dark_but_material": _agg(sel, ("surface", "dark_but_material")),
            "dark_but_material_excluding_rim": _agg(
                sel, ("surface", "dark_but_material_excluding_rim")),
        }
        for face in ("lower", "upper"):
            out[face] = {
                "error_abs_mean": _agg(sel, ("surface", face, "error_abs_mean")),
                "roughness_sa": _agg(sel, ("surface", face, "roughness_sa")),
                "requested_roughness_sa": _agg(sel, ("surface", face, "requested_roughness_sa")),
                "roughness_ratio_to_requested": _agg(
                    sel, ("surface", face, "roughness_ratio_to_requested")),
                "roughness_ratio_to_real_floor": _agg(
                    sel, ("surface", face, "roughness_ratio_to_real_floor")),
                "n_outliers": _agg(sel, ("surface", face, "outliers", "n_outliers")),
                "outlier_fraction": _agg(sel, ("surface", face, "outliers", "fraction")),
                "outlier_clusters": _agg(sel, ("surface", face, "outliers", "n_clusters")),
                "outliers_with_pore_fraction": _agg(
                    sel, ("surface", face, "outliers", "with_pore_at_face_fraction")),
            }
        # The gates differ by request type ON PURPOSE. A flat request asks a
        # controllability question and a rough one asks a realism question, and
        # scoring both the same way is how a razor-flat surface passed
        # everything at step 74000.
        if kind == "flat":
            errs = [out[f]["error_abs_mean"].get("mean") for f in ("lower", "upper")]
            out["gate"] = {
                "kind": "controllability",
                "position_error_lt_4vox": all(
                    e is not None and e < 4.0 for e in errs),
            }
        else:
            def within(key):
                vals = [out[f][key].get("mean") for f in ("lower", "upper")]
                return all(v is not None and 0.5 <= v <= 2.0 for v in vals)
            out["gate"] = {
                "kind": "realism",
                "sa_ratio_to_requested_in_0.5_2": within("roughness_ratio_to_requested"),
                "sa_ratio_to_real_floor_in_0.5_2": (
                    within("roughness_ratio_to_real_floor") if floor_sa else None),
                "real_floor_available": bool(floor_sa),
            }
        return out

    flat_rows = [r for r in rows if r["request"] == "flat"]
    rough_rows = [r for r in rows if r["request"] == "rough"]
    return {
        "assessment": "surface",
        "question": ("Does the model render air where the material map asks, put "
                     "the interface where requested, and give it the texture real "
                     "material has?"),
        "gates": {
            "flat (control)": "position |error| < 4 voxels — controllability",
            "rough": "Sa ratio to the requested field in [0.5, 2] AND to the real "
                     "floor in [0.5, 2] — realism",
        },
        "why_two_requests": (
            "At step 74000 the flat request produced Sa 0.03 voxels against a "
            "real floor of order 1 voxel — far flatter than any real surface — "
            "while passing every position gate. A flat request is a map of 0/1 "
            "cells, i.e. a plane exactly on the latent grid, so the model may "
            "simply have obeyed it. The rough request carries fractional rim "
            "cells like real material, and separates 'obeyed an unrealistic "
            "request' from 'cannot make a rough surface'."
        ),
        "real_surface_floor": floor,
        "floor_note": (None if floor else
                       "no real surface floor on disk — run `eval_v4 real-floor "
                       "--shapes surface`; without it the realism gate cannot be "
                       "scored and reports null rather than passing by default."),
        "per_case": rows,
        "summary": {
            "flat": block(flat_rows, "flat"),
            "rough": block(rough_rows, "rough"),
            **_failure_rate(rows),
        },
    }


# ---------------------------------------------------------------------------
# 10 - assembly when the volume spans chunks on every axis
# ---------------------------------------------------------------------------

def measure_multichunk(root, repo) -> dict:
    """Does the answer hold where three chunk planes cross?

    measure_core already separates the two seam families — window planes at
    period 64 and chunk planes at the case's own chunk period, both judged
    against the same interior baseline — so they are reported side by side
    here rather than recomputed.

    The chunk period is DERIVED from each case's chunk_tiles, not assumed. With
    the 2-tile chunks these cases use it is 128 voxels, so the planes fall at
    128 and 256 in a 384 canvas; a 3-tile chunk would put a single plane at 192.
    Hard-coding either would silently measure the wrong planes if the chunking
    ever changed.
    """
    cases = load_cases(root, "multichunk")
    if not cases:
        raise FileNotFoundError(f"no multichunk volumes under {root}")
    rows = []
    for case in cases:
        notes = case.manifest.notes or {}
        material = case.material_voxels()
        row = measure_core(case)
        row["request"] = notes.get("request")
        row["ddim_steps"] = case.manifest.ddim_steps
        row["scale"] = notes.get("scale")
        row["chunk_tiles"] = list(case.manifest.chunk_tiles or ())
        # Pore agreement ACROSS each chunk plane: the slabs either side are
        # produced by different chunk solves, so a mismatch here is assembly,
        # not texture.
        row["pore_dice_across_chunk_planes"] = M.pore_dice_across_planes(
            case.label, M.chunk_period(case.manifest))
        if row["request"] == "sphere":
            row["sphere"] = M.sphere_agreement(
                case.label, material,
                radius=float(notes.get("radius_vox", 160)),
                requested_phi=case.manifest.requested_global_phi,
                by_octant=True)
        if row["request"] == "rough":
            lo = M.height_map(material, face="lower")
            hi = M.height_map(material, face="upper")
            row["surface"] = M.surface_agreement(
                case.label, material, z_lo=lo, z_hi=hi, xct=case.xct)
        rows.append(row)

    def by_request(kind):
        sel = [r for r in rows if r["request"] == kind]
        if not sel:
            return None
        out = {
            "n_cases": len(sel),
            "window_plane_seam_xct": _agg(sel, ("seams", "seam_xct_ratio")),
            "chunk_plane_seam_xct": _agg(sel, ("seams", "seam_chunk_xct_ratio")),
            "window_plane_seam_pore": _agg(sel, ("seams", "seam_pore_ratio")),
            "chunk_plane_seam_pore": _agg(sel, ("seams", "seam_chunk_pore_ratio")),
            "pore_dice_across_chunk_planes": _agg(sel, ("pore_dice_across_chunk_planes", "mean")),
            **_failure_rate(sel),
        }
        if kind == "sphere":
            out["radial_surface_error_vox"] = _agg(sel, ("sphere", "radial_surface", "error_vox"))
            out["octant_spread_vox"] = _agg(sel, ("sphere", "radial_surface_by_octant", "spread_vox"))
        if kind == "rough":
            for face in ("lower", "upper"):
                out[f"{face}_roughness_ratio_to_requested"] = _agg(
                    sel, ("surface", face, "roughness_ratio_to_requested"))
        return out

    return {
        "assessment": "multichunk",
        "question": ("Does the assembly hold where the volume spans chunks on "
                     "ALL THREE axes, not only in x and y?"),
        "not_physics": (
            "No specimen in the dataset is thicker than about 330 voxels, so a "
            "384-voxel-deep volume asks for material that does not exist. These "
            "cases test ASSEMBLY across chunk planes in z; nothing here is "
            "evidence about thick-specimen microstructure."
        ),
        "chunk_plane_note": (
            "The chunk period is derived per case from chunk_tiles. With the "
            "2-tile chunks used here it is 128 voxels, so the planes fall at 128 "
            "and 256 in a 384 canvas — not at 192, which would be the 3-tile "
            "chunk the production sampler defaults to."
        ),
        "per_case": rows,
        "summary": {k: by_request(k) for k in ("box", "sphere", "rough")},
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


def measure_microstructure(root, repo, *, allow_busy_gpu: bool = False) -> dict:
    """Generated microstructure against real, and real against real.

    Every distance here is read three ways: what the generated set scores
    against real material, what two disjoint crops of one real panel score
    against each other, and the ratio.  The floor is not a formality — a
    Wasserstein distance between two finite samples of the SAME material is not
    zero, so without it a generated number cannot be called large or small.

    The memorisation block rides along in this assessment but is measured on
    other volumes: it searches the whole training store for the nearest
    neighbour of every 64-cubed patch of the ``sampler``, ``porosity_global``,
    ``multichunk`` and ``assembly_modes`` volumes.  The first two are the
    production operating point and there are 57 of them, so they are where a
    copy would matter; the last two are the only volumes that cross a chunk
    plane, and therefore the only ones whose windows are ever denoised with
    UNKNOWN neighbours.  The nine microstructure volumes are too few a sample
    to answer the question either way.
    """
    from poregen.eval_v4 import memorisation as MEMO  # noqa: PLC0415
    from poregen.eval_v4 import microstructure as MS  # noqa: PLC0415

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
            },
            "fid_generated_vs_real": fid_gen,
            "fid_real_vs_real": fid_floor,
            "profiles": [p.summary() for p in gen_p + a_p + b_p],
        }

    return {
        "assessment": "microstructure",
        "question": "Does the generated microstructure have the statistics of real material?",
        "note": (
            "Four distribution distances, each reported three ways: generated "
            "against real, real against real (two disjoint crops of ONE panel), "
            "and the ratio. A ratio of 1 means the generated set is as close to "
            "real material as real material is to itself, which is as close as "
            "this measurement can tell. Every level is scored against real crops "
            "matched to ITS porosity, because a porosity gap would otherwise be "
            "read as a texture gap. The memorisation block is the fifth "
            "statistic and reads differently: it is a full-store nearest-"
            "neighbour search over the sampler and porosity_global volumes, and "
            "what makes it readable is the real-val floor inside it, not the "
            "real-vs-real crops."
        ),
        "geometry": {
            "s2_window": MS.S2_WINDOW, "s2_r_max": MS.S2_R_MAX,
            "ripley_r_max": MS.RIPLEY_R_MAX,
            "fid_crop": MS.FID_CROP, "fid_extractor": MS.FID_EXTRACTOR,
            "connectivity": "6 (face-adjacent) for every connected-component count",
        },
        "per_case": rows,
        "levels": levels,
        "memorisation": MEMO.memorisation(root, repo=repo,
                                          allow_busy_gpu=allow_busy_gpu),
    }


# ---------------------------------------------------------------------------
# 11 - the statistics of the delivered porosity field
# ---------------------------------------------------------------------------

def _td_reference(repo: Path) -> dict:
    """The real correlation lengths the coherent request is BUILT from.

    Not a second measurement of the real material - it is the target
    :func:`poregen.diffusion.porosity_field.build_porosity_field` smooths with,
    read from the same file the generator reads.  It belongs in the results
    because it separates two different failures: a request that never carried
    the real lengths, and a request that did and a model that lost them.
    """
    from poregen.diffusion.porosity_field import (  # noqa: PLC0415
        DEFAULT_TD_RESULTS,
        load_corr_lengths_voxels,
    )

    path = Path(repo) / DEFAULT_TD_RESULTS
    if not path.exists():
        return {"source": str(DEFAULT_TD_RESULTS), "corr_length_vox": None}
    z, y, x = load_corr_lengths_voxels(path)
    return {
        "source": str(DEFAULT_TD_RESULTS),
        "corr_length_vox": {"z": z, "y": y, "x": x},
        "note": (
            "campaign 01 T-D, patch level, volume mean removed, on the SAME "
            "64-voxel window and 32-voxel stride this assessment uses."
        ),
    }


def _pool(fields: list[np.ndarray]) -> np.ndarray:
    return np.concatenate([np.asarray(f, float).ravel() for f in fields])


def _field_group(fields: list[np.ndarray], rows: list[dict], key: str,
                 stride: int) -> dict:
    """One group's pooled statistics, plus the per-field spread of each length."""
    per_axis = FS.axis_correlations(fields, stride=stride)
    return {
        "n_fields": len(fields),
        "field_grids": [list(f.shape) for f in fields],
        "n_windows": int(sum(int(np.isfinite(f).sum()) for f in fields)),
        "marginal": FS.marginal_stats(_pool(fields)),
        "per_axis": per_axis,
        "anisotropy": FS.anisotropy(per_axis),
        "corr_length_vox_per_field": {
            ax: _agg(rows, (key, "per_axis", ax, "corr_length_vox")) for ax in FS.AXES
        },
    }


def measure_field_stats(root, repo) -> dict:
    """Does the delivered field have the spatial statistics of the real one?"""
    rows: list[dict] = []
    delivered: dict[str, list[np.ndarray]] = defaultdict(list)
    requested: dict[str, list[np.ndarray]] = defaultdict(list)

    for assessment, key, want in FS.COHERENT_SOURCES:
        for case in load_cases(root, assessment):
            if (case.manifest.notes or {}).get(key) != want:
                continue
            group = f"generated/{assessment}"
            field = FS.delivered_field(case.label, case.material_voxels())
            row = {**case_identity(case), "kind": "generated", "group": group,
                   "source_assessment": assessment,
                   "delivered": FS.field_statistics(field)}
            delivered[group].append(field)
            req = case.requested_field
            if req is not None:
                # The request lives on the 64-voxel TILE grid it was painted on,
                # so it is read at that stride and the row says so.  Measuring it
                # on the delivered grid would mean resampling the request, which
                # is a statistic of the interpolation as much as of the field.
                row["requested"] = FS.field_statistics(req, window=TILE, stride=TILE)
                requested[group].append(req)
            rows.append(row)
            case.release()

    for case in load_cases(root, "real_floor"):
        notes = case.manifest.notes or {}
        if notes.get("shape_tag") not in FS.REAL_SHAPE_TAGS:
            continue
        group = f"real/{notes['shape_tag']}"
        field = FS.delivered_field(case.label, case.material_voxels())
        rows.append({"case": case.manifest.case, "kind": "real", "group": group,
                     "volume_shape": list(case.manifest.volume_shape), "notes": notes,
                     "delivered": FS.field_statistics(field)})
        delivered[group].append(field)
        case.release()

    if not rows:
        raise FileNotFoundError(
            f"no coherent-field volumes and no real crops under {root} - this "
            f"assessment measures volumes the other assessments wrote, so run "
            f"`eval_v4 measure porosity_local` and `eval_v4 real-floor` first."
        )

    by_group = _group(rows, lambda r: r["group"])
    groups, pools = {}, {}
    for name, fields in sorted(delivered.items()):
        groups[name] = _field_group(fields, by_group[name], "delivered", FS.STRIDE)
        pools[name] = _pool(fields)
    for name, fields in sorted(requested.items()):
        tag = f"requested/{name.split('/', 1)[1]}"
        groups[tag] = _field_group(fields, by_group[name], "requested", TILE)
        pools[tag] = _pool(fields)

    real_names = [n for n in groups if n.startswith("real/")]
    test_names = [n for n in groups if not n.startswith("real/")]
    comparisons = {}
    for a in test_names:
        for b in real_names:
            comparisons[f"{a} vs {b}"] = {
                "marginal": FS.marginal_distance(pools[a], pools[b]),
                "corr_length": FS.corr_length_gap(groups[a]["per_axis"],
                                                  groups[b]["per_axis"]),
            }
    # The real-vs-real floor. A distance has no reading without it: two halves of
    # the real material do not score zero against each other either.
    for b in real_names:
        half_a, half_b = delivered[b][0::2], delivered[b][1::2]
        if not half_a or not half_b:
            continue
        comparisons[f"{b} vs {b} (real floor)"] = {
            "marginal": FS.marginal_distance(_pool(half_a), _pool(half_b)),
            "corr_length": FS.corr_length_gap(FS.axis_correlations(half_a),
                                              FS.axis_correlations(half_b)),
        }

    return {
        "assessment": "field_stats",
        "question": (
            "Does the DELIVERED local porosity field have the spatial statistics "
            "of the real one - its marginal, and its correlation length per axis?"
        ),
        "note": (
            "This is the quantity a porosity field is claimed to be a sufficient "
            "descriptor of large-scale heterogeneity FOR (Naiff, Ramos and Wang, "
            "SSRN 10.2139/ssrn.7161201), so it is the direct comparison point. "
            "Every field is read from the LABEL, never from the request beside "
            "it; the request is measured separately and reported as its own row. "
            "Real and generated use the same window, and a correlation length "
            "longer than the crop is reported as absent rather than as the crop."
        ),
        "geometry": {
            "window_vox": FS.WINDOW,
            "stride_vox": FS.STRIDE,
            "requested_field_stride_vox": TILE,
            "min_material_frac": FS.MIN_MATERIAL_FRAC,
            "fixed_lags_vox": list(FS.FIXED_LAGS_VOX),
            "min_lag_pairs": M.MIN_LAG_PAIRS,
        },
        "t_d_reference": _td_reference(repo),
        "per_case": rows,
        "groups": groups,
        "comparisons": comparisons,
    }


# ---------------------------------------------------------------------------
# 11 - four ways to assemble the same request
# ---------------------------------------------------------------------------

#: The per-chunk quantities the arms are compared on.  Each is a path into one
#: row of ``metrics.chunk_profile``, and every one of them is defined for a real
#: crop too, which is what makes the floor row possible.
CHUNK_KEYS = {
    "chunk_plane_seam_xct": ("xct", "chunk_plane_ratio"),
    "tile_plane_seam_xct": ("xct", "tile_plane_ratio"),
    "chunk_plane_seam_pore": ("pore", "chunk_plane_ratio"),
    "tile_plane_seam_pore": ("pore", "tile_plane_ratio"),
    "phi_pore": ("porosity", "phi_pore"),
    "s2_relative_distance": ("s2", "s2_relative_distance"),
}


def _dig(row, path):
    v = row
    for p in path:
        v = (v or {}).get(p) if isinstance(v, dict) else None
    return v


def _chunk_series(case_rows: list[list[dict]]) -> dict:
    """Mean and sd ACROSS SEEDS at every chunk index, plus the trend in index.

    ``case_rows`` is one ``chunk_profile`` list per seed.  The volumes of one
    cell are the same shape, so chunk index k is the same block in all of them;
    the aggregation asserts that rather than assuming it.
    """
    n = {len(r) for r in case_rows}
    if len(n) != 1:
        raise ValueError(
            f"the seeds of one cell produced {sorted(n)} chunks; they must be the "
            "same shape to be aggregated per chunk index."
        )
    n_chunks = n.pop()
    out: dict = {"n_chunks": n_chunks, "by_chunk_index": {}, "trend": {}}
    for name, path in CHUNK_KEYS.items():
        per_index = []
        flat_x, flat_y = [], []
        for k in range(n_chunks):
            vals = [_dig(rows[k], path) for rows in case_rows]
            per_index.append(M.mean_sd(vals))
            for v in vals:
                if v is not None and np.isfinite(v):
                    flat_x.append(k)
                    flat_y.append(v)
        out["by_chunk_index"][name] = per_index
        # The fit is over every seed's every chunk, not over the per-index
        # means: a slope fitted on three points that are themselves means
        # would hide the seed spread the slope has to be read against.
        out["trend"][name] = M.fit_ols(flat_x, flat_y)
    return out


def _floor_chunk_profile(root, period) -> dict:
    """The same per-chunk profile on the real crops, by shape tag.

    A real volume was assembled by nothing, so what it scores at the reference
    planes is what the MEASUREMENT reads when there is no seam - the only thing
    a generated ratio can be called large or small against.
    """
    by_tag: dict[str, list] = {}
    for case in load_cases(root, "real_floor"):
        notes = case.manifest.notes or {}
        tag = notes.get("shape_tag")
        if tag not in ("small", "large"):
            continue
        rows = M.chunk_profile(
            case.xct, case.label, case.material_voxels(),
            manifest=case.manifest, period=period, pore_logit=case.pore_logit,
        )
        by_tag.setdefault(tag, []).append(rows)
    out = {}
    for tag, runs in sorted(by_tag.items()):
        flat = [r for rows in runs for r in rows]
        out[tag] = {
            "n_volumes": len(runs),
            "n_chunks_each": [len(r) for r in runs],
            **{name: M.mean_sd([_dig(r, path) for r in flat])
               for name, path in CHUNK_KEYS.items()},
        }
    return out


def measure_assembly_modes(root, repo) -> dict:
    """How much of the quality is the hybrid sampler, and how far is the ceiling?

    Every arm is read on the SAME reference chunk grid, taken from the cases and
    not from each volume's own ``chunk_tiles`` - the arms have different chunk
    geometries by construction, so their own grids would put four different sets
    of planes in one table.  Each volume also keeps its own-grid seam block from
    ``measure_core``, which is what the sampler itself reported.

    The per-chunk series is the point of the assessment.  A chunked sampler
    fails by compounding: chunk k assembles against material chunk k-1 already
    produced, so an arm can hold a good volume average and still degrade with
    distance from the first chunk.  ``trend`` is the OLS slope of each quantity
    against the chunk index, over every seed's every chunk.
    """
    from poregen.eval_v4.cases import (  # noqa: PLC0415
        ASSEMBLY_MODES_REFERENCE_TILES,
        TEACHER_MAX_DEPTH_VOX,
    )
    from poregen.eval_v4.io import TILE  # noqa: PLC0415

    cases = load_cases(root, "assembly_modes")
    if not cases:
        raise FileNotFoundError(f"no assembly_modes volumes under {root}")
    period = tuple(TILE * int(c) for c in ASSEMBLY_MODES_REFERENCE_TILES)

    rows = []
    profiles = []
    for case in cases:
        notes = case.manifest.notes or {}
        material = case.material_voxels()
        row = measure_core(case)
        row["arm"] = notes.get("arm")
        row["scale"] = notes.get("scale")
        row["neighbour_mode"] = notes.get("neighbour_mode")
        row["generated_chunk_tiles"] = list(case.manifest.chunk_tiles or ())
        row["reference_chunk_period"] = list(period)
        row["reference_latents"] = notes.get("reference_latents")
        # The whole-volume seam at the REFERENCE planes, so the four arms have
        # one comparable headline number beside the per-chunk series.
        grey = case.xct.astype(np.float32) / 255.0
        row["reference_seams"] = {
            **M.seam_discontinuity(grey, period, prefix="seam_ref_xct",
                                   interior_exclude=TILE),
            **M.seam_discontinuity(grey, TILE, prefix="seam_ref_tile_xct"),
        }
        profile = M.chunk_profile(
            case.xct, case.label, material, manifest=case.manifest,
            period=period, pore_logit=case.pore_logit,
        )
        row["n_chunks"] = len(profile)
        rows.append(row)
        profiles.append(profile)

    cells: dict[str, dict] = {}
    for (arm, scale), group in sorted(
        _group(zip(rows, profiles), lambda rp: (rp[0]["arm"], rp[0]["scale"])).items()
    ):
        grp_rows = [r for r, _ in group]
        cells[f"{arm}@{scale}"] = {
            "arm": arm,
            "scale": scale,
            "n_seeds": len(group),
            "seeds": sorted(r["seed"] for r in grp_rows),
            "generated_chunk_tiles": grp_rows[0]["generated_chunk_tiles"],
            "neighbour_mode": grp_rows[0]["neighbour_mode"],
            "volume_seam_xct_reference": _agg(grp_rows, ("reference_seams", "seam_ref_xct_ratio")),
            "volume_seam_tile_xct": _agg(grp_rows, ("reference_seams", "seam_ref_tile_xct_ratio")),
            "delivered_phi": _agg(grp_rows, ("porosity", "delivered_phi")),
            "air_fraction_interior": _agg(grp_rows, ("air_fraction_interior",)),
            "wall_time_s": _agg(grp_rows, ("wall_time_s",)),
            "peak_gpu_memory_bytes": _agg(grp_rows, ("peak_gpu_memory_bytes",)),
            **_failure_rate(grp_rows),
            **_chunk_series([p for _, p in group]),
        }

    return {
        "assessment": "assembly_modes",
        "question": ("How much of the generated quality comes from the hybrid "
                     "chunked sampler rather than its alternatives, and how far "
                     "is the assembly from an upper bound?"),
        "note": (
            "Four arms, one request, one seed set per scale. `joint` is one chunk "
            "with every neighbour UNKNOWN (the ldm05 MultiDiffusion sampler); "
            "`autoregressive` is chunk_tiles (1,1,1), patch at a time against "
            "finished material; `hybrid` is the production (3,3,3); "
            "`teacher_forced` is the production chunking with every neighbour "
            "replaced by the encoding of a REAL test volume at the same position. "
            "The last is a CONTROL, not a sampler: it is the ceiling the hybrid "
            "would reach if the material it assembles against were perfect."
        ),
        "reference_grid_note": (
            f"Every arm is measured on the same reference chunk grid, "
            f"{list(period)} voxels, whatever grid it was generated on. For the "
            "hybrid arm that IS its generation grid; for the others it is 'what "
            "happens at the planes the production sampler would have had to "
            "assemble across'. The joint arm has no chunk planes at all, so its "
            "row is the measurement's own no-seam reading."
        ),
        "teacher_forced_note": (
            f"The teacher-forced arm exists at the slab only. No specimen in the "
            f"dataset is thicker than about {TEACHER_MAX_DEPTH_VOX} voxels on the "
            "64-voxel tile grid, so there is no real material to teach with at "
            "384 deep, and repeating a block to fill the depth would put a fake "
            "join on a chunk plane - the one place this assessment measures. The "
            "384 cells carry no ceiling rather than a fabricated one."
        ),
        "chunk_keys": {k: list(v) for k, v in CHUNK_KEYS.items()},
        "real_floor": _floor_chunk_profile(root, period),
        "per_case": rows,
        "cells": cells,
    }


MEASURERS = {
    "sampler": measure_sampler,
    "porosity_global": measure_porosity_global,
    "porosity_local": measure_porosity_local,
    "cfg": measure_cfg,
    "layup": measure_layup,
    "assembly": measure_assembly,
    "geometry": measure_geometry,
    "surface": measure_surface,
    "multichunk": measure_multichunk,
    "microstructure": measure_microstructure,
    "field_stats": measure_field_stats,
    "assembly_modes": measure_assembly_modes,
}


def measure(root: str | Path, assessment: str, repo: str | Path | None = None,
            *, allow_busy_gpu: bool = False) -> dict:
    """Measure one assessment and write ``<root>/<assessment>/results.json``."""
    if assessment not in MEASURERS:
        raise KeyError(f"unknown assessment {assessment!r}; choose from {sorted(MEASURERS)}")
    repo = Path(repo) if repo else repo_root()
    # Only microstructure carries the full-store memorisation search, so only
    # microstructure has a reason to care whether the card is busy.  Naming the
    # one measurer is honest; giving every measurer a flag it ignores is not.
    extra = {"allow_busy_gpu": allow_busy_gpu} if assessment == "microstructure" else {}
    results = MEASURERS[assessment](Path(root), repo, **extra)
    results["n_cases_measured"] = len(results["per_case"])
    # A measure-only assessment generates nothing, so there is no case list to
    # be short of: what it measures is whatever the assessments it reads wrote.
    results["n_cases_expected"] = (
        len(build_cases(assessment, repo)) if assessment in ASSESSMENTS
        else results["n_cases_measured"]
    )
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
