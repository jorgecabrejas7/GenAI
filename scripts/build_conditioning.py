"""Build the ldm06 per-patch conditioning sidecar.

Writes ``<store>/<split>/cond.parquet`` — row-aligned with that split's
``index.parquet`` — holding ``cond_depth``, the six ``cond_dist6_*`` per-face
distances and ``cond_por_raw``, plus the ``conditioning`` block in the store's
``metadata.json``.  A sidecar (rather than new columns in ``index.parquet``)
keeps the store's encoded files byte-identical, so a rebuild of the
conditioning never means re-encoding 1.8 M patches.

The orientation field ``data/split_v2/orientation_field.json`` is REUSED as it
is.  It is per-VOLUME (theta(z) from the nominal ply sequence aligned to the
scan, plus each volume's foreground extent), and split_v3 changed which patches
exist, never the volumes themselves.  The script asserts that every volume in
the split_v3 index has a record before it uses one.  ``--rebuild-orientation``
re-derives the field from the T-I artefacts and the expert ground truth; it is
the provenance of that file and is only needed when its inputs change.

Six distances, not one
----------------------
ldm05 conditioned on a single ``cond_dist``: the distance from the patch CENTRE
to the nearest outer specimen face over all three axes.  One number cannot say
*which* face is near, so a patch under the top surface and a patch against a
side wall asked the model for the same thing.  ``cond_dist6`` is the gap from
each of the patch's own six faces to the matching face of the specimen box,
capped at 64 voxels and normalised, ordered
(z-, z+, y-, y+, x-, x+) by ``poregen.diffusion.conditioning.DIST6_DIRS``.
``poregen.diffusion.conditioning.dist6_from_box_array`` is the single
implementation; the sampler calls its scalar twin for generated volumes.

Run:  python scripts/build_conditioning.py [--store data/split_v3/latents_r08z4]
"""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from poregen.diffusion.conditioning import (
    DIST6_DIRS,
    DIST6_NAMES,
    DIST_CAP,
    NEIGHBOUR_DIRS,
    POR_LOG_EPS,
    dist6_from_box_array,
)

REPO = Path(__file__).resolve().parents[1]
DATA_ROOT = REPO / "data" / "split_v3"
# The orientation field is a per-VOLUME artefact and lives with the dataset
# root that first produced it; split_v3's volumes.zarr is a symlink to the same
# store, so the field describes both roots.
ORIENT_ROOT = REPO / "data" / "split_v2"
GT_PATH = REPO / "data" / "layup_ground_truth.json"
TI_DIR = REPO / "runs" / "campaigns" / "01-conditioning-design" / "T-I"
TA_PROFILES = REPO / "runs" / "campaigns" / "01-conditioning-design" / "T-A" / "fine_profiles.npz"
LAYUP_FIELD = TI_DIR / "layup_field.json"
ORIENT_OUT = ORIENT_ROOT / "orientation_field.json"

PATCH_SIZE = 64
VOXEL_SIZE_UM = 25.0
SPLITS = ("train", "val", "test")

# Three strides, kept separate.  neighbour_offset must be >= PATCH_SIZE so that
# face neighbours only TOUCH the target: at 32 they overlapped it by half and an
# opposite pair tiled it completely, which handed the denoiser the answer.
SAMPLE_STRIDE = 32        # spacing of the patches stored in the dataset
GENERATION_STRIDE = 64    # spacing of the tiling grid the sampler decodes on
NEIGHBOUR_OFFSET = 64     # voxel displacement of a face neighbour (touching)

DIST6_COLUMNS = tuple(f"cond_dist6_{n}" for n in DIST6_NAMES)


def geometry_metadata() -> dict:
    """The ``conditioning.geometry`` block written into the store metadata.

    Single definition so that the store always describes the geometry the
    training code enforces — ``LatentDataset`` refuses to run against a store
    whose recorded strides disagree with the config.
    """
    return {
        "grid_index_key": "(z0 // neighbour_offset, y0 // ..., x0 // ...)",
        "neighbour_dirs": [list(d) for d in NEIGHBOUR_DIRS],
        "sample_stride": SAMPLE_STRIDE,
        "generation_stride": GENERATION_STRIDE,
        "neighbour_offset": NEIGHBOUR_OFFSET,
        "patch_size": PATCH_SIZE,
        "neighbour_shared_voxels": max(PATCH_SIZE - NEIGHBOUR_OFFSET, 0) * PATCH_SIZE ** 2,
        "neighbour_relation": (
            "face neighbours TOUCH: neighbour_offset >= patch_size, so a neighbour "
            "shares no voxel with the target and cannot leak its content.  They are "
            "fed WHOLE and UNSHIFTED, and the training step noises each one to its "
            "own timestep (see poregen.training.ldm_engine.noise_neighbours)."
        ),
        "availability_states": {"OOB": 0, "EXISTS": 1, "UNKNOWN": 2},
        "availability_rule": (
            "the store serves EXISTS when it holds a patch at that position and OOB "
            "when it does not.  UNKNOWN comes from the training step's neighbour "
            "dropout (the CFG null) and from chunks the sampler has not generated."
        ),
    }


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


# ---------------------------------------------------------------------------
# 1. Orientation field
# ---------------------------------------------------------------------------

def foreground_extents() -> dict[str, dict]:
    """Per-volume outer-surface extent per axis, T-C's definition.

    T-C (``t_c_air_boundary.py::load_volume_geometry``) takes the T-A
    per-slice foreground fraction and thresholds it at half its maximum.  The
    The SAME extent is used for ``cond_depth`` and for all six ``cond_dist6``
    faces, so ``cond_depth`` 0/1 and ``cond_dist6_z{m,p}`` 0 describe the same
    two planes.
    """
    profiles = np.load(TA_PROFILES, allow_pickle=True)["profiles"].item()
    out: dict[str, dict] = {}
    for vid, p in profiles.items():
        e: dict[str, list[int]] = {}
        for a in ("z", "y", "x"):
            fg = np.asarray(p[a]["fg"], float)
            idx = np.where(fg > 0.5 * fg.max())[0]
            e[a] = [int(idx[0]), int(idx[-1])] if len(idx) else [0, len(fg) - 1]
        out[vid] = {"extent": e, "shape": [int(s) for s in p["shape"]]}
    return out


def verify_volume(vid: str, v: dict, gt: dict) -> list[str]:
    """Re-derive every step of the boundary <-> nominal-angle correspondence.

    Returns a list of failure strings; empty means the T-I record is internally
    consistent AND consistent with the expert ground truth.
    """
    fails: list[str] = []
    plies = np.asarray(gt["volumes"][vid]["plies"], float) % 180.0
    n = int(v["n_plies"])
    edges = np.asarray(v["ply_boundaries_z"], int)
    gt_ord = np.asarray(v["ply_angle_gt_deg"], float)
    img = np.asarray(v["ply_angle_image_deg"], float)
    ply_idx = np.asarray(v["slice_ply_index"], int)
    slice_ang = v["slice_angle_image_deg"]

    if len(edges) != n + 1 or not np.all(np.diff(edges) > 0):
        fails.append("ply_boundaries_z is not a strictly increasing (n_plies+1) list")
    if len(plies) != n:
        fails.append(f"expert sequence has {len(plies)} plies, field says {n}")
        return fails

    # face ordering: ply 0 sits at z_start; the expert list is reversed exactly
    # when the fit chose the reversed face hypothesis.
    expected = plies[::-1] if v["z_order_reversed_vs_expert_list"] else plies
    if not np.allclose(gt_ord, expected):
        fails.append("ply_angle_gt_deg does not match the expert sequence "
                     "under the recorded face ordering")
    # image angles = sign * (gt - offset) mod 180
    recomputed = (int(v["angle_sign"]) * (gt_ord - float(v["rotation_offset_deg"]))) % 180.0
    if not np.allclose(img, recomputed):
        fails.append("ply_angle_image_deg is not sign*(gt - offset) mod 180")
    if v["z_start"] != int(edges[0]) or v["z_end"] != int(edges[-1]) - 1:
        fails.append("z_start/z_end disagree with ply_boundaries_z")
    # slice -> ply index map
    expected_idx = np.full(int(v["shape"][0]), -1, int)
    for i in range(n):
        expected_idx[edges[i]:edges[i + 1]] = i
    if not np.array_equal(ply_idx, expected_idx):
        fails.append("slice_ply_index does not follow ply_boundaries_z")
    # slice -> angle map
    for z, pi in enumerate(ply_idx):
        a = slice_ang[z]
        if pi < 0:
            if a is not None:
                fails.append(f"slice {z} outside the laminate carries an angle")
                break
        elif a is None or abs(a - img[pi]) > 1e-9:
            fails.append(f"slice {z} angle does not match its ply's image angle")
            break
    return fails


def build_orientation_field() -> dict:
    gt = json.load(open(GT_PATH))
    lf = json.load(open(LAYUP_FIELD))
    geo = foreground_extents()

    out_vols: dict[str, dict] = {}
    for vid, v in lf["volumes"].items():
        g = geo.get(vid)
        base = {
            "shape": v.get("shape") or (g["shape"] if g else None),
            "extent_foreground": g["extent"] if g else None,
        }
        if not v.get("usable"):
            out_vols[vid] = {
                **base,
                "orientation_usable": False,
                "confidence": "none",
                "reason": v.get("reason", "no expert stacking sequence"),
                "theta_deg": None,
            }
            continue

        fails = verify_volume(vid, v, gt)
        plies = np.asarray(gt["volumes"][vid]["plies"], float) % 180.0
        palindromic = bool(np.allclose(plies, plies[::-1]))
        margin = float(v["hypothesis_margin_deg"])

        out_vols[vid] = {
            **base,
            "orientation_usable": len(fails) == 0,
            "verification_failures": fails,
            "confidence": v["confidence"],
            "hypothesis_margin_deg": margin,
            "hypothesis_margin_below_2deg": bool(margin < 2.0),
            "margin_degenerate_by_symmetry": palindromic,
            "fit_median_abs_error_deg": v["fit_median_abs_error_deg"],
            "rotation_offset_deg": v["rotation_offset_deg"],
            "rotation_offset_sd_deg": v["rotation_offset_sd_deg"],
            "angle_sign": int(v["angle_sign"]),
            "z_order_reversed_vs_expert_list": bool(v["z_order_reversed_vs_expert_list"]),
            "sequence_id": v["sequence_id"],
            "family": v["family"],
            "material": v["material"],
            "ply_thickness_mm": v["ply_thickness_mm"],
            "n_plies": int(v["n_plies"]),
            "z_start": int(v["z_start"]),
            "z_end": int(v["z_end"]),
            "ply_boundaries_z": v["ply_boundaries_z"],
            "ply_angle_gt_deg": v["ply_angle_gt_deg"],
            "ply_angle_image_deg": v["ply_angle_image_deg"],
            "theta_deg": v["slice_angle_image_deg"],
        }

    usable = [v for v in out_vols.values() if v["orientation_usable"]]
    conf = {c: sum(1 for v in usable if v["confidence"] == c)
            for c in ("high", "medium", "low")}
    field = {
        "description": (
            "Per-volume theta(z) for every z-slice in IMAGE coordinates, built "
            "from the NOMINAL ply sequence aligned to the scan (D32 section 2). "
            "Measured per-ply deviations are deliberately excluded."
        ),
        "created": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "angle_convention": (
            "degrees mod 180; 0 = image x axis, 90 = image y axis; axial "
            "(orientation, not direction) — encode as (cos 2t, sin 2t)"
        ),
        "ply_index_convention": "ply 0 at z = z_start, index increasing with z",
        "theta_convention": (
            "theta_deg[z] is null where slice z lies outside the fitted laminate "
            "extent [z_start, z_end]; consumers must encode null as the ZERO "
            "vector (unknown orientation), never as an angle"
        ),
        "extent_convention": (
            "extent_foreground[axis] = [lo, hi] outer specimen faces, from the "
            "T-A per-slice foreground fraction thresholded at half its maximum "
            "(the same definition T-C used for distance-to-surface)"
        ),
        "voxel_size_um": VOXEL_SIZE_UM,
        "provenance": {
            "builder": "scripts/build_conditioning.py",
            "ground_truth_file": str(GT_PATH.relative_to(REPO)),
            "ground_truth_sha256": sha256(GT_PATH),
            "estimator_run": str(TI_DIR.relative_to(REPO)),
            "estimator_script": "scripts/analysis/t_i_layup_validation.py",
            "layup_field_file": str(LAYUP_FIELD.relative_to(REPO)),
            "layup_field_sha256": sha256(LAYUP_FIELD),
            "foreground_extent_file": str(TA_PROFILES.relative_to(REPO)),
        },
        "summary": {
            "n_volumes": len(out_vols),
            "n_orientation_usable": len(usable),
            "n_orientation_unusable": len(out_vols) - len(usable),
            "confidence_counts": conf,
            "n_high_confidence": conf["high"],
            "n_flagged_low_confidence": conf["medium"] + conf["low"],
            "n_margin_below_2deg": sum(1 for v in usable
                                       if v["hypothesis_margin_below_2deg"]),
            "n_margin_below_2deg_excluding_symmetric": sum(
                1 for v in usable if v["hypothesis_margin_below_2deg"]
                and not v["margin_degenerate_by_symmetry"]),
            "n_verification_failures": sum(
                1 for v in out_vols.values()
                if v.get("verification_failures")),
        },
        "volumes": out_vols,
    }
    return field


# ---------------------------------------------------------------------------
# 2. Per-patch scalars
# ---------------------------------------------------------------------------

def check_volume_coverage(store: Path, field: dict) -> list[str]:
    """Every volume in the store's index must have an orientation record.

    The orientation field is a split_v2-era per-volume artefact; split_v3
    changed which patches exist, not which volumes do.  This is the assertion
    that the reuse is legitimate.
    """
    vols = field["volumes"]
    seen: set[str] = set()
    for split in SPLITS:
        df = pd.read_parquet(store / split / "index.parquet", columns=["volume_id"])
        seen |= set(df["volume_id"].unique().tolist())
    missing = sorted(v for v in seen if v not in vols)
    if missing:
        raise SystemExit(
            f"{len(missing)} volume(s) in {store} have no record in "
            f"{ORIENT_OUT.relative_to(REPO)}: {missing[:5]}.  Rebuild the "
            f"orientation field (--rebuild-orientation) before the sidecar."
        )
    no_extent = sorted(v for v in seen if not vols[v].get("extent_foreground"))
    if no_extent:
        raise SystemExit(
            f"{len(no_extent)} volume(s) have no foreground extent: "
            f"{no_extent[:5]}.  cond_depth and cond_dist6 are undefined without "
            f"the specimen box."
        )
    return sorted(seen)


def build_scalars(store: Path, field: dict) -> dict:
    """Per-patch cond_depth / cond_dist6 / cond_por_raw for every split."""
    vols = field["volumes"]
    stats: dict[str, dict] = {}
    frames: dict[str, pd.DataFrame] = {}

    for split in SPLITS:
        df = pd.read_parquet(store / split / "index.parquet",
                             columns=["source_row", "volume_id", "z0", "y0", "x0", "phi"])
        ext = df["volume_id"].map(lambda v: vols[v]["extent_foreground"])
        # extent_foreground is [lo, hi] INCLUSIVE slice indices; the specimen
        # box is [lo, hi + 1) so that a patch whose face sits on the last
        # foreground slice measures a gap of 0, not -1.
        box_lo = np.stack([np.array([e[a][0] for e in ext], float)
                           for a in ("z", "y", "x")], axis=1)
        box_hi = np.stack([np.array([e[a][1] + 1 for e in ext], float)
                           for a in ("z", "y", "x")], axis=1)
        origins = np.stack([df["z0"].to_numpy(float),
                            df["y0"].to_numpy(float),
                            df["x0"].to_numpy(float)], axis=1)

        centre_z = origins[:, 0] + PATCH_SIZE / 2
        span_z = np.maximum(box_hi[:, 0] - box_lo[:, 0], 1e-9)
        depth_raw = (centre_z - box_lo[:, 0]) / span_z
        depth = np.clip(depth_raw, 0.0, 1.0)

        dist6 = dist6_from_box_array(origins, PATCH_SIZE, box_lo, box_hi)
        raw_gaps = np.empty_like(dist6, dtype=np.float64)
        raw_gaps[:, 0::2] = origins - box_lo
        raw_gaps[:, 1::2] = box_hi - (origins + PATCH_SIZE)

        phi = df["phi"].to_numpy(float)
        por_raw = np.log(phi + POR_LOG_EPS)

        cols = {
            "source_row": df["source_row"].to_numpy(np.int64),
            "cond_depth": depth.astype(np.float32),
            "cond_por_raw": por_raw.astype(np.float32),
        }
        for k, name in enumerate(DIST6_COLUMNS):
            cols[name] = dist6[:, k]
        frames[split] = pd.DataFrame(cols)

        stats[split] = {
            "n": int(len(df)),
            "cond_depth": {"min": float(depth.min()), "max": float(depth.max()),
                           "mean": float(depth.mean())},
            "cond_dist6": {
                name: {"min": float(dist6[:, k].min()),
                       "max": float(dist6[:, k].max()),
                       "mean": float(dist6[:, k].mean()),
                       "frac_at_zero": float(np.mean(dist6[:, k] <= 0.0)),
                       "frac_at_cap": float(np.mean(raw_gaps[:, k] >= DIST_CAP))}
                for k, name in enumerate(DIST6_COLUMNS)
            },
            "cond_por_raw": {"min": float(por_raw.min()), "max": float(por_raw.max()),
                             "mean": float(por_raw.mean()), "std": float(por_raw.std())},
            "frac_depth_clipped": float(np.mean((depth_raw < 0) | (depth_raw > 1))),
            "frac_dist_clipped_low": float(np.mean(raw_gaps < 0)),
        }

    tr = frames["train"]["cond_por_raw"].to_numpy(np.float64)
    por_mean, por_std = float(tr.mean()), float(tr.std())

    for split, f in frames.items():
        f.to_parquet(store / split / "cond.parquet", index=False)

    return {"por_mean": por_mean, "por_std": por_std, "per_split": stats}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--store", default=str(DATA_ROOT / "latents_r08z4"))
    ap.add_argument("--rebuild-orientation", action="store_true",
                    help="Re-derive data/split_v2/orientation_field.json from the "
                         "T-I fit and the expert ground truth before building the "
                         "sidecar.  Only needed when those inputs change.")
    args = ap.parse_args()
    store = Path(args.store)

    if args.rebuild_orientation or not ORIENT_OUT.exists():
        print("[1/4] rebuilding orientation field ...", flush=True)
        field = build_orientation_field()
        ORIENT_OUT.parent.mkdir(parents=True, exist_ok=True)
        with open(ORIENT_OUT, "w") as fh:
            json.dump(field, fh, indent=1)
        sm = field["summary"]
        print(f"      wrote {ORIENT_OUT.relative_to(REPO)}")
        print(f"      {sm['n_orientation_usable']}/{sm['n_volumes']} volumes usable; "
              f"confidence {sm['confidence_counts']}; "
              f"{sm['n_margin_below_2deg']} with <2 deg hypothesis margin "
              f"({sm['n_margin_below_2deg_excluding_symmetric']} excluding symmetric "
              f"layups); {sm['n_verification_failures']} verification failures")
    else:
        print(f"[1/4] reusing {ORIENT_OUT.relative_to(REPO)} "
              f"(pass --rebuild-orientation to re-derive it)", flush=True)
        field = json.load(open(ORIENT_OUT))

    print("[2/4] checking volume coverage ...", flush=True)
    seen = check_volume_coverage(store, field)
    print(f"      {len(seen)} volumes in {Path(args.store).name}, all present in "
          f"the orientation field with a foreground extent")

    print("[3/4] building per-patch scalars ...", flush=True)
    sc = build_scalars(store, field)
    for split, st in sc["per_split"].items():
        d6 = st["cond_dist6"]
        print(f"      {split:5s} n={st['n']:>9,d} "
              f"depth[{st['cond_depth']['min']:.3f},{st['cond_depth']['max']:.3f}] "
              f"dist6 mean=" + "/".join(f"{d6[c]['mean']:.2f}" for c in DIST6_COLUMNS)
              + f" por_raw mean={st['cond_por_raw']['mean']:.3f}")
    print(f"      train standardisation: mean={sc['por_mean']:.6f} std={sc['por_std']:.6f}")

    print("[4/4] updating store metadata ...", flush=True)
    meta_path = store / "metadata.json"
    meta = json.load(open(meta_path))
    meta.pop("assembly", None)      # ldm05 parity schedule — gone with the sampler
    meta["voxel_size_um"] = VOXEL_SIZE_UM
    meta["conditioning"] = {
        "version": "ldm06-v1",
        "created": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "builder": "scripts/build_conditioning.py",
        "sidecar_file": "cond.parquet",
        "sidecar_alignment": "row i of cond.parquet is row i of index.parquet",
        "voxel_size_um": VOXEL_SIZE_UM,
        "orientation_field": str(ORIENT_OUT.relative_to(REPO)),
        "orientation_field_sha256": sha256(ORIENT_OUT),
        "n_volumes": len(seen),
        "porosity_transform": f"log(phi + {POR_LOG_EPS})",
        "porosity_denominator": f"{PATCH_SIZE}**3 (the full patch volume)",
        "por_standardisation": {
            "computed_over": "train",
            "mean": sc["por_mean"],
            "std": sc["por_std"],
        },
        "depth_definition": (
            "(z_centre - z_lo) / (z_hi - z_lo), z_lo/z_hi = outer specimen faces "
            "from the T-A foreground extent (half-max threshold), clipped to [0,1]"
        ),
        "dist6_definition": (
            f"per-face gap from the patch face to the matching specimen-box face, "
            f"min(d, {int(DIST_CAP)}) / {int(DIST_CAP)}; the box is the foreground "
            f"extent [lo, hi+1).  0 = the specimen ends at that face of the patch"
        ),
        "dist6_order": list(DIST6_NAMES),
        "dist6_dirs": [list(d) for d in DIST6_DIRS],
        "dist6_columns": list(DIST6_COLUMNS),
        "orient_definition": (
            "(cos 2t, sin 2t) over the patch's 64 depth voxels, COMPONENTS "
            "mean-pooled in groups of 4 to 16 depth planes (never renormalised), "
            "broadcast over the two in-plane latent axes -> (2,16,16,16)"
        ),
        "material_definition": (
            "material.bin / air.bin, written by scripts/build_latent_dataset.py; "
            "see the `material` metadata block"
        ),
        "geometry": geometry_metadata(),
        "scalar_stats": sc["per_split"],
    }
    tmp = meta_path.with_suffix(".json.tmp")
    with open(tmp, "w") as fh:
        json.dump(meta, fh, indent=2)
    tmp.replace(meta_path)
    print(f"      wrote {meta_path.relative_to(REPO)}")
    print("done.")


if __name__ == "__main__":
    main()
