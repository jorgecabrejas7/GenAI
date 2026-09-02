"""Build the ldm05 per-patch conditioning data (D32 §1, §2, §5).

Two artefacts, both canonical dataset content under ``data/`` (not ``runs/``):

1. ``data/split_v2/orientation_field.json`` — per-volume theta(z) for every
   z-slice in IMAGE coordinates, built from the **nominal** ply sequence
   (``data/layup_ground_truth.json``) aligned to the scan by the rotation
   offset / face order / sign convention fitted in T-I.  Measured per-ply
   angles are deliberately NOT used (D32 §2).  Carries the T-I confidence and
   hypothesis margin per volume plus a full re-verification of the
   boundary <-> ground-truth-angle correspondence.

2. ``data/split_v2/latents_r07z4/<split>/cond.parquet`` — a sidecar parquet,
   row-aligned with that split's ``index.parquet``, holding ``cond_depth``,
   ``cond_dist`` and ``cond_por_raw``.  A sidecar (rather than new columns in
   ``index.parquet``) keeps the latent store's existing files byte-identical,
   so runs already training against it are untouched.

The train-split standardisation statistics for ``cond_por_raw`` and
``voxel_size_um`` are written into the store's ``metadata.json`` under a new
``conditioning`` block (additive; nothing existing is modified).

Run:  python scripts/build_conditioning.py
"""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from poregen.diffusion.conditioning import NEIGHBOUR_DIRS, PARITY_GROUP_ORDER

REPO = Path(__file__).resolve().parents[1]
DATA_ROOT = REPO / "data" / "split_v2"
GT_PATH = REPO / "data" / "layup_ground_truth.json"
TI_DIR = REPO / "runs" / "campaigns" / "01-conditioning-design" / "T-I"
TA_PROFILES = REPO / "runs" / "campaigns" / "01-conditioning-design" / "T-A" / "fine_profiles.npz"
LAYUP_FIELD = TI_DIR / "layup_field.json"
ORIENT_OUT = DATA_ROOT / "orientation_field.json"

PATCH_SIZE = 64
VOXEL_SIZE_UM = 25.0
POR_EPS = 1e-3
DIST_CAP = 64.0
SPLITS = ("train", "val", "test")

# The three strides of D32 section 3.1, kept separate.  neighbour_offset must
# be >= PATCH_SIZE so that face neighbours only TOUCH the target: at 32 they
# overlapped it by half and an opposite pair tiled it completely, which handed
# the denoiser the answer.
SAMPLE_STRIDE = 32        # spacing of the patches stored in the dataset
GENERATION_STRIDE = 64    # spacing of the assembly grid the sampler walks
NEIGHBOUR_OFFSET = 64     # voxel displacement of a face neighbour (touching)


def assembly_metadata() -> dict:
    """The ``assembly`` block written into the latent store's metadata.json.

    Single definition so that the store always describes the geometry the
    training code enforces — ``LatentDataset`` refuses to run against a store
    whose recorded strides disagree with the config.
    """
    return {
        # read back by poregen.diffusion.conditioning.resolve_group_order, so
        # data and sampler use one ordering.  Rank = 4*pz + 2*py + px.
        "parity_group_order": [list(g) for g in PARITY_GROUP_ORDER],
        "grid_index_key": "(z0 // neighbour_offset, y0 // ..., x0 // ...)",
        "parity_group_key": "(iz % 2, iy % 2, ix % 2) of the grid index",
        "neighbour_dirs": [list(d) for d in NEIGHBOUR_DIRS],
        "sample_stride": SAMPLE_STRIDE,
        "generation_stride": GENERATION_STRIDE,
        "neighbour_offset": NEIGHBOUR_OFFSET,
        "patch_size": PATCH_SIZE,
        "neighbour_shared_voxels": max(PATCH_SIZE - NEIGHBOUR_OFFSET, 0) * PATCH_SIZE ** 2,
        "neighbour_relation": (
            "face neighbours TOUCH: neighbour_offset >= patch_size, so a neighbour "
            "shares no voxel with the target and cannot leak its content"
        ),
        "neighbour_shift": False,
        "neighbour_shift_note": (
            "neighbours are fed WHOLE and UNSHIFTED; the shift into the target "
            "frame only applies to overlapping neighbours and at this offset would "
            "produce an all-zero tensor"
        ),
        "blending": "none — generation_stride == patch_size, so patches tile exactly",
        "assembly_metric": (
            "seam discontinuity: mean |slice-to-slice difference| across each "
            "patch-to-patch plane, divided by the same quantity inside the patches"
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
    SAME extent is used here for both ``cond_depth`` and ``cond_dist`` so that
    ``cond_depth`` 0/1 and ``cond_dist`` 0 describe the same two faces.
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

def build_scalars(store: Path, field: dict) -> dict:
    vols = field["volumes"]
    stats: dict[str, dict] = {}
    frames: dict[str, pd.DataFrame] = {}

    for split in SPLITS:
        df = pd.read_parquet(store / split / "index.parquet",
                             columns=["source_row", "volume_id", "z0", "y0", "x0", "phi"])
        ext = df["volume_id"].map(lambda v: vols[v]["extent_foreground"])
        zlo = np.array([e["z"][0] for e in ext], float)
        zhi = np.array([e["z"][1] for e in ext], float)
        ylo = np.array([e["y"][0] for e in ext], float)
        yhi = np.array([e["y"][1] for e in ext], float)
        xlo = np.array([e["x"][0] for e in ext], float)
        xhi = np.array([e["x"][1] for e in ext], float)

        zc = df["z0"].to_numpy(float) + PATCH_SIZE / 2
        yc = df["y0"].to_numpy(float) + PATCH_SIZE / 2
        xc = df["x0"].to_numpy(float) + PATCH_SIZE / 2

        depth_raw = (zc - zlo) / np.maximum(zhi - zlo, 1e-9)
        depth = np.clip(depth_raw, 0.0, 1.0)

        d = np.minimum.reduce([
            np.minimum(zc - zlo, zhi - zc),
            np.minimum(yc - ylo, yhi - yc),
            np.minimum(xc - xlo, xhi - xc),
        ])
        dist = np.clip(d, 0.0, DIST_CAP) / DIST_CAP

        phi = df["phi"].to_numpy(float)
        por_raw = np.log(phi + POR_EPS)

        frames[split] = pd.DataFrame({
            "source_row": df["source_row"].to_numpy(np.int64),
            "cond_depth": depth.astype(np.float32),
            "cond_dist": dist.astype(np.float32),
            "cond_por_raw": por_raw.astype(np.float32),
        })
        stats[split] = {
            "n": int(len(df)),
            "cond_depth": {"min": float(depth.min()), "max": float(depth.max()),
                           "mean": float(depth.mean())},
            "cond_dist": {"min": float(dist.min()), "max": float(dist.max()),
                          "mean": float(dist.mean())},
            "cond_por_raw": {"min": float(por_raw.min()), "max": float(por_raw.max()),
                             "mean": float(por_raw.mean()), "std": float(por_raw.std())},
            "frac_depth_clipped": float(np.mean((depth_raw < 0) | (depth_raw > 1))),
            "frac_dist_clipped_low": float(np.mean(d < 0)),
            "frac_dist_at_cap": float(np.mean(d >= DIST_CAP)),
        }

    tr = frames["train"]["cond_por_raw"].to_numpy(np.float64)
    por_mean, por_std = float(tr.mean()), float(tr.std())

    for split, f in frames.items():
        f.to_parquet(store / split / "cond.parquet", index=False)

    return {
        "por_mean": por_mean,
        "por_std": por_std,
        "per_split": stats,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--store", default=str(DATA_ROOT / "latents_r07z4"))
    args = ap.parse_args()
    store = Path(args.store)

    print("[1/3] building orientation field ...", flush=True)
    field = build_orientation_field()
    ORIENT_OUT.parent.mkdir(parents=True, exist_ok=True)
    with open(ORIENT_OUT, "w") as fh:
        json.dump(field, fh, indent=1)
    s = field["summary"]
    print(f"      wrote {ORIENT_OUT.relative_to(REPO)}")
    print(f"      {s['n_orientation_usable']}/{s['n_volumes']} volumes usable; "
          f"confidence {s['confidence_counts']}; "
          f"{s['n_margin_below_2deg']} with <2 deg hypothesis margin "
          f"({s['n_margin_below_2deg_excluding_symmetric']} excluding symmetric layups); "
          f"{s['n_verification_failures']} verification failures")

    print("[2/3] building per-patch scalars ...", flush=True)
    sc = build_scalars(store, field)
    for split, st in sc["per_split"].items():
        print(f"      {split:5s} n={st['n']:>9,d} "
              f"depth[{st['cond_depth']['min']:.3f},{st['cond_depth']['max']:.3f}] "
              f"dist[{st['cond_dist']['min']:.3f},{st['cond_dist']['max']:.3f}] "
              f"por_raw mean={st['cond_por_raw']['mean']:.3f}")
    print(f"      train standardisation: mean={sc['por_mean']:.6f} std={sc['por_std']:.6f}")

    print("[3/3] updating store metadata ...", flush=True)
    meta_path = store / "metadata.json"
    meta = json.load(open(meta_path))
    meta["voxel_size_um"] = VOXEL_SIZE_UM
    meta["assembly"] = assembly_metadata()
    meta["conditioning"] = {
        "version": "ldm05-v1",
        "created": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "builder": "scripts/build_conditioning.py",
        "sidecar_file": "cond.parquet",
        "sidecar_alignment": "row i of cond.parquet is row i of index.parquet",
        "voxel_size_um": VOXEL_SIZE_UM,
        "orientation_field": str(ORIENT_OUT.relative_to(REPO)),
        "orientation_field_sha256": sha256(ORIENT_OUT),
        "porosity_transform": f"log(phi + {POR_EPS})",
        "porosity_denominator": "64**3 (unchanged — full patch volume)",
        "por_standardisation": {
            "computed_over": "train",
            "mean": sc["por_mean"],
            "std": sc["por_std"],
        },
        "depth_definition": (
            "(z_centre - z_top) / (z_bot - z_top), z_top/z_bot = outer specimen "
            "faces from the T-A foreground extent (half-max threshold), clipped to [0,1]"
        ),
        "dist_definition": (
            f"min(d, {int(DIST_CAP)}) / {int(DIST_CAP)}, d = distance in voxels from "
            "the patch centre to the nearest outer specimen surface over all three axes"
        ),
        "orient_definition": (
            "(cos 2t, sin 2t) over the patch's 64 depth voxels, COMPONENTS "
            "mean-pooled in groups of 4 to 16 depth planes (never renormalised), "
            "broadcast over the two in-plane latent axes -> (2,16,16,16)"
        ),
        "neighbour_schedule": {
            "n_neighbours": len(NEIGHBOUR_DIRS),
            "neighbour_dirs": [list(d) for d in NEIGHBOUR_DIRS],
            "group_ordering": [list(g) for g in PARITY_GROUP_ORDER],
            "group_rank": "4*pz + 2*py + px, generated in ascending rank",
            "availability_rule": (
                "a face neighbour flips exactly one parity bit; it is EXISTS if "
                "its group rank is lower than the target's, UNKNOWN if higher, "
                "OOB if the store holds no patch at that position.  Equivalently "
                "both neighbours on axis a are EXISTS iff the target's parity bit "
                "a is 1, giving 0/2/4/6 known neighbours by group"
            ),
            "shift": "none — the full neighbour latent is used as face-adjacent context",
        },
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
