"""The floor row: real test volumes through the same metrics, first.

Every table in this suite is read against a real-volume row.  A seam ratio of
0.96 is not "nearly perfect"; it is the number a real scan scores, and a
generated volume that reaches it has nothing left to fix.  A pore-cell spread
of 0.004 is not "good local control"; it is what real material does with no
request at all.

Crops come from the split_v3 TEST panels only - the same volumes the model was
never trained on - and are cut to the shapes the generated cases use, so the
comparison is like for like.  A crop carries its own ``sample_mask`` as its
requested material, so every fraction is taken inside real specimen exactly as
a generated fraction is taken inside the requested envelope.

Three metrics have no floor here, on purpose:

* **porosity error** - a real volume was not asked for a porosity.
* **geometry Dice** - it was not asked for a hole.
* **layup recovery** - campaign 08 already measured that floor on real scans
  with the nominal ply sequence as truth; repeating it with a worse truth would
  not improve it.
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

import numpy as np

from poregen.eval_v4 import metrics as M
from poregen.eval_v4.cases import (
    CHUNK_TILES,
    MICRO_TARGETS,
    SHAPE_LARGE,
    SHAPE_SMALL,
    WINDOW_STRIDE,
)
from poregen.eval_v4.generate import latent_material_map
from poregen.eval_v4.io import (
    load_cases,
    repo_root,
    save_case,
    volumes_dir,
    write_results,
)
from poregen.eval_v4.manifest import Manifest, head_commit
from poregen.eval_v4.microstructure import S2_WINDOW

logger = logging.getLogger(__name__)

ASSESSMENT = "real_floor"
#: The shapes the generated assessments use, so the floor is measured at the
#: same scale.  The geometry and assembly shapes reuse the small floor.
FLOOR_SHAPES = {"small": SHAPE_SMALL, "large": SHAPE_LARGE}

#: Side of a microstructure reference crop.  It is the S2 analysis window, and
#: it is the deepest clean box a real laminate actually holds — the generated
#: cases are 192 cubed, and only one test volume carries a clean 192-deep box.
#: Every microstructure statistic is defined on this window or on a
#: size-normalised quantity, so the two shapes are still comparable; see
#: :mod:`poregen.eval_v4.microstructure`.
MICRO_SIDE = S2_WINDOW
MICRO_SHAPE = (MICRO_SIDE, MICRO_SIDE, MICRO_SIDE)
#: A crop further than this from the requested level is written anyway, with the
#: miss recorded: it is the best real material there is, and hiding the miss
#: would let a porosity gap be read as a texture gap.
MICRO_PHI_TOLERANCE = 0.005
#: Below this share of usable 64-cubed cells a large window is not worth taking:
#: more than a tenth of it would be exterior air.
MIN_USABLE_CELL_FRACTION = 0.9
#: The shallowest crop worth measuring.  ``interior_mask`` excludes a 32-voxel
#: shell at each face, so a 64-deep crop would have no interior at all.
MIN_DEPTH = 128


def _real_windows(repo: Path):
    """The campaign-08 box finder, imported not forked.

    ``scripts/analysis`` is not an installable package, so it goes on the path
    here.  ``_real_windows`` is pure numpy over a ``sample_mask`` summary and
    has no campaign or sampler dependency.
    """
    analysis = repo / "scripts" / "analysis"
    if str(analysis) not in sys.path:
        sys.path.insert(0, str(analysis))
    import _real_windows  # noqa: PLC0415  (late by design - see the docstring)

    return _real_windows


def test_volume_ids(data_root: Path) -> list[str]:
    splits = json.loads((data_root / "splits.json").read_text())
    return sorted(v for v, s in splits["volumes"].items() if s == "test")


def _crop(zgroup, z0, y0, x0, shape):
    """Read one box of a real volume as (xct u8, 3-class label u8, sample_mask)."""
    from poregen.dataset.loader import build_label  # noqa: PLC0415

    d, h, w = shape
    sl = np.s_[z0:z0 + d, y0:y0 + h, x0:x0 + w]
    xct = np.asarray(zgroup["xct"][sl], np.uint8)
    mask = np.asarray(zgroup["mask"][sl]) > 0
    smask = np.asarray(zgroup["sample_mask"][sl]) > 0
    return xct, build_label(mask, smask), smask


def build_floor_volumes(
    root: str | Path,
    *,
    data_root: str | Path | None = None,
    repo: str | Path | None = None,
    shapes: tuple[str, ...] = ("small", "large"),
    max_volumes: int | None = None,
) -> list[dict]:
    """Cut and write the real crops.  Returns one record per crop attempted."""
    import zarr  # noqa: PLC0415

    repo = Path(repo) if repo else repo_root()
    data_root = Path(data_root) if data_root else repo / "data" / "split_v3"
    rw = _real_windows(repo)
    commit = head_commit(repo)

    g = zarr.open_group(str(data_root / "volumes.zarr"), mode="r")
    vol_ids = test_volume_ids(data_root)
    if max_volumes:
        vol_ids = vol_ids[:max_volumes]

    records: list[dict] = []
    box_shapes = [t for t in shapes if t in FLOOR_SHAPES]
    for vid in vol_ids:
        if vid not in g:
            records.append({"volume_id": vid, "skipped": "not in volumes.zarr"})
            continue
        if not box_shapes:
            continue
        ok_z = rw.cell_ok_by_slice(g[vid]["sample_mask"])
        for tag in box_shapes:
            shape = FLOOR_SHAPES[tag]
            rec = _one_crop(root, g[vid], vid, tag, shape, ok_z, rw, commit)
            records.append(rec)
            logger.info("%s %s: %s", vid, tag, rec.get("skipped") or "written")

    if SURFACE_TAG in shapes:
        build_surface_floor(root, g, vol_ids, commit=commit)

    if MICRO_TAG in shapes:
        records += build_micro_reference(
            root, g, vol_ids, data_root=data_root, rw=rw, commit=commit
        )
    return records


# ---------------------------------------------------------------------------
# Real surface roughness — the floor for the `surface` assessment
# ---------------------------------------------------------------------------

SURFACE_TAG = "surface"
#: Matches metrics.SURFACE_RIM_VOX so the real and generated numbers exclude
#: the same width of partial-volume rim and stay comparable.
SURFACE_RIM_VOX_FLOOR = 2
#: Written beside the crops. Not a Manifest-carrying case: a roughness floor is
#: a statistic over a whole real volume, not a cut volume, and inventing a
#: manifest for it would claim a provenance the number does not have.
SURFACE_FLOOR_FILE = "surface_floor.json"


def _height_maps_from_mask(zarr_mask, z_chunk: int = 32):
    """First and last material z per (y, x) column, scanning z in chunks.

    The whole ``sample_mask`` of a test volume is ~1 GB as bool; the answer is
    two (H, W) arrays, so it is read a slab at a time and never held.
    """
    d, h, w = zarr_mask.shape
    first = np.full((h, w), -1, dtype=np.int32)
    last = np.full((h, w), -1, dtype=np.int32)
    for z0 in range(0, d, z_chunk):
        blk = np.asarray(zarr_mask[z0:z0 + z_chunk]) > 0
        if not blk.any():
            continue
        present = blk.any(axis=0)
        idx_first = blk.argmax(axis=0).astype(np.int32) + z0
        idx_last = (blk.shape[0] - 1 - blk[::-1].argmax(axis=0)).astype(np.int32) + z0
        fresh = present & (first < 0)
        first[fresh] = idx_first[fresh]
        last[present] = idx_last[present]
    return first, last, d


def _detrend_plane(h: np.ndarray, valid: np.ndarray) -> np.ndarray:
    """Height with a least-squares plane removed, NaN outside *valid*.

    A real coupon is tilted and warped in the scanner frame, and that long-range
    shape is not roughness: measured about the mean plane, Sa on two crops of
    the same volume came out 0.479 and 3.295 voxels, which is the tilt talking,
    not the surface. Removing a plane leaves the short-range texture the
    generated surface is actually being compared against.
    """
    ys, xs = np.nonzero(valid)
    if ys.size < 16:
        return np.full(h.shape, np.nan)
    z = h[ys, xs].astype(np.float64)
    A = np.column_stack([ys.astype(np.float64), xs.astype(np.float64), np.ones(ys.size)])
    coef, *_ = np.linalg.lstsq(A, z, rcond=None)
    out = np.full(h.shape, np.nan)
    out[ys, xs] = z - A @ coef
    return out


def _correlation_length(h: np.ndarray, valid: np.ndarray, max_lag: int = 64) -> dict:
    """Lateral correlation length of a height map, in voxels, per axis.

    The lag at which the normalised autocovariance first falls below 1/e. A
    surface is not described by its amplitude alone: Sa says how far the height
    wanders, the correlation length says over what distance, and a random field
    built with the right Sa and the wrong correlation length looks nothing like
    the real thing.

    Computed on the mean-removed height over valid columns only, one axis at a
    time, so a coupon whose valid region is not rectangular still contributes.
    """
    out: dict = {}
    hh = np.where(valid, h.astype(np.float64), np.nan)
    hh = hh - np.nanmean(hh)
    for axis, name in ((1, "x"), (0, "y")):
        var = np.nanmean(hh ** 2)
        if not np.isfinite(var) or var <= 0:
            out[name] = None
            continue
        lag_at = None
        for lag in range(1, max_lag + 1):
            a_ = np.take(hh, np.arange(0, hh.shape[axis] - lag), axis=axis)
            b_ = np.take(hh, np.arange(lag, hh.shape[axis]), axis=axis)
            cov = np.nanmean(a_ * b_)
            if not np.isfinite(cov):
                break
            if cov / var < np.exp(-1.0):
                lag_at = lag
                break
        out[name] = int(lag_at) if lag_at is not None else None
    vals = [v for v in out.values() if v is not None]
    out["mean"] = float(np.mean(vals)) if vals else None
    return out


def surface_floor_stats(zarr_mask, zarr_xct=None, z_chunk: int = 32,
                        dark_threshold: int = 182) -> dict | None:
    """Sa and Sq of a real coupon's lower and upper faces, in voxels.

    Columns are excluded when they carry no material (the drilled holes and
    everything outside the coupon) and when the surface touches the first or
    last z slice — there the scan cut the specimen off, so what the height map
    records is the edge of the volume rather than the surface of the part.
    """
    first, last, d = _height_maps_from_mask(zarr_mask, z_chunk)
    have = first >= 0
    interior = have & (first > 0) & (last < d - 1)
    if interior.sum() < 100:
        return None
    out = {"n_columns": int(interior.sum()),
           "n_columns_with_material": int(have.sum()),
           "depth": int(d)}
    for face, arr in (("lower", first), ("upper", last)):
        v = arr[interior].astype(np.float64)
        dev = v - v.mean()
        out[face] = {
            "sa": float(np.abs(dev).mean()),
            "sq": float(np.sqrt((dev ** 2).mean())),
            "mean_z": float(v.mean()),
        }
        # Roughness about a FITTED PLANE, which is the quantity a synthetic
        # rough request has to match. See _detrend_plane.
        det = _detrend_plane(arr, interior)
        dv = det[np.isfinite(det)]
        out[face]["detrended"] = {
            "sa": float(np.abs(dv).mean()) if dv.size else None,
            "sq": float(np.sqrt((dv ** 2).mean())) if dv.size else None,
            "correlation_length_vox": _correlation_length(
                np.nan_to_num(det), np.isfinite(det)),
        }

    # Dark-but-material on REAL material, the floor for the generated number:
    # over all material, and again excluding a rim at each surface. The rim is
    # per-column here — a real coupon's faces are not flat planes, so a fixed z
    # band would exclude the wrong voxels.
    if zarr_xct is not None:
        n_dark = n_all = n_dark_core = n_core = 0
        for z0 in range(0, d, z_chunk):
            z1 = min(d, z0 + z_chunk)
            mblk = np.asarray(zarr_mask[z0:z1]) > 0
            if not mblk.any():
                continue
            xblk = np.asarray(zarr_xct[z0:z1])
            zz = np.arange(z0, z1)[:, None, None]
            keep = mblk & interior[None, :, :]
            dark = keep & (xblk < dark_threshold)
            n_all += int(keep.sum()); n_dark += int(dark.sum())
            rim = ((np.abs(zz - first[None, :, :]) < SURFACE_RIM_VOX_FLOOR)
                   | (np.abs(zz - last[None, :, :]) < SURFACE_RIM_VOX_FLOOR))
            core = keep & ~rim
            n_core += int(core.sum())
            n_dark_core += int((core & (xblk < dark_threshold)).sum())
        out["dark_but_material"] = {
            "all_material": (n_dark / n_all) if n_all else None,
            "excluding_face_rim": (n_dark_core / n_core) if n_core else None,
            "rim_vox": SURFACE_RIM_VOX_FLOOR,
            "n_voxels_all": n_all,
            "n_voxels_excluding_rim": n_core,
            "threshold": int(dark_threshold),
        }
    return out


def build_surface_floor(root, g, vol_ids, *, commit: str) -> dict:
    """Roughness of every test volume's own top and bottom face."""
    per_volume = {}
    for vid in vol_ids:
        if vid not in g:
            continue
        st = surface_floor_stats(g[vid]["sample_mask"],
                                 zarr_xct=g[vid].get("xct"))
        if st is None:
            logger.warning("%s: too few interior columns for a surface floor", vid)
            continue
        per_volume[vid] = st
        logger.info("%s surface: lower Sa %.3f  upper Sa %.3f  (%d columns)",
                    vid, st["lower"]["sa"], st["upper"]["sa"], st["n_columns"])
    dk = [st["dark_but_material"] for st in per_volume.values()
          if st.get("dark_but_material")]
    sa = [st[f]["sa"] for st in per_volume.values() for f in ("lower", "upper")]
    sq = [st[f]["sq"] for st in per_volume.values() for f in ("lower", "upper")]
    out = {
        "git_commit": commit,
        "n_volumes": len(per_volume),
        "n_faces": len(sa),
        "sa_mean": float(np.mean(sa)) if sa else None,
        "sa_sd": float(np.std(sa)) if sa else None,
        "sq_mean": float(np.mean(sq)) if sq else None,
        "sq_sd": float(np.std(sq)) if sq else None,
        "detrended_sa_mean": (
            float(np.mean([st[f]["detrended"]["sa"] for st in per_volume.values()
                           for f in ("lower", "upper")
                           if (st[f].get("detrended") or {}).get("sa") is not None]))
            if per_volume else None),
        "correlation_length_vox_mean": (
            float(np.mean([st[f]["correlation_length_vox"]["mean"]
                           for st in per_volume.values() for f in ("lower", "upper")
                           if st[f].get("correlation_length_vox", {}).get("mean") is not None]))
            if per_volume else None),
        "dark_but_material_all_material": (
            float(np.mean([x["all_material"] for x in dk if x["all_material"] is not None]))
            if dk else None),
        "dark_but_material_excluding_face_rim": (
            float(np.mean([x["excluding_face_rim"] for x in dk
                           if x["excluding_face_rim"] is not None])) if dk else None),
        "per_volume": per_volume,
        "definition": (
            "Sa and Sq of the lower and upper sample_mask surfaces of each test "
            "volume, in voxels, about each face's own mean plane. Columns with "
            "no material, and columns whose surface touches the first or last z "
            "slice, are excluded: there the scan cut the specimen off and the "
            "height map records the edge of the volume, not the part."
        ),
    }
    path = Path(root) / "real_floor" / SURFACE_FLOOR_FILE
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(out, indent=2))
    logger.info("wrote %s", path)
    return out


# ---------------------------------------------------------------------------
# Matched-porosity reference crops for the microstructure assessment
# ---------------------------------------------------------------------------

MICRO_TAG = "micro"


def micro_shape_tag(level: float) -> str:
    return f"{MICRO_TAG}_phi{level:g}"


def _panel_of(data_root: Path) -> dict[str, str]:
    """``volume_id -> panel_id``, from the patch index.

    The pair a microstructure floor compares must come from ONE panel: two
    panels differ in cure and in void population, so a cross-panel pair would
    fold the between-panel spread into the floor and flatter every generated
    number that is read against it.
    """
    import pandas as pd  # noqa: PLC0415

    df = pd.read_parquet(str(data_root / "patch_index.parquet"),
                         columns=["volume_id", "panel_id"])
    return dict(df.drop_duplicates("volume_id").itertuples(index=False, name=None))


def _cell_sums(arr, cell: int, z_chunk: int = 32) -> np.ndarray:
    """``(D, ny, nx)`` — True voxels in each ``cell``-square of each z slice.

    The same one-pass summary trick ``_real_windows.cell_ok_by_slice`` uses for
    the specimen mask, applied to the pore mask.  It makes the porosity of ANY
    64-aligned box a cumulative-sum lookup, so every candidate crop can be
    scored exactly without reading the volume once per candidate.
    """
    d, h, w = arr.shape
    ny, nx = h // cell, w // cell
    out = np.zeros((d, ny, nx), np.int64)
    for z0 in range(0, d, z_chunk):
        z1 = min(d, z0 + z_chunk)
        blk = np.asarray(arr[z0:z1, : ny * cell, : nx * cell]) > 0
        out[z0:z1] = blk.reshape(z1 - z0, ny, cell, nx, cell).sum(axis=(2, 4))
    return out


def _box_sum_2d(a: np.ndarray, k: int) -> np.ndarray:
    """Sum over every ``k x k`` cell window, per z slice."""
    c = np.pad(a.cumsum(1).cumsum(2), ((0, 0), (1, 0), (1, 0)))
    return c[:, k:, k:] - c[:, :-k, k:] - c[:, k:, :-k] + c[:, :-k, :-k]


def _z_window_sum(a: np.ndarray, side: int) -> np.ndarray:
    """Sum over every ``side``-slice run in z; axis 0 becomes ``D - side + 1``."""
    c = np.pad(a.cumsum(0), ((1, 0), (0, 0), (0, 0)))
    return c[side:] - c[:-side]


def _micro_candidates(zgroup, rw, side: int) -> tuple[np.ndarray, np.ndarray]:
    """``(clean, phi)`` over every ``side``-cubed box origin ``(z0, iy, ix)``.

    ``clean`` is True where the whole box lies inside ``sample_mask`` — no
    exterior air and none of the drilled holes.  ``phi`` is the box's exact
    pore fraction.  y and x origins are on the 64-cell grid; z is free, because
    a laminate is only ~200 voxels deep and a 64-aligned z rarely fits.
    """
    cell = rw.CELL
    k = side // cell
    ok = rw.cell_ok_by_slice(zgroup["sample_mask"]).astype(np.int64)
    pore = _cell_sums(zgroup["mask"], cell)
    d = ok.shape[0]
    if d < side or ok.shape[1] < k or ok.shape[2] < k:
        empty = np.zeros((0, 0, 0))
        return empty.astype(bool), empty
    clean = _z_window_sum(_box_sum_2d(ok, k), side) == side * k * k
    phi = _z_window_sum(_box_sum_2d(pore, k), side) / float(side ** 3)
    return clean, phi


def _pick_disjoint(candidates: dict, level: float, side: int, k: int, n: int = 2) -> list[dict]:
    """The ``n`` boxes closest to ``level`` that share no material.

    Greedy, and after each pick every box that overlaps it is struck out.  Two
    boxes in different volumes are disjoint by construction — they are
    different coupons — so a panel with several test volumes usually yields one
    box per volume; a panel with only one (JI_8) has to find two boxes that do
    not overlap inside it, which this handles the same way.
    """
    used = {vid: np.zeros(c.shape, bool) for vid, (c, _) in candidates.items()}
    picks: list[dict] = []
    while len(picks) < n:
        best = None
        for vid, (clean, phi) in candidates.items():
            valid = clean & ~used[vid]
            if not valid.any():
                continue
            miss = np.where(valid, np.abs(phi - level), np.inf)
            flat = int(np.argmin(miss))
            idx = np.unravel_index(flat, miss.shape)
            if best is None or miss[idx] < best[0]:
                best = (float(miss[idx]), vid, idx, float(phi[idx]))
        if best is None:
            break
        miss, vid, (z0, iy, ix), phi = best
        picks.append({"volume_id": vid, "z0": int(z0), "y0": int(iy) * (side // k),
                      "x0": int(ix) * (side // k), "phi": phi, "phi_miss": miss,
                      "cell_index": (int(iy), int(ix))})
        z_lo, z_hi = max(0, z0 - side + 1), z0 + side
        y_lo, y_hi = max(0, iy - k + 1), iy + k
        x_lo, x_hi = max(0, ix - k + 1), ix + k
        used[vid][z_lo:z_hi, y_lo:y_hi, x_lo:x_hi] = True
    return picks


def build_micro_reference(
    root,
    zroot,
    vol_ids,
    *,
    data_root: Path,
    rw,
    commit: str,
    levels=MICRO_TARGETS,
) -> list[dict]:
    """Cut the matched-porosity reference PAIRS the microstructure floor needs.

    For every requested porosity level and every test panel, two crops of the
    same panel whose measured porosity is as close to that level as real
    material gets, and which share no material.  Crop ``a`` is the reference
    the generated set is scored against; ``a`` against ``b`` is the floor.

    A level the panels cannot reach is still written, with the miss recorded in
    the manifest notes and carried into the report — the alternative is to
    compare a generated set against real material at another porosity and let
    the porosity gap be read as a texture gap.
    """
    panels = _panel_of(data_root)
    cache: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for vid in vol_ids:
        if vid not in zroot:
            continue
        clean, phi = _micro_candidates(zroot[vid], rw, MICRO_SIDE)
        if clean.size and clean.any():
            cache[vid] = (clean, phi)
        else:
            logger.info("%s: no clean %d-cubed box; not a microstructure reference",
                        vid, MICRO_SIDE)

    by_panel: dict[str, dict] = {}
    for vid, cands in cache.items():
        by_panel.setdefault(panels.get(vid, "unknown"), {})[vid] = cands

    k = MICRO_SIDE // rw.CELL
    records: list[dict] = []
    for level in levels:
        for panel, cands in sorted(by_panel.items()):
            picks = _pick_disjoint(cands, float(level), MICRO_SIDE, k)
            if len(picks) < 2:
                records.append({"panel_id": panel, "micro_level": float(level),
                                "skipped": "no two disjoint clean boxes in this panel"})
                continue
            for pair, pick in zip(("a", "b"), picks):
                records.append(_write_micro_crop(
                    root, zroot[pick["volume_id"]], panel, pair, float(level),
                    pick, commit,
                ))
                logger.info("micro phi=%g %s/%s: %s phi=%.4f (miss %.4f)",
                            level, panel, pair, pick["volume_id"],
                            pick["phi"], pick["phi_miss"])
    return records


def _write_micro_crop(root, zgroup, panel, pair, level, pick, commit) -> dict:
    """Write one matched-porosity reference crop as a ``real`` case."""
    xct, label, smask = _crop(
        zgroup, pick["z0"], pick["y0"], pick["x0"], MICRO_SHAPE
    )
    if not smask.all():
        raise ValueError(
            f"micro crop {panel}/{pair} at {pick['z0'], pick['y0'], pick['x0']} is "
            "not entirely inside sample_mask, but the candidate search said it "
            "was. The two disagree, so neither can be trusted."
        )
    tag = micro_shape_tag(level)
    manifest = Manifest(
        assessment=ASSESSMENT,
        case=f"{tag}__{panel}__{pair}",
        volume_shape=MICRO_SHAPE,
        git_commit=commit,
        sampler="real",
        chunk_tiles=CHUNK_TILES,
        window_stride=WINDOW_STRIDE,
        decode="overlapped",
        decode_overlap=32,
        requested_material="full",
        notes={
            "volume_id": pick["volume_id"],
            "panel_id": panel,
            "shape_tag": tag,
            "split": "test",
            "micro_level": level,
            "micro_pair": pair,
            "phi_from_cells": pick["phi"],
            "phi_miss": pick["phi_miss"],
            "phi_within_tolerance": bool(pick["phi_miss"] <= MICRO_PHI_TOLERANCE),
            "tolerance": MICRO_PHI_TOLERANCE,
            "requested_shape": list(MICRO_SHAPE),
            "origin_zyx": [pick["z0"], pick["y0"], pick["x0"]],
            "fully_inside_sample_mask": True,
            "usable_cell_fraction": 1.0,
        },
    )
    save_case(volumes_dir(root, ASSESSMENT) / manifest.case, manifest, xct, label)
    return {
        "case": manifest.case, "panel_id": panel, "micro_pair": pair,
        "micro_level": level, "volume_id": pick["volume_id"],
        "shape": list(MICRO_SHAPE), "origin_zyx": [pick["z0"], pick["y0"], pick["x0"]],
        "phi": pick["phi"], "phi_miss": pick["phi_miss"],
    }


def _one_crop(root, zgroup, vid, tag, shape, ok_z, rw, commit) -> dict:
    """Cut the deepest clean box of the requested in-plane size.

    A real laminate holds only ~185-212 voxels of continuous material and the
    outer z slices are the specimen surface, so a 192-deep box that is entirely
    inside ``sample_mask`` usually does not exist - campaign 08 hit the same
    wall and dropped to a 128-deep box.  The depth is therefore reduced a tile
    at a time until a clean box fits, never below :data:`MIN_DEPTH` (the
    interior needs more than two 32-voxel shells).  Every metric here is a
    ratio or a fraction, so a shallower crop does not bias it; the depth that
    was actually taken goes in the manifest.
    """
    d, h, w = shape
    cell = rw.CELL
    ky, kx = h // cell, w // cell
    if ok_z.shape[1] < ky or ok_z.shape[2] < kx:
        return {"volume_id": vid, "shape_tag": tag,
                "skipped": f"the scan is smaller than {h}x{w} in plane"}

    hit, depth = None, None
    for cand in range(d, MIN_DEPTH - 1, -cell):
        if cand > ok_z.shape[0]:
            continue
        hit = rw.find_region(None, cand, ky, kx, ok_z=ok_z)
        if hit is not None:
            depth = cand
            break

    usable = 1.0
    if hit is None:
        # No clean box at any depth: take the window with the most usable cells
        # and carry the sample_mask through as the crop's requested material, so
        # every fraction is still taken inside real specimen.
        depth = min(d, ok_z.shape[0])
        if depth < MIN_DEPTH:
            return {"volume_id": vid, "shape_tag": tag,
                    "skipped": f"the scan is shallower than {MIN_DEPTH} voxels"}
        best = None
        for z0 in range(0, ok_z.shape[0] - depth + 1):
            iy, ix, frac = rw.find_window_best(ok_z[z0:z0 + depth].all(axis=0), ky, kx)
            if best is None or frac > best[2]:
                best = (iy, ix, frac, z0)
        iy, ix, usable, z0 = best
        if usable < MIN_USABLE_CELL_FRACTION:
            return {"volume_id": vid, "shape_tag": tag, "usable_cell_fraction": usable,
                    "depth": depth,
                    "skipped": f"best window is only {usable:.0%} inside sample_mask"}
        hit = {"z0": z0, "y0": iy * cell, "x0": ix * cell, "z_aligned": z0 % cell == 0}

    shape = (depth, h, w)
    xct, label, smask = _crop(zgroup, hit["z0"], hit["y0"], hit["x0"], shape)
    full = bool(smask.all())
    material_map = None if full else latent_material_map(smask)

    manifest = Manifest(
        assessment=ASSESSMENT,
        case=f"{vid}__{tag}",
        volume_shape=shape,
        git_commit=commit,
        sampler="real",
        # For a real crop these are the grid the seam metric is evaluated ON,
        # not a claim about how the scan was made.  They match the generated
        # cases so the two seam numbers are comparable.
        chunk_tiles=CHUNK_TILES,
        window_stride=WINDOW_STRIDE,
        decode="overlapped",
        decode_overlap=32,
        requested_material="full" if full else "requested_material.npy",
        notes={
            "volume_id": vid,
            "shape_tag": tag,
            "split": "test",
            "requested_shape": list(FLOOR_SHAPES[tag]),
            "origin_zyx": [hit["z0"], hit["y0"], hit["x0"]],
            "z_64_aligned": bool(hit["z_aligned"]),
            "usable_cell_fraction": float(usable),
            "fully_inside_sample_mask": full,
        },
    )
    save_case(
        volumes_dir(root, ASSESSMENT) / manifest.case,
        manifest,
        xct,
        label,
        requested_material=material_map,
    )
    return {
        "volume_id": vid, "shape_tag": tag, "case": manifest.case,
        "shape": list(shape), "requested_shape": list(FLOOR_SHAPES[tag]),
        "origin_zyx": [hit["z0"], hit["y0"], hit["x0"]],
        "usable_cell_fraction": float(usable),
        "fully_inside_sample_mask": full,
    }


def measure_floor(root: str | Path, repo: str | Path | None = None) -> dict:
    """Run every request-free metric on the real crops and aggregate by shape."""
    repo = Path(repo) if repo else repo_root()
    cases = load_cases(root, ASSESSMENT)
    if not cases:
        raise FileNotFoundError(
            f"no real crops under {root}/{ASSESSMENT} - run `eval_v4 real-floor "
            f"--root {root}` first."
        )
    detector = M.grey_air_detector(repo)
    rows = []
    for case in cases:
        m = case.manifest
        material = case.material_voxels()
        row = {
            "case": m.case,
            "volume_shape": list(m.volume_shape),
            "notes": m.notes,
            **M.phase_fractions(case.label, material, manifest=m),
            "degenerate": M.degenerate_cells(case.label, material, manifest=m),
            "failure": M.failure_flags(case.label, material, manifest=m),
            "seams": M.seam_metrics(case.xct, manifest=m, pore_logit=None),
            "local": M.local_obedience(
                case.label, material, manifest=m, requested_tiles=None
            ),
            "cross_head": M.cross_head_disagreement(
                case.xct, case.label, material, manifest=m, detector=detector
            ),
        }
        rows.append(row)

    by_shape: dict[str, list] = {}
    for r in rows:
        by_shape.setdefault((r["notes"] or {}).get("shape_tag", "unknown"), []).append(r)

    summary = {}
    for tag, grp in by_shape.items():
        summary[tag] = {
            "n_volumes": len(grp),
            "volume_shape": grp[0]["volume_shape"],
            "phi_pore": M.mean_sd([r["phi_pore"] for r in grp]),
            "air_fraction": M.mean_sd([r["air_fraction"] for r in grp]),
            "air_fraction_interior": M.mean_sd([r["air_fraction_interior"] for r in grp]),
            "seam_xct_ratio": M.mean_sd([r["seams"]["seam_xct_ratio"] for r in grp]),
            "seam_chunk_xct_ratio": M.mean_sd(
                [r["seams"].get("seam_chunk_xct_ratio") for r in grp]),
            "cell_phi_sd": M.mean_sd([r["local"]["delivered_cell_sd"] for r in grp]),
            "cell_phi_mean": M.mean_sd([r["local"]["delivered_cell_mean"] for r in grp]),
            "cross_head_disagreement": M.mean_sd(
                [r["cross_head"]["disagreement_fraction"] for r in grp]),
            "cross_head_disagreement_interior": M.mean_sd(
                [r["cross_head"]["disagreement_fraction_interior"] for r in grp]),
        }

    results = {
        "assessment": ASSESSMENT,
        "question": "What do the metrics score on real volumes, where the answer is known?",
        "note": (
            "The floor row of every table. Porosity error, geometry Dice and layup "
            "recovery have no floor here: a real volume carries no request, and the "
            "layup floor is campaign 08's."
        ),
        "detector": {
            "t_abs": detector.t_abs, "min_cc": detector.min_cc,
            "edge_vox": detector.edge_vox, "source": detector.source,
        },
        "layup_floor": M.layup_floor(repo),
        "per_case": rows,
        "by_shape": summary,
    }
    write_results(root, ASSESSMENT, results)
    return results


def run(
    root: str | Path,
    *,
    data_root: str | Path | None = None,
    repo: str | Path | None = None,
    shapes: tuple[str, ...] = ("small", "large", MICRO_TAG),
    max_volumes: int | None = None,
    rebuild: bool = False,
) -> dict:
    """Cut the crops that are not there yet, then measure all of them.

    Missing shapes are cut one shape at a time rather than all-or-nothing:
    adding the microstructure references to a campaign that already carries the
    small and large crops must not mean re-cutting those, and must not mean
    quietly skipping the new ones because *something* is already there.
    """
    root = Path(root)
    existing = [] if rebuild else list(load_cases(root, ASSESSMENT))
    have = {(c.manifest.notes or {}).get("shape_tag", "") for c in existing}
    todo = tuple(
        s for s in shapes
        if s not in have and not (s == MICRO_TAG and any(t.startswith(MICRO_TAG) for t in have))
    )
    crops: list[dict] = []
    if todo:
        crops = build_floor_volumes(
            root, data_root=data_root, repo=repo, shapes=todo, max_volumes=max_volumes
        )
    results = measure_floor(root, repo)
    if crops:
        results["crops"] = crops
        write_results(root, ASSESSMENT, results)
    return results
