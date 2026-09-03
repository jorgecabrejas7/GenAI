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
from poregen.eval_v4.cases import CHUNK_TILES, SHAPE_LARGE, SHAPE_SMALL, WINDOW_STRIDE
from poregen.eval_v4.generate import latent_material_map
from poregen.eval_v4.io import (
    LATENT_DOWNSAMPLE,
    load_cases,
    repo_root,
    save_case,
    volumes_dir,
    write_results,
)
from poregen.eval_v4.manifest import Manifest, head_commit

logger = logging.getLogger(__name__)

ASSESSMENT = "real_floor"
#: The shapes the generated assessments use, so the floor is measured at the
#: same scale.  The geometry and assembly shapes reuse the small floor.
FLOOR_SHAPES = {"small": SHAPE_SMALL, "large": SHAPE_LARGE}
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
    for vid in vol_ids:
        if vid not in g:
            records.append({"volume_id": vid, "skipped": "not in volumes.zarr"})
            continue
        ok_z = rw.cell_ok_by_slice(g[vid]["sample_mask"])
        for tag in shapes:
            shape = FLOOR_SHAPES[tag]
            rec = _one_crop(root, g[vid], vid, tag, shape, ok_z, rw, commit)
            records.append(rec)
            logger.info("%s %s: %s", vid, tag, rec.get("skipped") or "written")
    return records


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
    shapes: tuple[str, ...] = ("small", "large"),
    max_volumes: int | None = None,
    rebuild: bool = False,
) -> dict:
    """Cut the crops if they are not there, then measure them."""
    root = Path(root)
    existing = list(load_cases(root, ASSESSMENT)) if not rebuild else []
    crops: list[dict] = []
    if not existing:
        crops = build_floor_volumes(
            root, data_root=data_root, repo=repo, shapes=shapes, max_volumes=max_volumes
        )
    results = measure_floor(root, repo)
    if crops:
        results["crops"] = crops
        write_results(root, ASSESSMENT, results)
    return results
