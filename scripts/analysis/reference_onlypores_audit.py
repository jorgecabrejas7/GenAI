"""Run the reference onlypores pipeline on a raw TIFF and compare it with the stored labels.

The reference is ``UTvsXCT-preprocessing/produccion/onlypores/onlypores_batch.ipynb``:
reslice for wall detection, ``aligner.crop_walls``, then
``onlypores.onlypores(volume, frontwall, backwall, 15, 0.2, 8)``.  Its code is
the installed, commit-pinned ``preprocess_tools`` package, called unchanged.
(At GenAI 826dbbe, when the audit ran, it was imported from the clone at
5d9da5b with ``cv2`` stubbed; the pinned commit is bit-identical to it.)

Three reference runs per volume:
  notebook        : the batch notebook's walls and parameters (15, 0.2, min size 8)
  single_notebook : ``onlypores.ipynb``'s walls and parameters (30, 0.125, min size 8);
                    the batch notebook claims to run the same pipeline but sets
                    other parameters, so both are measured
  defaults : the same code at its function defaults (30, 0.125, no walls,
             no filter) — the call PoreGen's ``compute_mask`` makes

Each is compared voxel by voxel with ``volumes.zarr[<key>]['mask']`` and
``['sample_mask']``, read in z-slabs.  The ``base`` stage saves one
through-thickness (z, x) slice through a drilled hole of the XCT and the stored
masks for the figure.  (At 826dbbe this stage was ``root`` and also computed the
repo-root ``onlypores.py`` column mask; that program is deleted.)

Usage (memory capped, CPU only, one stage per process, walls first):
  for st in walls notebook single_notebook defaults base; do
    systemd-run --user --scope -p MemoryMax=40G -p MemorySwapMax=0 env LOKY_MAX_CPU_COUNT=4 \
      python scripts/analysis/reference_onlypores_audit.py Na_04_2 $st --out <dir>
  done
LOKY_MAX_CPU_COUNT caps the reference's joblib Sauvola: it picks the parallel
path from psutil's system-wide free memory, which does not see the cgroup cap.
"""

from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path

import numpy as np
import zarr
from scipy import ndimage

GENAI = Path(__file__).resolve().parents[2]
ZARR = GENAI / "data/split_v3/volumes.zarr"
RAW = GENAI / "raw_data/MedidasDB"
HOLES = GENAI / "data/split_v3/holes.json"
SLAB = 64


def import_reference():
    from preprocess_tools import aligner, io, onlypores, reslicer
    return aligner, io, onlypores, reslicer


def clean_pores_lean(pores, min_size, *, drop_equal):
    """``preprocess_tools.onlypores.clean_pores`` with int32 labels and a lookup table.

    Same rule: 26-connected components with a bounding box of >= 2 voxels on
    every axis, and a size cut.  The size cut depends on scikit-image: before
    0.26 ``remove_small_objects(min_size=m)`` removed size < m; from 0.26 the
    deprecated ``min_size`` maps to ``max_size`` and removes size <= m.  The
    reference does not pin scikit-image, so *drop_equal* selects the rule.

    The reference labels in int64 and calls ``np.isin`` on the full label
    volume, which exceeds a 40 GB cap on a 1.1 G-voxel coupon.
    ``check_clean_pores`` bit-compares this function with the reference.
    """
    lab, n = ndimage.label(pores, structure=np.ones((3, 3, 3), bool), output=np.int32)
    sizes = np.bincount(lab.ravel(), minlength=n + 1)
    big = sizes > min_size if drop_equal else sizes >= min_size
    keep = np.zeros(n + 1, bool)
    for i, sl in enumerate(ndimage.find_objects(lab), start=1):
        if sl is not None and big[i] and all(s.stop - s.start >= 2 for s in sl):
            keep[i] = True
    return keep[lab]


def check_clean_pores(ref, pores, min_size, z0=0, n=48):
    """Bit-compare clean_pores_lean with the reference clean_pores on a slab (installed skimage)."""
    import skimage
    sub = np.ascontiguousarray(pores[z0:z0 + n, :1024, :1024])
    a = ref.clean_pores(sub, min_size=min_size)
    b = clean_pores_lean(sub, min_size, drop_equal=int(skimage.__version__.split(".")[1]) >= 26)
    return {"skimage": skimage.__version__, "slab_voxels": int(sub.size), "input_pores": int(sub.sum()),
            "ref_kept": int(a.sum()), "lean_kept": int(b.sum()), "identical": bool(np.array_equal(a, b))}


def compare(pores, sample, zg, hole_yx):
    """Slab-wise comparison against the stored arrays. Returns totals and z profiles."""
    D = pores.shape[0]
    prof = {k: np.zeros(D, np.int64) for k in (
        "shared", "ref_only_in_stored_sample", "ref_only_out_stored_sample", "labels_only",
        "sample_both", "sample_ref_only", "sample_stored_only", "labels_pores", "ref_pores")}
    in_hole = {"ref_only": 0, "labels_only": 0}
    for z0 in range(0, D, SLAB):
        z1 = min(z0 + SLAB, D)
        lab = np.asarray(zg["mask"][z0:z1]).astype(bool)
        ssm = np.asarray(zg["sample_mask"][z0:z1]).astype(bool)
        p = pores[z0:z1]
        s = sample[z0:z1]
        both = p & lab
        ro = p & ~lab
        lo = lab & ~p
        ax = (1, 2)
        prof["shared"][z0:z1] = both.sum(ax)
        prof["ref_only_in_stored_sample"][z0:z1] = (ro & ssm).sum(ax)
        prof["ref_only_out_stored_sample"][z0:z1] = (ro & ~ssm).sum(ax)
        prof["labels_only"][z0:z1] = lo.sum(ax)
        prof["labels_pores"][z0:z1] = lab.sum(ax)
        prof["ref_pores"][z0:z1] = p.sum(ax)
        prof["sample_both"][z0:z1] = (s & ssm).sum(ax)
        prof["sample_ref_only"][z0:z1] = (s & ~ssm).sum(ax)
        prof["sample_stored_only"][z0:z1] = (ssm & ~s).sum(ax)
        in_hole["ref_only"] += int((ro & hole_yx[None]).sum())
        in_hole["labels_only"] += int((lo & hole_yx[None]).sum())
    tot = {k: int(v.sum()) for k, v in prof.items()}
    tot["in_dilated_hole_footprint"] = in_hole
    sref = tot["sample_both"] + tot["sample_ref_only"]
    sst = tot["sample_both"] + tot["sample_stored_only"]
    tot["vvf_ref_pct"] = 100 * tot["ref_pores"] / sref
    tot["vvf_labels_pct"] = 100 * tot["labels_pores"] / sst
    tot["sample_ref_frac"] = sref / int(np.prod(pores.shape))
    tot["sample_stored_frac"] = sst / int(np.prod(pores.shape))
    return tot, {k: v.tolist() for k, v in prof.items()}


def ref_bbox(xct, margin=2):
    """The reference onlypores bounding box (np.where min/max + 2-voxel margin) without np.where."""
    lims = []
    for ax in range(3):
        other = tuple(i for i in range(3) if i != ax)
        idx = np.nonzero(np.any(xct > 0, axis=other))[0]
        lims.append((max(0, int(idx[0]) - margin), min(xct.shape[ax] - 1, int(idx[-1]) + margin)))
    return tuple(slice(lo, hi + 1) for lo, hi in lims)


def composed_onlypores(ref, volume, sample_cropped, frontwall=0, backwall=0,
                       sauvola_radius=30, sauvola_k=0.125):
    """``preprocess_tools.onlypores.onlypores`` (min_size_filtering=-1), steps 1-9, split so
    ``ref.material_mask`` can run in its own process.  Calls the reference's
    ``sauvola_thresholding``; wall and AND steps copy onlypores.py lines 358-383."""
    bb = ref_bbox(volume)
    binary_cropped = ref.sauvola_thresholding(volume[bb], window_size=sauvola_radius, k=sauvola_k)
    if frontwall > 0:
        binary_cropped[:frontwall, :, :] = True
    if backwall > 0:
        binary_cropped[backwall:, :, :] = True
    sample = np.zeros(volume.shape, bool)
    sample[bb] = sample_cropped
    pores = np.zeros(volume.shape, bool)
    pores[bb] = ~binary_cropped & sample_cropped
    return pores, sample


RUNS = {  # name -> (sauvola_radius, sauvola_k, min_size_filtering, use walls)
    "notebook": (15, 0.2, 8, True),          # onlypores_batch.ipynb cell 5
    "single_notebook": (30, 0.125, 8, True),  # onlypores.ipynb cell 6
    "defaults": (30, 0.125, -1, False),       # function defaults = PoreGen compute_mask
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("specimen", help="e.g. Na_04_2")
    ap.add_argument("stage", choices=["walls", "sample", *RUNS, "base"],
                    help="one stage per process: the reference's np.where bounding box alone "
                         "peaks near 30 GB on a full coupon")
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--composed", action="store_true",
                    help="use composed_onlypores with the 'sample' stage's ref.material_mask "
                         "(for coupons where the single onlypores call exceeds the memory cap)")
    a = ap.parse_args()
    out = a.out / a.specimen
    out.mkdir(parents=True, exist_ok=True)

    aligner, io, ref, reslicer = import_reference()
    g = zarr.open(str(ZARR), mode="r")
    key = next(k for k in g.group_keys() if f"_{a.specimen}_volume" in k)
    zg = g[key]
    tif = RAW / f"{key.split('__', 1)[1]}.tif"
    holes = json.load(open(HOLES))["volumes"][key]
    yc = int(round(holes["holes"][0]["centre_yx"][0]))
    volume = io.load_tif(tif)

    if a.stage == "walls":
        # ── notebook cell 7, process_volume: reslice, crop_walls ──
        report = {"key": key, "tif": str(tif), "dtype": str(volume.dtype), "shape": list(volume.shape),
                  "raw_tif_equals_zarr_xct": bool(all(
                      np.array_equal(volume[z0:z0 + SLAB], np.asarray(zg["xct"][z0:z0 + SLAB]))
                      for z0 in range(0, volume.shape[0], SLAB)))}
        resliced = reslicer.reslice(reslicer.rotate_90(volume, False), "Right")
        report["resliced_shape"] = list(resliced.shape)
        _, fw, bw = aligner.crop_walls(resliced)
        nz = np.nonzero(np.any(volume > 0, axis=(1, 2)))[0]
        crop_z0 = max(0, int(nz[0]) - 2)
        report["walls"] = {"frontwall": int(fw), "backwall": int(bw),
                           "content_min_z": int(nz[0]), "content_max_z": int(nz[-1]),
                           "crop_origin_z": crop_z0,
                           "forced_material_z_in_volume_coords": [[0, crop_z0 + int(fw)],
                                                                  [crop_z0 + int(bw), volume.shape[0]]]}
        json.dump(report, open(out / "walls.json", "w"), indent=1)

    elif a.stage == "sample":
        np.save(out / "ref_material_mask_cropped.npy", ref.material_mask(volume[ref_bbox(volume)]))

    elif a.stage in RUNS:
        radius, k, mins, use_walls = RUNS[a.stage]
        w = json.load(open(out / "walls.json"))["walls"]
        kw = dict(sauvola_radius=radius, sauvola_k=k, min_size_filtering=-1)
        if use_walls:
            kw.update(frontwall=w["frontwall"], backwall=w["backwall"])
        report = {"params": {**kw, "min_size_filtering": mins}}
        if a.composed:
            kw.pop("min_size_filtering")
            pores, sample = composed_onlypores(
                ref, volume, np.load(out / "ref_material_mask_cropped.npy"), **kw)
        else:
            pores, sample, binary = ref.onlypores(volume, **kw)
            del binary
        gc.collect()
        if mins > 0:
            report["clean_pores_check"] = check_clean_pores(ref, pores, mins, z0=volume.shape[0] // 2 - 24)
            report["pores_before_clean"] = int(pores.sum())
            p_le = clean_pores_lean(pores, mins, drop_equal=True)
            pores = clean_pores_lean(pores, mins, drop_equal=False)
            report["pores_size_eq_min_voxels"] = int(pores.sum() - p_le.sum())
            del p_le
            gc.collect()
        hole_yx = np.load(GENAI / "data/split_v3" / holes["mask_file"]).astype(bool)
        tot, prof = compare(pores, sample, zg, hole_yx)
        report.update(tot)
        tag = f"{a.stage}_composed" if a.composed else a.stage
        np.savez_compressed(out / f"profile_{tag}.npz", **{n: np.array(v) for n, v in prof.items()})
        np.savez_compressed(out / f"slice_{tag}.npz", y=yc, sample=sample[:, yc, :], pores=pores[:, yc, :])
        json.dump(report, open(out / f"summary_{tag}.json", "w"), indent=1)

    else:
        # ── the XCT and stored masks on the figure's slice ──
        np.savez_compressed(
            out / "slice_base.npz", y=yc, xct=volume[:, yc, :],
            stored_sample=np.asarray(zg["sample_mask"][:, yc, :]).astype(bool),
            stored_pores=np.asarray(zg["mask"][:, yc, :]).astype(bool))


if __name__ == "__main__":
    main()
