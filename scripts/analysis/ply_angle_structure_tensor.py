"""0c — how well can ply angles be read on REAL volumes, with nothing fitted?

The eval-v4 layup metric will be judged against generated volumes, so it needs
a floor: what does the SAME reader score on real scans, where the answer is
known and no offset, sign or face-order freedom is granted?  This script
measures that floor for three readers on the same ply blocks of the same
volumes:

    structure_tensor  3-D structure tensor of the grey volume.  Gaussian
                      gradients (sigma_grad), Gaussian integration
                      (sigma_int), tensor averaged over the ply block, the
                      in-plane (y, x) 2x2 part taken, and the fibre direction
                      read off the eigenvector of the SMALLEST eigenvalue —
                      intensity varies least along a tow.
    fft_slice         the T-I angular-spectrum reader, imported unchanged
                      through ``scripts/analysis/layup_roundtrip.measure_volume``.
    pore_axes         the T-I pore principal-axis reader, on the STORED mask,
                      same import.

Scores are DIRECT: measured vs ``ply_angle_image_deg`` from
``data/split_v2/orientation_field.json``, with no offset, no sign flip and no
face reversal.  Strict 4-class accuracy rounds both to the nearest of
0/45/90/135 degrees.

Geometry: one 1024x1024 in-plane window per volume — the size the T-I reader
was built for — chosen to be inside ``sample_mask`` over the whole laminate,
which also puts it away from the three drilled holes.  All three readers see
the same window and the same ply blocks, so the comparison is like for like.

Two aggregations of the structure tensor are reported.  ``tensor_mean`` is the
one specified: average the tensor over the ply block, then diagonalise.  A
block average of a Gaussian-smoothed field is the block average of the
unsmoothed one up to boundary effects, so ``sigma_int`` does not enter it and
its rows repeat across the sweep.  ``local`` applies the integration Gaussian,
reads an angle per (y, x) column and takes the coherence-weighted circular mean
of the doubled angle — that is the reader the sigma sweep actually informs.

Outputs -> ``runs/campaigns/08-pre-ldm06-diagnostics/angle_reader_floor/``:
results.json, per_ply.csv, findings.md.

Usage:
    python scripts/analysis/ply_angle_structure_tensor.py
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import zarr
from scipy import ndimage as ndi

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import REPO, ZARR_ROOT, write_findings, write_json  # noqa: E402
from _real_windows import CELL, cell_ok_by_slice, find_window_best  # noqa: E402

sys.path.insert(0, str(REPO / "src"))

import layup_roundtrip as lr  # noqa: E402  (T-I readers, imported not forked)

OUT_DIR = REPO / "runs/campaigns/08-pre-ldm06-diagnostics/angle_reader_floor"
ORIENT_FIELD = REPO / "data/split_v2/orientation_field.json"
WINDOW = 1024                       # the window the T-I reader was built for
SIGMA_GRAD = 1.5                    # Gaussian derivative scale, voxels
SIGMA_INT = (8.0, 12.0, 16.0)       # integration scales to sweep, voxels
ERODE_VOX = 3                       # sample_mask erosion before measuring

# 4 val + 1 test, all orientation_usable, spread over the three families and
# both confidence levels.  Na_09_4 and Na_01_3 are the high-confidence pair.
VOLUMES = {
    "Pegaso_1_26": ("val", "MedidasDB__Airbus_Panel_Pegaso_probetas_1_26_volumen_eq_aligned"),
    "Na_02_2": ("val", "MedidasDB__Fabricacion_Nacho_05_Probetas_Nacho_2025_probetas_Na_02_2_volume_eq_aligned"),
    "Na_08_4": ("val", "MedidasDB__Fabricacion_Nacho_05_Probetas_Nacho_2025_probetas_Na_08_4_volume_eq_aligned"),
    "Na_09_4": ("val", "MedidasDB__Fabricacion_Nacho_05_Probetas_Nacho_2025_probetas_Na_09_4_volume_eq_aligned"),
    "Na_01_3": ("test", "MedidasDB__Fabricacion_Nacho_05_Probetas_Nacho_2025_probetas_Na_01_3_volume_eq_aligned"),
}

_T0 = time.time()


def log(msg: str) -> None:
    print(f"[{time.time() - _T0:7.1f}s] {msg}", flush=True)


# ---------------------------------------------------------------------------
# The structure-tensor reader
# ---------------------------------------------------------------------------

def block_tensor(block: np.ndarray, valid: np.ndarray, sigma_grad: float
                 ) -> tuple[dict[tuple[int, int], np.ndarray], np.ndarray]:
    """Per-(y, x) column sums of the structure tensor over one ply block.

    Returns ``(num, den)``: ``num[(a, b)]`` is ``sum_z g_a g_b w`` as a
    (H, W) map and ``den`` is ``sum_z w``.  Collapsing z here is exact for the
    aggregations below: a ply block is ~20 voxels thick and the integration
    sigma is 8-16, so the z direction of the 3-D integration Gaussian is a
    block average already.  The in-plane direction, where it is not, is applied
    afterwards at each sigma — which makes the sigma sweep nearly free.

    Invalid voxels (outside the eroded ``sample_mask``) carry zero weight, so a
    window that clips a hole or the specimen edge does not bias the answer.
    """
    w = valid.astype(np.float32)
    img = block * w                       # zero outside; gradients see a flat field
    g = [ndi.gaussian_filter(img, sigma_grad, order=o, mode="nearest")
         for o in ((1, 0, 0), (0, 1, 0), (0, 0, 1))]   # d/dz, d/dy, d/dx
    num = {}
    for a in range(3):
        for b in range(a, 3):
            num[(a, b)] = (g[a] * g[b] * w).sum(axis=0)
    return num, w.sum(axis=0)


def angles_from_tensor(num: dict[tuple[int, int], np.ndarray],
                       den: np.ndarray, sigma_int: float) -> dict[str, float]:
    """In-plane fibre angle of one ply block, two aggregations.

    ``tensor_mean`` is the specified one: average the tensor over the whole
    block, take the in-plane (y, x) 2x2 part, and read the eigenvector of the
    SMALLEST eigenvalue — intensity varies least along a tow.  A block average
    of a Gaussian-smoothed field is the block average of the unsmoothed one up
    to boundary effects, so ``sigma_int`` does not enter it.

    ``local`` applies the in-plane integration Gaussian, reads an angle per
    (y, x) column, and takes the coherence-weighted circular mean of the
    doubled angle.  That is where the integration scale does work.

    Angles follow the image-frame convention of ``orientation_field.json``:
    0 = image x axis, 90 = image y axis, modulo 180.
    """
    tot = float(den.sum()) + 1e-12
    J = np.zeros((3, 3))
    for (a, b), m in num.items():
        J[a, b] = J[b, a] = float(m.sum()) / tot
    _, evecs = np.linalg.eigh(J[1:, 1:])              # in-plane (y, x) part
    vy, vx = evecs[:, 0]                              # smallest eigenvalue
    tensor_mean = float(np.degrees(np.arctan2(vy, vx)) % 180.0)

    ds = ndi.gaussian_filter(den, sigma_int, mode="nearest")
    safe = np.maximum(ds, 1e-6)
    jyy = ndi.gaussian_filter(num[(1, 1)], sigma_int, mode="nearest") / safe
    jyx = ndi.gaussian_filter(num[(1, 2)], sigma_int, mode="nearest") / safe
    jxx = ndi.gaussian_filter(num[(2, 2)], sigma_int, mode="nearest") / safe
    # Angle psi is measured FROM THE X AXIS, the convention of
    # orientation_field.json, so the direction (v_y, v_x) = (sin psi, cos psi)
    # and v^T J v = (Jyy+Jxx)/2 + (Jxx-Jyy)/2 cos 2psi + Jyx sin 2psi.  That is
    # maximal at 2 psi = atan2(2 Jyx, Jxx - Jyy) — note the denominator is
    # Jxx - Jyy, not Jyy - Jxx, which would give the angle from the Y axis.
    # The fibre is the MINIMUM, a quarter turn away: +pi in doubled angle.
    tr = jyy + jxx
    diff = jxx - jyy
    disc = np.sqrt(diff * diff + 4.0 * jyx * jyx)
    ang2 = np.arctan2(2.0 * jyx, diff)
    coh = np.where(tr > 1e-20, disc / np.maximum(tr, 1e-20), 0.0) * (ds > 0)
    z = (coh * np.exp(1j * (ang2 + np.pi))).sum()
    local = float(np.degrees(np.angle(z) / 2.0) % 180.0)
    return {"tensor_mean": tensor_mean,
            "local": local,
            "coherence": float(np.abs(z) / (coh.sum() + 1e-12))}


def self_test(tol_deg: float = 0.5) -> list[dict]:
    """Read back synthetic fibres at known angles — the reader's validation.

    A block whose intensity is constant along ``t`` and sinusoidal across it is
    a tow bundle at ``t``.  Both aggregations must return ``t``.  This pins the
    angle convention (0 = image x axis) and the smallest-eigenvalue choice; it
    caught a y-referenced doubled angle in the ``local`` aggregation.
    """
    out = []
    for true_deg in (0.0, 30.0, 45.0, 90.0, 135.0, 160.0):
        t = np.deg2rad(true_deg)
        h = w = 256
        yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
        across = -xx * np.sin(t) + yy * np.cos(t)      # perpendicular to the tow
        img = np.repeat((0.5 + 0.4 * np.sin(across * 2 * np.pi / 12.0))[None],
                        20, axis=0).astype(np.float32)
        num, den = block_tensor(img, np.ones(img.shape, bool), SIGMA_GRAD)
        a = angles_from_tensor(num, den, 12.0)
        err = {k: float(abs(lr.wrap180(a[k] - true_deg)))
               for k in ("tensor_mean", "local")}
        out.append({"true_deg": true_deg, **a, "abs_error_deg": err})
        if max(err.values()) > tol_deg:
            raise SystemExit(f"self-test failed at {true_deg} deg: {a}")
    return out


# ---------------------------------------------------------------------------
# Scoring — direct, nothing fitted
# ---------------------------------------------------------------------------

def strict_class(a: np.ndarray) -> np.ndarray:
    return np.round((np.asarray(a, float) % 180.0) / 45.0).astype(int) % 4


def score(measured: np.ndarray, truth: np.ndarray) -> dict:
    err = lr.wrap180(np.asarray(measured, float) - np.asarray(truth, float))
    return {
        "measured_deg": np.asarray(measured, float).tolist(),
        "errors_deg": err.tolist(),
        "median_abs_error_deg": float(np.median(np.abs(err))),
        "max_abs_error_deg": float(np.max(np.abs(err))),
        "frac_within_10": float(np.mean(np.abs(err) < 10.0)),
        "strict_class_accuracy": float(np.mean(
            strict_class(measured) == strict_class(truth))),
    }


# ---------------------------------------------------------------------------
# Per-volume driver
# ---------------------------------------------------------------------------

def run_volume(short: str, split: str, vid: str, g, field: dict) -> dict:
    t0 = time.time()
    rec = field[vid]
    edges = np.asarray(rec["ply_boundaries_z"], int)
    truth = np.asarray(rec["ply_angle_image_deg"], float)
    za, zb = int(edges[0]), int(edges[-1])
    depth = zb - za
    pitch = depth / float(rec["n_plies"])

    arr_x, arr_m, arr_s = g[vid]["xct"], g[vid]["mask"], g[vid]["sample_mask"]
    ok_z = cell_ok_by_slice(arr_s, CELL)
    ok = ok_z[za:zb].all(axis=0)
    k = WINDOW // CELL
    iy, ix, usable = find_window_best(ok, k, k)
    y0, x0 = iy * CELL, ix * CELL
    log(f"{short}: laminate z={za}:{zb} ({rec['n_plies']} plies, pitch "
        f"{pitch:.1f}), window y={y0} x={x0}, usable cells {usable:.3f}")

    sl = np.s_[za:zb, y0:y0 + WINDOW, x0:x0 + WINDOW]
    xct = np.asarray(arr_x[sl])
    mask = np.asarray(arr_m[sl])
    smask = np.asarray(arr_s[sl]) > 0
    valid_all = ndi.binary_erosion(smask, iterations=ERODE_VOX)
    log(f"{short}: loaded {xct.shape}, valid fraction {valid_all.mean():.4f}, "
        f"{time.time() - t0:.0f}s")

    # --- structure tensor, per ply block, per integration sigma -------------
    st: dict[str, dict[str, list[float]]] = {
        f"{s:g}": {"tensor_mean": [], "local": [], "coherence": []}
        for s in SIGMA_INT
    }
    t_st = time.time()
    for i in range(len(edges) - 1):
        z0, z1 = int(edges[i]) - za, int(edges[i + 1]) - za
        num, den = block_tensor(xct[z0:z1].astype(np.float32) / 255.0,
                                valid_all[z0:z1], SIGMA_GRAD)
        for sg in SIGMA_INT:
            out = angles_from_tensor(num, den, sg)
            for key, v in out.items():
                st[f"{sg:g}"][key].append(v)
    st_wall = time.time() - t_st
    log(f"{short}: structure tensor done ({len(SIGMA_INT)} sigmas), {st_wall:.0f}s")

    # --- T-I readers on the same blocks -------------------------------------
    t_ti = time.time()
    est = lr.measure_volume(xct, mask, edges - za, pitch)
    ti_wall = time.time() - t_ti
    log(f"{short}: T-I readers done, {ti_wall:.0f}s")

    readers: dict[str, dict] = {}
    for s in SIGMA_INT:
        key = f"{s:g}"
        readers[f"structure_tensor_mean_sig{key}"] = score(
            st[key]["tensor_mean"], truth)
        readers[f"structure_tensor_local_sig{key}"] = score(
            st[key]["local"], truth)
        readers[f"structure_tensor_local_sig{key}"]["mean_coherence"] = float(
            np.mean(st[key]["coherence"]))
    for name in ("fft_slice", "pore_axes", "combined"):
        if name in est:
            readers[name] = score(est[name]["angles"], truth)

    return {
        "volume": short, "split": split, "volume_id": vid,
        "confidence": rec["confidence"],
        "n_plies": int(rec["n_plies"]),
        "laminate_z": [za, zb],
        "ply_boundaries_z": edges.tolist(),
        "window": {"y0": y0, "x0": x0, "size": WINDOW,
                   "usable_cell_fraction": usable},
        "valid_voxel_fraction": float(valid_all.mean()),
        "truth_image_deg": truth.tolist(),
        "wall_s": {"total": round(time.time() - t0, 1),
                   "structure_tensor_all_sigmas": round(st_wall, 1),
                   "ti_readers": round(ti_wall, 1)},
        "readers": readers,
    }


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def reader_order(records: list[dict]) -> list[str]:
    return list(records[0]["readers"].keys())


def pooled(records: list[dict], name: str) -> dict:
    err = np.concatenate([np.abs(r["readers"][name]["errors_deg"])
                          for r in records])
    acc = [r["readers"][name]["strict_class_accuracy"] for r in records]
    return {"median_abs_error_deg": float(np.median(err)),
            "frac_within_10": float(np.mean(err < 10.0)),
            "strict_class_accuracy": float(np.mean(acc)),
            "n_plies": int(err.size)}


def build_findings(records: list[dict], agg: dict) -> str:
    names = reader_order(records)
    L = ["# 0c — real-volume floor for the ply-angle readers", "",
         f"{len(records)} real volumes ({sum(r['split'] == 'val' for r in records)} "
         f"val, {sum(r['split'] == 'test' for r in records)} test), all "
         "`orientation_usable`. One centred-as-possible "
         f"{WINDOW}x{WINDOW} in-plane window per volume, inside `sample_mask` "
         "over the whole laminate; the same window and the same ply blocks "
         "feed every reader. Truth is `ply_angle_image_deg`. Scores are "
         "DIRECT — no offset, sign or face-order fit.", "",
         f"Structure tensor: Gaussian derivatives at sigma_grad = "
         f"{SIGMA_GRAD}, integration sigma swept over "
         f"{', '.join(f'{s:g}' for s in SIGMA_INT)} voxels, `sample_mask` "
         f"eroded {ERODE_VOX} voxels.", "",
         "## Pooled over all plies of all volumes", "",
         "| reader | median \\|err\\| | <=10 deg | strict 4-class |",
         "|---|---|---|---|"]
    for n in names:
        a = agg[n]
        L.append(f"| {n} | {a['median_abs_error_deg']:.1f} deg "
                 f"| {100 * a['frac_within_10']:.0f}% "
                 f"| {100 * a['strict_class_accuracy']:.1f}% |")
    L += ["", "## Per volume — median |error| (deg) / strict 4-class (%)", "",
          "| reader | " + " | ".join(r["volume"] for r in records) + " |",
          "|---" * (len(records) + 1) + "|"]
    for n in names:
        cells = []
        for r in records:
            s = r["readers"][n]
            cells.append(f"{s['median_abs_error_deg']:.1f} / "
                         f"{100 * s['strict_class_accuracy']:.0f}")
        L.append(f"| {n} | " + " | ".join(cells) + " |")
    L += ["", "## Wall time per volume", "",
          "| volume | split | confidence | window usable | total | ST (all sigmas) "
          "| T-I readers |", "|---|---|---|---|---|---|---|"]
    for r in records:
        w = r["wall_s"]
        L.append(f"| {r['volume']} | {r['split']} | {r['confidence']} "
                 f"| {r['window']['usable_cell_fraction']:.3f} "
                 f"| {w['total']:.0f}s | {w['structure_tensor_all_sigmas']:.0f}s "
                 f"| {w['ti_readers']:.0f}s |")
    L += ["", "## Caveats", "",
          "- `tensor_mean` is the specified aggregation: the tensor is "
          "averaged over the whole ply block before diagonalising. A block "
          "average of a Gaussian-smoothed field equals the block average of "
          "the unsmoothed one up to boundary effects, so the integration "
          "sigma does not enter it and its rows are identical across the "
          "sweep. `local` applies the integration Gaussian in-plane, reads an "
          "angle per column, and takes the coherence-weighted circular mean "
          "of the doubled angle — that is where the scale does work.",
          "- A ply block is only ~20 voxels thick and sigma_int is 8-16, so "
          "the z direction of the 3-D integration Gaussian is a block average "
          "already; it is taken exactly (sum over z) and the Gaussian is "
          "applied in-plane. This is an approximation of the isotropic 3-D "
          "form, and it makes the sigma sweep nearly free.",
          "- `pore_axes` reads the STORED mask, not a predicted one, so its "
          "score is the best any mask-based reader could do here.",
          "- The reader is validated on synthetic tow bundles at six known "
          "angles (`self_test`): both aggregations read them back to under "
          "0.5 deg. That check pins the convention (0 = image x axis) and the "
          "smallest-eigenvalue choice.",
          "- Truth is the NOMINAL ply sequence rotated into the image frame "
          "(`orientation_field.json` excludes measured per-ply deviations), so "
          "part of the residual error is the truth's own.", ""]
    return "\n".join(L)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--volumes", nargs="*", default=list(VOLUMES))
    args = ap.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    st_check = self_test()
    log(f"self-test passed: {len(st_check)} synthetic angles read back to "
        f"<0.5 deg by both aggregations")
    g = zarr.open_group(str(ZARR_ROOT), mode="r")
    field = json.loads(ORIENT_FIELD.read_text())["volumes"]

    records = [run_volume(s, VOLUMES[s][0], VOLUMES[s][1], g, field)
               for s in args.volumes]
    agg = {n: pooled(records, n) for n in reader_order(records)}

    rows = []
    for r in records:
        for n, s in r["readers"].items():
            for i, (m, e) in enumerate(zip(s["measured_deg"], s["errors_deg"])):
                rows.append({"volume": r["volume"], "split": r["split"],
                             "reader": n, "ply": i,
                             "truth_deg": r["truth_image_deg"][i],
                             "measured_deg": m, "error_deg": e})
    pd.DataFrame(rows).to_csv(OUT_DIR / "per_ply.csv", index=False)

    write_json({
        "campaign": "08 — 0c real-volume floor for ply-angle readers",
        "question": "How well can ply angles be read on real volumes with no "
                    "offset/sign/face fit?",
        "window_vox": WINDOW,
        "sigma_grad": SIGMA_GRAD,
        "sigma_int_swept": list(SIGMA_INT),
        "erode_vox": ERODE_VOX,
        "truth_source": str(ORIENT_FIELD),
        "self_test": st_check,
        "ti_reader_source": "scripts/analysis/layup_roundtrip.py:measure_volume "
                            "(imported; t_i_layup_validation math unchanged)",
        "pooled": agg,
        "records": records,
    }, OUT_DIR)
    write_findings(build_findings(records, agg), OUT_DIR)
    log(f"wrote {OUT_DIR}/results.json, per_ply.csv and findings.md")


if __name__ == "__main__":
    main()
