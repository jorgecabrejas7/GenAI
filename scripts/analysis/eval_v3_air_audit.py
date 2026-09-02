"""Air audit v3 — calibrated ABSOLUTE threshold on correctly-decoded volumes.

Methodology and thresholds are those of ``scripts/analysis/air_audit_v2.py``
(imported, not forked): real-calibrated absolute u8 threshold ``T_abs = 182``
(Dice 0.842, precision 0.805, recall 0.883 on 8 real val/test volumes inside
``sample_mask``), minimum connected component 300 voxels, interior/edge split
at a 32-voxel shell, per-64^3-cell mask-collapse correlation.

What changed
------------
v2 had to run the detector through a scale map: the sampler's spurious
``expit`` compressed every generated volume into grey levels [133, 187], so an
absolute threshold calibrated on real scans could not be applied directly.
With the decode fixed, generated volumes carry the REAL u8 grey scale, so
``T_abs`` applies as calibrated — this is the first audit where the absolute
threshold is valid on generated data.

Reported per arm: unmasked air (detector says air, model mask says solid),
interior vs edge shell, mask capture (how much of the detected air the mask
head actually claims), largest components, and the per-cell correlation
between dark-air fraction and mask collapse.

Outputs -> ``runs/eval_v3/air_audit/``: results.json, per_volume.csv,
per_cell.csv, findings.md, figures (PDF + PNG, 300 dpi).

Usage:
    python scripts/analysis/eval_v3_air_audit.py
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import savefig, set_style, write_findings, write_json, plt  # noqa: E402
from _eval_v3 import (  # noqa: E402
    ARM_COLORS, ARMS, EDGE_VOX, MIN_CC, PATCH, ROOT, VOL_ROOT,
    analyse_modes, cc_filter, cellwise, edge_shell, hist_u8, intensity_stats,
    iter_volumes, largest_components, load_calibration, load_mask, load_u8,
    read_stats,
)
from air_audit_v2 import collapse_stats  # noqa: E402

OUT_DIR = ROOT / "air_audit"
_T0 = time.time()


def log(msg: str) -> None:
    print(f"[{time.time() - _T0:7.1f}s] {msg}", flush=True)


def audit_one(exp: str, arm: str, d: Path, t_abs: int
              ) -> tuple[dict, pd.DataFrame, np.ndarray]:
    """One volume: detector at T_abs, CC filter, interior/edge, per-cell."""
    stats = read_stats(d)
    u8, prov = load_u8(d / "volume.tif")
    mask = load_mask(d)
    n_vox = u8.size

    h = hist_u8(u8)
    info = analyse_modes(h)

    det = u8 < t_abs
    det_nocc = float(det.mean())
    det, n_kept, n_tot = cc_filter(det, MIN_CC)
    unmasked = det & ~mask
    captured = det & mask

    edge = edge_shell(u8.shape)
    n_edge = int(edge.sum())
    n_det = int(det.sum())
    n_um = int(unmasked.sum())
    um_edge = int((unmasked & edge).sum())

    comps = largest_components(unmasked)
    row = {
        "experiment": exp, "arm": arm, "name": d.name,
        "dtype_provenance": prov,
        "shape": "x".join(str(s) for s in u8.shape),
        "n_voxels": n_vox,
        "target": stats.get("target"),
        "seed": stats.get("seed"),
        "s_por": stats.get("s_por"),
        "layup": stats.get("layup"),
        "ddim_steps": stats.get("ddim_steps"),
        "raw_min_u8": int(np.nonzero(h)[0][0]),
        "raw_max_u8": int(np.nonzero(h)[0][-1]),
        "p01": info["p01"], "p50": info["p50"], "p99": info["p99"],
        "mean_u8": info["mean"], "std_u8": info["std"],
        "argmax_u8": info["argmax"],
        "n_modes": info["n_modes"], "bimodal": info["bimodal"],
        "dark_mode_u8": info.get("dark_mode"),
        "material_mode_u8": info.get("material_mode"),
        "mode_separation_u8": info.get("mode_separation"),
        "mask_porosity": float(mask.mean()),
        "delivered_mask_porosity": stats.get("delivered_mask_porosity"),
        "T_abs": int(t_abs),
        "detected_air_abs_nocc": det_nocc,
        "detected_air_abs": n_det / n_vox,
        "unmasked_air_abs": n_um / n_vox,
        "captured_air_abs": int(captured.sum()) / n_vox,
        # capture = fraction of detected air that the model mask does claim
        "mask_capture_of_detected": (int(captured.sum()) / n_det) if n_det else float("nan"),
        "n_components_abs": n_kept,
        "n_components_before_cc": n_tot,
        "unmasked_edge_abs": um_edge / n_vox,
        "unmasked_interior_abs": (n_um - um_edge) / n_vox,
        "unmasked_edge_local_abs": um_edge / n_edge,
        "unmasked_interior_local_abs": (n_um - um_edge) / (n_vox - n_edge),
        "edge_over_interior_ratio": ((um_edge / n_edge)
                                     / max((n_um - um_edge) / (n_vox - n_edge), 1e-12)),
        "largest_comp_voxels": comps[0]["voxels"] if comps else 0,
        "largest_comp_mm3": comps[0]["volume_mm3"] if comps else 0.0,
        "largest_comp_equiv_diam_mm": comps[0]["equiv_diameter_mm"] if comps else 0.0,
        "largest_comp_touches_face": comps[0]["touches_face"] if comps else False,
        "top_components": json.dumps(comps),
        "pore_stats": json.dumps(intensity_stats(hist_u8(u8, mask))),
        "material_stats": json.dumps(intensity_stats(hist_u8(u8, ~mask))),
    }

    cells = pd.DataFrame({
        "experiment": exp, "arm": arm, "name": d.name,
        "cell_dark_frac": cellwise(det).ravel(),
        "cell_mask_porosity": cellwise(mask).ravel(),
        "cell_unmasked_frac": cellwise(unmasked).ravel(),
    })
    del u8, mask, det, unmasked, captured
    return row, cells, h


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    set_style()
    cal = load_calibration()
    t_abs = cal["t_abs"]
    log(f"calibration: T_abs={t_abs} dice={cal['dice']:.3f} "
        f"(real volumes, {cal['n_calibration_volumes']}), min_cc={MIN_CC}, "
        f"edge_shell={EDGE_VOX}")

    vols = list(iter_volumes(VOL_ROOT))
    if not vols:
        raise SystemExit(f"no volumes under {VOL_ROOT}")
    rows, cell_frames, hists = [], [], {}
    for i, (exp, arm, d) in enumerate(vols, 1):
        row, cells, h = audit_one(exp, arm, d, t_abs)
        rows.append(row)
        cell_frames.append(cells)
        hists[f"{exp}/{arm}/{d.name}"] = h
        log(f"[{i:3d}/{len(vols)}] {exp}/{arm}/{d.name}  "
            f"range=[{row['raw_min_u8']},{row['raw_max_u8']}]  "
            f"mask={row['mask_porosity']:.4f}  "
            f"det={row['detected_air_abs']:.4f}  "
            f"unmasked={row['unmasked_air_abs']:.4f}  "
            f"int/edge={row['unmasked_interior_local_abs']:.4f}/"
            f"{row['unmasked_edge_local_abs']:.4f}")

    pv = pd.DataFrame(rows)
    cells = pd.concat(cell_frames, ignore_index=True)
    pv.to_csv(OUT_DIR / "per_volume.csv", index=False)
    cells.to_csv(OUT_DIR / "per_cell.csv", index=False)
    np.savez_compressed(OUT_DIR / "histograms.npz",
                        **{k.replace("/", "__"): v for k, v in hists.items()})

    groups = sorted(pv.groupby(["experiment", "arm"]).groups)
    agg = []
    for exp, arm in groups:
        s = pv[(pv.experiment == exp) & (pv.arm == arm)]
        agg.append({
            "experiment": exp, "arm": arm, "n": int(len(s)),
            "mask_porosity": float(s.mask_porosity.mean()),
            "detected_air_abs": float(s.detected_air_abs.mean()),
            "unmasked_air_abs": float(s.unmasked_air_abs.mean()),
            "unmasked_air_abs_std": float(s.unmasked_air_abs.std(ddof=1))
                                    if len(s) > 1 else 0.0,
            "unmasked_air_abs_max": float(s.unmasked_air_abs.max()),
            "unmasked_interior_local_abs": float(s.unmasked_interior_local_abs.mean()),
            "unmasked_edge_local_abs": float(s.unmasked_edge_local_abs.mean()),
            "edge_over_interior_ratio": float(s.edge_over_interior_ratio.mean()),
            "mask_capture_of_detected": float(s.mask_capture_of_detected.mean()),
            "largest_comp_mm3": float(s.largest_comp_mm3.mean()),
            "material_mode_u8": float(pd.to_numeric(s.material_mode_u8,
                                                    errors="coerce").mean()),
            "raw_min_u8": int(s.raw_min_u8.min()),
            "raw_max_u8": int(s.raw_max_u8.max()),
            "bimodal_fraction": float(s.bimodal.mean()),
        })

    arms_present = [a for a in ARMS + ["probe"] if (pv.arm == a).any()]
    collapse = collapse_stats(cells, arms_present)

    fp = cal["real_false_positive_baseline"]
    results = {
        "campaign": "eval v3 — air audit on correctly-decoded volumes",
        "volumes_root": str(VOL_ROOT),
        "n_volumes": len(pv),
        "detector": {
            "rule": "u8 < T_abs, then drop connected components < 300 voxels",
            "T_abs": t_abs,
            "scale": "raw-scan uint8 — generated volumes now share it with "
                     "the real scans, so the absolute threshold is valid "
                     "without any scale mapping",
            "real_dice": cal["dice"], "real_precision": cal["precision"],
            "real_recall": cal["recall"],
            "n_calibration_volumes": cal["n_calibration_volumes"],
            "min_cc_voxels": MIN_CC, "edge_shell_vox": EDGE_VOX,
            "cell_size_vox": PATCH,
            "calibration_source": cal["source"],
        },
        "real_false_positive_baseline": fp,
        "dtype_provenance": sorted(pv.dtype_provenance.unique().tolist()),
        "aggregate_by_group": agg,
        "mask_collapse": collapse,
        "figures": [],
    }

    figs = []
    # (1) unmasked air per group, with the real false-positive baseline
    fig, ax = plt.subplots(figsize=(7.4, 4.2), constrained_layout=True)
    labels = [f"{a['experiment']}\n{a['arm']}" for a in agg]
    xs = np.arange(len(agg))
    ax.bar(xs - 0.2, [a["detected_air_abs"] for a in agg], 0.38,
           color="#8a8a8a", label="detected air (T_abs, CC-filtered)")
    ax.bar(xs + 0.2, [a["unmasked_air_abs"] for a in agg], 0.38,
           color="#c2571a", label="unmasked air (detector yes, mask no)")
    ax.axhline(fp["mean"], color="#1b6ca8", ls="--", lw=1.2,
               label=f"real-volume false-positive mean ({fp['mean']:.4f})")
    ax.set_xticks(xs)
    ax.set_xticklabels(labels, fontsize=7.5)
    ax.set_ylabel("volume fraction")
    ax.set_title(f"Air audit v3 — absolute threshold T_abs = {t_abs} "
                 "(real-calibrated, Dice 0.842)")
    ax.legend(fontsize=8)
    figs += savefig(fig, OUT_DIR, "airv3_fig1_unmasked_air_by_group")

    # (2) interior vs edge
    fig, ax = plt.subplots(figsize=(7.4, 4.2), constrained_layout=True)
    ax.bar(xs - 0.2, [a["unmasked_interior_local_abs"] for a in agg], 0.38,
           color="#6a3d9a", label="interior (local fraction)")
    ax.bar(xs + 0.2, [a["unmasked_edge_local_abs"] for a in agg], 0.38,
           color="#c9a227", label=f"edge shell {EDGE_VOX} vox (local fraction)")
    ax.set_xticks(xs)
    ax.set_xticklabels(labels, fontsize=7.5)
    ax.set_ylabel("unmasked air, fraction of that region")
    ax.set_title("Unmasked air: interior vs edge shell")
    ax.legend(fontsize=8)
    figs += savefig(fig, OUT_DIR, "airv3_fig2_interior_vs_edge")

    # (3) per-cell mask collapse
    fig, axes = plt.subplots(1, len(arms_present),
                             figsize=(4.1 * len(arms_present), 3.9),
                             constrained_layout=True, sharex=True, sharey=True,
                             squeeze=False)
    for ax, arm in zip(axes[0], arms_present):
        c = cells[cells.arm == arm]
        ax.scatter(c.cell_dark_frac, c.cell_mask_porosity, s=5,
                   color=ARM_COLORS[arm], alpha=0.25, edgecolors="none")
        lim = [0, max(0.05, float(c.cell_dark_frac.max()) * 1.05)]
        ax.plot(lim, lim, color="0.4", ls="--", lw=1.0)
        st = collapse.get(arm, {})
        ax.set_title(f"{arm}\nr(dark, unmasked) = "
                     f"{st.get('pearson_dark_vs_unmasked', float('nan')):.3f}",
                     fontsize=9)
        ax.set_xlabel("cell dark-air fraction (T_abs)")
    axes[0][0].set_ylabel("cell mask porosity")
    fig.suptitle("Per-64³-cell mask collapse — the mask head stops claiming "
                 "voxels where air is massive")
    figs += savefig(fig, OUT_DIR, "airv3_fig3_mask_collapse_per_cell")

    # (4) intensity histograms — the decode fix made visible
    fig, ax = plt.subplots(figsize=(7.0, 4.2), constrained_layout=True)
    for arm in arms_present:
        keys = [k for k in hists if f"/{arm}/" in k]
        hh = np.sum([hists[k] for k in keys], axis=0).astype(float)
        hh /= hh.sum()
        ax.plot(np.arange(256), hh, color=ARM_COLORS[arm], lw=1.2,
                label=f"{arm} (n={len(keys)})")
    ax.axvline(t_abs, color="#c2571a", ls="--", lw=1.2,
               label=f"T_abs = {t_abs}")
    ax.axvline(cal["real_material_mode_mean"], color="0.35", ls=":", lw=1.2,
               label=f"real material mode ({cal['real_material_mode_mean']:.0f})")
    ax.set_yscale("log")
    ax.set_xlabel("grey level (uint8)")
    ax.set_ylabel("fraction of voxels")
    ax.set_title("Generated intensity distributions on the real u8 scale")
    ax.legend(fontsize=8)
    figs += savefig(fig, OUT_DIR, "airv3_fig4_intensity_histograms")

    results["figures"] = figs
    p_json = write_json(results, OUT_DIR)

    # ---------------- findings ----------------
    L = [f"# Air audit v3 — T_abs = {t_abs} on correctly-decoded volumes", "",
         f"{len(pv)} volumes under `{VOL_ROOT}`, stored dtype: "
         + "; ".join(results["dtype_provenance"]) + ".", "",
         f"Detector (methodology and constants imported from "
         f"`scripts/analysis/air_audit_v2.py`): `u8 < {t_abs}`, then connected "
         f"components below {MIN_CC} voxels dropped. `T_abs` is the "
         f"Dice-optimal absolute threshold on {cal['n_calibration_volumes']} "
         f"real val/test volumes inside `sample_mask` "
         f"(Dice {cal['dice']:.3f}, precision {cal['precision']:.3f}, "
         f"recall {cal['recall']:.3f}). Interior/edge split at a "
         f"{EDGE_VOX}-voxel shell; cells are {PATCH}³.", "",
         "**Why this is now valid.** The sampler used to apply `expit()` to "
         "the VAE XCT head, which regresses `xct/255` directly; every v2 "
         "volume was squashed into grey levels ~[133, 187] and an "
         "absolute real-calibrated threshold could only be applied through a "
         "scale map. The decode is fixed, so generated volumes carry the real "
         "u8 scale and `T_abs` transfers as calibrated.", "",
         f"Reference: on real volumes this detector flags "
         f"{fp['mean']:.4f} (max {fp['max']:.4f}) of voxels as air that the "
         f"stored mask does not — the false-positive floor any generated "
         f"number must be read against.", "",
         "## Per group", "",
         "| experiment | arm | n | mask φ | detected air | unmasked air "
         "| interior (local) | edge (local) | edge/interior | mask capture "
         "| grey range | material mode |",
         "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for a in agg:
        L.append(
            f"| {a['experiment']} | {a['arm']} | {a['n']} "
            f"| {a['mask_porosity']:.4f} | {a['detected_air_abs']:.4f} "
            f"| {a['unmasked_air_abs']:.4f} ± {a['unmasked_air_abs_std']:.4f} "
            f"| {a['unmasked_interior_local_abs']:.4f} "
            f"| {a['unmasked_edge_local_abs']:.4f} "
            f"| {a['edge_over_interior_ratio']:.2f} "
            f"| {a['mask_capture_of_detected']:.3f} "
            f"| [{a['raw_min_u8']}, {a['raw_max_u8']}] "
            f"| {a['material_mode_u8']:.0f} |")
    L += ["", "## Per-cell mask collapse", "",
          "| arm | n cells | r(dark, unmasked) | r(dark, mask φ) "
          "| median capture | capture, dark<2% | capture, 2–15% | capture, >15% |",
          "|---|---|---|---|---|---|---|---|"]
    for arm in arms_present:
        st = collapse.get(arm)
        if not st:
            continue
        r = st["regimes"]

        def _f(v):
            return "n/a" if v is None else f"{v:.4f}"
        L.append(f"| {arm} | {st['n_cells']} "
                 f"| {st['pearson_dark_vs_unmasked']:.4f} "
                 f"| {_f(st['pearson_dark_vs_mask_porosity'])} "
                 f"| {_f(st['capture_median_overall'])} "
                 f"| {_f(r['low']['capture_median'])} "
                 f"| {_f(r['mid']['capture_median'])} "
                 f"| {_f(r['high']['capture_median'])} |")
    L += ["",
          "`capture` = 1 − unmasked/dark per cell: the share of detected air "
          "the mask head does claim. A value near 0 in the high-dark regime "
          "is the mask-collapse signature.", "",
          f"Artefacts: `{OUT_DIR}` (results.json, per_volume.csv, "
          f"per_cell.csv, histograms.npz, 4 figures).", ""]
    p_md = write_findings("\n".join(L), OUT_DIR)
    log(f"wrote {p_json} and {p_md}")


if __name__ == "__main__":
    main()
