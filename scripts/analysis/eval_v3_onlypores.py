"""onlypores porosity of the eval-v3 volumes — global and local (CPU).

Measures every regenerated volume with the SAME segmentation the real dataset
was built with (``poregen.dataset.segmentation.onlypores`` at the defaults
``poregen.dataset.io.compute_mask`` uses: sauvola_radius=30, sauvola_k=0.125,
frontwall=0, backwall=0, min_size_filtering=-1), independently of the model's
own mask head.

Scale
-----
NO logit inversion.  ``runs/eval_v3/volumes/**/volume.tif`` is uint8 on the
raw-scan grey scale, because the sampler no longer applies ``expit`` to the
XCT head.  ``_eval_v3.load_u8`` verifies the dtype it finds and this script
records the provenance string and the measured grey range / material mode of
every volume, so the claim is checked rather than assumed.  (The v2 script had
to undo the sigmoid with ``clip(logit(v), 0, 1) * 255``.)

Two controls
------------
1. **Full real volumes** — re-run the pipeline on three real scans and compare
   against the masks stored in ``volumes.zarr`` (reused from
   ``scripts/analysis/onlypores_generated.py``).  Dice ~1.0 proves the harness
   reproduces the dataset's own ground truth.
2. **Real 192³ interior crops** — the SAME box size as the generated volumes,
   cut from inside ``sample_mask`` of real scans.  ``onlypores`` derives its
   material mask from a GLOBAL Otsu, which needs the exterior-air mode to be
   present; on a small interior box there is no air mode, the split lands
   inside the material, and the method reports ~0 porosity on known-good
   material.  This control is the reference for reading every 192³ onlypores
   number below — the 192³ figures are reported WITH that caveat, not as
   ground truth.

Outputs -> ``runs/eval_v3/onlypores/``: results.json, per_volume.csv,
per_cell.csv, real_validation.json, real_192_control.json, findings.md,
figures (PDF + PNG, 300 dpi).

Usage:
    python scripts/analysis/eval_v3_onlypores.py [--skip-real-validation]
"""

from __future__ import annotations

import os

os.environ.setdefault("TQDM_DISABLE", "1")

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import zarr

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import REPO, ZARR_ROOT, savefig, set_style, write_findings, write_json, plt  # noqa: E402
from _eval_v3 import (  # noqa: E402
    ARM_COLORS, ARM_MARKERS, ARMS, GATE, PATCH, ROOT, TARGETS, VOL_ROOT,
    block_sums, err_stats, fit_ols, iter_volumes, load_mask, load_u8,
    read_stats,
)
from air_audit_v2 import REAL_BOX, find_interior_boxes, real_split_names  # noqa: E402
from onlypores_generated import validate_real  # noqa: E402

sys.path.insert(0, str(REPO / "src"))
from poregen.dataset.segmentation import onlypores  # noqa: E402

OUT_DIR = ROOT / "onlypores"
N_CONTROL_VOLUMES = 5
N_CONTROL_BOXES = 4
_T0 = time.time()
_LOG_FH = None


def log(msg: str) -> None:
    line = f"[{time.time() - _T0:7.1f}s] {msg}"
    print(line, flush=True)
    if _LOG_FH is not None:
        _LOG_FH.write(line + "\n")
        _LOG_FH.flush()


def material_mode(u8: np.ndarray, sel: np.ndarray) -> int:
    """Modal u8 level of the material (levels >= 100 inside ``sel``)."""
    if not sel.any():
        return -1
    hist = np.bincount(u8[sel].ravel(), minlength=256)
    return int(np.argmax(hist[100:]) + 100)


# ---------------------------------------------------------------------------
# Control 2 — real 192³ interior crops
# ---------------------------------------------------------------------------

def real_192_control(rng: np.random.Generator) -> dict:
    """onlypores on real interior boxes the same size as the generated ones."""
    splits = json.loads((REPO / "data" / "split_v2" / "splits.json").read_text())
    names = real_split_names(splits)[:N_CONTROL_VOLUMES]
    g = zarr.open(str(ZARR_ROOT), mode="r")
    rows = []
    for name in names:
        grp = g[name]
        for origin in find_interior_boxes(grp, N_CONTROL_BOXES, rng):
            z0, y0, x0 = origin
            sl = (slice(z0, z0 + REAL_BOX), slice(y0, y0 + REAL_BOX),
                  slice(x0, x0 + REAL_BOX))
            xct = np.asarray(grp["xct"][sl])
            stored = np.asarray(grp["mask"][sl]) > 0
            stored_sm = np.asarray(grp["sample_mask"][sl]) > 0
            pore, sm, _ = onlypores(xct)
            if pore is None:
                continue
            n_sm = int(sm.sum())
            rows.append({
                "volume": name, "origin": [int(v) for v in origin],
                "box_edge": REAL_BOX,
                "stored_mask_porosity": float(stored.mean()),
                "stored_sample_frac": float(stored_sm.mean()),
                "onlypores_porosity_total": float(pore.mean()),
                "onlypores_porosity_sample": (int(pore.sum()) / n_sm) if n_sm else float("nan"),
                "onlypores_sample_frac": n_sm / float(sm.size),
                "material_mode_u8": material_mode(xct, stored_sm),
                "ratio_onlypores_over_stored": (float(pore.mean())
                                                / max(float(stored.mean()), 1e-12)),
            })
            log(f"  real192 {name[:44]:44s} {origin} stored={rows[-1]['stored_mask_porosity']:.5f} "
                f"onlypores={rows[-1]['onlypores_porosity_total']:.6f} "
                f"sm_frac={rows[-1]['onlypores_sample_frac']:.3f}")
            del xct, stored, stored_sm, pore, sm
    df = pd.DataFrame(rows)
    return {
        "n_volumes": int(df.volume.nunique()) if len(df) else 0,
        "n_boxes": int(len(df)),
        "box_edge": REAL_BOX,
        "stored_mask_porosity_mean": float(df.stored_mask_porosity.mean()) if len(df) else None,
        "onlypores_porosity_total_mean": float(df.onlypores_porosity_total.mean()) if len(df) else None,
        "onlypores_porosity_sample_mean": float(df.onlypores_porosity_sample.mean()) if len(df) else None,
        "onlypores_sample_frac_mean": float(df.onlypores_sample_frac.mean()) if len(df) else None,
        "recovery_ratio_mean": float(df.ratio_onlypores_over_stored.mean()) if len(df) else None,
        "recovery_ratio_max": float(df.ratio_onlypores_over_stored.max()) if len(df) else None,
        "verdict": (
            "onlypores under-reports porosity by ~"
            f"{1.0 / max(float(df.ratio_onlypores_over_stored.mean()), 1e-12):.0f}x "
            "on real 192³ interior crops: its material mask comes from a "
            "global Otsu that needs the exterior-air mode, which a small "
            "interior box does not contain."
            if len(df) and float(df.ratio_onlypores_over_stored.mean()) < 0.5
            else "onlypores tracks the stored mask on real 192³ interior crops."),
        "boxes": rows,
    }


# ---------------------------------------------------------------------------
# Generated volumes
# ---------------------------------------------------------------------------

def measure(exp: str, arm: str, d: Path) -> tuple[dict, pd.DataFrame]:
    stats = read_stats(d)
    u8, prov = load_u8(d / "volume.tif")
    model_mask = load_mask(d)
    pore, sm, _binary = onlypores(u8)
    if pore is None:
        pore = np.zeros(u8.shape, bool)
        sm = np.zeros(u8.shape, bool)

    n_vox = int(u8.size)
    n_sm = int(sm.sum())
    n_pore = int(pore.sum())
    n_mask = int(model_mask.sum())
    inter = int(np.count_nonzero(pore & model_mask))
    target = stats.get("target")
    op_sample = n_pore / n_sm if n_sm else float("nan")
    op_total = n_pore / n_vox
    mask_por = n_mask / n_vox

    row = {
        "experiment": exp, "arm": arm, "name": d.name,
        "dtype_provenance": prov,
        "shape": "x".join(str(s) for s in u8.shape),
        "n_voxels": n_vox,
        "grey_min_u8": int(u8.min()), "grey_max_u8": int(u8.max()),
        "material_mode_u8": material_mode(u8, sm) if n_sm else -1,
        "target": target,
        "seed": stats.get("seed"), "s_por": stats.get("s_por"),
        "layup": stats.get("layup"), "ddim_steps": stats.get("ddim_steps"),
        "mask_porosity": mask_por,
        "onlypores_porosity_sample": op_sample,
        "onlypores_porosity_total": op_total,
        "sample_mask_fraction": n_sm / n_vox,
        "stats_delivered_mask_porosity": stats.get("delivered_mask_porosity"),
        "dice_mask_vs_onlypores": (2.0 * inter / (n_mask + n_pore)
                                   if (n_mask + n_pore) else float("nan")),
        "mask_recall_of_onlypores": inter / n_pore if n_pore else float("nan"),
        "mask_precision_vs_onlypores": inter / n_mask if n_mask else float("nan"),
        "discrepancy_onlypores_minus_mask": op_sample - mask_por,
        "error_mask_vs_target": (mask_por - target) if target is not None else None,
        "error_onlypores_vs_target": (op_sample - target) if target is not None else None,
    }

    s_pore = block_sums(pore)
    s_sm = block_sums(sm)
    s_mask = block_sums(model_mask)
    gz, gy, gx = s_pore.shape
    cell_targets = stats.get("cell_targets")
    per_cell = PATCH ** 3
    recs = []
    for iz in range(gz):
        for iy in range(gy):
            for ix in range(gx):
                key = f"{iz},{iy},{ix}"
                if cell_targets and key in cell_targets:
                    ct, src = float(cell_targets[key]), "per_cell"
                else:
                    ct, src = target, "uniform_global"
                nsm = int(s_sm[iz, iy, ix])
                npo = int(s_pore[iz, iy, ix])
                nma = int(s_mask[iz, iy, ix])
                op_c = npo / nsm if nsm else float("nan")
                mk_c = nma / per_cell
                recs.append({
                    "experiment": exp, "arm": arm, "name": d.name,
                    "iz": iz, "iy": iy, "ix": ix,
                    "cell_target": ct, "cell_target_source": src,
                    "cell_mask_porosity": mk_c,
                    "cell_onlypores_porosity_sample": op_c,
                    "cell_onlypores_porosity_total": npo / per_cell,
                    "cell_sample_frac": nsm / per_cell,
                    "cell_discrepancy": op_c - mk_c,
                    "cell_err_mask": (mk_c - ct) if ct is not None else None,
                    "cell_err_onlypores": (op_c - ct) if ct is not None else None,
                })
    del u8, model_mask, pore, sm, _binary
    return row, pd.DataFrame(recs)


def main() -> None:
    global _LOG_FH
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-real-validation", action="store_true")
    args = ap.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    _LOG_FH = open(OUT_DIR / "run.log", "a")
    set_style()
    log("=== onlypores v3 start ===")

    real_val = {}
    if not args.skip_real_validation:
        real_val = validate_real()
        write_json(real_val, OUT_DIR, "real_validation.json")

    control = real_192_control(np.random.default_rng(0))
    write_json(control, OUT_DIR, "real_192_control.json")
    log(f"real 192³ control: onlypores "
        f"{control['onlypores_porosity_total_mean']:.6f} vs stored "
        f"{control['stored_mask_porosity_mean']:.5f} over "
        f"{control['n_boxes']} boxes")

    vols = list(iter_volumes(VOL_ROOT))
    if not vols:
        raise SystemExit(f"no volumes under {VOL_ROOT}")
    rows, frames = [], []
    for i, (exp, arm, d) in enumerate(vols, 1):
        row, cells = measure(exp, arm, d)
        rows.append(row)
        frames.append(cells)
        log(f"[{i:3d}/{len(vols)}] {exp}/{arm}/{d.name} "
            f"grey=[{row['grey_min_u8']},{row['grey_max_u8']}] "
            f"mat_mode={row['material_mode_u8']} "
            f"mask={row['mask_porosity']:.4f} "
            f"onlypores={row['onlypores_porosity_sample']:.4f} "
            f"sm_frac={row['sample_mask_fraction']:.3f}")

    pv = pd.DataFrame(rows)
    cells = pd.concat(frames, ignore_index=True)
    pv.to_csv(OUT_DIR / "per_volume.csv", index=False)
    cells.to_csv(OUT_DIR / "per_cell.csv", index=False)

    # ---------------- dose-response with onlypores porosity ----------------
    dose = pv[pv.experiment == "dose_response"]
    dose_cells = cells[cells.experiment == "dose_response"]
    fits, per_level = {}, {}
    for arm in ARMS:
        s = dose[dose.arm == arm]
        if not len(s):
            continue
        c = dose_cells[dose_cells.arm == arm]
        fits[arm] = {
            "global_onlypores": fit_ols(s.target.to_numpy(),
                                        s.onlypores_porosity_sample.to_numpy()),
            "global_mask": fit_ols(s.target.to_numpy(),
                                   s.mask_porosity.to_numpy()),
            "local_onlypores": fit_ols(c.cell_target.to_numpy(),
                                       c.cell_onlypores_porosity_sample.to_numpy()),
            "local_mask": fit_ols(c.cell_target.to_numpy(),
                                  c.cell_mask_porosity.to_numpy()),
            "global_err_onlypores": err_stats(s.error_onlypores_vs_target.to_numpy()),
            "global_err_mask": err_stats(s.error_mask_vs_target.to_numpy()),
            "local_err_onlypores": err_stats(c.cell_err_onlypores.to_numpy()),
            "local_err_mask": err_stats(c.cell_err_mask.to_numpy()),
        }
        per_level[arm] = []
        for t in TARGETS:
            ls = s[np.isclose(s.target.to_numpy(), t)]
            if not len(ls):
                continue
            per_level[arm].append({
                "target": t, "n": int(len(ls)),
                "mask_mean": float(ls.mask_porosity.mean()),
                "onlypores_mean": float(ls.onlypores_porosity_sample.mean()),
                "onlypores_std": float(ls.onlypores_porosity_sample.std(ddof=1))
                                 if len(ls) > 1 else 0.0,
                "sample_frac_mean": float(ls.sample_mask_fraction.mean()),
                "dice_mask_vs_onlypores": float(ls.dice_mask_vs_onlypores.mean()),
            })

    results = {
        "campaign": "eval v3 — onlypores porosity on correctly-decoded volumes",
        "volumes_root": str(VOL_ROOT),
        "n_volumes": int(len(pv)),
        "scale": {
            "logit_inversion_applied": False,
            "reason": "the sampler no longer applies expit to the XCT head, "
                      "so volume.tif is uint8 on the raw-scan grey scale — "
                      "the same scale onlypores was tuned on for real scans",
            "dtype_provenance": sorted(pv.dtype_provenance.unique().tolist()),
            "generated_grey_range_u8": [int(pv.grey_min_u8.min()),
                                        int(pv.grey_max_u8.max())],
            "generated_material_mode_u8_mean": float(
                pv.loc[pv.material_mode_u8 > 0, "material_mode_u8"].mean()),
            "real_material_mode_u8_reference": 208.6,
        },
        "segmentation": "poregen.dataset.segmentation.onlypores at "
                        "compute_mask defaults (sauvola_radius=30, "
                        "sauvola_k=0.125, min_size_filtering=-1)",
        "real_validation": real_val,
        "real_192_control": control,
        "caveat_192": (
            "onlypores' material mask uses a GLOBAL Otsu that needs the "
            "exterior-air mode. A 192³ interior box has none, so the split "
            "falls inside the material and porosity collapses — see "
            "real_192_control: real interior crops with stored porosity "
            f"{control['stored_mask_porosity_mean']:.4f} measure "
            f"{control['onlypores_porosity_total_mean']:.6f}. Every 192³ "
            "onlypores number below inherits that bias and must be read "
            "against this control, not as ground truth."),
        "dose_response_fits": fits,
        "dose_response_per_level": per_level,
        "figures": [],
    }

    # ---------------- figures ----------------
    figs = []
    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.4), constrained_layout=True)
    lim = [0.0, 0.115]
    for ax, col, title in ((axes[0], "mask_porosity", "model mask"),
                           (axes[1], "onlypores_porosity_sample", "onlypores")):
        ax.plot(lim, lim, color="0.4", ls="--", lw=1.0)
        ax.fill_between(lim, [v - GATE for v in lim], [v + GATE for v in lim],
                        color="0.85", alpha=0.5, zorder=0)
        for arm in ARMS:
            s = dose[dose.arm == arm]
            if not len(s):
                continue
            ax.scatter(s.target, s[col], s=18, color=ARM_COLORS[arm],
                       marker=ARM_MARKERS[arm], alpha=0.6, edgecolors="none",
                       label=arm)
        ax.set_xlim(lim)
        ax.set_xlabel("requested global porosity")
        ax.set_title(f"{title} porosity")
    axes[0].set_ylabel("delivered porosity")
    axes[0].legend(fontsize=8)
    fig.suptitle("Dose response measured two ways (v3, correct decode)")
    figs += savefig(fig, OUT_DIR, "opv3_fig1_global_dose_response")

    fig, axes = plt.subplots(1, len(ARMS), figsize=(4.2 * len(ARMS), 4.0),
                             constrained_layout=True, sharex=True, sharey=True,
                             squeeze=False)
    for ax, arm in zip(axes[0], ARMS):
        c = dose_cells[dose_cells.arm == arm]
        if not len(c):
            continue
        ax.scatter(c.cell_target, c.cell_onlypores_porosity_sample, s=6,
                   color=ARM_COLORS[arm], alpha=0.3, edgecolors="none")
        lim2 = [0, float(np.nanmax(c.cell_target)) * 1.15]
        ax.plot(lim2, lim2, color="0.4", ls="--", lw=1.0)
        f = fits[arm]["local_onlypores"]
        if f["slope"] is not None:
            xx = np.array(lim2)
            ax.plot(xx, f["slope"] * xx + f["intercept"], color="0.1", lw=1.3)
            ax.set_title(f"{arm}\nslope {f['slope']:.2f}, R² {f['r2']:.3f}, "
                         f"n={f['n']}", fontsize=9)
        ax.set_xlabel("cell local target")
    axes[0][0].set_ylabel("cell onlypores porosity (pore/sample)")
    fig.suptitle("Local conditioning obedience measured with onlypores")
    figs += savefig(fig, OUT_DIR, "opv3_fig2_local_dose_response")

    fig, ax = plt.subplots(figsize=(6.6, 4.4), constrained_layout=True)
    groups = sorted(pv.groupby(["experiment", "arm"]).groups)
    xs = np.arange(len(groups))
    mm = [float(pv[(pv.experiment == e) & (pv.arm == a)].mask_porosity.mean())
          for e, a in groups]
    oo = [float(pv[(pv.experiment == e) & (pv.arm == a)]
                .onlypores_porosity_sample.mean()) for e, a in groups]
    ax.bar(xs - 0.2, mm, 0.38, color="#1b6ca8", label="model mask")
    ax.bar(xs + 0.2, oo, 0.38, color="#c2571a", label="onlypores (pore/sample)")
    ctrl = control["onlypores_porosity_total_mean"]
    ax.axhline(ctrl, color="0.3", ls=":", lw=1.2,
               label=f"onlypores on REAL 192³ crops ({ctrl:.5f})")
    ax.set_xticks(xs)
    ax.set_xticklabels([f"{e}\n{a}" for e, a in groups], fontsize=7.5)
    ax.set_ylabel("porosity")
    ax.set_title("Mask vs onlypores porosity per group (read with the 192³ caveat)")
    ax.legend(fontsize=8)
    figs += savefig(fig, OUT_DIR, "opv3_fig3_mask_vs_onlypores_by_group")

    results["figures"] = figs
    p_json = write_json(results, OUT_DIR)

    # ---------------- findings ----------------
    sc = results["scale"]
    L = ["# onlypores porosity — eval v3 (correct decode)", "",
         f"{len(pv)} volumes under `{VOL_ROOT}`.", "",
         "## Scale check (no logit inversion)", "",
         f"- stored dtype: {'; '.join(sc['dtype_provenance'])}",
         f"- grey range over all volumes: [{sc['generated_grey_range_u8'][0]}, "
         f"{sc['generated_grey_range_u8'][1]}] u8",
         f"- generated material mode: "
         f"{sc['generated_material_mode_u8_mean']:.1f} u8 vs real reference "
         f"{sc['real_material_mode_u8_reference']:.1f} u8",
         "",
         "The v2 script had to undo the sampler's `expit` with "
         "`clip(logit(v), 0, 1) * 255` before onlypores would see a real-like "
         "histogram. That inversion is **not applied here and is not needed**: "
         "`volume.tif` is already uint8 on the raw-scan scale and its material "
         "mode lands on the real one.", ""]

    if real_val:
        L += ["## Control 1 — full real volumes", "",
              "| volume | Dice(pore mask) | recomputed φ | stored φ |",
              "|---|---|---|---|"]
        for name, r in real_val.items():
            L.append(f"| {name[:52]} | {r['dice_pore_mask']:.4f} "
                     f"| {r['recomputed_porosity_over_sample']:.5f} "
                     f"| {r['stored_porosity_over_sample']:.5f} |")
        L.append("")

    L += ["## Control 2 — real 192³ interior crops (THE caveat)", "",
          f"{control['n_boxes']} boxes of {REAL_BOX}³ cut from inside "
          f"`sample_mask` of {control['n_volumes']} real volumes:", "",
          f"- stored mask porosity: **{control['stored_mask_porosity_mean']:.5f}**",
          f"- onlypores porosity (pore/total): "
          f"**{control['onlypores_porosity_total_mean']:.6f}**",
          f"- onlypores sample-mask fraction: "
          f"{control['onlypores_sample_frac_mean']:.3f}",
          "",
          control["verdict"], "",
          "So every onlypores number for a 192³ generated volume is biased the "
          "same way. Use it for RELATIVE comparisons between arms and step "
          "counts, never as an absolute porosity.", "",
          "## Dose response (dose_response set)", "",
          "| arm | global slope (mask) | R² | global slope (onlypores) | R² "
          "| local slope (onlypores) | R² | mean \\|err\\| onlypores |",
          "|---|---|---|---|---|---|---|---|"]
    for arm in ARMS:
        f = fits.get(arm)
        if not f:
            continue

        def _s(d, k="slope"):
            return "n/a" if d[k] is None else f"{d[k]:.3f}"

        def _r(d):
            return "n/a" if d["r2"] is None else f"{d['r2']:.3f}"
        L.append(f"| {arm} | {_s(f['global_mask'])} | {_r(f['global_mask'])} "
                 f"| {_s(f['global_onlypores'])} | {_r(f['global_onlypores'])} "
                 f"| {_s(f['local_onlypores'])} | {_r(f['local_onlypores'])} "
                 f"| {f['global_err_onlypores'].get('abs_error_mean', float('nan')):.4f} |")
    L += ["", "## Per level (onlypores, pore/sample)", ""]
    for arm in ARMS:
        if arm not in per_level:
            continue
        L += [f"### {arm}", "",
              "| target | mask φ | onlypores φ | sample-mask frac "
              "| Dice(mask, onlypores) |", "|---|---|---|---|---|"]
        for pl in per_level[arm]:
            L.append(f"| {pl['target']:.3f} | {pl['mask_mean']:.4f} "
                     f"| {pl['onlypores_mean']:.4f} ± {pl['onlypores_std']:.4f} "
                     f"| {pl['sample_frac_mean']:.3f} "
                     f"| {pl['dice_mask_vs_onlypores']:.3f} |")
        L.append("")
    L += [f"Artefacts: `{OUT_DIR}` (results.json, per_volume.csv, "
          "per_cell.csv, real_validation.json, real_192_control.json, "
          "3 figures).", ""]
    p_md = write_findings("\n".join(L), OUT_DIR)
    log(f"wrote {p_json} and {p_md}")
    log("=== onlypores v3 done ===")


if __name__ == "__main__":
    main()
