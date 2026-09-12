"""findings.md and the figures, built from results.json alone.

The report never touches a volume.  Everything it says comes out of the
measure step's results file, so a table and the number behind it cannot drift
apart, and a report can be rebuilt after the volumes are deleted.

Every table that has a real-volume floor carries it as its first row, read from
``<root>/real_floor/results.json`` when that exists.  A table without a floor
says so rather than leaving the reader to assume 0 or 1 is the target.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from poregen.eval_v4.io import (  # noqa: E402
    assessment_dir,
    figures_dir,
    read_results,
    write_findings,
)

logger = logging.getLogger(__name__)

#: Categorical hues carried over from the campaign scripts, so a v3 figure and
#: a v4 figure of the same quantity are the same colour.
SERIES_COLORS = ("#1b6ca8", "#c2571a", "#2e7d32", "#6a3d9a", "#a11d33")
FLOOR_COLOR = "#555555"


def set_style() -> None:
    plt.rcParams.update({
        "figure.dpi": 120,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "font.size": 10,
        "axes.titlesize": 11,
        "axes.labelsize": 10,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "legend.fontsize": 8.5,
        "axes.grid": True,
        "grid.alpha": 0.25,
        "grid.linewidth": 0.6,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "lines.linewidth": 1.3,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })


def savefig(fig, out_dir: Path, name: str) -> list[str]:
    """Write one figure as PDF and PNG at 300 dpi."""
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for ext in ("pdf", "png"):
        p = out_dir / f"{name}.{ext}"
        fig.savefig(p, dpi=300)
        paths.append(str(p))
    plt.close(fig)
    return paths


# ---------------------------------------------------------------------------
# Markdown helpers
# ---------------------------------------------------------------------------

def fmt(v, digits: int = 4) -> str:
    if v is None:
        return "--"
    if isinstance(v, bool):
        return "yes" if v else "no"
    if isinstance(v, (int, np.integer)):
        return str(int(v))
    if isinstance(v, float) and not np.isfinite(v):
        return "--"
    return f"{v:.{digits}f}"


def ms(d: dict | None, digits: int = 4) -> str:
    """``mean +/- sd`` from a :func:`poregen.eval_v4.metrics.mean_sd` block."""
    if not d or d.get("mean") is None:
        return "--"
    return f"{d['mean']:.{digits}f} +/- {(d.get('sd') or 0.0):.{digits}f}"


def table(header: list[str], rows: list[list[str]]) -> str:
    """A markdown table with every cell's own pipes escaped.

    A metric name like ``|err|`` splits a row into the wrong number of columns
    and silently shifts every value one place left, which is a table that lies.
    """
    def cell(v) -> str:
        return str(v).replace("|", "\\|")

    out = ["| " + " | ".join(cell(h) for h in header) + " |",
           "|" + "|".join(["---"] * len(header)) + "|"]
    out += ["| " + " | ".join(cell(v) for v in r) + " |" for r in rows]
    return "\n".join(out)


def load_floor(root: Path) -> dict | None:
    path = assessment_dir(root, "real_floor") / "results.json"
    if not path.exists():
        return None
    return json.loads(path.read_text())



def _header(res: dict, root: Path, floor: dict | None) -> str:
    per = res.get("per_case") or [{}]
    ident = per[0]
    lines = [
        f"# eval v4 - {res['assessment']}",
        "",
        f"**Question.** {res.get('question', '')}",
        "",
    ]
    if res.get("note"):
        lines += [res["note"], ""]
    lines += [
        table(
            ["", ""],
            [
                ["campaign", f"`{root}`"],
                ["cases measured", f"{res.get('n_cases_measured')} of {res.get('n_cases_expected')}"],
                ["model run", f"`{ident.get('model_run', '--')}`"],
                ["checkpoint", f"`{(ident.get('notes') or {}).get('checkpoint', '--')}`"],
                ["checkpoint step", fmt(ident.get("checkpoint_step"))],
                ["weights", str(ident.get("weights"))],
                ["objective", str(ident.get("objective"))],
                ["cfg_rescale", fmt(ident.get("cfg_rescale"), 2)],
                ["code", f"`{ident.get('git_commit', '--')}`"],
                ["real floor", "yes" if floor else "not measured - run `eval_v4 real-floor`"],
            ],
        ),
        "",
        "",
    ]
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Per-assessment reports
# ---------------------------------------------------------------------------

def report_sampler(res, root, floor) -> tuple[str, list[str]]:
    cells = res["cells"]
    rows = []
    for name, c in sorted(cells.items(), key=lambda kv: (kv[1]["volume_shape"], kv[1]["ddim_steps"])):
        rows.append([
            name, str(c["ddim_steps"]),
            ms(c["porosity_error"]), ms(c["air_fraction_interior"]),
            ms(c["seam_xct_ratio"], 3), ms(c["seam_pore_ratio"], 3),
            ms(c["seam_chunk_xct_ratio"], 3), ms(c["seam_chunk_pore_ratio"], 3),
            ms(c["wall_time_s"], 1), fmt(c["failure_rate"], 2),
        ])
    floor_rows = []
    for tag, f in (floor or {}).get("by_shape", {}).items():
        floor_rows.append([
            f"real {tag}", "--", "--", ms(f["air_fraction_interior"]),
            ms(f["seam_xct_ratio"], 3), "--", ms(f["seam_chunk_xct_ratio"], 3), "--",
            "--", "--",
        ])
    body = table(
        ["cell", "DDIM", "phi error", "air (interior)", "seam xct", "seam pore",
         "chunk xct", "chunk pore", "wall s", "fail"],
        floor_rows + rows,
    )
    figs = _fig_sampler(cells, floor, root)
    return ("## Step count against scale\n\n" + body + "\n\n"
            "A seam ratio near the real row means the plane is indistinguishable "
            "from ordinary internal texture change. A 192-cubed volume is one "
            "chunk, so its chunk-plane columns are empty by construction.\n"), figs


def _fig_sampler(cells, floor, root) -> list[str]:
    set_style()
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.4))
    shapes = sorted({tuple(c["volume_shape"]) for c in cells.values()})
    for i, (key, ylabel) in enumerate([
        ("porosity_error", "delivered - requested phi"),
        ("seam_xct_ratio", "grey seam ratio"),
        ("wall_time_s", "wall time (s)"),
    ]):
        ax = axes[i]
        for j, shape in enumerate(shapes):
            pts = sorted(
                ((c["ddim_steps"], c[key]) for c in cells.values()
                 if tuple(c["volume_shape"]) == shape),
                key=lambda t: t[0],
            )
            x = [p[0] for p in pts]
            y = [p[1]["mean"] for p in pts]
            e = [p[1]["sd"] or 0.0 for p in pts]
            ax.errorbar(x, y, yerr=e, marker="o", capsize=3,
                        color=SERIES_COLORS[j % len(SERIES_COLORS)],
                        label=f"{shape[1]}x{shape[2]}x{shape[0]}")
        if key == "porosity_error":
            ax.axhline(0.0, color=FLOOR_COLOR, lw=0.9, ls="--")
        if key == "seam_xct_ratio" and floor:
            for f in floor.get("by_shape", {}).values():
                v = (f.get("seam_xct_ratio") or {}).get("mean")
                if v is not None:
                    ax.axhline(v, color=FLOOR_COLOR, lw=0.9, ls=":")
            ax.plot([], [], color=FLOOR_COLOR, ls=":", label="real floor")
        if key == "wall_time_s":
            ax.set_yscale("log")
        ax.set_xlabel("DDIM steps")
        ax.set_ylabel(ylabel)
        ax.legend(frameon=False)
    fig.tight_layout()
    return savefig(fig, figures_dir(root, "sampler"), "sampler_step_count")


def report_porosity_global(res, root, floor) -> tuple[str, list[str]]:
    dose = res["dose_response"]
    rows = [
        [k, fmt(c["requested"], 3), ms(c["delivered"]), ms(c["abs_error"]),
         fmt(c["within_gate"], 2), fmt(c["failure_rate"], 2)]
        for k, c in sorted(res["levels"].items(), key=lambda kv: kv[1]["requested"])
    ]
    off = res["off_manifold"]
    body = [
        "## Dose response",
        "",
        "Every phi here is MATERIAL porosity: pore voxels / material voxels, air "
        "outside the specimen envelope excluded from both.",
        "",
        table(["level", "requested phi (pore/material)", "delivered phi (pore/material)",
               "|error|", "within gate", "fail"], rows),
        "",
        f"OLS over every in-range volume: slope {fmt(dose['slope'], 3)}, "
        f"intercept {fmt(dose['intercept'], 4)}, R2 {fmt(dose['r2'], 3)}, "
        f"n {dose['n']}. Gate |error| < {dose['gate']}: "
        f"{fmt(dose['frac_within_gate'], 2)} of volumes.",
        "",
        "## Off-manifold request (failure mode, not part of the fit)",
        "",
        table(
            ["requested", "conditioned after clamp", "delivered", "|error|",
             "air (interior)", "fail"],
            [[fmt(off["requested"], 3), fmt(off.get("conditioned_phi_after_clamp"), 3),
              ms(off["delivered"]), ms(off["abs_error"]),
              ms(off["air_fraction_interior"]), fmt(off.get("failure_rate"), 2)]],
        ),
        "",
        off["note"],
    ]
    figs = _fig_dose(res, root)
    return "\n".join(body) + "\n", figs


def _fig_dose(res, root) -> list[str]:
    set_style()
    fig, ax = plt.subplots(figsize=(4.6, 4.2))
    req = np.array([r["porosity"]["requested_phi"] for r in res["per_case"]])
    got = np.array([r["porosity"]["delivered_phi"] for r in res["per_case"]])
    off = req == res["off_manifold"]["requested"]
    lim = float(max(req.max(), got.max())) * 1.08
    ax.plot([0, lim], [0, lim], color=FLOOR_COLOR, ls="--", lw=0.9, label="one to one")
    ax.fill_between([0, lim], [-res["dose_response"]["gate"], lim - res["dose_response"]["gate"]],
                    [res["dose_response"]["gate"], lim + res["dose_response"]["gate"]],
                    color=FLOOR_COLOR, alpha=0.10, lw=0, label="gate +/- 0.005")
    ax.scatter(req[~off], got[~off], s=26, color=SERIES_COLORS[0], zorder=3, label="in range")
    if off.any():
        ax.scatter(req[off], got[off], s=30, marker="X", color=SERIES_COLORS[4],
                   zorder=3, label="off manifold")
    fit = res["dose_response"]
    if fit["slope"] is not None:
        xs = np.array([0.0, lim])
        ax.plot(xs, fit["slope"] * xs + fit["intercept"], color=SERIES_COLORS[1],
                label=f"OLS slope {fit['slope']:.2f}")
    ax.set_xlim(0, lim)
    ax.set_ylim(0, lim)
    ax.set_xlabel("requested phi (pore / material)")
    ax.set_ylabel("delivered phi (pore / material)")
    ax.legend(frameon=False, loc="upper left")
    fig.tight_layout()
    return savefig(fig, figures_dir(root, "porosity_global"), "dose_response")


def report_porosity_local(res, root, floor) -> tuple[str, list[str]]:
    rows = []
    for name, f in sorted(res["fields"].items()):
        rows.append([
            name, str(f["n_seeds"]), ms(f["within_volume_slope"], 3),
            ms(f["within_volume_r2"], 3), ms(f["per_cell_abs_error"]),
            ms(f["per_cell_frac_within_gate"], 2),
            ms(f["requested_cell_sd"]), ms(f["delivered_cell_sd"]),
            fmt(f["pooled_over_seeds"].get("pooled_r2_not_obedience"), 3),
        ])
    floor_rows = []
    for tag, fl in (floor or {}).get("by_shape", {}).items():
        floor_rows.append([f"real {tag}", str(fl["n_volumes"]), "--", "--", "--", "--",
                           "--", ms(fl["cell_phi_sd"]), "--"])
    body = [
        "## Local obedience, within volume",
        "",
        "Per-tile phi is MATERIAL porosity: a tile's pore voxels over its material "
        "voxels. Tiles holding less than half material are not measured.",
        "",
        table(["field", "n", "slope", "R2", "per-cell |err|", "cells in gate",
               "requested cell sd", "delivered cell sd", "pooled R2 (not obedience)"],
              floor_rows + rows),
        "",
        "Slope and R2 are computed after each volume's own mean is removed from "
        "both the delivered and the requested cell values. The real row has no "
        "request; its delivered cell sd is the noise floor a slope has to beat. "
        "The pooled R2 column is context only - it mixes the global dose response "
        "into the local question.",
    ]
    figs = _fig_local(res, root, floor)
    return "\n".join(body) + "\n", figs


def _fig_local(res, root, floor) -> list[str]:
    set_style()
    names = sorted(res["fields"])
    fig, axes = plt.subplots(1, len(names) + 1, figsize=(3.2 * (len(names) + 1), 3.4))
    for i, name in enumerate(names):
        ax = axes[i]
        for j, row in enumerate(r for r in res["per_case"] if r["field"] == name):
            loc = row["local"]
            r = np.asarray(loc["cells_requested"], float)
            d = np.asarray(loc["cells_delivered"], float)
            ax.scatter(r - r.mean(), d - d.mean(), s=16, alpha=0.75,
                       color=SERIES_COLORS[j % len(SERIES_COLORS)],
                       label=f"seed {row['seed']}")
        lim = max(abs(np.asarray(ax.get_xlim())).max(), 1e-3)
        ax.plot([-lim, lim], [-lim, lim], color=FLOOR_COLOR, ls="--", lw=0.9)
        ax.set_title(name)
        ax.set_xlabel("requested phi (pore/material) - volume mean")
        if i == 0:
            ax.set_ylabel("delivered phi (pore/material) - volume mean")
        ax.legend(frameon=False)
    ax = axes[-1]
    xs = np.arange(len(names))
    ax.bar(xs, [res["fields"][n]["within_volume_slope"]["mean"] or 0.0 for n in names],
           yerr=[res["fields"][n]["within_volume_slope"]["sd"] or 0.0 for n in names],
           color=SERIES_COLORS[0], capsize=3)
    ax.axhline(1.0, color=FLOOR_COLOR, ls="--", lw=0.9)
    ax.set_xticks(xs)
    ax.set_xticklabels(names, rotation=20)
    ax.set_ylabel("within-volume slope")
    ax.set_title("obedience (1.0 = exact)")
    fig.tight_layout()
    return savefig(fig, figures_dir(root, "porosity_local"), "local_obedience")


def report_cfg(res, root, floor) -> tuple[str, list[str]]:
    rows = [
        [k, fmt(c["s_por"], 2), fmt(c["requested"], 3), ms(c["delivered_phi"]),
         ms(c["abs_error"]), ms(c["air_fraction_interior"]),
         ms(c["degenerate_cell_fraction"], 3), ms(c["seam_xct_ratio"], 3),
         fmt(c["failure_rate"], 2)]
        for k, c in sorted(res["s_por_cells"].items())
    ]
    nb_rows = [
        [k, fmt(c["s_nb"], 1), ms(c["delivered_phi"]), ms(c["air_fraction_interior"]),
         ms(c["seam_chunk_xct_ratio"], 3), ms(c["seam_chunk_pore_ratio"], 3)]
        for k, c in sorted(res["s_nb_cells"].items())
    ]
    summ = res["s_nb_summary"]
    body = [
        "## Porosity guidance",
        "",
        table(["cell", "s_por", "requested phi (pore/material)",
               "delivered phi (pore/material)", "|error|", "air (interior)",
               "degenerate cells", "seam xct", "fail"], rows),
        "",
        "## Neighbour guidance - do the neighbours act?",
        "",
        table(["cell", "s_nb", "delivered", "air (interior)", "chunk seam xct",
               "chunk seam pore"], nb_rows),
        "",
        table(["comparison", "value"], [
            ["pore Dice across the chunk plane, s_nb 0 vs 1",
             ms(summ["pore_dice_chunk_plane"], 3)],
            ["pore Dice over the whole volume", ms(summ["pore_dice_whole_volume"], 3)],
        ]),
        "",
        summ["reading"],
    ]
    figs = _fig_cfg(res, root)
    return "\n".join(body) + "\n", figs


def _fig_cfg(res, root) -> list[str]:
    set_style()
    fig, axes = plt.subplots(1, 2, figsize=(8.4, 3.4))
    ax = axes[0]
    targets = sorted({c["requested"] for c in res["s_por_cells"].values()})
    for j, target in enumerate(targets):
        pts = sorted(
            ((c["s_por"], c["abs_error"]) for c in res["s_por_cells"].values()
             if c["requested"] == target), key=lambda t: t[0])
        ax.errorbar([p[0] for p in pts], [p[1]["mean"] for p in pts],
                    yerr=[p[1]["sd"] or 0.0 for p in pts], marker="o", capsize=3,
                    color=SERIES_COLORS[j % len(SERIES_COLORS)],
                    label=f"target {target:g}")
    ax.axhline(0.005, color=FLOOR_COLOR, ls="--", lw=0.9, label="gate 0.005")
    ax.set_xlabel("s_por")
    ax.set_ylabel("|delivered - requested|")
    ax.legend(frameon=False)

    ax = axes[1]
    pairs = res.get("s_nb_pairs") or []
    if pairs:
        xs = np.arange(len(pairs))
        ax.bar(xs - 0.18, [p["pore_dice_chunk_plane"] for p in pairs], 0.36,
               color=SERIES_COLORS[0], label="chunk plane")
        ax.bar(xs + 0.18, [p["pore_dice_whole_volume"] for p in pairs], 0.36,
               color=SERIES_COLORS[1], label="whole volume")
        ax.set_xticks(xs)
        ax.set_xticklabels([f"seed {p['seed']}" for p in pairs])
        ax.axhline(1.0, color=FLOOR_COLOR, ls="--", lw=0.9)
        ax.set_ylim(0, 1.05)
    ax.set_ylabel("pore Dice, s_nb 0 vs 1")
    ax.set_title("1.0 = the neighbour arm did nothing")
    ax.legend(frameon=False)
    fig.tight_layout()
    return savefig(fig, figures_dir(root, "cfg"), "guidance")


def report_layup(res, root, floor) -> tuple[str, list[str]]:
    fl = res["real_floor"]
    rows = []
    for name, entry in sorted(res["layups"].items()):
        for reader, r in entry["readers"].items():
            if not r.get("available"):
                rows.append([name, reader, "--", "--", "--", "--", "--"])
                continue
            f = r.get("real_floor") or {}
            rows.append([
                name, reader,
                ms(r["median_abs_error_deg"], 2), fmt(f.get("median_abs_error_deg"), 2),
                ms(r["strict_class_accuracy"], 3), fmt(f.get("strict_class_accuracy"), 3),
                ms(r["recovered_ply_count"], 1) + f" of {entry['requested_ply_count']}",
            ])
    ply_tables = []
    for name, entry in sorted(res["layups"].items()):
        req = entry["requested_deg"]
        hdr = ["ply", "requested"]
        cols = {}
        for reader, r in entry["readers"].items():
            if r.get("available"):
                hdr.append(f"{reader} hit rate")
                cols[reader] = r["per_ply_hit_rate"]
        body = [[str(i + 1), f"{req[i]:g}"] + [fmt(cols[k][i], 2) for k in cols]
                for i in range(len(req))]
        ply_tables.append(f"### {name}\n\n" + table(hdr, body))

    text = [
        "## Angle recovery against the real-volume floor",
        "",
        table(["layup", "reader", "median |err| deg", "real floor", "4-class",
               "real floor", "plies recovered"], rows),
        "",
        f"Floors read from `{fl['source']}` - {fl.get('fft_slice', {}).get('n_plies')} "
        "real plies, direct scoring, `pore_axes` on the STORED mask.",
        "",
        "## Per ply",
        "",
        *ply_tables,
    ]
    figs = _fig_layup(res, root)
    return "\n".join(text) + "\n", figs


def _fig_layup(res, root) -> list[str]:
    set_style()
    names = sorted(res["layups"])
    fig, axes = plt.subplots(1, 2, figsize=(9.0, 3.6))
    readers = ("fft_slice", "pore_axes")
    width = 0.36
    for i, (key, ylabel, better) in enumerate([
        ("median_abs_error_deg", "median |error| (deg)", "lower"),
        ("strict_class_accuracy", "strict 4-class accuracy", "higher"),
    ]):
        ax = axes[i]
        xs = np.arange(len(names))
        for j, reader in enumerate(readers):
            vals = [(res["layups"][n]["readers"].get(reader) or {}).get(key) for n in names]
            mean = [(v or {}).get("mean") or np.nan for v in vals]
            sd = [(v or {}).get("sd") or 0.0 for v in vals]
            ax.bar(xs + (j - 0.5) * width, mean, width, yerr=sd, capsize=3,
                   color=SERIES_COLORS[j], label=reader)
            fl = (res["real_floor"] or {}).get(reader, {}).get(key)
            if fl is not None:
                ax.axhline(fl, color=SERIES_COLORS[j], ls=":", lw=1.1)
        ax.plot([], [], color=FLOOR_COLOR, ls=":", label="real floor")
        ax.set_xticks(xs)
        ax.set_xticklabels(names)
        ax.set_ylabel(f"{ylabel} ({better} is better)")
        ax.legend(frameon=False)
    fig.tight_layout()
    return savefig(fig, figures_dir(root, "layup"), "layup_recovery")


def report_assembly(res, root, floor) -> tuple[str, list[str]]:
    rows = [
        [k, ms(c["seam_xct_ratio"], 3), ms(c["seam_pore_ratio"], 3),
         ms(c["seam_chunk_xct_ratio"], 3), ms(c["seam_chunk_pore_ratio"], 3),
         ms(c["cross_head_disagreement"], 4),
         ms(c["cross_head_disagreement_interior"], 4)]
        for k, c in sorted(res["cells"].items())
    ]
    control = res["vae_tile_decode_control"]
    ctrl_rows = [
        [name, ms(v["seam_xct_ratio"], 3), ms(v["seam_mask_ratio"], 3)]
        for name, v in sorted(control["assemblies"].items())
    ]
    floor_rows = []
    for tag, f in (floor or {}).get("by_shape", {}).items():
        floor_rows.append([f"real {tag}", ms(f["seam_xct_ratio"], 3), "--",
                           ms(f["seam_chunk_xct_ratio"], 3), "--",
                           ms(f["cross_head_disagreement"], 4),
                           ms(f["cross_head_disagreement_interior"], 4)])
    phase = res["window_phase"]
    text = [
        "## Window plane against chunk plane, on the sampler volumes",
        "",
        table(["cell", "seam xct", "seam pore", "chunk xct", "chunk pore",
               "cross-head", "cross-head (interior)"], floor_rows + rows),
        "",
        "Cross-head disagreement is the share of requested material where the grey "
        f"detector (u8 < {res['detector']['t_abs']}, components of at least "
        f"{res['detector']['min_cc']} voxels) says dark and the label says material.",
        "",
        "## VAE tile-decode control (campaign 08, real data, no LDM)",
        "",
        table(["assembly", "seam xct", "seam mask"], ctrl_rows),
        "",
        f"Source `{control['source']}`, {control['n_volumes']} real volumes. "
        "It separates an assembly-side seam from a model-side one: whatever the "
        "tiled row scores is what the decoder alone contributes.",
        "",
        "## Window-phase sensitivity",
        "",
    ]
    if phase.get("available"):
        ref = phase["reference_offset"]
        text += [
            table(["offset", "seed", "pore Dice", f"phi at offset {ref}",
                   "phi at offset", "phi difference"],
                  [[str(p["offset"]), str(p["seed"]), fmt(p["pore_dice"], 3),
                    fmt(p["phi_reference"]), fmt(p["phi_offset"]),
                    fmt(p["phi_difference"])]
                   for p in phase["pairs"]]),
            "",
            table(["offset", "seeds", "pore Dice", "phi difference"],
                  [[off, str(v["n_seeds"]), ms(v["pore_dice"], 3),
                    ms(v["phi_difference"])]
                   for off, v in sorted(phase["by_offset"].items(), key=lambda kv: int(kv[0]))]),
            "",
            phase["note"],
        ]
    else:
        text += [phase.get("note", "not measured")]
    figs = _fig_assembly(res, root, floor)
    return "\n".join(text) + "\n", figs


def _fig_assembly(res, root, floor) -> list[str]:
    set_style()
    fig, ax = plt.subplots(figsize=(7.6, 3.8))
    names = sorted(res["cells"])
    xs = np.arange(len(names))
    keys = [("seam_xct_ratio", "window, grey"), ("seam_chunk_xct_ratio", "chunk, grey"),
            ("seam_pore_ratio", "window, pore"), ("seam_chunk_pore_ratio", "chunk, pore")]
    width = 0.2
    for j, (key, lab) in enumerate(keys):
        vals = [(res["cells"][n][key] or {}).get("mean") or np.nan for n in names]
        errs = [(res["cells"][n][key] or {}).get("sd") or 0.0 for n in names]
        ax.bar(xs + (j - 1.5) * width, vals, width, yerr=errs, capsize=2,
               color=SERIES_COLORS[j], label=lab)
    ax.axhline(1.0, color=FLOOR_COLOR, ls="--", lw=0.9, label="no seam")
    ctrl = res["vae_tile_decode_control"]["assemblies"].get("A_tiled", {})
    v = (ctrl.get("seam_xct_ratio") or {}).get("mean")
    if v is not None:
        ax.axhline(v, color=FLOOR_COLOR, ls=":", lw=1.1, label="VAE tiled control")
    ax.set_xticks(xs)
    ax.set_xticklabels(names, rotation=25, ha="right")
    ax.set_ylabel("seam / interior ratio")
    ax.legend(frameon=False, ncol=3)
    fig.tight_layout()
    return savefig(fig, figures_dir(root, "assembly"), "seams")


def report_geometry(res, root, floor) -> tuple[str, list[str]]:
    s = res["summary"]
    rows = [
        ["air Dice against the requested air", ms(s["dice_air"], 3)],
        ["precision", ms(s["precision_air"], 3)],
        ["recall", ms(s["recall_air"], 3)],
        ["air fraction inside the requested material", ms(s["air_fraction_inside_material"])],
        ["air fraction outside it", ms(s["air_fraction_outside_material"], 3)],
        ["pore fraction inside the material (pore / material)",
         ms(s["phi_pore_inside_material"])],
        ["failure rate", fmt(s["failure_rate"], 2)],
    ]
    floor_note = ""
    if floor:
        fl = next(iter(floor.get("by_shape", {}).values()), None)
        if fl:
            floor_note = (
                "\nReal floor for the air fraction inside specimen: "
                f"{ms(fl['air_fraction'])} over {fl['n_volumes']} test crops. "
                "There is no real floor for the Dice: a real volume was never asked "
                "for a hole.\n"
            )
    text = [
        "## Requested geometry",
        "",
        table(["quantity", "mean +/- sd over seeds"], rows),
        "",
        "The material map carries a 64-voxel notch cut into the y = 0 face and a "
        "200-voxel cylindrical hole through z - the size of the real drilled "
        "registration holes. A model that obeys the map scores a high Dice, a high "
        "air fraction outside the material and a low one inside.",
        floor_note,
    ]
    figs = _fig_geometry(res, root)
    return "\n".join(text) + "\n", figs


def _fig_geometry(res, root) -> list[str]:
    set_style()
    fig, axes = plt.subplots(1, 2, figsize=(8.2, 3.4))
    s = res["summary"]
    ax = axes[0]
    keys = ["dice_air", "precision_air", "recall_air"]
    ax.bar(np.arange(len(keys)), [(s[k] or {}).get("mean") or np.nan for k in keys],
           yerr=[(s[k] or {}).get("sd") or 0.0 for k in keys], capsize=3,
           color=SERIES_COLORS[0])
    ax.set_xticks(np.arange(len(keys)))
    ax.set_xticklabels(["Dice", "precision", "recall"])
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("predicted air vs requested air")

    ax = axes[1]
    keys = ["air_fraction_inside_material", "air_fraction_outside_material"]
    ax.bar(np.arange(2), [(s[k] or {}).get("mean") or np.nan for k in keys],
           yerr=[(s[k] or {}).get("sd") or 0.0 for k in keys], capsize=3,
           color=[SERIES_COLORS[1], SERIES_COLORS[2]])
    ax.set_xticks(np.arange(2))
    ax.set_xticklabels(["inside material", "outside material"])
    ax.set_ylabel("air fraction")
    fig.tight_layout()
    return savefig(fig, figures_dir(root, "geometry"), "requested_geometry")


def report_real_floor(res, root, floor) -> tuple[str, list[str]]:
    rows = []
    for tag, f in sorted(res["by_shape"].items()):
        rows.append([
            tag, str(f["n_volumes"]), "x".join(str(s) for s in f["volume_shape"]),
            ms(f["phi_pore"]), ms(f["air_fraction"]), ms(f["air_fraction_interior"]),
            ms(f["seam_xct_ratio"], 3), ms(f["cell_phi_sd"]),
            ms(f["cross_head_disagreement"], 4),
        ])
    fl = res["layup_floor"]
    layup_rows = [
        [k, fmt(v["median_abs_error_deg"], 2), fmt(v["strict_class_accuracy"], 3),
         fmt(v["frac_within_10"], 2), str(v["n_plies"])]
        for k, v in fl.items() if isinstance(v, dict)
    ]
    text = [
        "## The floor row of every table",
        "",
        table(["shape", "n", "voxels", "phi (pore/material)", "air", "air (interior)",
               "seam xct", "cell phi sd", "cross-head"], rows),
        "",
        res["note"],
        "",
        "## Layup reader floor (campaign 08)",
        "",
        table(["reader", "median |err| deg", "strict 4-class", "within 10 deg", "plies"],
              layup_rows),
        "",
        f"Source `{fl['source']}`.",
    ]
    return "\n".join(text) + "\n", []


def memorisation_section(memo: dict) -> str:
    """The full-store nearest-neighbour search, beside its two real-val floors.

    The generated row alone says nothing: real held-out material also has near
    neighbours in the train set, so a floor row is what turns the ratio into a
    reading.  Grey space gets TWO floors and the difference between them is the
    VAE decoder: the round-tripped row is the like-for-like one, the raw row is
    what the ratio would be with no decoder error at all.
    """
    if not memo:
        return ""
    if not memo.get("available"):
        return ("\n## Memorisation\n\nThe memorisation check was skipped: "
                f"{memo.get('reason')}.\n")

    bank = memo["bank"]
    crit = memo["criterion"]

    def row(label: str, space: str, s: dict) -> list[str] | None:
        if not s or not s.get("n"):
            return None
        return [
            label, space, str(s["n"]),
            fmt(s["ratio_mean"], 3), fmt(s["ratio_median"], 3),
            fmt(s["ratio_min"], 3), fmt(s["ratio_p5"], 3),
            f"{s['n_memorised']} ({s['frac_memorised']:.1%})",
        ]

    def block(label: str, cell: dict) -> list[list[str]]:
        out = []
        for i, space in enumerate(("latent", "grey")):
            r = row(label if i == 0 else "", space, cell.get(space) or {})
            if r:
                out.append(r)
        return out

    header = ["set", "space", "patches", "ratio mean", "median", "min", "p5",
              f"below {crit['threshold']:.3f}"]
    floor = memo["real_val_floor"]
    rows = block("generated", memo["generated"])
    rows += block("real val floor (VAE round trip)", floor)
    raw_row = row("real val floor (raw scan)", "grey", floor.get("grey_raw") or {})
    if raw_row:
        rows.append(raw_row)

    by_phi = [
        r for key, cell in sorted(memo["by_requested_phi"].items(),
                                  key=lambda kv: float(kv[0]))
        for r in block(f"phi {key}", cell)
    ]
    by_nb = [
        r for key, cell in sorted(memo["by_neighbours"].items())
        for r in block(key.replace("_", " "), cell)
    ]

    shapes = sorted({"x".join(str(v) for v in c["volume_shape"])
                     for c in memo["per_case"].values() if c.get("volume_shape")})
    coarse = memo.get("cases_bucketed_at_volume_level") or []
    coarse_note = (
        f"\n**{len(coarse)} volume(s) could not be bucketed per window** and fall "
        f"back to the volume-level bucket: {', '.join(coarse)}. Their patch "
        "positions were never window origins, so no honest per-window state "
        "exists for them.\n"
        if coarse else ""
    )

    text = [
        "",
        "## Memorisation - the full-store nearest-neighbour search",
        "",
        f"Every 64-cubed patch of every generated volume in "
        f"{', '.join(memo['assessments_found'])} at {', '.join(shapes)} voxels "
        f"({memo['n_patches']} patches from {memo['n_cases']} volumes), searched "
        f"against ALL {bank['n_rows']} stride-{bank['stride']} rows of the train "
        f"split (of {bank['n_rows_in_split']} rows in it). Not a sample: the "
        f"whole bank. Volumes above {memo['max_tiles_per_volume']} tiles are left "
        f"out on cost - {len(memo.get('skipped_too_large') or [])} of them.",
        "",
        f"Statistic `{crit['statistic']}`. {crit['reading']}",
        "",
        table(header, rows),
        "",
        floor["note"],
        "",
        "### By requested porosity",
        "",
        table(header, by_phi),
        "",
        "### By neighbour availability when the window was denoised",
        "",
        f"Neighbour state is {memo['neighbour_state_source']}. A face counts as "
        "UNKNOWN only when the chunk it reaches into had not been solved yet, so "
        "this is an ordering fact and not a geometric one: the same position in "
        "the last chunk of a volume has every neighbour it needs.",
        "",
        table(header, by_nb),
        coarse_note,
        "",
        f"Latent space is per-channel normalised with the store's own train "
        f"statistics; the grey bank is the raw source patches at "
        f"`{bank['grey_source']}`.",
        "",
    ]
    return "\n".join(text)


def report_microstructure(res, root, floor) -> tuple[str, list[str]]:
    """Four distances against the real-vs-real floor, then the memorisation search."""
    geom = res["geometry"]
    rows, notes = [], []
    for key, cell in sorted(res["levels"].items(), key=lambda kv: kv[1]["requested"]):
        if not cell.get("available"):
            rows.append([key, "--", "--", "--", "--", cell.get("reason", "not measured")])
            continue
        got, fl, rat = cell["generated_vs_real"], cell["real_vs_real"], cell["ratio"]
        stats = [
            ("S2(r) W1, voxels", got["s2_w1"], fl["s2_w1"], rat["s2_w1"], 3),
            ("pore-size W1, voxels", got["psd_w1"], fl["psd_w1"], rat["psd_w1"], 3),
            ("Ripley K, mean |log ratio|", got["ripley_log_ratio"],
             fl["ripley_log_ratio"], rat["ripley_log_ratio"], 3),
            ("FID, 2-D slices", cell["fid_generated_vs_real"].get("mean"),
             cell["fid_real_vs_real"].get("mean"), rat["fid"], 1),
        ]
        for i, (name, a, b, r, digits) in enumerate(stats):
            rows.append([
                key if i == 0 else "", name, fmt(a, digits), fmt(b, digits),
                fmt(r, 2), "1 is the floor, lower is better",
            ])
        miss = cell.get("real_phi_miss_max")
        if miss is not None and miss > 0.005:
            notes.append(
                f"- At the {key} level the closest real crop the test panels hold is "
                f"{miss:.4f} away in porosity. Part of every distance in that row is "
                "that porosity gap, not a texture difference."
            )

    fid_note = ""
    first = next((c for c in res["levels"].values() if c.get("available")), None)
    if first and not first["fid_generated_vs_real"].get("available"):
        fid_note = ("\nFID was not computed: "
                    f"{first['fid_generated_vs_real'].get('reason')}.\n")
    text = [
        "## Microstructure statistics against the real-vs-real floor",
        "",
        table(["phi", "statistic", "generated vs real", "real vs real (floor)",
               "ratio", "reading"], rows),
        "",
        fid_note,
        memorisation_section(res.get("memorisation") or {}),
        "## What was measured on what",
        "",
        table(["setting", "value"], [
            ["S2 analysis window", f"{geom['s2_window']} cubed, r up to "
                                   f"{geom['s2_r_max']} voxels"],
            ["Ripley r range", f"1 to {geom['ripley_r_max']} voxels, border-corrected"],
            ["connected components", geom["connectivity"]],
            ["FID crops", f"{geom['fid_crop']}x{geom['fid_crop']} at native resolution"],
            ["FID feature extractor", geom["fid_extractor"]],
        ]),
        "",
        "The generated volumes are 192 cubed and the real reference crops are 128 "
        "cubed, because no test panel holds a clean 192-deep box. Every statistic "
        "above is defined either on the fixed analysis window or on a "
        "size-normalised quantity, so the two shapes are compared like for like.",
        "",
        *notes,
    ]
    figs = _fig_microstructure(res, root)
    return "\n".join(text) + "\n", figs


def _fig_microstructure(res, root) -> list[str]:
    set_style()
    cells = [(k, c) for k, c in sorted(res["levels"].items(),
                                       key=lambda kv: kv[1]["requested"])
             if c.get("available")]
    if not cells:
        return []
    fig, axes = plt.subplots(1, 4, figsize=(15.0, 3.5))

    ax = axes[0]
    for j, (key, c) in enumerate(cells):
        cur = c["generated_vs_real"]["curves"]
        col = SERIES_COLORS[j % len(SERIES_COLORS)]
        ax.plot(cur["s2_r"], cur["s2_a"], color=col, label=f"gen {key}")
        ax.plot(cur["s2_r"], cur["s2_b"], color=col, ls="--", label=f"real {key}")
    ax.set_yscale("log")
    ax.set_xlabel("r (voxels)")
    ax.set_ylabel("S2(r)")
    ax.set_title("two-point correlation")
    ax.legend(frameon=False, ncol=2)

    ax = axes[1]
    for j, (key, c) in enumerate(cells):
        cur = c["generated_vs_real"]["curves"]
        edges = np.asarray(cur["psd_bin_edges"], float)
        mid = 0.5 * (edges[:-1] + edges[1:])
        col = SERIES_COLORS[j % len(SERIES_COLORS)]
        ax.plot(mid, cur["psd_density_a"], color=col, label=f"gen {key}")
        ax.plot(mid, cur["psd_density_b"], color=col, ls="--", label=f"real {key}")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("equivalent diameter (voxels)")
    ax.set_ylabel("density")
    ax.set_title("pore-size distribution")
    ax.legend(frameon=False, ncol=2)

    ax = axes[2]
    for j, (key, c) in enumerate(cells):
        cur = c["generated_vs_real"]["curves"]
        r = np.asarray(cur["ripley_r"], float)
        csr = (4.0 / 3.0) * np.pi * r ** 3
        col = SERIES_COLORS[j % len(SERIES_COLORS)]
        ax.plot(r, np.asarray(cur["k_a"], float) / csr, color=col, label=f"gen {key}")
        ax.plot(r, np.asarray(cur["k_b"], float) / csr, color=col, ls="--",
                label=f"real {key}")
    ax.axhline(1.0, color=FLOOR_COLOR, ls=":", lw=0.9)
    ax.set_xlabel("r (voxels)")
    ax.set_ylabel("K(r) / CSR")
    ax.set_title("clustering (1 = Poisson)")
    ax.legend(frameon=False, ncol=2)

    ax = axes[3]
    keys = ["s2_w1", "psd_w1", "ripley_log_ratio", "fid"]
    labels = ["S2", "PSD", "Ripley", "FID"]
    xs = np.arange(len(keys))
    width = 0.8 / max(len(cells), 1)
    for j, (key, c) in enumerate(cells):
        vals = [c["ratio"].get(k) for k in keys]
        ax.bar(xs + (j - (len(cells) - 1) / 2) * width,
               [np.nan if v is None else v for v in vals], width,
               color=SERIES_COLORS[j % len(SERIES_COLORS)], label=f"phi {key}")
    ax.axhline(1.0, color=FLOOR_COLOR, ls="--", lw=0.9, label="the real floor")
    ax.set_xticks(xs)
    ax.set_xticklabels(labels)
    ax.set_ylabel("distance / real-vs-real floor")
    ax.set_title("1.0 = as close as real is to real")
    ax.legend(frameon=False)

    fig.tight_layout()
    return savefig(fig, figures_dir(root, "microstructure"), "microstructure")


def report_surface(res, root, floor) -> tuple[str, list[str]]:
    s = res["summary"]
    fl = res.get("real_surface_floor")

    def block(key, title):
        b = s.get(key)
        if b is None:
            return [f"### {title}", "", "Not generated.", ""]
        rows = [["air fraction OUTSIDE the box", ms(b["air_fraction_outside_box"], 4)],
                ["air fraction INSIDE the box", ms(b["air_fraction_inside_box"], 4)],
                ["dark-but-material (all)", ms(b["dark_but_material"], 4)],
                ["dark-but-material (excl. 2-vox face rim)",
                 ms(b["dark_but_material_excluding_rim"], 4)]]
        for face in ("lower", "upper"):
            f = b[face]
            rows += [
                [f"{face}: |position error| (vox)", ms(f["error_abs_mean"], 3)],
                [f"{face}: roughness Sa (vox)", ms(f["roughness_sa"], 3)],
                [f"{face}: REQUESTED Sa (vox)", ms(f["requested_roughness_sa"], 3)],
                [f"{face}: Sa ratio to request", ms(f["roughness_ratio_to_requested"], 3)],
                [f"{face}: Sa ratio to real floor", ms(f["roughness_ratio_to_real_floor"], 3)],
                [f"{face}: outlier columns", ms(f["n_outliers"], 1)],
                [f"{face}: outlier fraction", ms(f["outlier_fraction"], 5)],
                [f"{face}: outlier clusters", ms(f["outlier_clusters"], 1)],
                [f"{face}: outliers with a pore at the face", ms(f["outliers_with_pore_fraction"], 3)],
            ]
        g = b.get("gate", {})
        gate_txt = "  ".join(f"`{k}` = {v}" for k, v in g.items() if k != "kind")
        return [f"### {title} ({b['n_cases']} cases, gate: {g.get('kind','-')})", "",
                table(["quantity", "mean +/- sd"], rows), "",
                f"Gate: {gate_txt}", ""]

    floor_txt = res.get("floor_note") or ""
    if fl:
        floor_txt = (
            f"Real surface floor over {fl.get('n_faces')} faces of the test "
            f"volumes: Sa {fl.get('sa_mean'):.3f} about the mean plane, "
            f"{fl.get('detrended_sa_mean'):.3f} about a fitted plane, lateral "
            f"correlation length {fl.get('correlation_length_vox_mean')} voxels. "
            "The detrended figure is the one the ratios use: a real coupon is "
            "tilted in the scanner frame, and that tilt is not roughness."
        )

    text = [
        "## The specimen surface", "",
        res.get("why_two_requests", ""), "",
        "Position error is measured against the REQUESTED FIELD, not against a "
        "plane. Against a plane, a correctly followed rough request would read "
        "as position error equal to the requested roughness, and the model would "
        "be marked wrong for obeying.", "",
        *block("flat", "Flat request — controllability (control row)"),
        *block("rough", "Rough request — realism"),
        floor_txt, "",
        f"Failure rate: {fmt(s['failure_rate'], 2)}",
    ]
    return "\n".join(text) + "\n", []


def _group_order(groups: dict) -> list[str]:
    """Real groups first - they are the floor every other row is read against."""
    rank = {"real": 0, "requested": 1, "generated": 2}
    return sorted(groups, key=lambda n: (rank.get(n.split("/", 1)[0], 3), n))


def _axis_cell(ax: dict) -> str:
    """A correlation length, or the reach that could not find one."""
    v = ax.get("corr_length_vox")
    if v is None:
        return f"> {ax.get('reach_vox', 0)}"
    return f"{v:.0f}"


def report_field_stats(res, root, floor) -> tuple[str, list[str]]:
    """The marginal and the per-axis correlation length, real row first."""
    geom = res["geometry"]
    groups = res["groups"]
    order = _group_order(groups)

    marg_rows = []
    for name in order:
        g = groups[name]
        m, q = g["marginal"], g["marginal"]["quantiles"]
        marg_rows.append([
            name, str(g["n_fields"]), str(m["n"]), fmt(m["mean"]), fmt(m["sd"]),
            fmt(m["cv"], 3), fmt(q.get("0.05")), fmt(q.get("0.5")), fmt(q.get("0.95")),
        ])

    corr_rows = []
    for name in order:
        g = groups[name]
        pa, an = g["per_axis"], g["anisotropy"]
        corr_rows.append([
            name,
            _axis_cell(pa["z"]), _axis_cell(pa["y"]), _axis_cell(pa["x"]),
            fmt(an.get("y_over_z"), 2), fmt(an.get("x_over_z"), 2),
            "/".join(str(pa[a].get("reach_vox", 0)) for a in ("z", "y", "x")),
            fmt(pa["z"]["r_at_lag_vox"].get("64"), 3),
            fmt(pa["y"]["r_at_lag_vox"].get("128"), 3),
            fmt(pa["x"]["r_at_lag_vox"].get("128"), 3),
        ])
    ref = (res.get("t_d_reference") or {}).get("corr_length_vox")
    if ref:
        corr_rows.insert(0, [
            "T-D target (real patch grid)", f"{ref['z']:.0f}", f"{ref['y']:.0f}",
            f"{ref['x']:.0f}", fmt(ref["y"] / ref["z"], 2), fmt(ref["x"] / ref["z"], 2),
            "whole dataset", "--", "--", "--",
        ])

    cmp_rows = []
    for key, c in sorted(res["comparisons"].items()):
        m, cl = c["marginal"], c["corr_length"]
        cmp_rows.append([
            key, fmt(m["w1"], 5), fmt(m["w1_ratio"], 3), fmt(m["ks"], 3),
            *[fmt(cl[a]["difference_vox"], 0) for a in ("z", "y", "x")],
        ])

    body = [
        "## What the delivered field looks like",
        "",
        f"Local porosity is measured over **{geom['window_vox']}-voxel windows "
        f"every {geom['stride_vox']} voxels**, pore voxels over material voxels, "
        f"on real crops and generated volumes alike. A window holding less than "
        f"{geom['min_material_frac']:.0%} material is dropped. The requested field "
        f"is read on the {geom['requested_field_stride_vox']}-voxel tile grid it "
        "was painted on.",
        "",
        table(["group", "fields", "windows", "mean phi", "sd", "cv",
               "p5", "p50", "p95"], marg_rows),
        "",
        "## How far it stays correlated, per axis",
        "",
        table(["group", "L z (vox)", "L y (vox)", "L x (vox)", "Ly/Lz", "Lx/Lz",
               "reach z/y/x", "r(z, 64)", "r(y, 128)", "r(x, 128)"], corr_rows),
        "",
        "`L` is the lag at which the field's correlation falls to 1/e, "
        "interpolated between lags. `> N` means the field had NOT decorrelated "
        "by the longest lag the crop reaches: the length is longer than the crop, "
        "not absent. Compare two rows on one axis only when both found a length "
        "inside their common reach - the `r(axis, lag)` columns are the "
        "comparison that always holds.",
        "",
        "## Distance to the real marginal",
        "",
        table(["comparison", "W1", "W1 (mean-normalised)", "KS",
               "dL z", "dL y", "dL x"], cmp_rows),
        "",
        "`W1 (mean-normalised)` divides each sample by its own mean first, so it "
        "measures the SHAPE of the heterogeneity and not the global porosity the "
        "two sets sit at. The `real floor` row is real material against real "
        "material: no generated row can be expected below it. `dL` is "
        "generated minus real, blank where either side had no measurable length "
        "inside the common reach.",
    ]
    note = (res.get("t_d_reference") or {}).get("note")
    if note:
        body += ["", f"T-D target row: {note} Source "
                     f"`{res['t_d_reference']['source']}`."]
    return "\n".join(body) + "\n", _fig_field_stats(res, root)


def _fig_field_stats(res, root) -> list[str]:
    set_style()
    groups = res["groups"]
    order = [n for n in _group_order(groups) if not n.startswith("requested/")]
    fig, axes = plt.subplots(1, 4, figsize=(15.0, 3.5))

    for ai, name in enumerate(("z", "y", "x")):
        ax = axes[ai]
        for j, gname in enumerate(order):
            pa = groups[gname]["per_axis"][name]
            lag = np.asarray(pa["lag_vox"], float)
            r = np.asarray([np.nan if v is None else v for v in pa["correlation"]],
                           float)
            real = gname.startswith("real/")
            ax.plot(lag, r, color=FLOOR_COLOR if real
                    else SERIES_COLORS[j % len(SERIES_COLORS)],
                    ls="--" if real else "-", label=gname)
        ax.axhline(float(np.exp(-1.0)), color=FLOOR_COLOR, ls=":", lw=0.9)
        ax.axhline(0.0, color=FLOOR_COLOR, lw=0.6)
        ax.set_xlabel(f"lag along {name} (voxels)")
        if ai == 0:
            ax.set_ylabel("correlation of window phi")
            ax.legend(frameon=False)
        ax.set_title(f"{name} axis")

    ax = axes[3]
    for j, gname in enumerate(order):
        q = groups[gname]["marginal"]["quantiles"]
        levels = sorted(float(k) for k in q)
        vals = [q[f"{lv:g}"] for lv in levels]
        real = gname.startswith("real/")
        ax.plot(vals, levels, marker="o", ms=3,
                color=FLOOR_COLOR if real else SERIES_COLORS[j % len(SERIES_COLORS)],
                ls="--" if real else "-", label=gname)
    ax.set_xlabel("window phi (pore/material)")
    ax.set_ylabel("cumulative fraction of windows")
    ax.set_title("marginal")
    ax.legend(frameon=False)

    fig.tight_layout()
    return savefig(fig, figures_dir(root, "field_stats"), "field_stats")


def report_multichunk(res, root, floor) -> tuple[str, list[str]]:
    s = res["summary"]

    def block(kind, title):
        b = s.get(kind)
        if b is None:
            return [f"### {title}", "", "Not generated.", ""]
        rows = [
            ["window-plane seam, grey", ms(b["window_plane_seam_xct"], 3)],
            ["CHUNK-plane seam, grey", ms(b["chunk_plane_seam_xct"], 3)],
            ["window-plane seam, pore logit", ms(b["window_plane_seam_pore"], 3)],
            ["CHUNK-plane seam, pore logit", ms(b["chunk_plane_seam_pore"], 3)],
            ["pore Dice across the chunk planes", ms(b["pore_dice_across_chunk_planes"], 3)],
        ]
        if kind == "sphere":
            rows += [["radial surface error (vox)", ms(b["radial_surface_error_vox"], 2)],
                     ["octant spread (vox)", ms(b["octant_spread_vox"], 2)]]
        if kind == "rough":
            rows += [[f"{f} Sa ratio to the request", ms(b[f"{f}_roughness_ratio_to_requested"], 3)]
                     for f in ("lower", "upper")]
        return [f"### {title} ({b['n_cases']} cases)", "",
                table(["quantity", "mean +/- sd"], rows), ""]

    text = [
        "## Assembly across chunk planes on all three axes", "",
        res.get("question", ""), "",
        "Every other large case is 1024x1024x192, which is a single chunk deep: "
        "the z axis never crosses a chunk plane. These do, on all three.", "",
        res.get("chunk_plane_note", ""), "",
        "**" + res.get("not_physics", "") + "**", "",
        "The two seam families are reported separately. A window plane is two "
        "overlapping windows inside ONE chunk solve; a chunk plane is two "
        "independent solves meeting. Pooling them would let a good window "
        "average away a bad chunk boundary.", "",
        *block("box", "Full-material box"),
        *block("sphere", "Sphere, radius 160"),
        *block("rough", "Rough surface slab"),
        f"Failure rate: {fmt(s.get('failure_rate'), 2)}",
    ]
    return "\n".join(text) + "\n", []
#: Row order of the arm tables.  Fixed, not alphabetical: the three samplers in
#: increasing chunk size and then the ceiling, so the table reads as the
#: argument it is.
ARM_ORDER = ("joint", "autoregressive", "hybrid", "teacher_forced")
#: Per-chunk quantities the findings show, and how many digits each deserves.
CHUNK_ROWS = (
    ("chunk_plane_seam_xct", "chunk-plane seam (grey)", 3),
    ("tile_plane_seam_xct", "tile-plane seam (grey)", 3),
    ("chunk_plane_seam_pore", "chunk-plane seam (pore)", 3),
    ("tile_plane_seam_pore", "tile-plane seam (pore)", 3),
    ("phi_pore", "porosity per chunk", 4),
    ("s2_relative_distance", "S2 across vs inside", 4),
)


def _mean_sd(values) -> dict:
    """Mean and sd of already-aggregated numbers, in the shape ``ms`` reads.

    Local on purpose: importing ``metrics`` for one function would pull the
    sampler, and with it torch, into a stage that reads ``results.json`` and
    nothing else.
    """
    v = [x for x in values if x is not None and np.isfinite(x)]
    if not v:
        return {"mean": None, "sd": None, "n": 0}
    a = np.asarray(v, float)
    return {"mean": float(a.mean()),
            "sd": float(a.std(ddof=1)) if a.size > 1 else 0.0,
            "n": int(a.size)}


def _cell_order(cells: dict) -> list[str]:
    def key(name):
        arm = cells[name]["arm"]
        rank = ARM_ORDER.index(arm) if arm in ARM_ORDER else len(ARM_ORDER)
        return (str(cells[name]["scale"]), rank)

    return sorted(cells, key=key)


def report_assembly_modes(res, root, floor) -> tuple[str, list[str]]:
    cells = res["cells"]
    order = _cell_order(cells)

    head = table(
        ["arm @ scale", "chunk tiles", "neighbours", "seeds", "seam at ref planes",
         "seam at tile planes", "delivered phi", "air (interior)", "wall s", "fail"],
        [[name, str(c["generated_chunk_tiles"]), str(c["neighbour_mode"]),
          str(c["n_seeds"]), ms(c["volume_seam_xct_reference"], 3),
          ms(c["volume_seam_tile_xct"], 3), ms(c["delivered_phi"]),
          ms(c["air_fraction_interior"]), ms(c["wall_time_s"], 1),
          fmt(c["failure_rate"], 2)]
         for name, c in ((n, cells[n]) for n in order)],
    )

    fl = res.get("real_floor") or {}
    floor_rows = [
        [f"real {tag}", str(f["n_volumes"]),
         *[ms(f.get(key), dig) for key, _, dig in CHUNK_ROWS]]
        for tag, f in sorted(fl.items())
    ]

    def mean_over_chunks(cell, key):
        vals = [b.get("mean") for b in cell["by_chunk_index"][key]]
        return _mean_sd(vals)

    body = [
        "## The four arms", "",
        res["note"], "",
        head, "",
        res["reference_grid_note"], "",
        res["teacher_forced_note"], "",
        "## Per chunk, along the generation order", "",
        "Every quantity below is a mean over the seeds at each chunk index, then "
        "summarised two ways: its mean over all chunks, and the OLS slope against "
        "the chunk index. The slope is the number that matters — a chunked "
        "sampler fails by compounding, so an arm can hold a good volume average "
        "and still degrade with distance from the first chunk. A slope of zero "
        "means the last chunk is as good as the first.", "",
        table(
            ["quantity", "n"] + [lab for _, lab, _ in CHUNK_ROWS],
            floor_rows,
        ) if floor_rows else "No real floor on disk — run `eval_v4 real-floor`.",
        "",
        "The real rows are the floor: real material was assembled by nothing, so "
        "what it scores at the reference planes is what the measurement reads "
        "when there is no seam.", "",
    ]

    for name in order:
        c = cells[name]
        rows = []
        for key, label, dig in CHUNK_ROWS:
            series = c["by_chunk_index"][key]
            trend = c["trend"][key]
            first = series[0].get("mean") if series else None
            last = series[-1].get("mean") if series else None
            rows.append([
                label, ms(mean_over_chunks(c, key), dig),
                fmt(first, dig), fmt(last, dig),
                fmt(trend.get("slope"), dig + 1), fmt(trend.get("r2"), 2),
            ])
        body += [
            f"### {name} ({c['n_chunks']} chunks, seeds {c['seeds']})", "",
            table(["quantity", "mean over chunks", "chunk 0", "last chunk",
                   "slope / chunk", "r2"], rows),
            "",
        ]

    figs = _fig_assembly_modes(res, root)
    return "\n".join(body) + "\n", figs


def _fig_assembly_modes(res, root) -> list[str]:
    """One panel per quantity: the per-chunk series of every arm, at each scale."""
    set_style()
    cells = res["cells"]
    scales = sorted({c["scale"] for c in cells.values()})
    keys = [("chunk_plane_seam_xct", "chunk-plane seam (grey)"),
            ("phi_pore", "porosity per chunk"),
            ("s2_relative_distance", "S2 across vs inside")]
    fig, axes = plt.subplots(len(keys), len(scales),
                             figsize=(5.2 * len(scales), 3.0 * len(keys)),
                             squeeze=False)
    for i, (key, ylabel) in enumerate(keys):
        for j, scale in enumerate(scales):
            ax = axes[i][j]
            for name in _cell_order(cells):
                c = cells[name]
                if c["scale"] != scale:
                    continue
                series = c["by_chunk_index"][key]
                # Chunk 0 owns no chunk plane, so its value is legitimately
                # absent; NaN leaves a gap in the line rather than drawing a
                # point that was never measured.
                y = [np.nan if b.get("mean") is None else b["mean"] for b in series]
                e = [b.get("sd") or 0.0 for b in series]
                colour = SERIES_COLORS[ARM_ORDER.index(c["arm"]) % len(SERIES_COLORS)]
                ax.errorbar(range(len(y)), y, yerr=e, marker="o", ms=3, capsize=2,
                            color=colour, label=c["arm"])
            fl = (res.get("real_floor") or {}).get("large" if scale == "1024" else "small")
            v = (fl or {}).get(key, {}).get("mean")
            if v is not None:
                ax.axhline(v, color=FLOOR_COLOR, ls=":", lw=1.1, label="real floor")
            ax.set_xlabel("chunk index (generation order)")
            ax.set_ylabel(ylabel)
            ax.set_title(f"{scale}")
            if i == 0 and j == 0:
                ax.legend(frameon=False, ncol=2)
    fig.tight_layout()
    return savefig(fig, figures_dir(root, "assembly_modes"), "per_chunk_series")


# ---------------------------------------------------------------------------
# 12 - stress geometries (EXPLORATORY)
# ---------------------------------------------------------------------------

def _request_order(res) -> list[str]:
    """The designed order, not the alphabet.

    The list runs from the requests nearest the training coupons to the ones
    furthest from them, and read in that order the table says where the
    conditioning stops. Sorted alphabetically it says nothing.
    """
    from poregen.eval_v4.cases import STRESS_GEOMETRIES  # noqa: PLC0415

    designed = [g[0] for g in STRESS_GEOMETRIES]
    seen = list(res.get("summary") or {})
    return ([r for r in designed if r in seen]
            + sorted(r for r in seen if r not in designed))


def _stress_rows(rows) -> list[list[str]]:
    out = []
    for r in sorted(rows, key=lambda r: r.get("ddim_steps") or 0):
        band = r.get("chunk_band") or {}
        geo = r.get("geometry_agreement") or {}
        surf = r.get("surface_agreement") or {}
        out.append([
            fmt(r.get("ddim_steps")),
            "x".join(str(v) for v in r["volume_shape"]),
            fmt((r.get("phase_fractions") or {}).get("phi_pore")),
            fmt((r.get("phase_fractions") or {}).get("air_fraction_all"), 3),
            fmt(geo.get("dice_air"), 3) if geo.get("available") is not False else "n/a",
            fmt(band.get("ratio_-8"), 2),
            fmt(band.get("ratio_+0"), 2),
            fmt(band.get("ratio_-8_terminal"), 2),
            fmt(((r.get("seams") or {}).get("seam_chunk_xct_ratio")), 3),
            fmt((surf.get("lower") or {}).get("error_abs_mean"), 1),
            fmt((surf.get("upper") or {}).get("error_abs_mean"), 1),
            fmt((r.get("failure_flags") or {}).get("failed")),
        ])
    return out


STRESS_HEADER = ["DDIM", "shape", "phi pore", "air frac", "air Dice",
                 "band -8", "band +0", "band -8 term", "chunk seam grey",
                 "err lo", "err hi", "failed"]


def report_stress_geometry(res, root, floor) -> tuple[str, list[str]]:
    by_request: dict[str, list] = {}
    for r in res["per_case"]:
        by_request.setdefault(r.get("request"), []).append(r)
    text = [
        "## Requests the training material never contained",
        "",
        "**EXPLORATORY. Nothing here is gated and no threshold is applied.** "
        + res["off_gates_because"],
        "",
        "`band -8` and `band +0` are the material porosity in the 8 voxels "
        "either side of a chunk frontier, as a ratio to the volume's own mean, "
        "over NON-TERMINAL planes; 1.00 is flat. `band -8 term` is the last "
        "plane on each axis, whose successor's far face is the volume edge - "
        "the one frontier configuration the model was trained on, and so the "
        "anchor the other columns are read against. `err lo` / `err hi` are the "
        "mean absolute distance in voxels between the generated specimen "
        "surface and the requested one, over the columns the request fills.",
        "",
    ]
    for request in _request_order(res):
        rows = by_request.get(request) or []
        if not rows:
            continue
        note = (rows[0].get("geometry_note") or "").strip()
        text += [f"### {request}", ""]
        if note:
            text += [note, ""]
        text += [table(STRESS_HEADER, _stress_rows(rows)), ""]
        text += _stress_extras(rows)

    # The reason itself is one sentence repeated per row, so it is stated once
    # above and the table carries only what differs: the shape that prevented
    # the reading. The full sentence stays in results.json, where a reader has
    # no note beside it.
    missing = [[r["case"],
                "x".join(str(v) for v in ((r["layup_recovery"] or {})
                                          .get("in_plane") or []))]
               for r in res["per_case"]
               if (r.get("layup_recovery") or {}).get("available") is False]
    if missing:
        text += [
            "## What could not be measured, and why",
            "",
            res["layup_window_note"],
            "",
            table(["case with no layup reading", "in-plane shape"], missing),
            "",
        ]
    figs = _fig_stress_geometry(res, root) + _stress_montage(res, root)
    return "\n".join(text) + "\n", figs


def _stress_extras(rows) -> list[str]:
    """The blocks only some requests carry: the painted ramp and the two legs."""
    out = []
    ramp_rows = [[fmt(r.get("ddim_steps")),
                  fmt((r["ramp"] or {}).get("requested_slope_per_tile"), 5),
                  fmt((r["ramp"] or {}).get("delivered_slope_per_tile"), 5),
                  fmt((r["ramp"] or {}).get("slope_ratio"), 2),
                  fmt((r.get("local_obedience") or {}).get("within_volume_slope"), 2),
                  fmt((r.get("local_obedience") or {}).get("within_volume_r2"), 3)]
                 for r in sorted(rows, key=lambda r: r.get("ddim_steps") or 0)
                 if r.get("ramp")]
    if ramp_rows:
        out += [
            "The painted field, read two ways: as a slope along y (the ramp the "
            "request carries) and as the within-volume tile fit assessment 3 "
            "uses. A model that delivers its own mean everywhere scores near "
            "zero on both.",
            "",
            table(["DDIM", "requested slope/tile", "delivered slope/tile",
                   "slope ratio", "within-vol slope", "within-vol R2"], ramp_rows),
            "",
        ]
    leg_rows = []
    for r in sorted(rows, key=lambda r: r.get("ddim_steps") or 0):
        for name, leg in (r.get("legs") or {}).items():
            leg_rows.append([fmt(r.get("ddim_steps")), name,
                             "x".join(str(v) for v in leg["shape"]),
                             fmt(leg.get("phi_pore")),
                             fmt(leg.get("air_fraction_inside_material"), 4),
                             fmt((leg.get("layup_recovery") or {}).get("available"))])
    if leg_rows:
        out += [
            "The two legs, cropped apart with the corner in neither: no ply "
            "orientation is defined through a corner, so a reader run across it "
            "would score the model against a request that does not exist there.",
            "",
            table(["DDIM", "leg", "shape", "phi pore", "air inside material",
                   "layup read"], leg_rows),
            "",
        ]
    return out


def _fig_stress_geometry(res, root) -> list[str]:
    """Two panels: how well each shape was carved, and the band at its frontiers."""
    set_style()
    order = _request_order(res)
    s = res["summary"]
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 3.8))
    x = np.arange(len(order))

    ax = axes[0]
    dice = [((s[r] or {}).get("dice_air") or {}).get("mean") for r in order]
    ax.bar(x, [d if d is not None else np.nan for d in dice],
           color=SERIES_COLORS[0])
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("air Dice vs the requested shape")
    ax.set_title("Was the shape carved where it was asked for?")

    ax = axes[1]
    for i, key in enumerate(("chunk_band_ratio_-8", "chunk_band_ratio_+0")):
        v = [((s[r] or {}).get(key) or {}).get("mean") for r in order]
        ax.bar(x + (i - 0.5) * 0.38,
               [b if b is not None else np.nan for b in v], width=0.38,
               color=SERIES_COLORS[i + 1], label=key.split("_")[-1])
    ax.axhline(1.0, color=FLOOR_COLOR, ls="--", lw=1.0, label="flat")
    ax.set_ylabel("band phi / volume phi")
    ax.set_title("The chunk-plane band, by request")
    ax.legend()

    for ax in axes:
        ax.set_xticks(x)
        ax.set_xticklabels(order, rotation=30, ha="right")
    fig.tight_layout()
    return savefig(fig, figures_dir(root, "stress_geometry"), "stress_overview")


def _stress_montage(res, root) -> list[str]:
    """Three mid-slices per case, grey over label, into ``<root>/inspection``.

    ``eval_v4_inspection_pack.render_case`` already draws exactly this panel for
    the gated assessments, and it is imported rather than reproduced: a second
    renderer would drift from the first and a reader comparing a stress case
    with a gated one would be comparing two different figures. The import is
    late because the pack is a script, not a package module.
    """
    import sys  # noqa: PLC0415

    analysis = Path(__file__).resolve().parents[3] / "scripts" / "analysis"
    if str(analysis) not in sys.path:
        sys.path.insert(0, str(analysis))
    try:
        import eval_v4_inspection_pack as pack  # noqa: PLC0415
    except Exception as exc:                    # noqa: BLE001
        logger.warning("stress montage skipped: %s", exc)
        return []

    out = Path(root) / "inspection"
    out.mkdir(parents=True, exist_ok=True)
    paths = []
    for r in res["per_case"]:
        case_dir = Path(root) / "stress_geometry" / "volumes" / r["case"]
        if not (case_dir / "manifest.json").exists():
            continue
        why = (r.get("geometry_note") or r.get("request") or "")[:90]
        info = pack.render_case(case_dir, "stress_geometry", r["case"], why, out)
        if info:
            paths.append(info["png"])
    return paths


# ---------------------------------------------------------------------------
# 13 - conditioning out of distribution (EXPLORATORY)
# ---------------------------------------------------------------------------

def _hit_string(hits) -> str:
    """Per-ply agreement as a row of marks, which reads faster than a list."""
    if not hits:
        return "--"
    return "".join("#" if h else "." for h in hits)


def _ood_layup_table(rows) -> str:
    out = []
    for r in sorted(rows, key=lambda r: r["case"]):
        lr = r.get("layup_recovery") or {}
        if not lr.get("available"):
            out.append([r["case"], "--", "no reading", "--", "--", "--"])
            continue
        for reader in ("fft_slice", "pore_axes"):
            h = (r.get("ply_hits") or {}).get(reader) or {}
            if not h.get("available"):
                out.append([r["case"], reader, "unavailable", "--", "--", "--"])
                continue
            out.append([
                r["case"], reader,
                f"{h['n_hit']}/{h['n_plies']}",
                fmt(h.get("class_accuracy"), 2),
                fmt(h.get("median_abs_error_deg"), 1),
                _hit_string(h.get("per_ply_hit")),
            ])
    return table(["case", "reader", "plies hit", "class acc",
                  "median |err| deg", "per ply"], out)


def report_ood_conditioning(res, root, floor) -> tuple[str, list[str]]:
    groups: dict[str, list] = {}
    for r in res["per_case"]:
        groups.setdefault(r.get("group"), []).append(r)

    text = [
        "## Conditioning asked for what the training set does not contain",
        "",
        "**EXPLORATORY. Nothing here is gated and no threshold is applied.** "
        + res["off_gates_because"],
        "",
        "Four groups. Each moves ONE axis of the conditioning and holds the "
        "others at their trained values, so a failure can be attributed.",
        "",
        f"*Not measured here:* {res['not_measured']}",
        "",
    ]

    if groups.get("sequence"):
        text += [
            "### 1 — stacking sequences the training set does not contain",
            "",
            "Every angle is inside the trained set {0, 45, -45, 90}: what is out "
            "of distribution is the ORDER, so a failure cannot be blamed on an "
            "orientation the model never saw. `per ply` reads from the z = 0 "
            "face, `#` for a ply whose class came back and `.` for one that did "
            "not.",
            "",
            _ood_layup_table(groups["sequence"]),
            "",
        ]

    if groups.get("pitch"):
        blocks = sorted({(r.get("requested_ply_blocks"),
                          (r.get("notes") or {}).get("pitch_vox"))
                         for r in groups["pitch"]})
        made = ", ".join(f"{p:g} vox -> {n} plies" for n, p in blocks if p)
        text += [
            "### 2 — ply pitch above and below the trained ones",
            "",
            f"Trained pitches are 10.0 and 19.6 voxels. Here: {made}. "
            "The two requests are the same thing as \"6 thick plies and 24 thin "
            "plies filling the depth\", so they are one set of two cases.",
            "",
            _ood_layup_table(groups["pitch"]),
            "",
        ]

    if groups.get("porosity"):
        rows = []
        for r in sorted(groups["porosity"], key=lambda r: (
                (r.get("conditioning") or {}).get("requested_phi") or 0, r["case"])):
            c = r.get("conditioning") or {}
            pe = r.get("porosity_error") or {}
            rows.append([
                r["case"],
                fmt(c.get("requested_phi"), 3),
                "lifted" if c.get("porosity_clamped") is False else "held",
                fmt(c.get("phi_conditioned"), 3),
                fmt(c.get("cond_por"), 2),
                fmt(pe.get("delivered_phi"), 4),
                fmt(pe.get("error"), 4),
                fmt((r.get("failure_flags") or {}).get("failed")),
            ])
        text += [
            "### 3 — requested porosity outside the clamped range",
            "",
            res["clamp_note"],
            "",
            "`cond_por` is in standard deviations of the training "
            "distribution's own log-porosity, so it says HOW FAR off the "
            "manifold each request is rather than only that it is off.",
            "",
            table(["case", "requested", "clamp", "conditioned", "cond_por (sd)",
                   "delivered", "error", "failed"], rows),
            "",
            "**The 0.150 and 0.200 rows are a stated failure mode, not a "
            "target.** They ask for a porosity no training volume has, and a "
            "large error there is the answer rather than a defect.",
            "",
        ]

    if groups.get("correlation"):
        rows = []
        for r in sorted(groups["correlation"],
                        key=lambda r: r.get("requested_corr_vox") or 0):
            per = ((r.get("field") or {}).get("per_axis") or {})
            lo = r.get("local_obedience") or {}
            rows.append([
                r["case"], fmt(r.get("requested_corr_vox"), 0),
                *[fmt((per.get(ax) or {}).get("corr_length_vox"), 0)
                  for ax in ("z", "y", "x")],
                fmt(lo.get("within_volume_slope"), 2),
                fmt(lo.get("within_volume_r2"), 3),
            ])
        text += [
            "### 4 — the field's correlation length",
            "",
            "The request is ISOTROPIC, which is itself off the manifold: the "
            "measured lengths are (79, 414, 901) voxels in (z, y, x), because a "
            "laminate is not isotropic. So each case asks two things at once — "
            "a length the model never saw AND the same length on every axis — "
            "and the delivered lengths are reported per axis for that reason.",
            "",
            table(["case", "requested (vox)", "delivered z", "delivered y",
                   "delivered x", "obedience slope", "R2"], rows),
            "",
        ]

    figs = _fig_ood_conditioning(res, root)
    return "\n".join(text) + "\n", figs


def _fig_ood_conditioning(res, root) -> list[str]:
    """Delivered against requested porosity, with the clamp drawn on it."""
    from poregen.diffusion.conditioning import POR_MAX, POR_MIN  # noqa: PLC0415

    sel = [r for r in res["per_case"] if r.get("group") == "porosity"]
    if not sel:
        return []
    set_style()
    fig, ax = plt.subplots(figsize=(5.4, 4.2))
    req = np.array([(r["conditioning"] or {}).get("requested_phi") or 0.0 for r in sel])
    got = np.array([
        (v if (v := (r.get("porosity_error") or {}).get("delivered_phi")) is not None
         else np.nan) for r in sel])
    if not np.isfinite(got).any():
        logger.warning("ood_conditioning: no delivered porosity to plot")
        plt.close(fig)
        return []
    lifted = np.array([(r["conditioning"] or {}).get("porosity_clamped") is False
                       for r in sel])
    lo, hi = 0.0, max(float(np.nanmax(req)), float(np.nanmax(got))) * 1.08
    ax.axvspan(POR_MIN, POR_MAX, color=FLOOR_COLOR, alpha=0.12,
               label=f"training range [{POR_MIN}, {POR_MAX}]")
    ax.plot([lo, hi], [lo, hi], ls="--", lw=1.0, color=FLOOR_COLOR,
            label="delivered = requested")
    ax.scatter(req[~lifted], got[~lifted], s=34, color=SERIES_COLORS[0],
               label="clamp held")
    ax.scatter(req[lifted], got[lifted], s=34, marker="^",
               color=SERIES_COLORS[1], label="clamp lifted")
    ax.set_xlabel("requested porosity")
    ax.set_ylabel("delivered porosity (pore / material)")
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.legend(loc="upper left")
    fig.tight_layout()
    return savefig(fig, figures_dir(root, "ood_conditioning"), "porosity_extremes")


REPORTERS = {
    "sampler": report_sampler,
    "porosity_global": report_porosity_global,
    "porosity_local": report_porosity_local,
    "cfg": report_cfg,
    "layup": report_layup,
    "assembly": report_assembly,
    "geometry": report_geometry,
    "multichunk": report_multichunk,
    "surface": report_surface,
    "microstructure": report_microstructure,
    "field_stats": report_field_stats,
    "assembly_modes": report_assembly_modes,
    "real_floor": report_real_floor,
    "stress_geometry": report_stress_geometry,
    "ood_conditioning": report_ood_conditioning,
}


def report_one(root: str | Path, assessment: str) -> Path:
    root = Path(root)
    res = read_results(root, assessment)
    floor = load_floor(root) if assessment != "real_floor" else None
    body, figs = REPORTERS[assessment](res, root, floor)
    text = _header(res, root, floor) + body
    if figs:
        text += "\n## Figures\n\n" + "\n".join(
            f"- `{Path(p).relative_to(root)}`" for p in figs
        ) + "\n"
    return write_findings(root, assessment, text)


def report(root: str | Path, assessments: list[str] | None = None) -> list[Path]:
    """Write ``findings.md`` and the figures for every measured assessment."""
    root = Path(root)
    names = assessments or [
        a for a in REPORTERS
        if (assessment_dir(root, a) / "results.json").exists()
    ]
    out = []
    for name in names:
        out.append(report_one(root, name))
        logger.info("wrote %s", out[-1])
    if not out:
        raise FileNotFoundError(
            f"no results.json under {root} - run `eval_v4 measure <assessment>` first."
        )
    return out
