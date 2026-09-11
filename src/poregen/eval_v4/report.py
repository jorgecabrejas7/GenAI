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


def report_microstructure(res, root, floor) -> tuple[str, list[str]]:
    """Five distances, each against the real-vs-real floor that makes it readable."""
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
            ("memorisation NN distance",
             cell["memorisation_generated"].get("nn_distance_mean"),
             cell["memorisation_real"].get("nn_distance_mean"),
             rat["memorisation_nn_distance"], 2),
        ]
        for i, (name, a, b, r, digits) in enumerate(stats):
            memo = name.startswith("memorisation")
            rows.append([
                key if i == 0 else "", name, fmt(a, digits), fmt(b, digits),
                fmt(r, 2),
                "1 or more is healthy" if memo else "1 is the floor, lower is better",
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
    memo_note = ""
    if first and not first["memorisation_generated"].get("available"):
        memo_note = ("\nThe memorisation check was skipped: "
                     f"{first['memorisation_generated'].get('reason')}.\n")

    text = [
        "## Microstructure statistics against the real-vs-real floor",
        "",
        table(["phi", "statistic", "generated vs real", "real vs real (floor)",
               "ratio", "reading"], rows),
        "",
        fid_note,
        memo_note,
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
    keys = ["s2_w1", "psd_w1", "ripley_log_ratio", "fid", "memorisation_nn_distance"]
    labels = ["S2", "PSD", "Ripley", "FID", "memo"]
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


REPORTERS = {
    "sampler": report_sampler,
    "porosity_global": report_porosity_global,
    "porosity_local": report_porosity_local,
    "cfg": report_cfg,
    "layup": report_layup,
    "assembly": report_assembly,
    "geometry": report_geometry,
    "surface": report_surface,
    "microstructure": report_microstructure,
    "assembly_modes": report_assembly_modes,
    "real_floor": report_real_floor,
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
