"""Collect every number shown in the group-meeting deck, with its source.

Writes deck/media/numbers.json: {key: {"value": ..., "source": "<file>#<json path>"}}.
The deck's speaker notes cite these sources. Nothing here is computed beyond
selection and simple aggregation (mean, min, max over seeds), stated per key.

Usage: python talks/poregen-group/scripts/extract_numbers.py
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
CAMP = ROOT / "runs" / "campaigns"
FINAL = CAMP / "18-eval-v4-final-v4-run0005"          # final model, full evaluation
A2S = CAMP / "35-a2s-seam-v4"                          # overlapped chunk scheme
ABL = CAMP / "24-ablation-v4-run0005"                  # one-input-at-a-time ablation
PHI_ONLY = CAMP / "25-ldm-phi-only-v4-run0002"         # porosity-only LDM baseline
SLICEGAN = CAMP / "22-slicegan-baseline-v4"
DDPM3D = CAMP / "23-ddpm3d-baseline-v4"
VAE_TABLE = CAMP / "33-advperc-ldm" / "ablation" / "table.md"
OUT = Path(__file__).resolve().parents[1] / "deck" / "media" / "numbers.json"

numbers: dict[str, dict] = {}


def put(key, value, source):
    numbers[key] = {"value": value, "source": str(Path(source).relative_to(ROOT)) if str(source).startswith(str(ROOT)) else source}


def load(p: Path):
    return json.loads(p.read_text())


def rel(p: Path, path: str) -> str:
    return f"{p.relative_to(ROOT)}#{path}"


# --- porosity control: dose response -------------------------------------
pg = load(FINAL / "porosity_global" / "results.json")
dr = pg["dose_response"]
put("dose_slope", dr["slope"], rel(FINAL / "porosity_global/results.json", "dose_response.slope"))
put("dose_r2", dr["r2"], rel(FINAL / "porosity_global/results.json", "dose_response.r2"))
put("dose_n", dr["n"], rel(FINAL / "porosity_global/results.json", "dose_response.n"))
put("dose_gate", dr["gate"], rel(FINAL / "porosity_global/results.json", "dose_response.gate"))
put("dose_gate_frac", dr["frac_within_gate"], rel(FINAL / "porosity_global/results.json", "dose_response.frac_within_gate"))
pts = [(c["requested_global_phi"], c["phi_pore"], c["ddim_steps"]) for c in pg["per_case"]]
put("dose_points", pts, rel(FINAL / "porosity_global/results.json", "per_case[].(requested_global_phi, phi_pore, ddim_steps)"))

# --- guidance scale --------------------------------------------------------
cfg = load(FINAL / "cfg" / "results.json")
rows = {}
for c in cfg["per_case"]:
    m = re.match(r"spor([\d.]+)_target([\d.]+)", c["case"])
    if m:
        rows.setdefault((float(m[1]), float(m[2])), []).append(c["phi_pore"])
put("cfg_por", [[s, t, float(np.mean(v)), float(np.std(v))] for (s, t), v in sorted(rows.items())],
    rel(FINAL / "cfg/results.json", "per_case[spor*_target*].phi_pore, mean/sd over seeds"))

# --- microstructure ratios to the real floor ------------------------------
ms = load(FINAL / "microstructure" / "results.json")
for lv, d in ms["levels"].items():
    for k in ("s2_w1", "psd_w1", "ripley_log_ratio", "fid"):
        put(f"micro_{lv}_{k}", d["ratio"][k], rel(FINAL / "microstructure/results.json", f"levels.{lv}.ratio.{k}"))
    cur = d["generated_vs_real"]["curves"]
    put(f"s2curve_{lv}", {"r": cur["s2_r"], "gen": cur["s2_a"], "real": cur["s2_b"]},
        rel(FINAL / "microstructure/results.json", f"levels.{lv}.generated_vs_real.curves (a = generated, b = real)"))
mem = ms["memorisation"]
put("mem_n_patches", mem["n_patches"], rel(FINAL / "microstructure/results.json", "memorisation.n_patches"))
put("mem_bank_rows", mem["bank"]["n_rows"], rel(FINAL / "microstructure/results.json", "memorisation.bank.n_rows"))
put("mem_threshold", mem["criterion"]["threshold"], rel(FINAL / "microstructure/results.json", "memorisation.criterion.threshold"))
put("mem_generated", mem["generated"], rel(FINAL / "microstructure/results.json", "memorisation.generated"))
put("mem_real_floor", mem["real_val_floor"], rel(FINAL / "microstructure/results.json", "memorisation.real_val_floor"))

# --- geometry, layup, local field ------------------------------------------
geo = load(FINAL / "geometry" / "results.json")
d = [c for c in geo["per_case"] if c["case"].startswith("notch_hole")]
put("geom_air_dice", float(np.mean([c["geometry"]["dice_air"] for c in d])),
    rel(FINAL / "geometry/results.json", "per_case[notch_hole*].geometry.dice_air, mean over seeds"))

lay = load(FINAL / "layup" / "results.json")
for name in ("A", "B16", "C"):
    cs = [c for c in lay["per_case"] if c["case"].startswith(name + "_")]
    for reader in ("fft_slice", "pore_axes"):
        med = [c["recovery"]["readers"][reader]["median_abs_error_deg"] for c in cs]
        put(f"layup_{name}_{reader}_median", float(np.mean(med)),
            rel(FINAL / "layup/results.json", f"per_case[{name}_*].recovery.readers.{reader}.median_abs_error_deg, mean over seeds"))
    put(f"layup_{name}_requested_deg", cs[0]["recovery"]["requested_deg"],
        rel(FINAL / "layup/results.json", f"per_case[{name}_seed101].recovery.requested_deg"))
    put(f"layup_{name}_recovered_pore_axes_seed101", cs[0]["recovery"]["readers"]["pore_axes"]["recovered_deg"],
        rel(FINAL / "layup/results.json", f"per_case[{name}_seed101].recovery.readers.pore_axes.recovered_deg"))

loc = load(FINAL / "porosity_local" / "results.json")
for field in ("checkerboard", "halves", "coherent"):
    cs = [c for c in loc["per_case"] if c["case"].startswith(field)]
    put(f"local_{field}_slope", float(np.mean([c["local"]["within_volume_slope"] for c in cs])),
        rel(FINAL / "porosity_local/results.json", f"per_case[{field}*].local.within_volume_slope, mean over cases"))
    put(f"local_{field}_r2", float(np.mean([c["local"]["within_volume_r2"] for c in cs])),
        rel(FINAL / "porosity_local/results.json", f"per_case[{field}*].local.within_volume_r2, mean over cases"))

# --- seams: production chunking against the overlapped scheme --------------
def chunk_axes(results: Path, shape_y=1024, ddim=50):
    out = []
    for c in load(results)["per_case"]:
        if c.get("volume_shape", [0, 0])[1] == shape_y and c.get("ddim_steps") == ddim:
            s = c["seams"]
            out.append({"case": c["case"], "y": s["seam_chunk_pore_y_ratio"], "x": s["seam_chunk_pore_x_ratio"],
                        "grey_y": s["seam_chunk_xct_y_ratio"], "grey_x": s["seam_chunk_xct_x_ratio"]})
    return out

put("seam_production", chunk_axes(FINAL / "sampler" / "results.json"),
    rel(FINAL / "sampler/results.json", "per_case[1024_ddim50_*].seams.seam_chunk_{pore,xct}_{y,x}_ratio"))
put("seam_overlapped", chunk_axes(A2S / "sampler" / "results.json"),
    rel(A2S / "sampler/results.json", "per_case[1024_ddim50_*].seams.seam_chunk_{pore,xct}_{y,x}_ratio (3 of 18 cases measured)"))

# --- baselines: ratios to the real floor -----------------------------------
for name, root in (("phi_only", PHI_ONLY), ("slicegan", SLICEGAN), ("ddpm3d", DDPM3D)):
    p = root / ("slicegan" if name == "slicegan" else "ddpm3d" if name == "ddpm3d" else "microstructure") / "results.json"
    if not p.exists():
        p = root / "microstructure" / "results.json"
    r = load(p)
    for lv, d in r["levels"].items():
        if not d.get("available", True):
            continue
        for k in ("s2_w1", "psd_w1", "ripley_log_ratio", "fid"):
            put(f"base_{name}_{lv}_{k}", d["ratio"].get(k), rel(p, f"levels.{lv}.ratio.{k}"))
        put(f"base_{name}_{lv}_floor_s2", d["real_vs_real"]["s2_w1"], rel(p, f"levels.{lv}.real_vs_real.s2_w1"))
    put(f"base_{name}_floor_check", None, "see base_*_floor_s2 against micro floor in the same level")
phi_only_pg = load(PHI_ONLY / "porosity_global" / "results.json")["dose_response"]
put("base_phi_only_dose_slope", phi_only_pg["slope"], rel(PHI_ONLY / "porosity_global/results.json", "dose_response.slope"))
put("base_phi_only_gate_frac", phi_only_pg["frac_within_gate"], rel(PHI_ONLY / "porosity_global/results.json", "dose_response.fraction_within_gate"))
for lv, d in ms["levels"].items():
    put(f"final_{lv}_floor_s2", d["real_vs_real"]["s2_w1"], rel(FINAL / "microstructure/results.json", f"levels.{lv}.real_vs_real.s2_w1"))

# --- ablation ---------------------------------------------------------------
arms = load(ABL / "ablation" / "results.json")["arms"]
ABR = ABL / "ablation/results.json"
for reader in ("fft_slice", "pore_axes"):
    put(f"abl_ply_swap_scored_{reader}", arms["ply_swap"]["recovery_of_the_scored_layup"][reader]["median_abs_error_deg"]["mean"],
        rel(ABR, f"arms.ply_swap.recovery_of_the_scored_layup.{reader}.median_abs_error_deg.mean"))
    put(f"abl_ply_swap_fed_{reader}", arms["ply_swap"]["recovery_of_the_fed_layup"][reader]["median_abs_error_deg"]["mean"],
        rel(ABR, f"arms.ply_swap.recovery_of_the_fed_layup.{reader}.median_abs_error_deg.mean"))
put("abl_material_full_air_dice", arms["material_full"]["dice_air"]["mean"] if isinstance(arms["material_full"]["dice_air"], dict) else arms["material_full"]["dice_air"],
    rel(ABR, "arms.material_full.dice_air"))
put("abl_position_swap_air_dice", arms["position_swap_geometry"]["dice_air"] if not isinstance(arms["position_swap_geometry"]["dice_air"], dict) else arms["position_swap_geometry"]["dice_air"]["mean"],
    rel(ABR, "arms.position_swap_geometry.dice_air"))
for f in ("checkerboard", "halves", "coherent"):
    put(f"abl_field_global_{f}_r2", arms[f"field_global_{f}"]["local_r2"]["mean"], rel(ABR, f"arms.field_global_{f}.local_r2.mean"))

# --- autoencoder compression study (table A, disc ON column) ---------------
rows = []
for line in VAE_TABLE.read_text().split("## B.")[0].splitlines():
    m = re.match(r"\| (rf-\d+) \| 16³×(\d+) \|", line)
    if m:
        cells = [x.strip() for x in line.strip("|").split("|")]
        on = lambda s: float(s.split("→")[1])
        rows.append({"rf": m[1], "c": int(m[2]), "grey_l1": on(cells[4]), "lpips": on(cells[5]),
                     "pore_dice": on(cells[6]), "air_dice": on(cells[7]), "phi_mae": on(cells[8])})
put("vae_rungs", rows, f"{VAE_TABLE.relative_to(ROOT)}#table A, ON column (whole val split, last step)")

OUT.parent.mkdir(parents=True, exist_ok=True)
OUT.write_text(json.dumps(numbers, indent=1, default=float))
print(f"{len(numbers)} numbers -> {OUT}")
