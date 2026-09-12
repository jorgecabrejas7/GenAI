#!/usr/bin/env python
"""Build the eval-v4 inspection notebook.

The notebook is generated, not hand-edited: every cell is a string in this file,
so the notebook the author runs on a laptop is traceable to a commit.  Run:

    python scripts/analysis/build_eval_v4_notebook.py

It writes
  runs/campaigns/12-eval-v4/inspect_eval_v4.ipynb   (beside the data it reads)
  runs/campaigns/12-eval-v4/requirements-inspect.txt
  notebooks/eval_v4_inspection.ipynb                (committed copy, no outputs)

The notebook itself is self-contained (json / tifffile / numpy / pandas /
plotly / ipywidgets).  If the poregen package is importable it uses it for the
paper figures and for on-the-fly metric recomputation; otherwise those cells
say so and skip.  Every section is guarded so the notebook runs end to end on
a partial campaign.
"""
from __future__ import annotations

import subprocess
from pathlib import Path

import nbformat
from nbformat.v4 import new_code_cell, new_markdown_cell, new_notebook

REPO = Path(__file__).resolve().parents[2]
CAMPAIGN = REPO / "runs" / "campaigns" / "12-eval-v4"
OUT_CAMPAIGN = CAMPAIGN / "inspect_eval_v4.ipynb"
OUT_REQ = CAMPAIGN / "requirements-inspect.txt"
OUT_REPO = REPO / "notebooks" / "eval_v4_inspection.ipynb"

REQUIREMENTS = """\
numpy>=1.26
pandas>=2.0
tifffile>=2023.7
scipy>=1.11
scikit-image>=0.21
matplotlib>=3.7
plotly>=5.20
ipywidgets>=8.0
jupyterlab>=4.0
jupyterlab_widgets>=3.0
anywidget>=0.9
nbformat>=5.9
pyyaml>=6.0
"""

# ---------------------------------------------------------------------------
# cell list.  Each entry is ("md", text) or ("code", text).  Sections register
# an anchor so the index is generated from the same list and cannot drift.
# ---------------------------------------------------------------------------

CELLS: list[tuple[str, str]] = []
SECTIONS: list[tuple[str, str, str]] = []   # (anchor, title, what you will see)
_sec = [0]


def md(text: str) -> None:
    CELLS.append(("md", text.strip("\n")))


def code(text: str) -> None:
    CELLS.append(("code", text.strip("\n")))


def section(title: str, what: str, level: int = 2) -> str:
    """Start a section: heading with anchor + back link; registers the index row."""
    _sec[0] += 1
    anchor = f"sec-{_sec[0]:02d}"
    SECTIONS.append((anchor, title, what))
    hashes = "#" * level
    md(f'<a id="{anchor}"></a>\n{hashes} {_sec[0]}. {title}\n\n[↑ back to index](#index)')
    return anchor


def subsection(title: str) -> None:
    md(f"### {title}\n\n[↑ back to index](#index)")


def index_cell_text() -> str:
    rows = ["| # | section | what you will see |", "|---|---|---|"]
    for i, (anchor, title, what) in enumerate(SECTIONS, 1):
        rows.append(f"| {i} | [{title}](#{anchor}) | {what} |")
    return '<a id="index"></a>\n## Index\n\n' + "\n".join(rows)


def git_remote() -> str:
    try:
        out = subprocess.run(["git", "remote", "get-url", "origin"], cwd=REPO,
                             capture_output=True, text=True, check=True).stdout.strip()
        return out or "git@github.com:jorgecabrejas7/GenAI.git"
    except Exception:  # noqa: BLE001
        return "git@github.com:jorgecabrejas7/GenAI.git"


# ===========================================================================
# 0. Setup instructions
# ===========================================================================

def build_setup() -> None:
    remote_ssh = git_remote()
    remote_https = "https://github.com/jorgecabrejas7/GenAI.git"
    md(f"""
# PoreGen — eval v4 inspection notebook

This notebook inspects the evaluation campaign of the ldm06 model (conditional 3D latent diffusion that
generates synthetic X-ray CT volumes of CFRP laminates together with a voxel label map).
Every section first explains, in plain English, what the model does to produce the thing being looked at,
how each number is computed, and how to read the figure. Then it shows the data.

**Label colours everywhere:** grey = material, red = pore, blue = air. Label codes on disk: 0 material, 1 pore, 2 air.

**Three rules of eval v4** (they explain why the files look the way they do):
1. Every generated volume carries a `manifest.json` that says exactly what was requested. Nothing is inferred from a file name.
2. Every metric declares which manifest fields it needs, and refuses a volume that lacks them.
3. Real held-out scans go through every metric first. Their number is the *floor*: the value a perfect generator would get,
   because the measurement itself is not zero on real material. Every table shows the floor beside the generated value.

**The notebook runs on a partial campaign.** A section whose inputs are not on disk yet prints a yellow
"NOT YET AVAILABLE" box and moves on. Re-run the copy commands later and re-run the notebook to fill it in.

---
## Step 1 — copy the campaign to your laptop

Resumable, re-runnable: the same commands later only fetch what is new.

```bash
mkdir -p ~/Dev/poregen-eval/campaigns && cd ~/Dev/poregen-eval
H=jorgecabrejas@192.168.8.91:/home/jorgecabrejas/Dev/GenAI
rsync -avP --partial $H/runs/campaigns/12-eval-v4            ./campaigns/
rsync -avP --partial $H/runs/campaigns/13-label-uncertainty  ./campaigns/
rsync -avP --partial $H/runs/campaigns/10-eval-v4-real-floor ./campaigns/
rsync -avP $H/runs/ldm/ldm06-run-0001-20260907-145657-z8-c128-bs256-lr1e-04/convergence_check.jsonl ./campaigns/ldm06_convergence_check.jsonl
# these two appear later in the queue; the commands fail harmlessly until then
rsync -avP --partial $H/runs/campaigns/17-chunk-band-trial   ./campaigns/
rsync -avP --partial $H/runs/campaigns/16-window-rim          ./campaigns/
rsync -avP --partial $H/runs/campaigns/19-stress-geometry     ./campaigns/
rsync -avP --partial $H/runs/campaigns/18-eval-v4-final       ./campaigns/
rsync -avP --partial --exclude 'r08_*' --exclude 'calibration_probe*' --exclude 'gate_*' $H/runs/campaigns/09-r08-latent-sweep ./campaigns/
rsync -avP --partial $H/runs/campaigns/11-decoder-ft         ./campaigns/
rsync -avP --partial $H/runs/campaigns/14-downstream-utility ./campaigns/
```
Campaign 12 is about 25 GB now and about 35 GB when finished. To skip the accidental step-77k preview
add `--exclude preview_77k` to the first command.

## Step 2 — clone the code (needed for the paper figures and metric recomputation cells)

```bash
cd ~/Dev/poregen-eval
git clone {remote_ssh} poregen        # or: git clone {remote_https} poregen
# the default branch (main) is the one to use
```

## Step 3 — conda environment

```bash
conda create -n poregen-inspect python=3.12 -y
conda activate poregen-inspect
pip install -r campaigns/12-eval-v4/requirements-inspect.txt
pip install torch --index-url https://download.pytorch.org/whl/cpu     # CPU torch is enough here
pip install -e ./poregen
```

## Step 4 — run

```bash
cd ~/Dev/poregen-eval
jupyter lab campaigns/12-eval-v4/inspect_eval_v4.ipynb
```
Run all cells (Run ▸ Run All Cells). Widgets need JupyterLab ≥ 4 with ipywidgets ≥ 8, both in the requirements file.
If a Plotly figure shows as blank, restart the kernel once after the first install; if a widget shows as text,
run `pip install "ipywidgets>=8" jupyterlab_widgets` and reload the browser tab.
The notebook finds the campaign automatically when it sits inside `campaigns/12-eval-v4/`; otherwise set `ROOT` in the configuration cell.
""")


# ===========================================================================
# 1. Configuration and helpers
# ===========================================================================

def build_config() -> None:
    section("Configuration and helpers",
            "where the data is, and the small library every later cell uses")
    md("""
**What this cell does.** It locates the campaign folders, and defines the helpers the rest of the notebook uses:
`have()` tests for files, `unavailable()` prints the yellow box, `load_json()` reads a JSON file
(manifests can contain bare `NaN` tokens, which strict parsers reject; Python's `json` accepts them),
and `manifests()` builds one table with one row per generated volume.

Nothing here computes a result. If this cell prints an error, the path in `ROOT` is wrong.
""")
    code(r'''
import json, math, os, sys, re, warnings
from pathlib import Path
from functools import lru_cache

import numpy as np
import pandas as pd
import tifffile
import plotly.graph_objects as go
import plotly.express as px
import plotly.io as pio
from plotly.subplots import make_subplots
from IPython.display import display, HTML, Markdown, Image
import ipywidgets as W

warnings.filterwarnings("ignore", category=FutureWarning)
pd.set_option("display.width", 200); pd.set_option("display.max_columns", 60)
pio.templates.default = "plotly_white"

# ---- where the data is -----------------------------------------------------
_here = Path.cwd()
_cands = [_here, _here / "campaigns" / "12-eval-v4", Path.home() / "Dev" / "poregen-eval" / "campaigns" / "12-eval-v4",
          Path.home() / "Dev" / "GenAI" / "runs" / "campaigns" / "12-eval-v4"]
ROOT = next((p for p in _cands if (p / "real_floor").exists() or (p / "sampler").exists()), _cands[1])
ROOT = Path(os.environ.get("EVAL_V4_ROOT", ROOT)).resolve()
# EVAL_V4_HEADLESS=1 is set only by the headless verification run (nbconvert): widget cells then build
# their controls but do not display them, because nbclient can stall waiting on widget comm traffic.
# In JupyterLab this variable is unset and every widget shows as normal.
HEADLESS = bool(os.environ.get("EVAL_V4_HEADLESS"))
def show_widget(box, first_draw):
    if HEADLESS:
        print("headless run: widget built but not displayed (set by EVAL_V4_HEADLESS)")
        return
    display(box); first_draw()
CAMPAIGNS = ROOT.parent
LABEL_ROOT = CAMPAIGNS / "13-label-uncertainty"
FLOOR10_ROOT = CAMPAIGNS / "10-eval-v4-real-floor"
DECODER_FT_ROOT = CAMPAIGNS / "11-decoder-ft"
DOWNSTREAM_ROOT = CAMPAIGNS / "14-downstream-utility"
TRIAL_ROOT = CAMPAIGNS / "17-chunk-band-trial"
RIM_ROOT = CAMPAIGNS / "16-window-rim"
STRESS_ROOT = CAMPAIGNS / "19-stress-geometry"
FINAL_ROOT = CAMPAIGNS / "18-eval-v4-final"
RUNGS_ROOT = CAMPAIGNS / "09-r08-latent-sweep"
CONV_JSONL = next((p for p in [CAMPAIGNS / "ldm06_convergence_check.jsonl",
                               ROOT.parents[2] / "ldm" / "ldm06-run-0001-20260907-145657-z8-c128-bs256-lr1e-04" / "convergence_check.jsonl"]
                   if p.exists()), CAMPAIGNS / "ldm06_convergence_check.jsonl")
print("campaign root :", ROOT, "(exists)" if ROOT.exists() else "(MISSING - set ROOT)")
for _n, _p in [("label uncertainty", LABEL_ROOT), ("real-floor campaign 10", FLOOR10_ROOT),
               ("decoder fine-tune", DECODER_FT_ROOT), ("downstream utility", DOWNSTREAM_ROOT), ("convergence jsonl", CONV_JSONL)]:
    print(f"{_n:24s}: {_p}  {'ok' if _p.exists() else 'not on disk yet'}")

# ---- constants of the evaluation --------------------------------------------
TILE = 64                 # one latent window = 64 voxels; the model sees 64^3 at a time
LATENT_DOWN = 4           # 4 voxels per latent cell
EDGE_VOX = 32             # "interior" = more than 32 voxels from every face
POROSITY_GATE = 0.005     # |delivered - requested| below this passes
SAMPLED_STD_REF = 1.863   # std of sampled train latents (mu + sigma*eps), in mu units
LABEL_NAMES = {0: "material", 1: "pore", 2: "air"}
LABEL_COLORS = ["#bfbfbf", "#d81b1b", "#3366d9"]
ASSESSMENT_ORDER = ["sampler", "porosity_global", "porosity_local", "cfg", "layup", "assembly",
                    "geometry", "microstructure", "surface", "multichunk", "assembly_modes", "field_stats"]
EXPECTED_CASES = {"sampler": 18, "porosity_global": 48, "porosity_local": 18, "cfg": 24, "layup": 9,
                  "assembly": 9, "geometry": 8, "microstructure": 9, "surface": 12, "multichunk": 5,
                  "assembly_modes": 17, "field_stats": 0, "real_floor": 39}

# ---- guards ------------------------------------------------------------------
def have(*paths) -> bool:
    return all(Path(p).exists() for p in paths)

def unavailable(what: str, waiting_for: str) -> None:
    display(HTML(f'<div style="background:#fff7cc;border-left:6px solid #e0a800;padding:8px 12px;margin:6px 0">'
                 f'<b>NOT YET AVAILABLE</b> — {what}<br><small>waiting for: {waiting_for}</small></div>'))

def note(text: str) -> None:
    display(HTML(f'<div style="background:#eef5ff;border-left:6px solid #3b6fd8;padding:6px 12px;margin:6px 0">{text}</div>'))

def load_json(path):
    with open(path) as fh:
        return json.load(fh)            # accepts bare NaN tokens

def case_dir(assessment: str, case: str) -> Path:
    return ROOT / assessment / "volumes" / case

def case_dirs(assessment: str) -> list:
    d = ROOT / assessment / "volumes"
    return sorted(p for p in d.iterdir() if (p / "manifest.json").exists()) if d.exists() else []

@lru_cache(maxsize=None)
def manifests(assessment: str) -> pd.DataFrame:
    """One row per case: manifest fields + flattened notes + generation_stats (gs_*)."""
    rows = []
    for d in case_dirs(assessment):
        m = load_json(d / "manifest.json")
        notes = m.pop("notes", None) or {}
        gs = notes.pop("generation_stats", None) or {}
        row = {**m, **{f"note_{k}": v for k, v in notes.items() if not isinstance(v, (dict, list))},
               **{f"gs_{k}": v for k, v in gs.items()}}
        row["dir"] = str(d)
        row["shape_str"] = "x".join(str(s) for s in m["volume_shape"])
        row["n_voxels"] = int(np.prod(m["volume_shape"]))
        row["has_probs"] = (d / "probs.npz").exists(); row["has_latents"] = (d / "latents.npy").exists()
        row["has_field"] = (d / "requested_field.npy").exists(); row["has_material"] = (d / "requested_material.npy").exists()
        rows.append(row)
    df = pd.DataFrame(rows)
    if len(df) and "requested_global_phi" in df:
        df["requested_global_phi"] = pd.to_numeric(df["requested_global_phi"], errors="coerce")
    return df

def results(assessment: str):
    p = ROOT / assessment / "results.json"
    return load_json(p) if p.exists() else None

def per_case_df(assessment: str):
    r = results(assessment)
    return pd.json_normalize(r["per_case"], sep=".") if r else None

def ms(d, digits=4):
    """Format a {mean, sd, n} triple."""
    if not isinstance(d, dict) or d.get("mean") is None:
        return "—"
    sd = d.get("sd"); n = d.get("n")
    return f"{d['mean']:.{digits}f}" + (f" ± {sd:.{digits}f}" if sd is not None else "") + (f" (n={n})" if n else "")

def cells_table(cells: dict, keys: list, digits=4) -> pd.DataFrame:
    """A {cell_name: {key: {mean,sd,n}}} dict -> tidy table."""
    out = []
    for name, c in cells.items():
        row = {"cell": name}
        for k in keys:
            v = c.get(k)
            row[k] = ms(v, digits) if isinstance(v, dict) and "mean" in v else v
        out.append(row)
    return pd.DataFrame(out)

def mean_sd_frame(cells: dict, key: str) -> pd.DataFrame:
    out = []
    for name, c in cells.items():
        v = c.get(key) or {}
        out.append({"cell": name, "mean": v.get("mean"), "sd": v.get("sd"), "n": v.get("n"), **{k: c[k] for k in c if not isinstance(c[k], (dict, list))}})
    return pd.DataFrame(out)

def show_findings(assessment: str, root=None) -> bool:
    """Display findings.md and its figures if `eval_v4 report` has run."""
    root = Path(root or ROOT)
    f = root / assessment / "findings.md"
    if not f.exists():
        unavailable(f"{assessment}/findings.md", f"eval_v4 report --assessment {assessment}")
        return False
    display(Markdown(f.read_text()))
    for png in sorted((root / assessment / "figures").glob("*.png")):
        display(Image(filename=str(png)))
    return True

def floor_hline(fig, y, text="real floor", row=None, col=None, color="#555"):
    if y is None or (isinstance(y, float) and math.isnan(y)):
        return
    kw = dict(row=row, col=col) if row else {}
    fig.add_hline(y=y, line_dash="dash", line_color=color, annotation_text=text, annotation_position="top left", **kw)

print("helpers ready")
''')
    md("""
**Reading volumes without loading them.** A 1024×1024×192 grey volume is 200 MB. The TIFFs are uncompressed,
so one z-page can be read on its own, and a y- or x-slice is assembled by reading one line from every page.
The helpers below do that, so the slice viewers stay fast even on the panel-wide cases.
`read_volume()` loads a whole volume; it prints the size first and refuses above 400 MB unless forced.
""")
    code(r'''
def _tif(path):
    return tifffile.TiffFile(str(path))

def volume_shape(d) -> tuple:
    return tuple(load_json(Path(d) / "manifest.json")["volume_shape"])

def read_slice(d, which: str, axis: str, idx: int) -> np.ndarray:
    """One 2-D slice of volume.tif ('grey') or label.tif ('label'), axis z/y/x, page-wise."""
    path = Path(d) / ("volume.tif" if which == "grey" else "label.tif")
    with _tif(path) as tf:
        series = tf.series[0]
        nz, ny, nx = series.shape[:3]
        if axis == "z":
            return np.asarray(series.asarray(key=int(idx)))
        pages = tf.pages
        if axis == "y":
            return np.stack([np.asarray(pages[k].asarray())[int(idx), :] for k in range(nz)])
        return np.stack([np.asarray(pages[k].asarray())[:, int(idx)] for k in range(nz)])

def read_volume(d, which="label", force=False):
    path = Path(d) / ("volume.tif" if which == "grey" else "label.tif")
    mb = path.stat().st_size / 1e6
    if mb > 400 and not force:
        print(f"{path.name} is {mb:.0f} MB; pass force=True to load it"); return None
    return tifffile.imread(str(path))

def chunk_planes(d) -> dict:
    """Voxel indices of the chunk planes on each axis, from the manifest (chunk_tiles x 64)."""
    m = load_json(Path(d) / "manifest.json")
    ct = m.get("chunk_tiles"); shape = m["volume_shape"]
    out = {}
    for ax, n, c in zip("zyx", shape, ct or (None, None, None)):
        period = (c or 0) * TILE
        out[ax] = [k for k in range(period, n, period)] if period and period < n else []
    return out

def tile_planes(d) -> dict:
    shape = volume_shape(d)
    return {ax: list(range(TILE, n, TILE)) for ax, n in zip("zyx", shape)}

def middle_indices(d) -> dict:
    """Per axis: the chunk plane nearest the middle if there is one, else the centre."""
    shape = volume_shape(d); cp = chunk_planes(d); out = {}
    for ax, n in zip("zyx", shape):
        planes = cp[ax]
        out[ax] = (min(planes, key=lambda p: abs(p - n // 2)), True) if planes else (n // 2, False)
    return out

def tile_phi_map(label: np.ndarray, tile=TILE) -> np.ndarray:
    """Porosity per 64-voxel tile = pore / (pore + material); NaN where the tile is < 50 % material+pore."""
    nz, ny, nx = (s // tile for s in label.shape)
    out = np.full((nz, ny, nx), np.nan)
    for i in range(nz):
        for j in range(ny):
            for k in range(nx):
                t = label[i*tile:(i+1)*tile, j*tile:(j+1)*tile, k*tile:(k+1)*tile]
                pore = (t == 1).sum(); mat = (t == 0).sum()
                if pore + mat >= 0.5 * t.size:
                    out[i, j, k] = pore / (pore + mat)
    return out

def phase_fractions(label: np.ndarray) -> dict:
    pore, mat, air = [(label == c).sum() for c in (1, 0, 2)]
    inner = label[EDGE_VOX:-EDGE_VOX, EDGE_VOX:-EDGE_VOX, EDGE_VOX:-EDGE_VOX] if min(label.shape) > 2 * EDGE_VOX else label
    return {"phi_pore (pore/material)": pore / max(pore + mat, 1), "phi_pore_all (pore/all)": pore / label.size,
            "air_fraction": air / label.size, "air_fraction_interior": (inner == 2).mean()}

print("volume helpers ready")
''')
    md("""
**Optional: the poregen package.** If the repo is installed (Step 2 and 3 of the setup), this cell imports the
evaluation code itself. That enables two kinds of cell later: the *paper figures* exactly as `eval_v4 report` draws
them, and *recomputation* of a metric on one chosen case, so you can check a number in a table against the code
that produced it. If the import fails the notebook still runs; those cells print a note.
""")
    code(r'''
HAVE_POREGEN = False
try:
    import poregen.eval_v4.metrics as PM
    import poregen.eval_v4.report as PR
    import poregen.eval_v4.io as PIO
    import matplotlib
    matplotlib.use("module://matplotlib_inline.backend_inline")   # report.py may have set Agg
    import matplotlib.pyplot as plt
    HAVE_POREGEN = True
    print("poregen available:", Path(PM.__file__).parents[2])
except Exception as exc:  # noqa: BLE001
    import matplotlib
    import matplotlib.pyplot as plt
    print("poregen not importable (fine; package-dependent cells will skip):", type(exc).__name__, str(exc)[:120])
%matplotlib inline
''')


# ===========================================================================
# 2. Progress, 3. reading a case, 4. slice viewer, 5. compare viewer
# ===========================================================================

def build_progress() -> None:
    section("Campaign progress", "which assessments are generated, measured and reported")
    md("""
**The pipeline has three stages per assessment.** *Generate* runs the model and writes one folder per case
(`volume.tif`, `label.tif`, `manifest.json`, and for smaller volumes `probs.npz` with the class probabilities and
`latents.npy` with the finished latent). *Measure* reads those folders and writes `results.json`. *Report* turns
`results.json` into `findings.md` and figures. The queue runs generation for all assessments first, then measure,
then report, so for a while there are volumes with no numbers. This cell shows where each assessment stands.

**How to read it.** A full bar means every planned case is on disk. The three ticks say whether results,
findings and figures exist. `field_stats` generates nothing (it re-reads other volumes), so its bar is always empty.
""")
    code(r'''
rows = []
for a in ASSESSMENT_ORDER + ["real_floor"]:
    n = len(case_dirs(a)); exp = EXPECTED_CASES.get(a, 0)
    rows.append({"assessment": a, "cases on disk": n, "expected": exp,
                 "results.json": have(ROOT / a / "results.json"),
                 "findings.md": have(ROOT / a / "findings.md"),
                 "figures": (ROOT / a / "figures").exists() and any((ROOT / a / "figures").glob("*.png"))})
progress = pd.DataFrame(rows)
display(progress)
fig = go.Figure()
fig.add_bar(x=progress["assessment"], y=progress["cases on disk"], name="on disk", marker_color="#1b6ca8")
fig.add_bar(x=progress["assessment"], y=(progress["expected"] - progress["cases on disk"]).clip(lower=0), name="missing", marker_color="#e0a800")
fig.update_layout(barmode="stack", title="Generated cases per assessment", height=350, yaxis_title="cases")
fig.show()
''')


def build_case_reading() -> None:
    section("How to read one case", "a manifest explained field by field, and what one volume folder holds")
    md("""
**How the model produces a volume.** The request is a set of maps and numbers: the target porosity φ, a map of where
material should be (the *material map*; air is wanted outside it), a stacking sequence of ply angles (the *layup*),
and optionally a painted porosity *field* on a 64-voxel grid. The diffusion model works in a compressed space
(*latent*: 8 numbers per 4×4×4 voxel block). It starts from noise and removes noise in `ddim_steps` rounds.
It only ever sees a 64³ voxel window (16³ latent cells), so a bigger volume is built from overlapping windows
(*stride* 32 voxels) that are averaged at every round, in groups called *chunks* of `chunk_tiles` windows
(3×3×3 = 192³ voxels). Chunks are finished one after another; a finished chunk becomes the *neighbour* of the next.
At the end the VAE decoder turns the latent into the grey volume and the 3-class label, decoding overlapping
blocks so no block edges show (`decode = overlapped`).

**The manifest** records exactly that: shape, steps, chunk tiles, stride, decode mode, the guidance scales
`s_por` (how strongly the porosity request is pushed) and `s_nb` (how strongly the neighbours are pushed),
the seed, and the requested φ / layup / material. `notes.generation_stats` holds numbers the sampler computed
as it finished: the delivered porosity and air, and the seam statistics (explained in the assembly section).
""")
    code(r'''
_ex = next((d for a in ASSESSMENT_ORDER for d in case_dirs(a)), None)
if _ex is None:
    unavailable("no generated case on disk", "eval_v4 generate")
else:
    m = load_json(_ex / "manifest.json")
    gs = (m.get("notes") or {}).pop("generation_stats", {})
    print("example case:", _ex.relative_to(ROOT)); print()
    print(json.dumps({k: v for k, v in m.items() if k != "notes"}, indent=1)[:3000])
    print("\nnotes:", json.dumps(m.get("notes"), indent=1)[:1200])
    print("\ngeneration_stats (first keys):", {k: gs[k] for k in list(gs)[:8]})
    print("\nfiles:", sorted(p.name for p in _ex.iterdir()))
''')
    md("""
**Two porosity definitions.** Inside eval v4, φ is *pore voxels divided by material-plus-pore voxels*: air is not
material, so a coupon with air around it must not look less porous. The model was trained on a different bookkeeping,
*pore voxels divided by all 64³ voxels of a window*; the sampler converts one into the other per window using the
material map. The table below shows both for every generated case, from the label map, so you can see when they differ
(only when there is air in the volume).
""")
    code(r'''
rows = []
for a in ASSESSMENT_ORDER:
    df = manifests(a)
    if len(df) == 0: continue
    for _, r in df.iterrows():
        rows.append({"assessment": a, "case": r["case"], "shape": r["shape_str"], "requested φ": r.get("requested_global_phi"),
                     "delivered φ (sampler, pore/material)": r.get("gs_actual_label_porosity"),
                     "air fraction": r.get("gs_actual_label_air"), "ddim": r.get("ddim_steps"), "seed": r.get("seed"),
                     "wall time s": r.get("wall_time_s"), "peak GPU GB": (r.get("peak_gpu_memory_bytes") or 0) / 1e9})
ALL_CASES = pd.DataFrame(rows)
if len(ALL_CASES) == 0:
    unavailable("no generated cases", "eval_v4 generate")
else:
    display(ALL_CASES.head(12)); print(len(ALL_CASES), "generated cases in total")
''')


def build_slice_viewer() -> None:
    section("Slice viewer", "look at any generated or real volume, any axis, grey / label / probability")
    md("""
**What you see.** A 2-D slice through a 3-D volume. Axis *z* is the through-thickness direction of the laminate
(plies are stacked along z); *y* and *x* are in-plane. The grey image is the synthetic CT (0 = black, 255 = white);
the label image colours each voxel material / pore / air; the probability view shows how sure the decoder was that a
voxel is pore (0 to 1), which exists only for volumes up to 192³ (larger ones store the pore log-odds only).

**Chunk planes and tile planes.** The model built the volume in 192³ chunks and 64-voxel tiles. The "jump to chunk
plane" button moves the slice onto the nearest plane where two chunks meet; any visible line there is an assembly seam.
Dashed lines mark the tile planes (every 64 voxels) and solid lines the chunk planes.

**What good looks like.** Plies visible as alternating texture along z, pores as small dark blobs of realistic size,
no straight lines at 64 or 192 voxel multiples, air only outside the requested material.
""")
    code(r'''
def _png_uri(rgb: np.ndarray) -> str:
    """Encode an RGB uint8 image as a PNG data URI: Plotly renders it as one blob instead of a 3-million-number JSON."""
    import base64, io
    from PIL import Image as _PILImage
    buf = io.BytesIO(); _PILImage.fromarray(np.ascontiguousarray(rgb.astype(np.uint8))).save(buf, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()

def _overlay_rgb(grey, label, alpha=0.45):
    rgb = np.stack([grey]*3, -1).astype(np.float32) / 255.0
    cols = np.array([[0.75,0.75,0.75],[0.85,0.1,0.1],[0.2,0.4,0.85]])
    lab = cols[np.clip(label, 0, 2)]
    mask = (label > 0)[..., None]
    return np.where(mask, (1-alpha)*rgb + alpha*lab, rgb)

def slice_figure(d, axis="z", idx=None, mode="grey + label", alpha=0.45, width=650):
    d = Path(d); shape = volume_shape(d)
    n = dict(zip("zyx", shape))[axis]
    if idx is None: idx = middle_indices(d)[axis][0]
    idx = int(np.clip(idx, 0, n-1))
    grey = read_slice(d, "grey", axis, idx); label = read_slice(d, "label", axis, idx)
    if mode == "grey":
        img = np.stack([grey]*3, -1)
    elif mode == "label":
        img = (np.array([[191,191,191],[216,27,27],[51,102,217]])[np.clip(label,0,2)]).astype(np.uint8)
    elif mode == "pore probability":
        pz = d / "probs.npz"
        if not pz.exists():
            print("no probs.npz for this case"); img = np.stack([grey]*3, -1)
        else:
            z = np.load(pz)
            if "class_probs" in z.files:
                cp = z["class_probs"]
                sl = {"z": cp[1, idx], "y": cp[1, :, idx, :], "x": cp[1, :, :, idx]}[axis].astype(np.float32)
            else:
                lg = z["pore_logit"]
                sl = {"z": lg[idx], "y": lg[:, idx, :], "x": lg[:, :, idx]}[axis].astype(np.float32); sl = 1/(1+np.exp(-sl))
            import matplotlib.cm as _cm
            img = (_cm.get_cmap("viridis")(np.clip(sl, 0, 1))[..., :3] * 255).astype(np.uint8)
    else:
        img = (_overlay_rgb(grey, label, alpha) * 255).astype(np.uint8)
    fig = go.Figure(go.Image(source=_png_uri(img)))
    h, w = img.shape[:2]
    others = [a for a in "zyx" if a != axis]           # rows = first other axis, cols = second
    cp = chunk_planes(d); tp = tile_planes(d)
    for p in tp[others[0]]: fig.add_hline(y=p-0.5, line_dash="dot", line_color="yellow", line_width=0.6)
    for p in tp[others[1]]: fig.add_vline(x=p-0.5, line_dash="dot", line_color="yellow", line_width=0.6)
    for p in cp[others[0]]: fig.add_hline(y=p-0.5, line_color="orange", line_width=1.2)
    for p in cp[others[1]]: fig.add_vline(x=p-0.5, line_color="orange", line_width=1.2)
    m = load_json(d / "manifest.json"); gs = (m.get("notes") or {}).get("generation_stats") or {}
    title = (f"{d.parent.parent.name}/{d.name} — {axis}={idx} of {n}"
             + (" (CHUNK PLANE)" if idx in cp[axis] else "")
             + f" | shape {shape} | φ req {m.get('requested_global_phi')} deliv {gs.get('actual_label_porosity', float('nan')):.4f}"
             if gs else f"{d.parent.parent.name}/{d.name} — {axis}={idx} of {n} | shape {shape}")
    fig.update_layout(title=dict(text=title, font=dict(size=12)), width=width, height=int(width * h / w) + 80,
                      margin=dict(l=10, r=10, t=60, b=10), xaxis=dict(title=others[1], range=[0, w]), yaxis=dict(title=others[0], range=[h, 0], scaleanchor="x"))
    return fig

def all_case_options():
    opts = []
    for a in ASSESSMENT_ORDER + ["real_floor"]:
        for d in case_dirs(a):
            opts.append((f"{a} / {d.name}", str(d)))
    # volumes of the other campaigns on disk (stress geometries, the regenerated final set, the band trial arms)
    for label, root in (("stress", STRESS_ROOT / "volumes"), ("final", FINAL_ROOT), ("trial", TRIAL_ROOT)):
        if root.exists():
            for d in sorted(root.rglob("manifest.json")):
                if (d.parent / "label.tif").exists():
                    opts.append((f"{label} / {d.parent.relative_to(root)}", str(d.parent)))
    return opts

_OPTS = all_case_options()
if not _OPTS:
    unavailable("no volumes on disk", "eval_v4 generate / real-floor")
else:
    _default = next((v for k, v in _OPTS if k == "sampler / 192_ddim200_seed101"), _OPTS[0][1])
    w_case = W.Dropdown(options=_OPTS, value=_default, description="case", layout=W.Layout(width="70%"))
    w_axis = W.ToggleButtons(options=["z", "y", "x"], description="axis")
    w_idx = W.IntSlider(value=96, min=0, max=191, description="slice", continuous_update=False, layout=W.Layout(width="60%"))
    w_mode = W.Dropdown(options=["grey + label", "grey", "label", "pore probability"], description="view")
    w_alpha = W.FloatSlider(value=0.45, min=0, max=1, step=0.05, description="label α", continuous_update=False)
    b_chunk = W.Button(description="jump to chunk plane"); b_mid = W.Button(description="middle")
    out = W.Output()
    def _sync_range(*_):
        n = dict(zip("zyx", volume_shape(w_case.value)))[w_axis.value]
        w_idx.max = n - 1; w_idx.value = min(w_idx.value, n - 1)
    def _draw(*_):
        with out:
            out.clear_output(wait=True)
            slice_figure(w_case.value, w_axis.value, w_idx.value, w_mode.value, w_alpha.value).show()
    def _chunk(_):
        cp = chunk_planes(w_case.value)[w_axis.value]
        if cp: w_idx.value = min(cp, key=lambda p: abs(p - w_idx.value))
        else: print("this volume is a single chunk on this axis")
    def _mid(_): w_idx.value = middle_indices(w_case.value)[w_axis.value][0]
    w_case.observe(lambda c: (_sync_range(), _draw()), names="value"); w_axis.observe(lambda c: (_sync_range(), _draw()), names="value")
    for w in (w_idx, w_mode, w_alpha): w.observe(_draw, names="value")
    b_chunk.on_click(_chunk); b_mid.on_click(_mid)
    _sync_range(); show_widget(W.VBox([w_case, W.HBox([w_axis, w_mode, w_alpha]), W.HBox([w_idx, b_chunk, b_mid]), out]), _draw)
''')
    md("""
**Three orthogonal middle slices of one case, in one figure.** The same rendering the inspection pack uses:
top row grey, bottom row label, one column per axis, each slice on the chunk plane nearest the middle when the volume
has one. Change `CASE` to any `assessment/case` string from the dropdown above.
""")
    code(r'''
def three_view(d, mode="grey + label", width=1100):
    d = Path(d); mids = middle_indices(d)
    fig = make_subplots(rows=2, cols=3, subplot_titles=[f"{ax}={mids[ax][0]}{' (chunk plane)' if mids[ax][1] else ''}" for ax in "zyx"]*1 + ["label"]*3,
                        horizontal_spacing=0.03, vertical_spacing=0.06)
    for j, ax in enumerate("zyx", 1):
        idx = mids[ax][0]
        grey = read_slice(d, "grey", ax, idx); label = read_slice(d, "label", ax, idx)
        fig.add_trace(go.Image(source=_png_uri(np.stack([grey]*3, -1))), row=1, col=j)
        fig.add_trace(go.Image(source=_png_uri(np.array([[191,191,191],[216,27,27],[51,102,217]])[np.clip(label,0,2)])), row=2, col=j)
    fig.update_layout(title=f"{d.parent.parent.name}/{d.name}  shape {volume_shape(d)}", width=width, height=int(width*0.62), margin=dict(l=10,r=10,t=60,b=10))
    fig.update_yaxes(autorange="reversed")
    return fig

CASE = "sampler/192_ddim200_seed101"
_d = ROOT / CASE.split("/")[0] / "volumes" / CASE.split("/")[1]
if _d.exists(): three_view(_d).show()
else: unavailable(CASE, "eval_v4 generate sampler")
''')


def build_compare_viewer() -> None:
    section("Compare viewer", "two to four volumes side by side, same slice, with presets for every comparison the paper makes")
    md("""
**Why compare.** Most questions in this evaluation are differences: more denoising steps versus fewer, one seed versus
another, a generated volume versus a real crop of the same size, a flat surface request versus a rough one, and the
four ways of assembling chunks. The viewer shows the chosen cases at the *same* slice index and axis so your eye
compares like with like. For two cases of identical shape the *difference* view shows |grey A − grey B| and the map of
voxels whose label differs.

**Presets** fill the case list for the standard comparisons. Pick a preset, then move the slider.
Note that different seeds are different random draws: they should look alike in *statistics* (pore density, ply texture)
and not in detail. A real crop has scanner noise the model may render differently; look at pore shapes and ply texture
rather than at pixel noise.
""")
    code(r'''
def _find(assessment, pattern):
    return [str(d) for d in case_dirs(assessment) if re.search(pattern, d.name)]

def _real(tag="small", n=1):
    return [str(d) for d in case_dirs("real_floor") if d.name.endswith("__" + tag)][:n]

PRESETS = {
    "DDIM 50 / 100 / 200 at 192³ (seed 101)": _find("sampler", r"^192_ddim(50|100|200)_seed101$"),
    "DDIM 50 / 200 at 1024 wide (seed 101)": _find("sampler", r"^1024_ddim(50|200)_seed101$"),
    "three seeds, 192³ DDIM-200": _find("sampler", r"^192_ddim200_seed"),
    "generated 192³ vs real crop 128×192×192": _find("sampler", r"^192_ddim200_seed101$") + _real("small", 1),
    "generated 1024 vs real crop 1024": _find("sampler", r"^1024_ddim200_seed101$") + _real("large", 1),
    "flat vs rough surface request (192³)": _find("surface", r"^(flat|rough)_192_ddim50_seed101$"),
    "notch+hole, sphere 192, sphere 256": _find("geometry", r"^(notch_hole_seed101|sphere_192_ddim50_seed101|sphere_256_ddim50_seed101)$"),
    "multichunk box / sphere / rough (384)": _find("multichunk", r"^(box384_ddim50_seed101|sphere384_ddim50_seed101|rough384_ddim50_seed101)$"),
    "assembly modes at 384³: joint / autoregressive / hybrid": _find("assembly_modes", r"^(joint|autoregressive|hybrid)_384_seed101$"),
    "assembly modes at 1024: joint / autoregressive / hybrid / teacher-forced": _find("assembly_modes", r"^(joint|autoregressive|hybrid|teacher_forced)_1024_seed101$"),
    "porosity 0.005 / 0.03 / 0.10 / 0.15 (DDIM-200, seed 101)": _find("porosity_global", r"^target(0\.005|0\.03|0\.1|0\.15)_ddim200_seed101$"),
    "local field: halves / checkerboard / coherent": _find("porosity_local", r"^(halves|checkerboard|coherent)_ddim200_seed101$"),
    "guidance s_por 1 / 1.5 / 2 at target 0.05": _find("cfg", r"^spor(1|1\.5|2)_target0\.05_seed101$"),
    "neighbour arm on / off (s_nb 1 vs 0)": _find("cfg", r"^snb(0|1)_seed101$"),
    "assembly offsets 0 / 16 / 32": _find("assembly", r"^offset(0|16|32)_seed101$"),
    "layups A / C / B16": _find("layup", r"^(A|C|B16)_seed101$"),
}
if DECODER_FT_ROOT.exists():
    _re = sorted(DECODER_FT_ROOT.glob("**/volumes/192_ddim200_seed101"))
    if _re: PRESETS["original decoder vs fine-tuned decoder (192³ DDIM-200 seed 101)"] = _find("sampler", r"^192_ddim200_seed101$") + [str(_re[0])]
PRESETS = {k: v for k, v in PRESETS.items() if len(v) >= 2}

def compare_figure(dirs, axis="z", idx=96, mode="grey + label", diff=False, width=1200):
    dirs = [Path(d) for d in dirs]; n = len(dirs)
    cols = n + (1 if diff and n == 2 and volume_shape(dirs[0]) == volume_shape(dirs[1]) else 0)
    fig = make_subplots(rows=1, cols=cols, subplot_titles=[f"{d.parent.parent.name}/{d.name}" for d in dirs] + (["|A−B| grey (top) — label disagreement (red)"] if cols > n else []),
                        horizontal_spacing=0.02)
    imgs = []
    for j, d in enumerate(dirs, 1):
        nax = dict(zip("zyx", volume_shape(d)))[axis]; i = int(np.clip(idx, 0, nax - 1))
        grey = read_slice(d, "grey", axis, i); label = read_slice(d, "label", axis, i); imgs.append((grey, label))
        if mode == "grey": img = np.stack([grey]*3, -1)
        elif mode == "label": img = (np.array([[191,191,191],[216,27,27],[51,102,217]])[np.clip(label,0,2)]).astype(np.uint8)
        else: img = (_overlay_rgb(grey, label, 0.45) * 255).astype(np.uint8)
        fig.add_trace(go.Image(source=_png_uri(img)), row=1, col=j)
        for p in chunk_planes(d)[[a for a in "zyx" if a != axis][0]]: fig.add_hline(y=p-0.5, line_color="orange", line_width=1, row=1, col=j)
        for p in chunk_planes(d)[[a for a in "zyx" if a != axis][1]]: fig.add_vline(x=p-0.5, line_color="orange", line_width=1, row=1, col=j)
    if cols > n:
        (ga, la), (gb, lb) = imgs
        dg = np.abs(ga.astype(np.int16) - gb.astype(np.int16)).astype(np.uint8)
        img = np.stack([dg]*3, -1); img[la != lb] = [216, 27, 27]
        fig.add_trace(go.Image(source=_png_uri(img)), row=1, col=cols)
    h, w = imgs[0][0].shape
    fig.update_layout(width=width, height=int(width / cols * h / w) + 90, margin=dict(l=10, r=10, t=50, b=10), title=f"{axis} = {idx}")
    fig.update_yaxes(autorange="reversed")
    return fig

if not PRESETS:
    unavailable("no comparable pairs on disk", "eval_v4 generate")
else:
    c_preset = W.Dropdown(options=list(PRESETS), description="preset", layout=W.Layout(width="80%"))
    c_cases = W.SelectMultiple(options=_OPTS, rows=4, description="cases", layout=W.Layout(width="80%"))
    c_axis = W.ToggleButtons(options=["z", "y", "x"], description="axis")
    c_idx = W.IntSlider(value=96, min=0, max=1023, description="slice", continuous_update=False, layout=W.Layout(width="60%"))
    c_mode = W.Dropdown(options=["grey + label", "grey", "label"], description="view")
    c_diff = W.Checkbox(value=True, description="difference (2 same-shape cases)")
    c_out = W.Output()
    def _apply_preset(*_): c_cases.value = tuple(PRESETS[c_preset.value])
    def _cdraw(*_):
        with c_out:
            c_out.clear_output(wait=True)
            if len(c_cases.value) < 2: print("pick at least two cases"); return
            compare_figure(list(c_cases.value)[:4], c_axis.value, c_idx.value, c_mode.value, c_diff.value).show()
    c_preset.observe(lambda c: (_apply_preset(), _cdraw()), names="value")
    for w in (c_cases, c_axis, c_idx, c_mode, c_diff): w.observe(_cdraw, names="value")
    _apply_preset(); show_widget(W.VBox([c_preset, c_cases, W.HBox([c_axis, c_mode, c_diff]), c_idx, c_out]), _cdraw)
''')


# ===========================================================================
# 6. real floors, 7. sampler, 8. porosity global, 9. porosity local
# ===========================================================================

def build_primer() -> None:
    section("Primer: what this work is, for a reader with no background",
            "the problem, the data, the model, the sampler and the evaluation philosophy, at the level of the paper's introduction and methods")
    md("""
**The problem.** Carbon-fibre laminates (CFRP) are made of stacked layers ("plies") of parallel fibres in resin, each ply
oriented at an angle (0°, 45°, 90°, 135° …). During manufacturing, small gas pockets stay trapped: **pores**. Porosity (the
pore volume fraction) degrades strength, and industry inspects parts with **X-ray computed tomography (CT)**: a 3D grey
image where material is bright, pores and the air around the part are dark. Our scans are at 25 µm per voxel; a coupon
5 mm thick is about 192 voxels deep. Training a pore detector, or studying how porosity affects a part, needs many scans
with known pores. Real scans are expensive, few, and come with **no ground truth**: the pore label is itself the output of
a thresholding algorithm (Sauvola, local adaptive threshold) whose uncertainty we measure in campaign 13.

**What PoreGen does.** It generates synthetic CT volumes of laminates **together with a voxel-aligned label** (material /
pore / air), at any size, with the porosity, the local porosity field, the ply stacking sequence and the part geometry
chosen by the user. The label comes from the same model as the grey image, so a generated volume is a complete training
example, not an image that still needs labelling.
""")
    md("""
**The data (split_v3).** 80 CT volumes from 12 laminate panels, three manufacturing families, two ply stacking sequences.
Each volume is cut into overlapping 64³-voxel windows (1.6 million windows at stride 32, about 220 k without overlap).
The split is by **panel**, not by volume, so a test panel is never seen in training: test = three panels, validation =
three, train = the rest. Registration holes drilled in the coupons were removed (dilated by 32 voxels) because a model
trained with them learned to invent air inside material. Every window carries its porosity, the ply angle at each depth,
its depth in the stack, the distance to the six faces of the coupon, and a material map (what is inside the part).

**The model, in two stages.**
1. **Compressor (VAE r08).** A convolutional autoencoder squeezes a 64³ grey+label window into a latent block of 16³ cells
   with 8 numbers each (8× fewer numbers than the grey voxels). Its decoder outputs the grey image and a 3-class
   probability per voxel (material / pore / air). Campaign 09 chose 8 channels: pore Dice saturates between 8× and 4×
   reduction. The latent is regularised to be roughly Gaussian so the second stage can model it.
2. **Diffusion model (ldm06).** A 3D U-Net (83 M parameters) learns to turn Gaussian noise into latent blocks, step by step
   ("denoising"), *conditioned* on what we ask for: porosity (a number), the ply-angle profile per depth plane, the depth,
   the six face distances, the material map, and the latents of the **six neighbouring windows** so that adjacent windows
   agree. It predicts "v" (a mix of noise and signal) on a cosine noise schedule; sampling uses DDIM with 50–200 steps.
""")
    md("""
**The sampler: how a window becomes a part.** The model only knows 64³ windows. To make a 1024×1024×192 plate:
- **Windows** of 64 voxels are placed every 32 voxels (so each voxel is inside up to 8 windows) and denoised *together*:
  at every step each window predicts its content, and the predictions are averaged with cosine weights (MultiDiffusion).
  This is the **joint** mode; it keeps neighbouring windows consistent but needs the whole region in memory.
- **Chunks** of 192³ (3×3×3 windows) are solved one after another in raster order. A finished chunk is re-noised to the
  current noise level and fed to the next chunk as its neighbours (RePaint rule), so the new chunk continues the old one.
  This is the **hybrid** mode the paper proposes: joint inside a chunk, sequential between chunks, unbounded size.
- **Decoding** runs the VAE decoder on overlapping 64³ tiles with a Tukey window, because tile-by-tile decoding leaves
  visible seams in the label (campaign 08).
- **Classifier-free guidance**: the model is trained sometimes without the porosity input and sometimes without the
  neighbours, so at sampling time either can be turned off (the "null" condition) or amplified (s_por, s_nb > 1).

**The evaluation philosophy (eval v4).** Three rules: every generated volume carries a manifest that records exactly
what was asked and with which code; every metric declares what it needs and refuses a volume that lacks it; and every
table has a **real floor** row: the same metric computed on real held-out volumes, because a seam measure or a texture
statistic is only meaningful against what real material scores. We never claim realism from appearance. We ask: does the
delivered porosity follow the requested one (dose response)? do pores land where the field asked (local obedience)? can
the requested ply sequence be read back (two independent angle readers with known floors)? are window and chunk joins
invisible (seam ratios vs real)? does the model carve air where the material map says (geometry Dice)? are the surfaces
as rough as real ones? are the pore statistics (two-point correlation S2, pore-size distribution, Ripley's K, slice FID)
within the real-vs-real floor? does it copy training windows (nearest neighbour over the full train store)? and, the
test of use: does a segmentation network trained on synthetic data work on real scans (downstream utility)?
""")
    md("""
**The competitor and the claims.** The closest prior work generates porous rock with a latent diffusion model conditioned
on porosity (Naiff et al., *Computers & Geosciences* 2026; a field-controlled follow-up at 1024³ is a 2026 preprint). It
emits a binary phase, not grey + multi-class label, and does not condition on part geometry or through-thickness ply
structure. PoreGen's contributions, as framed for the paper: joint grey+label generation of a layered, non-stationary
material; explicit structural and geometric conditioning; realism measured by statistics and by use against real floors;
local porosity and unbounded assembly as measured requirements; and the small-dataset practices reported as findings
(panel split, hole removal, latent-spread gate, external code audit, retraction of earlier over-claims, and the
chunk-plane band with its root cause).

**How to read the rest of this notebook.** Sections 6 onwards are one per assessment: each says what it asks, how the
number is computed, what the floor is, and what good and bad look like, then shows the data. The research log (next)
tells what happened in which order. The chunk-band section is the one finding you must know before quoting any
chunk-plane number.
""")


def build_stress_geometry() -> None:
    section("Stress geometries (campaign 19): tubes, brackets, tapers, gradients, a cube, a gyroid, letters",
            "nine geometries far from the training data, each at DDIM 50 and 200: does porosity, ply placement, air and surface control survive?")
    md("""
**Why this campaign exists.** Everything in campaigns 12 and 18 is a flat plate or a simple cut (notch, hole, sphere).
Real parts are tubes, brackets and tapered skins. The model has never seen any of them: every training window comes from
a flat 5 mm coupon. This campaign asks the model for nine shapes it cannot have memorised and measures the same things
as the paper's tables: delivered porosity, where the pores are, whether the ply sequence can be read back, whether air
is carved exactly where the material map says, how the surfaces sit, and whether the chunk joins show. It is
**exploratory and off the gates**: a failure here is a finding about the limits of the conditioning, not a bug.

**What each geometry tests.** *Tube* and *hollow sphere*: curved surfaces on both sides of a thin wall, interior air the
data never has. *L-bracket*: a corner — the ply conditioning is per depth plane, so the plies cannot bend around it;
expect flat plies cutting the corner. *Tapered plate*: thickness changing along the part, stressing the depth and face
distance inputs. *Gradient + hot spots*: local porosity control at part scale. *Cube 1024³*: five times the volume of
anything else, chunk joins on all three axes, eight chunks deep. *Gyroid*: surfaces everywhere, no interior far from a
face. *Two coupons*: independence of two parts with clean air between. *Letters*: controllability, for the figure.

**Why DDIM 50 and 200.** Fifty steps is the production setting; two hundred is the slow, careful one. Where they differ,
the difference is the sampler's, not the model's. Each geometry is generated once at each.
""")
    code(r'''
SR = STRESS_ROOT / "results.json"
if not SR.exists():
    unavailable("campaign 19 results.json", "stress_geometry generation (after the regeneration) and measure")
else:
    D = load_json(SR); rows = []
    for c in D.get("per_case", []):
        pf = c.get("phase_fractions") or c; ga = c.get("geometry_agreement") or {}; sm = c.get("seams") or {}; band = c.get("band") or c.get("chunk_band") or {}
        rows.append({"case": c.get("case"), "geometry": c.get("geometry") or (c.get("notes") or {}).get("geometry"), "ddim": c.get("ddim_steps"),
                     "shape": "×".join(map(str, c.get("volume_shape") or [])), "φ requested": c.get("requested_global_phi"), "φ delivered": pf.get("phi_pore"),
                     "air inside material": pf.get("air_fraction_interior", pf.get("air_fraction")), "air Dice": ga.get("dice"), "air precision": ga.get("precision"), "air recall": ga.get("recall"),
                     "seam window grey": sm.get("seam_xct_ratio"), "seam chunk grey": sm.get("seam_chunk_xct_ratio"), "seam chunk pore": sm.get("seam_chunk_pore_ratio"),
                     "band −8": band.get("ratio_-8"), "band +0": band.get("ratio_+0"), "failed": (c.get("failure") or {}).get("failed"), "min": (c.get("wall_time_s") or float("nan")) / 60})
    ST = pd.DataFrame(rows).sort_values(["geometry", "ddim"])
    display(ST.round(4))
    fig = make_subplots(rows=1, cols=3, subplot_titles=["delivered φ vs requested", "air Dice vs requested geometry", "chunk-plane band (−8), 1 = none"])
    for dd, g in ST.groupby("ddim"):
        fig.add_trace(go.Bar(x=g["geometry"], y=g["φ delivered"], name=f"DDIM {dd}"), row=1, col=1)
        fig.add_trace(go.Bar(x=g["geometry"], y=g["air Dice"], name=f"DDIM {dd}", showlegend=False), row=1, col=2)
        fig.add_trace(go.Bar(x=g["geometry"], y=g["band −8"], name=f"DDIM {dd}", showlegend=False), row=1, col=3)
    if ST["φ requested"].notna().any(): fig.add_hline(y=float(ST["φ requested"].dropna().iloc[0]), line_dash="dash", row=1, col=1)
    fig.add_hrect(y0=0.8, y1=1.2, fillcolor="green", opacity=0.1, line_width=0, row=1, col=3)
    fig.update_layout(height=420, barmode="group"); fig.show()
''')
    md("""
**Reading the first table.** φ delivered should sit at the requested 0.03 in every shape; a shape where it drifts tells you
the conditioning (depth, face distances) is being misread. Air Dice near 1 with air-inside-material near 0 means the
material map is obeyed; watch the thin-wall cases (tube, gyroid, shell). The band column shows whether the chunk-plane
fix holds on shapes where many windows touch a surface *and* a chunk frontier at once.
""")
    code(r'''
if SR.exists():
    D = load_json(SR); rows = []
    for c in D.get("per_case", []):
        L = c.get("layup") or c.get("layup_recovery") or {}
        for reader, r in L.items():
            if isinstance(r, dict) and ("median_abs_error_deg" in r or "median_abs_error" in r):
                rows.append({"case": c.get("case"), "ddim": c.get("ddim_steps"), "region": r.get("region", "whole"), "reader": reader,
                             "median |err| deg": r.get("median_abs_error_deg", r.get("median_abs_error")), "4-class acc": r.get("strict_class_accuracy"), "plies recovered": r.get("n_recovered")})
        S = c.get("surface") or c.get("surface_agreement") or {}
        for face, f in (S.items() if isinstance(S, dict) else []):
            if isinstance(f, dict):
                rows.append({"case": c.get("case"), "ddim": c.get("ddim_steps"), "region": face, "reader": "surface", "position error vox": f.get("error_abs_mean", f.get("radial_error_vox")), "Sa vox": f.get("roughness_sa")})
    LS = pd.DataFrame(rows)
    if LS.empty: note("no layup / surface rows yet in results.json")
    else:
        display(LS.round(3))
        note("Layup floors on real volumes: fft_slice 8.2° / 74 %, pore_axes 4.2° / 86 %. On the L-bracket each leg is read separately; the corner is expected to fail because plies are conditioned per depth plane.")
''')
    md("""
**Reading the second table.** Ply recovery should match the flat-plate numbers of section *layup* on the plates and the
bracket legs. The corner of the bracket is the known limit: a per-depth-plane ply conditioning cannot bend. Surface
rows: radial error of the tube and the shell, per-column thickness error of the taper, and position error of the plate
faces, all in voxels (25 µm).
""")
    code(r'''
if SR.exists():
    D = load_json(SR)
    grad = [c for c in D.get("per_case", []) if "gradient" in str(c.get("case", "")) or "gradient" in str(c.get("geometry", ""))]
    if not grad: note("gradient + hot-spot case not measured yet")
    for c in grad:
        lo = c.get("local") or c.get("local_obedience") or {}
        print(c.get("case"), "DDIM", c.get("ddim_steps"), "| within-volume slope", lo.get("slope"), "R²", lo.get("r2"), "| per-cell |err|", lo.get("cell_abs_error_mean"), "| ramp slope delivered/requested", lo.get("ramp_slope_ratio"))
        req, dlv = lo.get("requested_cells"), lo.get("delivered_cells")
        if req and dlv:
            fig = make_subplots(rows=1, cols=2, subplot_titles=["requested field (mid-depth tiles)", "delivered field"])
            fig.add_trace(go.Heatmap(z=np.asarray(req), colorscale="Viridis", zmin=0, zmax=0.1), row=1, col=1)
            fig.add_trace(go.Heatmap(z=np.asarray(dlv), colorscale="Viridis", zmin=0, zmax=0.1), row=1, col=2)
            fig.update_layout(height=380, title=f"{c.get('case')}: gradient + hot spots, DDIM {c.get('ddim_steps')}"); fig.show()
''')
    md("""
**Reading the gradient maps.** Left is the porosity asked per 64-voxel tile, right what was delivered. A good result keeps
the ramp direction, reaches the 10 % spots, and stays at 0.5 % in the low corner. Recall from section *porosity local*
that tile-scale contrast is halved by the overlapping windows; a smooth ramp is the regime where local control works.
""")
    code(r'''
# Inspection montage for campaign 19 (three mid-slices per case, grey + label) if the reporter wrote it
INS = STRESS_ROOT / "inspection"
if INS.exists():
    pngs = sorted(INS.glob("*.png"))
    print(len(pngs), "montages")
    w_p = W.Dropdown(options=[(p.name, str(p)) for p in pngs], description="case")
    out_p = W.Output()
    def _showp(*_):
        with out_p:
            out_p.clear_output(wait=True); display(Image(filename=w_p.value))
    w_p.observe(_showp, names="value")
    show_widget(W.VBox([w_p, out_p]), _showp)
else:
    unavailable("campaign 19 inspection montages", "eval_v4 report stress_geometry")
''')
    md("""
**What to look for in the montages.** Crisp air/material boundaries on curved walls; plies visible as layers in the grey;
flat plies cutting through the bracket corner (the known limitation); clean air in the gap between the two coupons; the
letters readable in the label. The full volumes are in `campaigns/19-stress-geometry/volumes/<case>/` and open in the
slice viewer (section 5) like any other case.
""")


def build_research_log() -> None:
    section("Research log since the ldm06 training finished",
            "a dated account of everything that happened after step 130k: results, defects found, decisions taken and pending, queue state")
    md("""
**Purpose of this section.** If you read nothing else, read this. It is the narrative the numbers below belong to, written
by the supervising session and updated with the builder (date of this text: **2026-09-12 09:00**). Each entry names the
section of this notebook or the vault note where the evidence lives. Dates are local machine time.

**Where we were when training ended.** ldm06 is the conditional 3D latent diffusion model of this paper: VAE r08 (8 latent
channels, 4× spatial compression, 3-class head material/pore/air) + a 83 M-parameter UNet trained 130k steps on the
split_v3 store (1.6 M overlapping 64³ windows, holes removed, panel-level split) with v-prediction, noised-neighbour
conditioning and a hybrid chunked sampler (windows of 64 voxels at stride 32, fused every step; chunks of 192³ solved in
order; finished chunks fed back as neighbours). Training ended **2026-09-11 03:45** at step 130 000.
""")
    md("""
**Timeline.**

| when (2026) | what | where |
|---|---|---|
| 09-11 03:45 | ldm06 reaches 130k. Exit gates on the final weights: porosity error 0.0010–0.0011, latent spread 0.97–0.98 of the sampled reference, zero interior air, surface position ±0.03 vox, kill-switch 0.0049/0.0500. EMA = raw. | section *training-time diagnostics*; vault *LDM Experiments → ldm06* |
| 09-11 03:50 → 20:35 | Eval v4 generation (217 volumes, 13 assessments), measure, report, two inspection packs. **All on `best.ckpt` = step 119k**, hardcoded in the runner; discovered 23:50. | sections 6–19; caveat in *chunk-plane band* |
| 09-11 12:45 | Label uncertainty (campaign 13): Sauvola k ±20 % moves real porosity by 0.014–0.016 (3× the gate, 14× the model error); Yen excluded as a different segmentation; between-k pore Dice min 0.28 / median 0.56. | section *label uncertainty*; vault E9 |
| 09-11 12:42 / 20:35 | Inspection packs built (13 cases; then with layup and assembly-mode cases). Author's visual inspection = the gate for the decoder fine-tune; **not yet given**. | `12-eval-v4/inspection/` |
| 09-11 14:3x | Memorisation smoke test killed twice: memory cap counted page cache; then the machine killed it for low memory while a CUDA job ran. Rule adopted: on this GB10 unified-memory machine, nothing streams the 195 GiB store beside a GPU job; readers drop pages. Passed in its own slot (3.5 min). | `docs/DEVELOPMENT.md`; vault LDM Experiments incident 6 |
| 09-11 18:5x | Branch `refactor` fast-forwarded into `main` and deleted. `main` is the only branch. | git |
| 09-11 19:00–23:30 | This notebook built (builder committed), verified headless three times. | `scripts/analysis/build_eval_v4_notebook.py` |
| 09-11 21:40 → 09-12 01:07 | VAE rung reports (base, rf-8, rf-32, rf-4) and the compare table. rf-2 and rf-64 still owed. | section *VAE compressor rungs* |
| 09-11 23:50 | **Defect found:** pore-logit seam ratio at chunk planes 0.39 and cross-plane pore Dice 0.27 trace to one thing — a pore-depleted band in the 32 voxels on each side of every chunk plane (φ at −8 ≈ 0.2 of the mean; real crops flat). Explains the hybrid arm's φ deficit of 0.005. | section *chunk-plane band* |
| 09-12 00:30–02:00 | Four-arm and s_nb diagnosis: the band is caused by the neighbour conditioning (s_nb 0 removes it); not re-noising, not edge fusion, not drift. Latent spread in the band normal → content, not blur. | same section, arms figure |
| 09-12 02:40 | Single-window rim test negative: a window beside six consistent neighbours dips ≤ 16 % in 8 voxels. | section *single-window rim tests* |
| 09-12 03:30 | Single-window test with 3 present + 3 missing faces: **pores move away from the missing face** (0.05–0.15 of the window mean), present faces untouched; inheritance from a pore-poor neighbour 1.7–2.5×. | same |
| 09-12 04:00 | **Root cause in the training code:** neighbour dropout dropped all six faces per sample; 64 % of training windows have one missing face and it is always the specimen surface; "present + unknown" never occurred. Learned rule: nothing behind a face = surface = no pores. The one healthy plane per volume is the terminal one (next chunk's far face = volume edge = trained case). | same; `12-eval-v4/README.md` |
| 09-12 04:00–08:49 | Fix trial (campaign 17) on 130k, scored on non-terminal planes (trailing / leading strip, worst plane, cost): overlap+blend (a) 0.31 fail; s_nb 0.5 (b) 0.51 fail; both (c) 0.56 fail at 3.8×; drop-neighbours-when-mixed (f) 0.85 / 0.75 fail at 1×; **(g) = (f)+overlap 0.82 / 1.00, worst 0.74, PASS on the mean at 1.36×**; **(a2s) overlap 64 with 32 pinned, successor writes the strip: 0.95 / 0.91, worst 0.84, PASS on every plane at 2.05×**; (a2b) blend ≈ a2s. Pinned-only (e) 0.19 = the steering control. | section *chunk-plane band*, trial table |
| 09-12 09:00 | GPU idle after the trial. Launched the training-side fix `ldm06/facedrop` (per-face neighbour dropout, warm start from 130k, 15k steps ≈ 7 h). After it: production sampler and a2s re-tested on the new weights + the mixed-set rim test as mechanism check. Campaign 12 is NOT regenerated yet; that choice (a2s on 130k vs production sampler on facedrop) is the author's. | `configs/experiments/ldm06/facedrop.yaml`; campaign 17 README |
""")
    md("""
**Decisions taken in this period** (supervisor, within the author's standing instructions): move layup to the end so
the inspection pack arrived at 12:30 instead of 19:50; label uncertainty reports only the perturbation of our own method;
the memorisation check searches the full store and the multi-chunk volumes; `refactor` merged into `main`; campaign 12
stays on 119k and is documented as such, the next generation is on 130k; band criterion = mean over non-terminal planes
plus worst plane; downstream utility runs last so it trains on the regenerated set.

**Decisions that belong to the author and are open:**
1. Visual inspection of the pack → go / no-go for the decoder fine-tune (D43). Recommendation: judge it on the regenerated set.
2. Which band fix: sampler rule (f) or (g) today, or the per-face fine-tune (mechanism-clean, ≈ 7 h), or both.
3. Regenerate campaign 12 on 130k with the chosen fix (≈ 12 h generation + measure). Recommended.
4. Whether the "hybrid unifies joint and autoregressive" claim is withdrawn: on seams hybrid = joint = teacher-forced at the real floor; it beats autoregressive; it does not beat joint. Framing v2 says withdraw; replace with "unbounded size at joint quality, no drift".
5. Train longer (vault L2): parked; the exit gates did not move between 74k and 130k.

**What the eval says for the paper, in one paragraph.** Global porosity control (slope 1.07, R² 0.999, 90 % in gate) and
layup control (1.3–2.2° for the trained sequences, beyond the reader floors) hold. Geometry (Dice 0.992), sphere (1 vox),
rough-surface roughness (0.965 of request) hold. Grey seams at window and chunk planes are at the real floor. No
memorisation (4,199 patches vs the full train store, zero copies). Failure rate zero. Microstructure statistics are 1.5–6×
the real-vs-real floor, worst at low porosity. Local control follows large patterns (slope 0.94) but halves tile-scale
contrast (0.49). The pore band at chunk planes is a documented defect with a known cause and a fix under test; all
chunk-plane pore numbers above must be read with it.
""")
    code(r'''
# Live state of the queue and decisions, read from disk (no narrative here)
state = {
    "decoder-ft gate file present": (CAMPAIGNS / "decoder_ft_go").exists(),
    "decoder-ft campaign (11) exists": DECODER_FT_ROOT.exists(),
    "downstream campaign (14) exists": DOWNSTREAM_ROOT.exists(),
    "band trial report (17)": TRIAL_ROOT.joinpath("trial_report.json").exists(),
    "rim tests (16)": RIM_ROOT.exists(),
    "VAE rung compare (09) final table": RUNGS_ROOT.joinpath("decision_table.json").exists(),
    "campaign-12 checkpoint step (from one manifest)": (load_json(next(ROOT.glob("sampler/volumes/*/manifest.json"))).get("checkpoint_step") if list(ROOT.glob("sampler/volumes/*/manifest.json")) else None),
}
display(pd.Series(state, name="state").to_frame())
if TRIAL_ROOT.joinpath("trial_report.json").exists():
    print("trial arms present:", ", ".join(load_json(TRIAL_ROOT / "trial_report.json").keys()))
''')


def build_rungs() -> None:
    section("VAE compressor rungs (campaign 09)",
            "the six-rung reduction-factor table behind the choice of 8 latent channels: Dice, porosity MAE per bin, dense panels, seams")
    md("""
**What this is.** Before ldm06, the compressor (VAE r08) was trained at several *reduction factors*: how many numbers the
latent keeps per 4×4×4 block of voxels. With `z` channels at 4× spatial compression, the reduction is 64/z: z=16 → 4×,
z=8 → 8×, z=4 → 16×, z=2 → 32×. rf-2 (z=32) and rf-64 (z=1) are still owed. Every rung is the same architecture, the
same data, the same loss; only z changes. **Why it matters:** the diffusion model lives in this latent; whatever the
compressor cannot reconstruct, the generator cannot produce. The paper's claim is that pore Dice is monotone in latent
width and saturates between 8× and 4×, which is why 8× (z=8) was chosen.

**How to read the table.** *val φ MAE* = mean absolute error of the porosity of a reconstructed window vs the labelled
one. *pore Dice* = overlap of reconstructed vs labelled pore voxels (1 = perfect). *DENSE pore Dice* = the same on the
≥ 6 % porosity panels only, the hard case. *tau* = the calibrated pore threshold on validation. *seam* ratios are the
tile-decode discontinuity (near 1 = invisible). *drift 2σ* = how far the latent statistics move over training.
""")
    code(r'''
F = RUNGS_ROOT / "findings.md"; DT = RUNGS_ROOT / "decision_table.json"
if not F.exists():
    unavailable("campaign 09 findings.md", "eval_v4 owed rung reports + COMPARE (ran 2026-09-12 01:07)")
else:
    display(Markdown(F.read_text()))
    if DT.exists():
        R = load_json(DT)["rungs"]
        T = pd.DataFrame([{"rung": r["experiment"], "z": r["z_channels"], "reduction": 64 // r["z_channels"], "step": r["step"], "wall_h": r["wall_h"],
                           "pore Dice val": (r.get("val_full") or {}).get("dice_pore") or (r.get("val_full") or {}).get("class_dice_1"),
                           "pore Dice test": (r.get("test_full") or {}).get("dice_pore") or (r.get("test_full") or {}).get("class_dice_1"),
                           "phi MAE val": (r.get("val_full") or {}).get("phi_mae"), "stopped": r.get("stopped")} for r in R]).sort_values("reduction")
        display(T.round(4))
        fig = go.Figure()
        for col in ("pore Dice val", "pore Dice test"):
            if T[col].notna().any(): fig.add_trace(go.Scatter(x=T["reduction"], y=T[col], mode="lines+markers", name=col, text=T["rung"]))
        fig.update_layout(title="Pore Dice vs reduction factor (log x) — the saturation argument for z=8", xaxis_type="log", xaxis_title="reduction factor (64 / z)", yaxis_title="Dice", height=380); fig.show()
''')
    md("""
**What to look for.** Dice should fall as the reduction grows; the knee between 4× and 8× is the argument for 8×. If rf-4
is not better than rf-8 by more than the seed-to-seed spread, 8× is the right choice (half the latent, same quality).
Compare the *DENSE* column: that is where compressors fail first.
""")


def build_rim_tests() -> None:
    section("Single-window rim tests (campaign 16)",
            "one window denoised alone, with different neighbour sets: where do the pores go relative to each face?")
    md("""
**What this is.** The chunk band (previous section) needed a test at the smallest scale: one 64-voxel window denoised on
its own (no fusion, no chunks), with its six neighbours fed in different ways, then the porosity measured in 8-voxel
shells by distance from each face, divided by the window's own mean. 64 interior windows, DDIM-50, the 130k weights.

Three runs: **real** = six real neighbours from a validation volume (consistent set); **generated** = six neighbours taken
from a generated volume; **mixed** = the natural canvas set of a raster order, three faces present and three missing.
Each run has arms: `reference_decoded` (the real window through the VAE, no diffusion: the flat control), `s_nb=1`
(neighbour guidance on) and `s_nb=0` (off). A shell value of 1.0 means no effect; below 1 fewer pores near that face.
""")
    code(r'''
if not RIM_ROOT.exists():
    unavailable("campaign 16-window-rim", "scripts/analysis/window_rim_test.py (ran 2026-09-12 02:40 and 03:30)")
else:
    rows = []
    for run in ("real", "generated", "mixed"):
        f = RIM_ROOT / run / "results.json"
        if not f.exists(): continue
        d = load_json(f)
        for arm, A in d["arms"].items():
            mean = A.get("phi_interior_windows") or float("nan")
            for shell, v in (A.get("phi_by_shell") or {}).items():
                rows.append({"run": run, "arm": arm, "face": "all", "shell_vox": int(shell), "phi": v, "ratio": v / mean if mean else float("nan")})
            for key in ("phi_by_face_shell", "by_face", "faces"):          # per-face block, name differs between versions
                if isinstance(A.get(key), dict):
                    for face, sh in A[key].items():
                        shells = sh.get("phi_by_shell", sh) if isinstance(sh, dict) else {}
                        for shell, v in shells.items():
                            try: rows.append({"run": run, "arm": arm, "face": face, "shell_vox": int(shell), "phi": v, "ratio": v / mean if mean else float("nan")})
                            except Exception: pass
    RIM = pd.DataFrame(rows)
    if RIM.empty:
        unavailable("rim results", "results.json with arms/phi_by_shell")
    else:
        display(RIM[RIM.face == "all"].pivot_table(index=["run", "arm"], columns="shell_vox", values="ratio").round(3))
        fig = make_subplots(rows=1, cols=3, subplot_titles=["real neighbours", "generated neighbours", "mixed (3 present + 3 missing)"])
        for i, run in enumerate(("real", "generated", "mixed"), 1):
            for arm, g in RIM[(RIM.run == run) & (RIM.face == "all")].groupby("arm"):
                fig.add_trace(go.Scatter(x=g["shell_vox"], y=g["ratio"], mode="lines+markers", name=f"{run}: {arm}"), row=1, col=i)
            fig.add_hline(y=1.0, line_dash="dash", line_color="grey", row=1, col=i)
        fig.update_layout(height=380, title="φ(shell)/φ(window) by distance from the face (all faces pooled)"); fig.show()
        PF = RIM[RIM.face != "all"]
        if not PF.empty:
            display(PF[PF.shell_vox == 0].pivot_table(index=["run", "arm"], columns="face", values="ratio").round(3))
            note("Per-face row at shell 0: in the mixed run the UNKNOWN (+) faces are the depleted ones; the EXISTS (−) faces are not. The z row is not interpretable (a single depth).")
''')
    md("""
**What to look for.** In the *real* run the dip is small (≤ 16 % in the first shell) and barely changes with s_nb: a single
window beside consistent neighbours is fine. In the *mixed* run the per-face table shows the mechanism: faces with
nothing behind them lose almost all their pores (0.05–0.15), faces with a neighbour do not. This is what happens at every
chunk frontier, and it is why the fix is either "do not show the model a mixed set" (drop the neighbour arm for those
windows) or "teach it mixed sets" (per-face dropout in training).
""")


def build_real_floor() -> None:
    section("Real floors", "what every metric scores on real held-out scans, by volume shape")
    md("""
**Why a floor.** Suppose a metric says "the seam ratio is 0.97". Is that good? Only if we know what the same
metric says on a real scan, where there is no seam at all. So `eval_v4 real-floor` cuts crops from the held-out
*test* panels at the same shapes the generator produces, and runs every request-free metric on them. Those numbers are
the floors. A generated value at the floor is as good as the measurement can tell.

**Shapes.** *small* is 128×192×192 (a 192-deep clean box does not exist in a real laminate, so depth is reduced one
64-voxel tile at a time, never below 128), *large* is 128×1024×1024, and *micro_φ* are pairs of 128³ crops from one
panel matched to a target porosity, used by the microstructure section. Porosity differs between shapes (large crops
hold more of the porous regions), so a generated volume must always be compared with the floor of its own shape.
""")
    code(r'''
RF = results("real_floor")
if RF is None:
    unavailable("real_floor/results.json", "eval_v4 real-floor")
else:
    keys = ["phi_pore", "air_fraction", "air_fraction_interior", "seam_xct_ratio", "seam_chunk_xct_ratio", "cell_phi_sd", "cross_head_disagreement", "cross_head_disagreement_interior"]
    tbl = cells_table(RF["by_shape"], keys)
    tbl.insert(1, "n", [RF["by_shape"][k]["n_volumes"] for k in RF["by_shape"]])
    tbl.insert(2, "shape", [RF["by_shape"][k]["volume_shape"] for k in RF["by_shape"]])
    display(tbl)
    print("air detector:", RF.get("detector"))
''')
    md("""
**What each floor column means.**
- `phi_pore` — pore / (pore + material). The natural porosity of real test material at that crop size.
- `air_fraction_interior` — air voxels more than 32 voxels from every face. Exactly 0 on every real crop, so
  *any* interior air in a generated volume is the model's, not the detector's.
- `seam_xct_ratio` — grey-level jump across the 64-voxel tile planes divided by the same jump measured on planes in
  between (details in the assembly section). Real material gives about 0.97: the reading of "no seam" is not 1.0.
- `seam_chunk_xct_ratio` — same, at the 192-voxel chunk planes. Undefined (n = 0) on 192-deep shapes, which hold one chunk.
- `cell_phi_sd` — spread of porosity between 64³ tiles. How patchy real material is.
- `cross_head_disagreement` — voxels the grey head renders dark like air while the label head calls material.

The per-volume view below shows how the floors vary between real specimens: the spread is part of the floor.
""")
    code(r'''
if RF is not None:
    pc = pd.json_normalize(RF["per_case"], sep=".")
    pc["shape_tag"] = pc["notes.shape_tag"] if "notes.shape_tag" in pc else pc["volume_shape"].astype(str)
    pc["volume"] = pc["case"].str.replace("MedidasDB__Fabricacion_Nacho_05_Probetas_Nacho_2025_probetas_", "").str.replace("_volume_eq_aligned", "")
    fig = make_subplots(rows=1, cols=3, subplot_titles=["porosity φ (pore/material)", "tile-plane seam ratio (grey)", "cross-head disagreement"])
    for j, col in enumerate(["phi_pore", "seams.seam_xct_ratio", "cross_head.disagreement_fraction"], 1):
        if col in pc:
            for tag, g in pc.groupby("shape_tag"):
                fig.add_trace(go.Box(y=g[col], name=str(tag), boxpoints="all", text=g["volume"], hovertemplate="%{text}<br>%{y:.4f}", showlegend=(j == 1)), row=1, col=j)
    fig.update_layout(height=380, title="Real-floor spread across test specimens")
    fig.show()
''')
    subsection("Surface floor and layup reader floors")
    md("""
**Surface floor.** Real coupons have a top and a bottom face. For each real volume the height of the first material
voxel in every column gives a height map; its roughness *Sa* (mean absolute deviation from the mean plane) and *Sq*
(root mean square) are the floor for the surface section. *Detrended Sa* removes the tilt of the coupon first.
The *correlation length* says over how many voxels the height stays correlated, i.e. how coarse the roughness is.

**Layup reader floors.** The ply orientation is *read back* from a volume by two readers: `fft_slice` looks at the
grey texture of each in-plane slice (Fourier transform, dominant direction), and `pore_axes` looks at the elongation
direction of the pores in the label. Both were run on real volumes where the true layup is known: their median error
and how often they get the 4-class angle (0/45/−45/90) right are the best any generated volume can score.
""")
    code(r'''
sf = ROOT / "real_floor" / "surface_floor.json"
if sf.exists():
    S = load_json(sf)
    display(pd.DataFrame([{k: S.get(k) for k in ["n_volumes", "n_faces", "sa_mean", "sa_sd", "sq_mean", "sq_sd", "detrended_sa_mean", "correlation_length_vox_mean", "dark_but_material_all_material", "dark_but_material_excluding_face_rim"]}]).T.rename(columns={0: "value"}))
    pv = S.get("per_volume") or {}
    if pv:
        rows = [{"volume": k[-40:], "face": f, "Sa": v[f]["sa"], "Sq": v[f]["sq"], "detrended Sa": v[f]["detrended"]["sa"], "corr len": v[f]["detrended"]["correlation_length_vox"]["mean"]} for k, v in pv.items() for f in ("lower", "upper") if f in v]
        display(pd.DataFrame(rows))
else:
    unavailable("real_floor/surface_floor.json", "eval_v4 real-floor --shapes surface")
if RF is not None and RF.get("layup_floor"):
    display(pd.DataFrame(RF["layup_floor"]).T)
''')


def build_sampler() -> None:
    section("Sampler: DDIM step count and scale", "porosity error, seams and cost at 50/100/200 steps, 192³ and 1024 wide")
    md("""
**What DDIM steps are.** The model removes noise in rounds. With 50 rounds each round takes a bigger jump than with 200.
Fewer rounds are cheaper (time grows linearly with rounds) but the result can be less accurate: at 40k training steps
the porosity error at 200 rounds was five times the error at 50, and at 74k that gap was gone. This assessment
asks, on the final checkpoint, what 100 and 200 rounds buy over 50, at the production size 192³ and at a full panel
width 1024×1024×192, with three seeds each.

**Numbers shown here come from the sampler's own bookkeeping** (`generation_stats` in each manifest), so they exist
before `eval_v4 measure` runs. Delivered φ is pore/material from the label; the seam ratios are explained in the
assembly section (1.0 = no seam, real floor ≈ 0.97).
""")
    code(r'''
SM = manifests("sampler")
if len(SM) == 0:
    unavailable("sampler volumes", "eval_v4 generate sampler")
else:
    SM["scale"] = SM["shape_str"].map(lambda s: "192³" if s.startswith("192x192") else "1024 wide")
    SM["phi_error"] = SM["gs_actual_label_porosity"] - SM["requested_global_phi"]
    g = SM.groupby(["scale", "ddim_steps"]).agg(n=("seed", "size"), delivered_phi=("gs_actual_label_porosity", "mean"),
        abs_error=("phi_error", lambda s: s.abs().mean()), sd_error=("phi_error", "std"),
        seam_tile=("gs_seam_xct_ratio", "mean"), seam_chunk=("gs_seam_chunk_xct_ratio", "mean"),
        wall_s=("wall_time_s", "mean"), gpu_GB=("peak_gpu_memory_bytes", lambda s: s.mean() / 1e9)).reset_index()
    display(g.round(5))
''')
    md("""
**Porosity error versus steps.** Each point is one seed; the line joins the mean. The grey band is the ±0.005 gate.
A good result: all points inside the band at every step count, and no trend with steps (then 50 steps are enough).
""")
    code(r'''
if len(SM):
    fig = px.strip(SM, x="ddim_steps", y="phi_error", color="scale", hover_data=["case", "seed"], stripmode="overlay")
    for sc, gg in SM.groupby("scale"):
        mm = gg.groupby("ddim_steps")["phi_error"].mean()
        fig.add_scatter(x=mm.index, y=mm.values, mode="lines+markers", name=f"{sc} mean")
    fig.add_hrect(y0=-POROSITY_GATE, y1=POROSITY_GATE, fillcolor="grey", opacity=0.15, line_width=0, annotation_text="±0.005 gate")
    fig.update_layout(title="Delivered − requested porosity vs DDIM steps (request 0.03)", xaxis_title="DDIM steps", yaxis_title="φ error", height=400)
    fig.show()
''')
    md("""
**Seam ratios versus steps, with the real floor.** The dashed line is the real floor for tile planes (≈0.97) and, for
the 1024-wide case, chunk planes (≈0.90). Values above 1 mean the planes are more discontinuous than the interior,
i.e. a visible seam; 1.2 is a faint but real line. Look for whether more steps reduce a chunk seam.
""")
    code(r'''
if len(SM):
    fig = make_subplots(rows=1, cols=2, subplot_titles=["tile-plane seam ratio (grey)", "chunk-plane seam ratio (grey)"])
    for j, col in enumerate(["gs_seam_xct_ratio", "gs_seam_chunk_xct_ratio"], 1):
        for sc, gg in SM.groupby("scale"):
            fig.add_trace(go.Scatter(x=gg["ddim_steps"], y=gg[col], mode="markers", name=f"{sc}", text=gg["case"], showlegend=(j == 1)), row=1, col=j)
    if RF is not None:
        floor_hline(fig, RF["by_shape"]["small"]["seam_xct_ratio"]["mean"], "floor small", row=1, col=1)
        floor_hline(fig, RF["by_shape"]["large"]["seam_xct_ratio"]["mean"], "floor large", row=1, col=1, color="#999")
        floor_hline(fig, RF["by_shape"]["large"]["seam_chunk_xct_ratio"]["mean"], "floor large", row=1, col=2)
    fig.add_hline(y=1.0, line_color="black", line_width=0.5, row=1, col=2)
    fig.update_layout(height=400, title="Seam ratios from the sampler's own statistics")
    fig.show()
''')
    md("""
**Cost.** Wall time per volume and peak GPU memory, from the manifests. Time should grow linearly with steps and with
the number of windows (a 1024-wide volume has 28× the windows of a 192³ one). This is what the paper quotes as the
price of each setting.
""")
    code(r'''
if len(SM):
    fig = make_subplots(rows=1, cols=2, subplot_titles=["wall time per volume (s)", "peak GPU memory (GB)"])
    for sc, gg in SM.groupby("scale"):
        m1 = gg.groupby("ddim_steps")["wall_time_s"].mean(); m2 = gg.groupby("ddim_steps")["peak_gpu_memory_bytes"].mean() / 1e9
        fig.add_trace(go.Scatter(x=m1.index, y=m1.values, mode="lines+markers", name=sc), row=1, col=1)
        fig.add_trace(go.Scatter(x=m2.index, y=m2.values, mode="lines+markers", name=sc, showlegend=False), row=1, col=2)
    fig.update_layout(height=350); fig.update_yaxes(type="log", row=1, col=1); fig.show()
''')
    md("""
**Latent statistics (derived view).** `latents.npy` is the finished latent before decoding: 8 channels per 4³ block.
During training the model saw latents sampled as μ + σ·ε from the VAE, whose per-channel standard deviation
(averaged over the training store) is 1.863 in μ units. A generator that produces latents with the right spread is
neither collapsed (too small) nor exploding (too large). The plot shows the per-channel std of each generated latent
divided by the store reference; the training-time gate was 0.9–1.1 of the reference. Values in that band are good.
""")
    code(r'''
if len(SM) and SM["has_latents"].any():
    rows = []
    for _, r in SM.iterrows():
        if not r["has_latents"]: continue
        z = np.load(Path(r["dir"]) / "latents.npy", mmap_mode="r")
        std = np.asarray(z, dtype=np.float32).reshape(z.shape[0], -1).std(axis=1)
        rows.append({"case": r["case"], "scale": r["scale"], "ddim": r["ddim_steps"], "seed": r["seed"], "std_ratio_all": float(std.mean() / SAMPLED_STD_REF), **{f"ch{i}": float(s) for i, s in enumerate(std)}})
    LZ = pd.DataFrame(rows); display(LZ.round(3))
    fig = px.scatter(LZ, x="ddim", y="std_ratio_all", color="scale", hover_data=["case"], title="latent std / sampled-store reference (1.863)")
    fig.add_hrect(y0=0.9, y1=1.1, fillcolor="green", opacity=0.1, line_width=0); fig.update_layout(height=350); fig.show()
else:
    unavailable("latents.npy on sampler cases", "eval_v4 generate sampler --save-latents")
''')
    subsection("Sampler: measured results and paper figure")
    md("""
**Once `eval_v4 measure sampler` has run**, `results.json` holds the same quantities from the official metric code,
aggregated as mean ± sd over seeds per cell (`cells`), plus `air_fraction_interior`, the failure flags
(collapsed φ < 1e-4, saturated φ > 0.5, air inside material > 20 %) and, when `probs.npz` exists, the seam ratios
on the pore log-odds (`seam_pore_*`), which catch seams in the label that the grey does not show.
""")
    code(r'''
R = results("sampler")
if R is None:
    unavailable("sampler/results.json", "eval_v4 measure sampler")
else:
    display(cells_table(R["cells"], ["n_seeds", "porosity_error", "porosity_abs_error", "delivered_phi", "air_fraction_interior", "seam_xct_ratio", "seam_pore_ratio", "seam_chunk_xct_ratio", "seam_chunk_pore_ratio", "failure_rate", "wall_time_s"]))
    show_findings("sampler")
''')


def build_porosity_global() -> None:
    section("Global porosity: dose–response", "requested vs delivered porosity from 0.005 to 0.10, plus the off-manifold 0.15")
    md("""
**How the request enters the model.** The target porosity is one number, φ. It is turned into log φ, standardised,
and injected into every layer of the denoiser (*FiLM* conditioning: it scales and shifts the feature maps). During
training the value came from the label of the training patch, so the model learned "this much conditioning ⇒ this many
pore voxels". The request is *clamped* at φ = 0.107, the top of the training range; 0.15 is deliberately beyond it and
shows what happens off the training manifold.

**The measurement.** For each request (8 levels × 2 step counts × 3 seeds) the delivered φ is pore/material in the label.
The plot is requested (x) against delivered (y); perfect control is the diagonal. The grey band is ±0.005. The slope of
a straight-line fit through the in-range points is the *dose response*; 1.0 means one-to-one control, below 1 means the
model under-delivers at high porosity.
""")
    code(r'''
PG = manifests("porosity_global")
if len(PG) == 0:
    unavailable("porosity_global volumes", "eval_v4 generate porosity_global")
else:
    PG["delivered"] = PG["gs_actual_label_porosity"]; PG["error"] = PG["delivered"] - PG["requested_global_phi"]
    fig = px.scatter(PG, x="requested_global_phi", y="delivered", color=PG["ddim_steps"].astype(str), symbol="seed", hover_data=["case"],
                     labels={"color": "DDIM steps", "requested_global_phi": "requested φ", "delivered": "delivered φ"})
    xs = np.linspace(0, 0.16, 50)
    fig.add_scatter(x=xs, y=xs, mode="lines", name="y = x", line=dict(color="black", width=1))
    fig.add_scatter(x=np.r_[xs, xs[::-1]], y=np.r_[xs + POROSITY_GATE, (xs - POROSITY_GATE)[::-1]], fill="toself", fillcolor="rgba(120,120,120,0.15)", line=dict(width=0), name="±0.005 gate")
    fig.add_vline(x=0.107, line_dash="dot", annotation_text="clamp 0.107")
    fig.update_layout(title="Dose–response: requested vs delivered porosity", height=480); fig.show()
    ins = PG[PG["requested_global_phi"] < 0.12]
    for steps, gg in ins.groupby("ddim_steps"):
        slope, icpt = np.polyfit(gg["requested_global_phi"], gg["delivered"], 1)
        print(f"DDIM {steps}: slope {slope:.3f}, intercept {icpt:.4f}, mean |error| {gg['error'].abs().mean():.4f}, within gate {(gg['error'].abs() < POROSITY_GATE).mean():.0%}")
''')
    md("""
**Residuals and seed spread.** The same data as error (delivered − requested) per level. Look for a pattern:
a constant offset means a bias, a growing error means a slope below 1, and the width of the three seeds at each level
is the intrinsic randomness of a 192³ draw (the real floor for this is the spread between real crops of the same size).
""")
    code(r'''
if len(PG):
    fig = px.box(PG, x=PG["requested_global_phi"].astype(str), y="error", color=PG["ddim_steps"].astype(str), points="all", hover_data=["case"],
                 labels={"x": "requested φ", "color": "DDIM steps"})
    fig.add_hrect(y0=-POROSITY_GATE, y1=POROSITY_GATE, fillcolor="grey", opacity=0.15, line_width=0)
    fig.update_layout(title="Residual per requested level (three seeds)", height=400); fig.show()
''')
    md("""
**Off-manifold request 0.15.** The conditioning value is clamped at 0.107, so the model is asked for 0.107 while the
table compares to 0.15. This row is a failure mode by design; the questions are whether the volume stays sane
(no interior air, no collapse) and what porosity it settles at.
""")
    code(r'''
if len(PG):
    off = PG[PG["requested_global_phi"] >= 0.12]
    display(off[["case", "requested_global_phi", "gs_conditioned_porosity", "delivered", "gs_actual_label_air", "gs_seam_xct_ratio"]].round(4) if len(off) else "no off-manifold case on disk")
''')
    subsection("Global porosity: measured results and paper figure")
    md("""
`results.json` adds the official OLS fit (`dose_response`: slope, intercept, r², fraction within gate), per-level
mean ± sd (`levels`) and the off-manifold block. The paper figure is the dose–response plot with error bars.
""")
    code(r'''
R = results("porosity_global")
if R is None:
    unavailable("porosity_global/results.json", "eval_v4 measure porosity_global")
else:
    print("dose_response:", {k: (round(v, 4) if isinstance(v, float) else v) for k, v in R["dose_response"].items()})
    display(cells_table(R["levels"], ["requested", "n_seeds", "delivered", "abs_error", "within_gate", "failure_rate"]))
    print("off-manifold:", {k: R["off_manifold"][k] for k in ("requested", "conditioned_phi_after_clamp", "n_seeds")}, ms(R["off_manifold"]["delivered"]))
    show_findings("porosity_global")
''')


def build_porosity_local() -> None:
    section("Local porosity: painted fields", "halves, checkerboard and a sampled coherent field on the 3×3×3 tile grid")
    md("""
**How a local field enters the model.** Instead of one φ for the volume, the request is a 3×3×3 grid of φ values, one
per 64-voxel tile. Each window the model denoises reads the φ of the tile under it (the mean of the field over the
window footprint, which is why a window straddling two tiles gets a value in between). Three fields are tested:
*halves* (0.01 on one side, 0.05 on the other), *checkerboard* (alternating 0.01/0.05) and *coherent* (a smooth random
field with the correlation lengths of real material).

**Obedience, and why it is measured within the volume.** For each tile we compare delivered φ (from the label) with
requested φ. But a volume whose *mean* is right for the wrong reasons would score well on a naive fit, so the
obedience score first subtracts the volume's own mean from both sides and fits delivered on requested *within* the volume.
Slope 1 means the tiles differ by exactly what was asked; slope 0.5 means the contrast was halved.

The cell below recomputes the tile map from the label (cheap on 192³) and shows requested against delivered per tile.
""")
    code(r'''
PL = manifests("porosity_local")
if len(PL) == 0:
    unavailable("porosity_local volumes", "eval_v4 generate porosity_local")
else:
    rows = []
    for _, r in PL.iterrows():
        d = Path(r["dir"])
        if not (d / "requested_field.npy").exists(): continue
        req = np.load(d / "requested_field.npy"); lab = read_volume(d, "label")
        deliv = tile_phi_map(lab)
        for idx in np.ndindex(req.shape):
            rows.append({"case": r["case"], "field": r.get("note_field"), "ddim": r["ddim_steps"], "seed": r["seed"], "tile": str(idx), "requested": float(req[idx]), "delivered": float(deliv[idx])})
    TILES = pd.DataFrame(rows)
    fig = px.scatter(TILES, x="requested", y="delivered", color="field", symbol=TILES["ddim"].astype(str), hover_data=["case", "tile"], opacity=0.7)
    xs = np.linspace(0, 0.07, 10); fig.add_scatter(x=xs, y=xs, mode="lines", name="y = x", line=dict(color="black", width=1))
    fig.update_layout(title="Per-tile requested vs delivered porosity (recomputed from label.tif)", height=450); fig.show()
    fit = []
    for (c, f), g in TILES.groupby(["case", "field"]):
        x = g["requested"] - g["requested"].mean(); y = g["delivered"] - g["delivered"].mean()
        slope = float((x * y).sum() / (x * x).sum()) if (x * x).sum() > 0 else float("nan")
        r2 = float(1 - ((y - slope * x) ** 2).sum() / max((y ** 2).sum(), 1e-12))
        fit.append({"case": c, "field": f, "within-volume slope": slope, "within-volume r²": r2, "mean |error|": float((g["delivered"] - g["requested"]).abs().mean())})
    FIT = pd.DataFrame(fit); display(FIT.groupby("field")[["within-volume slope", "within-volume r²", "mean |error|"]].agg(["mean", "std"]).round(3))
''')
    md("""
**Tile maps side by side.** For one case: the requested field (left) and the delivered tile porosity (right),
as three z-layers of the 3×3×3 grid. The pattern should be visible in the delivered map; its contrast is the slope above.
""")
    code(r'''
if len(PL):
    LOCAL_CASE = "checkerboard_ddim200_seed101"
    d = case_dir("porosity_local", LOCAL_CASE)
    if d.exists() and (d / "requested_field.npy").exists():
        req = np.load(d / "requested_field.npy"); deliv = tile_phi_map(read_volume(d, "label"))
        fig = make_subplots(rows=2, cols=3, subplot_titles=[f"requested z-tile {k}" for k in range(3)] + [f"delivered z-tile {k}" for k in range(3)])
        vmax = float(np.nanmax([req.max(), np.nanmax(deliv)]))
        for k in range(3):
            fig.add_trace(go.Heatmap(z=req[k], zmin=0, zmax=vmax, colorscale="Reds", showscale=(k == 2)), row=1, col=k + 1)
            fig.add_trace(go.Heatmap(z=deliv[k], zmin=0, zmax=vmax, colorscale="Reds", showscale=False), row=2, col=k + 1)
        fig.update_layout(height=520, title=f"{LOCAL_CASE}: requested field vs delivered tile porosity"); fig.show()
    else:
        unavailable(LOCAL_CASE, "eval_v4 generate porosity_local")
''')
    subsection("Local porosity: measured results and paper figure")
    md("""
`results.json` holds, per field type, the official within-volume slope and r² (mean ± sd over seeds), the per-tile
absolute error and fraction within the gate, and the delivered versus requested tile spread. The `pooled_over_seeds`
fit is *not* obedience (it is dominated by the global dose response) and is reported only as context.
""")
    code(r'''
R = results("porosity_local")
if R is None:
    unavailable("porosity_local/results.json", "eval_v4 measure porosity_local")
else:
    display(cells_table(R["fields"], ["n_seeds", "within_volume_slope", "within_volume_r2", "per_cell_abs_error", "per_cell_frac_within_gate", "delivered_cell_sd", "requested_cell_sd"]))
    show_findings("porosity_local")
''')


# ===========================================================================
# 10. cfg, 11. layup, 12. assembly, 13. geometry, 14. surface
# ===========================================================================

def build_cfg() -> None:
    section("Guidance scales (CFG)", "what pushing the porosity request harder does, and whether the neighbour arm acts")
    md("""
**Classifier-free guidance in one paragraph.** During training the model sometimes saw the request blanked out
(10 % of the time for the porosity number, 10 % for the neighbours). At sampling time it therefore knows two
predictions: with the request and without. Guidance scale *s* uses `without + s·(with − without)`: s = 1 is the plain
conditional model, s = 2 doubles the pull towards the request. Too much guidance sharpens obedience but can distort
texture and create artefacts (saturated latents, degenerate tiles). Two arms are tested:
- `s_por` ∈ {1, 1.5, 2} at targets 0.02 and 0.05: does harder guidance improve porosity accuracy, and at what cost in seams and degenerate tiles?
- `s_nb` ∈ {0, 1} on 256³ volumes (which have a chunk plane at 192): with the neighbour guidance off, does the volume
  change *across the chunk plane*? If the pore pattern there is identical (Dice ≈ 1), the neighbour arm does nothing.

The cells below use the sampler's own statistics; the measured results add the chunk-plane Dice pairs.
""")
    code(r'''
CF = manifests("cfg")
if len(CF) == 0:
    unavailable("cfg volumes", "eval_v4 generate cfg")
else:
    CF["arm"] = CF.get("note_arm", CF["case"].str.extract(r"^(snb|spor)")[0])
    por = CF[CF["case"].str.startswith("spor")].copy()
    por["error"] = por["gs_actual_label_porosity"] - por["requested_global_phi"]
    fig = make_subplots(rows=1, cols=3, subplot_titles=["φ error vs s_por", "tile-plane seam ratio vs s_por", "air fraction vs s_por"])
    for tgt, g in por.groupby("requested_global_phi"):
        mm = g.groupby("s_por").agg(err=("error", "mean"), seam=("gs_seam_xct_ratio", "mean"), air=("gs_actual_label_air", "mean")).reset_index()
        fig.add_trace(go.Scatter(x=g["s_por"], y=g["error"], mode="markers", name=f"target {tgt} (seeds)", marker=dict(opacity=0.5), showlegend=False), row=1, col=1)
        fig.add_trace(go.Scatter(x=mm["s_por"], y=mm["err"], mode="lines+markers", name=f"target {tgt}"), row=1, col=1)
        fig.add_trace(go.Scatter(x=mm["s_por"], y=mm["seam"], mode="lines+markers", name=f"target {tgt}", showlegend=False), row=1, col=2)
        fig.add_trace(go.Scatter(x=mm["s_por"], y=mm["air"], mode="lines+markers", name=f"target {tgt}", showlegend=False), row=1, col=3)
    fig.add_hrect(y0=-POROSITY_GATE, y1=POROSITY_GATE, fillcolor="grey", opacity=0.15, line_width=0, row=1, col=1)
    fig.update_layout(height=380, title="Porosity guidance arm (sampler statistics)"); fig.show()
    nb = CF[CF["case"].str.startswith("snb")]
    display(nb[["case", "s_nb", "requested_global_phi", "gs_actual_label_porosity", "gs_seam_xct_ratio", "gs_seam_chunk_xct_ratio"]].round(4))
''')
    md("""
**Neighbour arm, recomputed here.** For each seed, the pore label of the s_nb = 0 volume is compared with the s_nb = 1
volume in a slab around the chunk plane (64 voxels either side of z, y, x = 192) and in the whole volume, with the
Dice coefficient (2·overlap / total pore voxels of both). Same seed means same starting noise, so any difference is
the effect of the neighbour guidance. Dice near 1 in the slab = the arm is inert; clearly below 1 = it acts.
""")
    code(r'''
def pore_dice(a, b, mask=None):
    pa, pb = (a == 1), (b == 1)
    if mask is not None: pa, pb = pa & mask, pb & mask
    den = pa.sum() + pb.sum(); return float(2 * (pa & pb).sum() / den) if den else float("nan")

if len(CF):
    rows = []
    for seed in sorted(nb["seed"].unique()):
        a, b = case_dir("cfg", f"snb0_seed{seed}"), case_dir("cfg", f"snb1_seed{seed}")
        if not (a.exists() and b.exists()): continue
        la, lb = read_volume(a, "label"), read_volume(b, "label")
        shp = la.shape; slab = np.zeros(shp, bool)
        for ax, n in enumerate(shp):
            for p in range(192, n, 192):
                sl = [slice(None)] * 3; sl[ax] = slice(max(p - 32, 0), min(p + 32, n)); slab[tuple(sl)] = True
        rows.append({"seed": seed, "dice chunk-plane slab": pore_dice(la, lb, slab), "dice whole volume": pore_dice(la, lb), "slab voxels": int(slab.sum())})
    display(pd.DataFrame(rows).round(4) if rows else "no s_nb pairs on disk")
''')
    subsection("Guidance: measured results and paper figure")
    code(r'''
R = results("cfg")
if R is None:
    unavailable("cfg/results.json", "eval_v4 measure cfg")
else:
    display(cells_table(R["s_por_cells"], ["s_por", "requested", "n_seeds", "delivered_phi", "abs_error", "air_fraction_interior", "degenerate_cell_fraction", "seam_xct_ratio", "seam_pore_ratio", "failure_rate"]))
    display(cells_table(R["s_nb_cells"], ["s_nb", "n_seeds", "delivered_phi", "seam_chunk_xct_ratio", "seam_chunk_pore_ratio"]))
    print("s_nb pairs:", pd.DataFrame(R["s_nb_pairs"]).round(4).to_string()); print(R["s_nb_summary"]["reading"])
    show_findings("cfg")
''')


def build_layup() -> None:
    section("Layup: reading the stacking sequence back", "requested ply angles vs the angles two readers recover, with their real floors")
    md("""
**How orientation enters the model.** A laminate is a stack of plies, each with fibres at 0°, 45°, −45° or 90°.
The request is a per-depth profile: for each z-plane the model receives cos 2θ and sin 2θ of the ply that plane
belongs to (two channels, constant within a ply, changing at ply boundaries), plus its relative depth. So the model
knows, at every latent cell, which fibre direction the texture around it should have.

**How it is read back.** Two readers estimate θ per ply from the *generated* volume: `fft_slice` finds the dominant
texture direction of each in-plane grey slice with a Fourier transform; `pore_axes` fits the elongation direction of
pores in the label (pores are elongated along the fibres). Their errors on real volumes with known layup are the floors
(median error 8.2° / 4.2°, strict 4-class accuracy 74 % / 86 %). Scoring is direct: no shift, no sign flip allowed.
Three layups are tested at panel width and 200 steps: A and C are training sequences, B16 is an unseen permutation.

Nothing can be shown before `eval_v4 measure layup`, except which cases exist and the sampler statistics.
""")
    code(r'''
LU = manifests("layup")
if len(LU) == 0:
    unavailable("layup volumes", "eval_v4 generate layup (runs last in the queue, ~6 h)")
else:
    display(LU[["case", "note_layup", "requested_layup", "requested_ply_thickness_vox", "ddim_steps", "seed", "gs_actual_label_porosity", "gs_seam_chunk_xct_ratio", "wall_time_s"]].round(4))
''')
    md("""
**Per-ply recovery.** Once measured, each case carries the requested angle per ply, the recovered angle per reader,
and a hit flag. The plot shows requested (black steps) against recovered per reader along depth; the table gives the
median error and 4-class accuracy against the floors. A reader at or near its floor means the model's texture carries
the layup as well as real material does; a result far above the floor (large errors) means the plies are not rendered
with the requested fibre direction.
""")
    code(r'''
R = results("layup")
if R is None:
    unavailable("layup/results.json", "eval_v4 measure layup")
else:
    rows = []
    for name, L in R["layups"].items():
        for reader, rd in L["readers"].items():
            if rd.get("available"):
                rows.append({"layup": name, "reader": reader, "median |err| deg": ms(rd["median_abs_error_deg"], 2), "strict 4-class acc": ms(rd["strict_class_accuracy"], 3),
                             "frac within 10°": ms(rd["frac_within_10"], 3), "floor median": (rd.get("real_floor") or {}).get("median_abs_error_deg"), "floor acc": (rd.get("real_floor") or {}).get("strict_class_accuracy")})
    display(pd.DataFrame(rows))
    pc = R["per_case"]
    fig = make_subplots(rows=1, cols=len(R["layups"]), subplot_titles=list(R["layups"]))
    for j, name in enumerate(R["layups"], 1):
        rows_l = [r for r in pc if r["layup"] == name]
        if not rows_l: continue
        rec = rows_l[0]["recovery"]; req = rec["requested_deg"]
        fig.add_trace(go.Scatter(y=req, x=list(range(len(req))), mode="lines+markers", name="requested", line=dict(color="black"), showlegend=(j == 1)), row=1, col=j)
        for k, (reader, col) in enumerate([("fft_slice", "#1b6ca8"), ("pore_axes", "#c2571a")]):
            for r in rows_l:
                rd = r["recovery"]["readers"].get(reader, {})
                if rd.get("available") and rd.get("recovered_deg"):
                    fig.add_trace(go.Scatter(y=rd["recovered_deg"], x=list(range(len(rd["recovered_deg"]))), mode="markers", name=f"{reader} seed {r['seed']}", marker=dict(color=col, opacity=0.6), showlegend=(j == 1)), row=1, col=j)
    fig.update_layout(height=380, title="Requested vs recovered ply angle per ply index"); fig.update_yaxes(title="angle (deg)"); fig.update_xaxes(title="ply index"); fig.show()
    show_findings("layup")
''')


def build_assembly() -> None:
    section("Assembly: seams and window phase", "is the volume one object or a grid of blocks? seam ratios, cross-head disagreement, offset test")
    md("""
**Where seams can come from.** Two places: the 64-voxel *tile* planes, where overlapping windows are averaged during
denoising and where the decoder's blocks meet, and the 192-voxel *chunk* planes, where one chunk was finished before
the next started. Campaign 08 showed that the mask seams of the old pipeline came from tile-by-tile decoding, cured by
overlapped decoding; the chunk seam is the one the neighbour conditioning must remove.

**The seam ratio.** Take the grey volume. On each plane of interest compute the mean absolute difference between the
two voxel layers on either side of the plane (a jump). Compute the same quantity on all the planes *in between*
(the interior baseline). Ratio = jump on the planes / jump in the interior. 1.0 = the planes are indistinguishable;
the real floor is about 0.97 for tile planes and 0.90 for chunk planes (the planes happen to fall where real texture is
slightly smoother). A visible line gives 1.2 or more. The same ratio on the pore log-odds (`seam_pore_*`) catches seams
in the label. Both come per axis, so an assembly problem in z only is visible.

**Cross-head disagreement.** The decoder has two heads: one draws grey, one draws the label. A voxel the grey head
renders as dark as air, in a region the label head calls material, is a disagreement between the heads (detector:
grey < 182, components ≥ 300 voxels). The real floor is ~0.0001.
""")
    code(r'''
if len(SM):
    axes_rows = []
    for _, r in SM.iterrows():
        for ax in "zyx":
            axes_rows.append({"case": r["case"], "scale": r["scale"], "ddim": r["ddim_steps"], "axis": ax, "tile seam (grey)": r.get(f"gs_seam_xct_{ax}_ratio"),
                              "chunk seam (grey)": r.get(f"gs_seam_chunk_xct_{ax}_ratio"), "tile seam (pore)": r.get(f"gs_seam_pore_{ax}_ratio"), "chunk seam (pore)": r.get(f"gs_seam_chunk_pore_{ax}_ratio")})
    AX = pd.DataFrame(axes_rows)
    fig = make_subplots(rows=1, cols=4, subplot_titles=["tile seam (grey)", "chunk seam (grey)", "tile seam (pore)", "chunk seam (pore)"])
    for j, col in enumerate(["tile seam (grey)", "chunk seam (grey)", "tile seam (pore)", "chunk seam (pore)"], 1):
        for sc, g in AX.groupby("scale"):
            fig.add_trace(go.Box(x=g["axis"], y=g[col], name=sc, boxpoints="all", text=g["case"], showlegend=(j == 1)), row=1, col=j)
        fig.add_hline(y=1.0, line_color="black", line_width=0.5, row=1, col=j)
    fig.update_layout(boxmode="group", height=380, title="Seam ratios per axis on the sampler volumes (sampler statistics)"); fig.show()
else:
    unavailable("sampler volumes", "eval_v4 generate sampler")
''')
    md("""
**Seam profile along an axis (derived view).** Instead of one ratio, the mean absolute grey jump between consecutive
slices, for every slice position along one axis. Tile planes are marked dotted, chunk planes solid. A seam shows as
a spike exactly on a marked plane; a healthy volume shows only the ply texture (regular bumps along z with the ply period)
and noise. Change `PROFILE_CASE` to any case; the 192³ cases load fully, the 1024 ones use one z-profile.
""")
    code(r'''
PROFILE_CASE = ("sampler", "192_ddim200_seed101")
d = case_dir(*PROFILE_CASE)
if not d.exists():
    unavailable("/".join(PROFILE_CASE), "eval_v4 generate sampler")
else:
    vol = read_volume(d, "grey")
    if vol is not None:
        fig = make_subplots(rows=1, cols=3, subplot_titles=[f"axis {ax}" for ax in "zyx"])
        for j, ax in enumerate("zyx"):
            v = np.moveaxis(vol, j, 0).astype(np.float32)
            jump = np.abs(np.diff(v, axis=0)).mean(axis=(1, 2))
            fig.add_trace(go.Scatter(y=jump, x=np.arange(1, len(jump) + 1), mode="lines", name=ax, showlegend=False), row=1, col=j + 1)
            for p in tile_planes(d)[ax]: fig.add_vline(x=p, line_dash="dot", line_color="#e0a800", row=1, col=j + 1)
            for p in chunk_planes(d)[ax]: fig.add_vline(x=p, line_color="orange", row=1, col=j + 1)
        fig.update_layout(height=340, title=f"Mean |grey jump| between consecutive slices — {PROFILE_CASE[1]}"); fig.show()
''')
    md("""
**The offset test.** The same region of material is requested three times inside a 256³ canvas, shifted by 0, 16 and
32 voxels. The noise is translated with the request, so the *only* thing that changes is where the window and chunk
grid falls relative to the material. Offset 32 is a whole window stride: it keeps the window phase and moves only the
chunk alignment. Offset 16 is half a stride: it moves the window phase too. If the assembly grid does not matter, the
pore pattern in the region is the same in all three (Dice near 1 after shifting back). Below: the sampler statistics
of the three offsets; the Dice pairs need `eval_v4 measure assembly`.
""")
    code(r'''
AS = manifests("assembly")
if len(AS) == 0:
    unavailable("assembly volumes", "eval_v4 generate assembly")
else:
    AS["offset"] = AS["case"].str.extract(r"offset(\d+)")[0].astype(int)
    display(AS[["case", "offset", "region_offset", "region_shape", "gs_actual_label_porosity", "gs_actual_label_air", "gs_seam_xct_ratio", "gs_seam_chunk_xct_ratio"]].round(4).sort_values(["offset", "case"]))
''')
    subsection("Assembly: measured results and paper figure")
    code(r'''
R = results("assembly")
if R is None:
    unavailable("assembly/results.json", "eval_v4 measure assembly")
else:
    display(cells_table(R["cells"], ["n_seeds", "seam_xct_ratio", "seam_pore_ratio", "seam_chunk_xct_ratio", "seam_chunk_pore_ratio", "cross_head_disagreement", "cross_head_disagreement_interior", "grey_air_claimed_by_label"]))
    wp = R.get("window_phase") or {}
    if wp.get("available"):
        display(pd.DataFrame(wp["pairs"]).round(4)); display(pd.DataFrame(wp["by_offset"]).T)
    print("VAE tile-decode control:", R.get("vae_tile_decode_control"))
    show_findings("assembly")
''')


def build_geometry() -> None:
    section("Geometry: carving air where the material map asks", "notch, drilled hole, and spheres the model never saw")
    md("""
**How geometry enters the model.** The material map is a 0/1 (or fractional) map on the latent grid: 1 where the
specimen is, 0 where there should be air. It enters as one input channel, and in addition each latent cell receives six
*face distances* (how far to the specimen boundary along ±z, ±y, ±x), computed from the *bounding box* of the material.
For a box the two agree; for a sphere the face distances describe a cube while the map describes a ball, so the sphere
is an off-manifold probe of which signal the model follows.

**Metrics.** Predicted air (label = 2) versus requested air (material map = 0): Dice, precision (of the voxels called
air, how many were requested), recall (of the requested air, how much was rendered), plus the air fraction *inside*
the material (should be 0) and *outside* (should be 1), and the pore fraction inside. For the sphere: the radius at which
the air fraction crosses one half, compared with the requested radius, per octant (`spread_vox` = how uneven).
Notch + hole is a gated case at 192×512×512 (a real-shaped coupon with a notch and a through hole, no real floor because
the request is synthetic); the spheres are exploratory.
""")
    code(r'''
GE = manifests("geometry")
if len(GE) == 0:
    unavailable("geometry volumes", "eval_v4 generate geometry")
else:
    rows = []
    for _, r in GE.iterrows():
        d = Path(r["dir"]); mat = np.load(d / "requested_material.npy") if (d / "requested_material.npy").exists() else None
        lab = read_volume(d, "label", force=True)
        if mat is None: continue
        req_mat = np.kron(mat, np.ones((4, 4, 4)))[:lab.shape[0], :lab.shape[1], :lab.shape[2]] >= 0.5
        air = lab == 2; req_air = ~req_mat
        inter = (air & req_air).sum(); dice = 2 * inter / max(air.sum() + req_air.sum(), 1)
        rows.append({"case": r["case"], "request": r.get("note_request", "notch_hole"), "shape": r["shape_str"], "ddim": r["ddim_steps"], "dice_air": dice,
                     "precision_air": inter / max(air.sum(), 1), "recall_air": inter / max(req_air.sum(), 1),
                     "air inside material": float(air[req_mat].mean()), "air outside material": float(air[req_air].mean()) if req_air.any() else float("nan"),
                     "φ inside material": float((lab[req_mat] == 1).mean() / max((lab[req_mat] != 2).mean(), 1e-9))})
    GEO = pd.DataFrame(rows); display(GEO.round(4))
''')
    md("""
**Air components (derived view).** Connected components of air (6-connectivity) inside the requested material, sorted by
size. Real material has none. A few tiny components at the surface rim are a decoder edge effect; a large component deep
inside is the failure that the drilled holes in the old dataset used to teach.
""")
    code(r'''
if len(GE):
    from scipy import ndimage
    GEO_CASE = "notch_hole_seed101"; d = case_dir("geometry", GEO_CASE)
    if d.exists():
        lab = read_volume(d, "label", force=True); mat = np.load(d / "requested_material.npy")
        req_mat = np.kron(mat, np.ones((4, 4, 4)))[:lab.shape[0], :lab.shape[1], :lab.shape[2]] >= 0.5
        inside_air = (lab == 2) & req_mat
        cc, n = ndimage.label(inside_air); sizes = np.sort(ndimage.sum(inside_air, cc, range(1, n + 1)))[::-1] if n else np.array([])
        print(f"{GEO_CASE}: {n} air components inside requested material, total {inside_air.sum()} voxels ({inside_air.mean():.2e} of volume)")
        if n: print("largest components (voxels):", sizes[:10].astype(int).tolist())
        three_view(d).show()
''')
    subsection("Geometry: measured results and paper figure")
    code(r'''
R = results("geometry")
if R is None:
    unavailable("geometry/results.json", "eval_v4 measure geometry")
else:
    print("notch + hole summary:"); display(pd.DataFrame({k: [ms(v) if isinstance(v, dict) and "mean" in v else v] for k, v in R["summary"].items()}).T.rename(columns={0: "value"}))
    sp = R.get("sphere_exploratory")
    if sp:
        print("sphere (exploratory, no gate):", sp["no_gate_because"]); print(sp["cond_dist6_note"])
        display(pd.DataFrame([{ "case": r["case"], "scale": r.get("scale"), "ddim": r.get("ddim_steps"), "dice_air": r["sphere"]["dice_air"], "air inside": r["sphere"]["air_fraction_inside_sphere"], "air outside": r["sphere"]["air_fraction_outside_sphere"],
                               "φ inside": r["sphere"]["phi_pore_inside_sphere"], "radial error vox": (r["sphere"].get("radial_surface") or {}).get("error_vox")} for r in sp["per_case"]]).round(4))
    show_findings("geometry")
''')


def build_surface() -> None:
    section("Surface: position and roughness of the specimen faces", "flat request (controllability) vs rough request (realism), against the real Sa floor")
    md("""
**The two requests.** A *flat* request is a material map with the top face at z = 32 and bottom at z = 160, exactly on
the latent grid. A *rough* request adds a random height field with the roughness (Sa) and correlation length of real
faces, so the map has fractional rim cells like real material. Why both: at step 74k the flat request produced a face
60× smoother than any real one while passing every position gate. The flat case tests whether the model puts the
face where asked (*controllability*: position error < 4 voxels); the rough case tests whether it can make a realistic
surface (*realism*: delivered Sa within 0.5–2× the requested Sa and within 0.5–2× the real floor).

**How it is measured.** For every (y, x) column, the first material voxel from the top gives the upper face height and
the last gives the lower face height (*height map*). Position error = height map minus requested height. Sa = mean
absolute deviation of the height map from its own mean plane; Sq = root mean square. Outlier columns (more than 4 voxels
off) are counted and clustered. `dark_but_material` counts voxels the grey head rendered as dark as air inside the
label's material, with and without a 2-voxel rim at the face.
""")
    code(r'''
SU = manifests("surface")
if len(SU) == 0:
    unavailable("surface volumes", "eval_v4 generate surface")
else:
    def height_maps(lab):
        mat = (lab == 0) | (lab == 1)
        anym = mat.any(axis=0)
        top = np.where(anym, mat.argmax(axis=0), np.nan)
        bot = np.where(anym, lab.shape[0] - 1 - mat[::-1].argmax(axis=0), np.nan)
        return top, bot
    def sa_sq(h):
        h = h[np.isfinite(h)]; dev = h - h.mean(); return float(np.abs(dev).mean()), float(np.sqrt((dev ** 2).mean()))
    rows = []; HM = {}
    for _, r in SU.iterrows():
        if r["n_voxels"] > 16e6: continue
        d = Path(r["dir"]); lab = read_volume(d, "label"); mat = np.load(d / "requested_material.npy")
        req = np.kron(mat, np.ones((4, 4, 4)))[:lab.shape[0], :lab.shape[1], :lab.shape[2]] >= 0.5
        req_lab = np.where(req, 0, 2)
        top, bot = height_maps(lab); rtop, rbot = height_maps(req_lab); HM[r["case"]] = (top, bot, rtop, rbot)
        for face, h, rh in (("upper", top, rtop), ("lower", bot, rbot)):
            sa, sq = sa_sq(h); rsa, _ = sa_sq(rh); err = np.nanmean(np.abs(h - rh))
            rows.append({"case": r["case"], "request": r.get("note_request"), "ddim": r["ddim_steps"], "seed": r["seed"], "face": face, "mean |position error| vox": err, "Sa": sa, "Sq": sq, "requested Sa": rsa, "Sa ratio to requested": sa / rsa if rsa > 0 else float("nan"),
                         "outlier columns (>4 vox)": int((np.abs(h - rh) > 4).sum()), "air inside box": float((lab[req] == 2).mean()), "air outside box": float((lab[~req] == 2).mean())})
    SURF = pd.DataFrame(rows); display(SURF.round(3))
    sf = load_json(ROOT / "real_floor" / "surface_floor.json") if (ROOT / "real_floor" / "surface_floor.json").exists() else None
    if sf: print(f"real floor: Sa {sf['sa_mean']:.3f} ± {sf['sa_sd']:.3f} vox (detrended {sf['detrended_sa_mean']:.3f}), correlation length {sf['correlation_length_vox_mean']:.1f} vox")
''')
    md("""
**Sa against the floor.** Each bar is a generated face; the dashed line is the real Sa. The flat request is expected
*below* the floor (it asked for a plane); the rough request should land within the green band (0.5–2× the floor).
""")
    code(r'''
if len(SU) and len(SURF):
    fig = px.bar(SURF, x="case", y="Sa", color="face", barmode="group", hover_data=["request", "requested Sa"], title="Face roughness Sa per generated case (192³ cases)")
    if sf:
        fig.add_hline(y=sf["sa_mean"], line_dash="dash", annotation_text="real Sa"); fig.add_hrect(y0=0.5 * sf["sa_mean"], y1=2 * sf["sa_mean"], fillcolor="green", opacity=0.1, line_width=0)
    fig.update_layout(height=400); fig.show()
''')
    md("""
**Height maps.** For one flat and one rough case: the delivered upper-face height map next to the requested one.
A flat request should give a nearly uniform map; a rough one should reproduce the large-scale bumps of the request
(same correlation length) with the small-scale texture of real material on top.
""")
    code(r'''
if len(SU) and HM:
    picks = [c for c in ("flat_192_ddim50_seed101", "rough_192_ddim50_seed101") if c in HM]
    fig = make_subplots(rows=len(picks), cols=2, subplot_titles=[t for c in picks for t in (f"{c}: delivered upper face", "requested upper face")])
    for i, c in enumerate(picks, 1):
        top, _, rtop, _ = HM[c]; zmin, zmax = np.nanmin(rtop) - 3, np.nanmax(rtop) + 3
        fig.add_trace(go.Heatmap(z=top, zmin=zmin, zmax=zmax, colorscale="Viridis", showscale=(i == 1)), row=i, col=1)
        fig.add_trace(go.Heatmap(z=rtop, zmin=zmin, zmax=zmax, colorscale="Viridis", showscale=False), row=i, col=2)
    fig.update_layout(height=380 * len(picks), title="Upper-face height maps (z index of first material voxel)"); fig.show()
''')
    subsection("Surface: measured results and paper figure")
    code(r'''
R = results("surface")
if R is None:
    unavailable("surface/results.json", "eval_v4 measure surface")
else:
    for kind in ("flat", "rough"):
        b = R["summary"].get(kind)
        if not b: continue
        print(f"--- {kind}: n={b['n_cases']}, gate:", b.get("gate"))
        display(pd.DataFrame({face: {k: ms(v, 3) for k, v in b[face].items()} for face in ("lower", "upper")}))
        print("air inside box", ms(b["air_fraction_inside_box"]), "| outside", ms(b["air_fraction_outside_box"]), "| dark-but-material", ms(b["dark_but_material"]), "| excl. rim", ms(b["dark_but_material_excluding_rim"]))
    print("real surface floor:", R.get("real_surface_floor")); print(R.get("why_two_requests"))
    show_findings("surface")
''')


# ===========================================================================
# 15. multichunk, 16. assembly modes, 17. microstructure + memorisation, 18. field stats
# ===========================================================================

def build_multichunk() -> None:
    section("Multi-chunk volumes: 384³ box, sphere r = 160, rough slab", "chunk planes on all three axes, and a curved surface crossing them")
    md("""
**Why 384³.** The production chunk is 192³, so a 384³ volume has one chunk plane on every axis (at 192) and eight
chunks; the previous assessments only had chunk planes in y and x (the 1024-wide cases are 192 deep). This is the
first place a chunk is assembled against finished neighbours *above and below* it. Three requests: a plain box,
a sphere of radius 160 whose curved surface crosses the planes, and a rough slab 384×384×192.
No real specimen is 384 voxels thick, so nothing here is evidence about thick material; it tests assembly only.

**Metrics.** The seam ratios split into *window planes* (period 64) and *chunk planes* (period from `chunk_tiles`),
the pore Dice across each chunk plane (the two slabs either side come from different chunk solves, so a mismatch is
assembly, not texture), and for the sphere the radial surface error per octant.
""")
    code(r'''
MC = manifests("multichunk")
if len(MC) == 0:
    unavailable("multichunk volumes", "eval_v4 generate multichunk")
else:
    cols = ["case", "note_request", "shape_str", "chunk_tiles", "ddim_steps", "seed", "gs_actual_label_porosity", "gs_actual_label_air", "gs_seam_xct_ratio", "gs_seam_chunk_xct_ratio", "gs_seam_pore_ratio", "gs_seam_chunk_pore_ratio", "wall_time_s"]
    display(MC[[c for c in cols if c in MC]].round(4))
    fig = go.Figure()
    for _, r in MC.iterrows():
        fig.add_bar(name=r["case"], x=[f"chunk {ax}" for ax in "zyx"] + [f"tile {ax}" for ax in "zyx"],
                    y=[r.get(f"gs_seam_chunk_xct_{ax}_ratio") for ax in "zyx"] + [r.get(f"gs_seam_xct_{ax}_ratio") for ax in "zyx"])
    fig.add_hline(y=1.0, line_color="black", line_width=0.6)
    if RF is not None: floor_hline(fig, RF["by_shape"]["large"]["seam_chunk_xct_ratio"]["mean"], "real chunk floor (large)")
    fig.update_layout(barmode="group", height=380, title="Grey seam ratio per axis: chunk planes vs tile planes (sampler statistics)"); fig.show()
''')
    md("""
**Look at the chunk planes.** Three views of the 384³ box on the planes z = y = x = 192 (top grey, bottom label),
then the sphere. Any straight line along the orange markers is an assembly seam; the sphere's boundary should be a
clean circle in every view, with no step where it crosses a plane.
""")
    code(r'''
for c in ("box384_ddim50_seed101", "sphere384_ddim50_seed101", "rough384_ddim50_seed101"):
    d = case_dir("multichunk", c)
    if d.exists(): three_view(d, width=1000).show()
    else: unavailable(c, "eval_v4 generate multichunk")
''')
    md("""
**Pore Dice across the chunk planes (recomputed).** For each chunk plane, the pore pattern in the 8-voxel slab just
before the plane is compared with the slab just after it, and the same for tile planes and for planes in between
(floor). Because pores are 3-D objects that cross planes, consecutive slabs overlap in pore identity; a chunk plane where
the two solves disagree gives a lower Dice than the interior baseline.
""")
    code(r'''
if len(MC):
    def plane_dice(lab, ax, p, half=4):
        a = np.take(lab, range(p - half, p), axis=ax); b = np.take(lab, range(p, p + half), axis=ax)
        return pore_dice(a, b)
    rows = []
    for _, r in MC.iterrows():
        lab = read_volume(Path(r["dir"]), "label")
        cp = chunk_planes(r["dir"]); tp = tile_planes(r["dir"])
        for j, ax in enumerate("zyx"):
            n = lab.shape[j]
            chunk = [plane_dice(lab, j, p) for p in cp[ax]]
            tile = [plane_dice(lab, j, p) for p in tp[ax] if p not in cp[ax]]
            between = [plane_dice(lab, j, p) for p in range(40, n - 8, 24) if p % 64]
            rows.append({"case": r["case"], "axis": ax, "chunk planes": np.mean(chunk) if chunk else np.nan, "tile planes": np.mean(tile) if tile else np.nan, "interior (floor)": np.mean(between)})
    display(pd.DataFrame(rows).round(3))
''')
    subsection("Multi-chunk: measured results and paper figure")
    code(r'''
R = results("multichunk")
if R is None:
    unavailable("multichunk/results.json", "eval_v4 measure multichunk")
else:
    print(R["not_physics"]); print(R["chunk_plane_note"])
    for kind, b in R["summary"].items():
        if b: print("---", kind); display(pd.DataFrame({k: [ms(v, 3) if isinstance(v, dict) and "mean" in v else v] for k, v in b.items()}).T.rename(columns={0: "value"}))
    show_findings("multichunk")
''')


def build_assembly_modes() -> None:
    section("Assembly modes: joint / autoregressive / hybrid / teacher-forced", "how much the hybrid sampler buys, per chunk index, against a ceiling")
    md("""
**The four arms, one request, same seeds.**
- *joint*: the whole volume is one chunk; every window is averaged with its overlaps at every step, and every neighbour
  input is UNKNOWN (the classifier-free null). This is the MultiDiffusion sampler of the old model.
- *autoregressive*: chunk size one tile (64³); each tile is denoised on its own against the finished tiles around it.
- *hybrid*: the production sampler, 3×3×3 tile chunks, joint inside a chunk, finished chunks re-noised and fed as
  neighbours to the next.
- *teacher-forced*: production chunking but every neighbour is replaced by the encoding of a *real* test volume at the
  same position. It is a control, not a sampler: the ceiling the hybrid would reach if what it assembles against were
  perfect. It exists only at the 192-deep slab (no real material is 384 deep).

**How they are compared.** Every arm is measured on the *same* reference chunk grid (192 voxels), whatever grid it was
generated on. Per chunk index (chunks in generation order) the seam ratio on chunk planes and on tile planes (grey and
pore), the porosity of the chunk, and the S2 texture distance across the plane relative to inside are read, with the real
floor. Then a straight line is fitted against chunk index: a positive slope means quality degrades as the volume grows
(compounding error), the failure a chunked sampler is prone to.
""")
    code(r'''
AM = manifests("assembly_modes")
if len(AM) == 0:
    unavailable("assembly_modes volumes", "eval_v4 generate assembly_modes")
else:
    AM["arm"] = AM["case"].str.extract(r"^(joint|autoregressive|hybrid|teacher_forced)")[0]
    AM["scale"] = AM["case"].str.extract(r"_(384|1024)_")[0]
    cols = ["case", "arm", "scale", "chunk_tiles", "note_neighbour_mode", "seed", "gs_actual_label_porosity", "gs_actual_label_air", "gs_seam_xct_ratio", "gs_seam_chunk_xct_ratio", "gs_seam_pore_ratio", "gs_seam_chunk_pore_ratio", "wall_time_s"]
    display(AM[[c for c in cols if c in AM]].round(4).sort_values(["scale", "arm", "seed"]))
    fig = make_subplots(rows=1, cols=3, subplot_titles=["tile-plane seam (grey, own grid)", "delivered φ", "wall time (s)"])
    for sc, g in AM.groupby("scale"):
        m = g.groupby("arm").agg(seam=("gs_seam_xct_ratio", "mean"), phi=("gs_actual_label_porosity", "mean"), t=("wall_time_s", "mean")).reset_index()
        fig.add_trace(go.Bar(x=m["arm"], y=m["seam"], name=f"{sc}"), row=1, col=1)
        fig.add_trace(go.Bar(x=m["arm"], y=m["phi"], name=f"{sc}", showlegend=False), row=1, col=2)
        fig.add_trace(go.Bar(x=m["arm"], y=m["t"], name=f"{sc}", showlegend=False), row=1, col=3)
    fig.update_layout(barmode="group", height=380, title="Assembly modes — sampler statistics (own chunk grid; the measured results use the common grid)"); fig.show()
''')
    md("""
**Recomputed on the common 192 grid (grey).** The sampler statistics above use each arm's own grid, which is what the
paper must *not* compare. Here the grey seam ratio at the planes z, y, x = 192, 384, … is recomputed for every 384³
case, so the four arms are on one grid. A ratio at the floor (≈0.90–0.97) for joint is the no-seam reading of the
measurement, since joint has no chunk planes at all.
""")
    code(r'''
def seam_ratio_grey(vol, period, axis, exclude=64):
    v = vol.astype(np.float32); n = v.shape[axis]
    jumps = np.abs(np.diff(v, axis=axis)).mean(axis=tuple(a for a in range(3) if a != axis))   # jump between slice i and i+1
    planes = [p for p in range(period, n, period)]
    on = [jumps[p - 1] for p in planes]
    interior = [jumps[i] for i in range(exclude, n - 1 - exclude) if all(abs(i + 1 - p) > 8 for p in planes)]
    return float(np.mean(on) / np.mean(interior)) if on and interior else float("nan")

if len(AM):
    rows = []
    for _, r in AM[AM["scale"] == "384"].iterrows():
        vol = read_volume(Path(r["dir"]), "grey")
        rows.append({"case": r["case"], "arm": r["arm"], "seed": r["seed"], **{f"chunk seam {ax}": seam_ratio_grey(vol, 192, j) for j, ax in enumerate("zyx")}, **{f"tile seam {ax}": seam_ratio_grey(vol, 64, j) for j, ax in enumerate("zyx")}})
    CG = pd.DataFrame(rows); display(CG.round(3).sort_values(["arm", "seed"]))
    if len(CG):
        m = CG.groupby("arm")[[c for c in CG if "seam" in c]].mean()
        fig = go.Figure([go.Bar(name=c, x=m.index, y=m[c]) for c in m])
        fig.add_hline(y=1.0, line_color="black", line_width=0.6); fig.update_layout(barmode="group", height=380, title="Grey seam ratio on the COMMON 192 / 64 grids, 384³ cases, mean over seeds"); fig.show()
''')
    subsection("Assembly modes: measured results and paper figure")
    md("""
The measured results give, per arm and scale, the volume-level seam on the reference grid, φ, the per-chunk-index series
for six quantities with mean ± sd over seeds, the OLS *trend* against chunk index, and the real floor for each quantity.
The plot draws the per-chunk series for each arm with the floor band. The claim "the hybrid beats its parts" is true only
if the hybrid line sits below joint and autoregressive on the chunk-plane seams and shows no upward trend.
""")
    code(r'''
R = results("assembly_modes")
if R is None:
    unavailable("assembly_modes/results.json", "eval_v4 measure assembly_modes")
else:
    print(R["note"]); print(R["reference_grid_note"])
    display(cells_table(R["cells"], ["arm", "scale", "n_seeds", "generated_chunk_tiles", "neighbour_mode", "volume_seam_xct_reference", "volume_seam_tile_xct", "delivered_phi", "air_fraction_interior", "wall_time_s", "failure_rate", "n_chunks"]))
    keys = list(R["chunk_keys"]); floor = R.get("real_floor") or {}
    fig = make_subplots(rows=2, cols=3, subplot_titles=keys)
    for i, key in enumerate(keys):
        row, col = i // 3 + 1, i % 3 + 1
        for name, c in R["cells"].items():
            ser = c["by_chunk_index"].get(key, [])
            fig.add_trace(go.Scatter(x=list(range(len(ser))), y=[s.get("mean") for s in ser], error_y=dict(array=[s.get("sd") or 0 for s in ser]), mode="lines+markers", name=name, legendgroup=name, showlegend=(i == 0)), row=row, col=col)
        for tag, f in floor.items():
            if f.get(key, {}).get("mean") is not None: fig.add_hline(y=f[key]["mean"], line_dash="dash", line_color="#555", annotation_text=f"floor {tag}", row=row, col=col)
    fig.update_layout(height=700, title="Per-chunk-index series per arm (mean ± sd over seeds)"); fig.update_xaxes(title="chunk index (generation order)"); fig.show()
    trend = pd.DataFrame({name: {k: c["trend"][k].get("slope") for k in keys} for name, c in R["cells"].items()}).T
    print("OLS slope vs chunk index (positive = degrades with distance from the first chunk):"); display(trend.round(4))
    show_findings("assembly_modes")
''')


def build_chunk_band() -> None:
    section("The chunk-plane pore band: defect, cause, fix trial",
            "what eval v4 found after the tables: a pore-free band at every chunk frontier, its root cause in training, and the sampler fixes tried (campaign 17)")
    md("""
**Read this section before trusting any chunk-plane number above.** When the assembly tables were read, two numbers
looked odd at once: the *pore-logit* seam ratio at chunk planes was far **below** 1 (0.39 at 1024 width) while the grey
ratio sat at the real floor, and the pore Dice across chunk planes was low. Both turned out to be one defect of the
generated volumes: **the model leaves a pore-depleted band in the ~32 voxels on each side of every chunk plane.**

**Important caveat on the checkpoint.** Every table in campaign 12 was generated with `best.ckpt` = step **119 000**
(minimum validation loss), because the runner hard-coded `--ckpt best`. The final weights are step 130 000. The
training-time gates (section "ldm06 training-time diagnostics") are on 130k. The band trial below runs on 130k. The next
full generation will be on 130k.
""")
    md("""
**How the band is measured.** For each chunk plane (every 192 voxels, the chunk size the production sampler uses) we take
8-voxel-thick slabs at offsets −32 … +24 from the plane (negative = the chunk generated *earlier*, the trailing side;
positive = the chunk generated *later*, the leading side). In each slab we compute the porosity as pore voxels / material
voxels, and divide by the volume's overall porosity. A flat line at 1.0 means the plane is invisible. The same profile is
computed on a **real** test volume at the same plane positions: real laminate is flat, so anything not flat is the model's.
""")
    code(r'''
PROF = ROOT / "chunk_plane_profile.json"; PROF_REAL = ROOT / "chunk_plane_profile_realfloor.json"
OFFS = ["-32", "-24", "-16", "-8", "0", "8", "16", "24"]
def _profile_rows(path, label=None):
    """Flatten a chunk_plane_profile json into rows: case, arm, axis, plane, offset, phi/phi_volume."""
    d = load_json(path); rows = []
    cases = d if isinstance(d, list) else d.get("per_case", [])        # two file generations: a bare list, or {"per_case": [...]}
    for c in cases:
        pv = c.get("phi_volume") or c.get("phi_whole_volume") or float("nan")
        if "axes" in c:                                                   # newer layout: axes -> {ax: {per_plane: [{plane, phi}]}}
            axes = [(ax, A.get("per_plane", []), "phi") for ax, A in (c.get("axes") or {}).items()]
        else:                                                             # older layout: profiles -> [{axis, per_plane: [{plane, phi_by_offset}]}]
            axes = [(str(pr.get("axis")), pr.get("per_plane", []), "phi_by_offset") for pr in c.get("profiles", []) if not pr.get("skipped")]
        for ax, per_plane, key in axes:
            for pp in per_plane:
                for o in OFFS:
                    v = (pp.get(key) or {}).get(o)
                    if v is not None:
                        rows.append({"case": c["case"], "arm": c.get("arm") or label or "generated", "s_nb": c.get("s_nb"),
                                     "axis": ax, "plane": pp["plane"], "offset": int(o), "phi": v, "ratio": v / pv if pv else float("nan")})
    return pd.DataFrame(rows)
if not PROF.exists():
    unavailable("chunk_plane_profile.json", "scripts/analysis/chunk_plane_profile.py on the campaign (see campaign README)")
else:
    P = _profile_rows(PROF, "generated (campaign 12)")
    if PROF_REAL.exists(): P = pd.concat([P, _profile_rows(PROF_REAL, "real test crop")])
    G = P.groupby(["arm", "case", "offset"], as_index=False)["ratio"].mean()
    fig = go.Figure()
    for (arm, case), g in G.groupby(["arm", "case"]):
        fig.add_trace(go.Scatter(x=g["offset"], y=g["ratio"], mode="lines+markers", name=f"{arm}: {case.split('/')[-1]}",
                                 line=dict(dash="dot" if "real" in arm else "solid")))
    fig.add_hrect(y0=0.8, y1=1.2, fillcolor="green", opacity=0.08, line_width=0); fig.add_vline(x=0, line_dash="dash", line_color="grey")
    fig.update_layout(title="Porosity by offset from the chunk plane, relative to the volume mean (mean over planes and in-plane axes)",
                      xaxis_title="offset from chunk plane (voxels; negative = earlier chunk)", yaxis_title="φ slab / φ volume", height=450)
    fig.show()
    display(P[P.offset == -8].groupby(["arm", "case"])["ratio"].agg(["mean", "min", "count"]).round(3))
''')
    md("""
**How to read it.** The dashed line is real material: flat at 1.0 within noise. The generated volumes dip to roughly
0.2 at −8 (the last 8 voxels of the earlier chunk) and recover to 1.0 about 24 voxels into the later chunk. The green band
is the ±20 % tolerance used later as the pass criterion. The count column says how many plane×axis profiles were averaged.

**Why it matters.** (1) It is why the pore-logit seam ratio was below 1: where there are no pores, slice-to-slice change
is small. (2) It is why the hybrid sampler delivered 0.025 for a request of 0.030 at 1024 width: the missing pores are
the band. (3) It is not drift: the band is the same depth at the first and the last plane of a volume.
""")
    md("""
**Finding the cause: the four sampler arms.** The same profile was computed on the assembly-mode ablation volumes, which
share one request and differ only in how neighbours are fed: *joint* (one chunk, every neighbour face UNKNOWN), *hybrid*
(neighbours from finished chunks, the frontier face UNKNOWN), *autoregressive* (one tile at a time), *teacher-forced*
(real neighbours on every face), plus the guidance pair from the cfg assessment (s_nb 0 = neighbour signal off, s_nb 1 =
on). A line that is flat in one arm and banded in another tells you which ingredient makes the band.
""")
    code(r'''
files = {"assembly-mode arms @1024": ROOT / "chunk_plane_profile_arms1024.json", "assembly-mode arms @384": ROOT / "chunk_plane_profile_arms384.json",
         "cfg pair s_nb 0 vs 1": ROOT / "chunk_plane_profile_snb.json"}
have_any = False
for title, f in files.items():
    if not f.exists(): continue
    have_any = True
    P = _profile_rows(f)
    P["label"] = P["arm"] + P["s_nb"].map(lambda v: f" (s_nb={v})" if "cfg" in title else "")
    G = P.groupby(["label", "offset"], as_index=False)["ratio"].mean()
    fig = go.Figure()
    for lab, g in G.groupby("label"):
        fig.add_trace(go.Scatter(x=g["offset"], y=g["ratio"], mode="lines+markers", name=lab))
    fig.add_hrect(y0=0.8, y1=1.2, fillcolor="green", opacity=0.08, line_width=0); fig.add_vline(x=0, line_dash="dash", line_color="grey")
    fig.update_layout(title=f"{title}: φ slab / φ volume by offset from the chunk plane", xaxis_title="offset (voxels)", yaxis_title="ratio", height=400); fig.show()
if not have_any:
    unavailable("chunk_plane_profile_arms*.json / _snb.json", "scripts/analysis/chunk_plane_profile.py on assembly_modes and cfg")
''')
    md("""
**What the arms say.** Joint (no neighbour signal anywhere) is flat. Turning the neighbour guidance off (s_nb 0) is flat.
Every arm that feeds a neighbour signal is banded, deepest with generated neighbours, shallower with real ones. So the
band is caused by the neighbour conditioning itself, not by re-noising, not by window fusion at the edge, and not by the
frontier being unknown as such.

**The single-window test** (64 real validation windows denoised one at a time, three neighbour faces present and three
missing) showed the mechanism inside one window: pores move **away from the missing face** (porosity at that face 0.05–0.15
of the window mean) while faces with a neighbour are untouched. A window conditioned on a pore-poor neighbour face is
also poorer at that face (inheritance), which is why the later chunk's first voxels are depleted too.

**Root cause, in the training code.** Neighbour dropout in ldm06 dropped all six faces *together* per sample. In the
training data 64 % of windows have five neighbours and one missing, and that missing face is **always the specimen
surface** (outside the sample). "Five present, one unknown" never occurred. The model therefore learned: *a face with
nothing behind it is a surface; real laminates have few pores near the surface; put the pores elsewhere.* At a chunk
frontier the not-yet-generated side looks exactly like that. The one healthy chunk plane in every volume is the last one,
where the next chunk's far face really is the volume edge, i.e. the trained configuration. The fix in training is one
line: per-face independent dropout (prepared as `configs/experiments/ldm06/facedrop.yaml`, a short warm-started run).
""")
    md("""
**The fix trial (campaign 17).** Sampler-side candidates, each generated on the **130k** weights at 1024×1024×192 (2 seeds)
and 384³ (3 seeds), scored on the *non-terminal* planes (the terminal plane is healthy for the reason above and would
flatter small volumes): **(a)** chunks overlap by 32 voxels and the shared strip is blended; **(b)** neighbour guidance
s_nb 0.5; **(c)** both; **(e)** overlap with the strip pinned (control); **(f)** drop the neighbour arm only for windows
whose neighbour set is mixed (some faces present, some missing), which is exactly the null condition the model was
trained on; **(g)** (f) + overlap. Pass = φ in both strips (−8 trailing, +0 leading) within ±20 % of the volume mean,
grey seam at the real floor, delivered φ within the 0.005 gate. *Cost* is minutes per volume relative to production.
""")
    code(r'''
TR = TRIAL_ROOT / "trial_report.json"
if not TR.exists():
    unavailable("campaign 17 trial_report.json", "the chunk-band fix trial (runs after the VAE rung reports)")
else:
    R = load_json(TR); rows = []
    for arm, A in R.items():
        S = A.get("summary") or {}
        rows.append({"arm": arm, "φ": S.get("phi_volume"), "φ error": S.get("phi_error"), "trailing −8": S.get("ratio_-8"), "worst −8": S.get("worst_-8"),
                     "leading +0": S.get("ratio_+0"), "terminal −8": S.get("terminal_-8"), "grey seam": S.get("grey_chunk_ratio"), "pore seam": S.get("pore_chunk_ratio"),
                     "min/vol (1024)": (S.get("wall_time_s") or float("nan")) / 60, "n": S.get("n_cases_with_nonterminal"), "PASS": S.get("PASS")})
    T = pd.DataFrame(rows).set_index("arm")
    display(T.round(3))
    fig = make_subplots(rows=1, cols=2, subplot_titles=["band: φ(slab)/φ(volume) on non-terminal planes", "cost, min per volume"])
    fig.add_trace(go.Bar(x=T.index, y=T["trailing −8"], name="trailing strip (−8)"), row=1, col=1)
    fig.add_trace(go.Bar(x=T.index, y=T["leading +0"], name="leading strip (+0)"), row=1, col=1)
    fig.add_hrect(y0=0.8, y1=1.2, fillcolor="green", opacity=0.1, line_width=0, row=1, col=1)
    fig.add_trace(go.Bar(x=T.index, y=T["min/vol (1024)"], name="min/volume", showlegend=False), row=1, col=2)
    fig.update_layout(height=420, barmode="group"); fig.show()
''')
    md("""
**How to read the trial table.** Columns "trailing −8" and "leading +0" are the two single-window strips; 1.0 is perfect,
the green band is the pass zone. "worst −8" is the worst single plane (a fix must hold everywhere). "terminal −8" is the
last plane, shown separately because it is in-distribution and healthy by construction. Grey and pore seams are ratios to
interior texture change; the real floor is about 0.91 at the chunk period, so near 1 is right. Cost is GPU minutes for
one 1024×1024×192 volume at DDIM-50 (production is 5.8). The per-plane series below shows that the band does not grow
along the generation order and that only the terminal plane differs.
""")
    code(r'''
if TR.exists():
    R = load_json(TR)
    arms = [a for a in R if R[a].get("per_case")]
    w_arm = W.Dropdown(options=arms, value=arms[-1] if arms else None, description="arm")
    out_pp = W.Output()
    def _pp(*_):
        with out_pp:
            out_pp.clear_output(wait=True)
            fig = make_subplots(rows=1, cols=2, subplot_titles=["trailing strip (−8), per plane in generation order", "leading strip (+0), per plane"])
            for c in R[w_arm.value]["per_case"]:
                pv = c.get("phi_volume") or float("nan")
                for side, col in (("trailing", 1), ("leading", 2)):
                    for ax, S in (c.get("per_plane") or {}).get(side, {}).items():
                        fig.add_trace(go.Scatter(x=S["planes"], y=[v / pv for v in S["phi"]], mode="lines+markers", name=f"{c['case']} axis {ax}"), row=1, col=col)
            for col in (1, 2): fig.add_hrect(y0=0.8, y1=1.2, fillcolor="green", opacity=0.1, line_width=0, row=1, col=col)
            fig.update_layout(height=400, title=f"{w_arm.value}: φ(slab)/φ(volume) at every chunk plane"); fig.show()
    w_arm.observe(_pp, names="value")
    show_widget(W.VBox([w_arm, out_pp]), _pp)
''')


def build_microstructure() -> None:
    section("Microstructure statistics and memorisation", "S2, pore-size distribution, Ripley's K, FID vs real, and nearest-neighbour search of the whole training store")
    md("""
**The idea.** Appearance is not evidence. Instead, four statistics of the pore structure are computed on generated
volumes and on real crops matched to the same porosity, and — crucially — on two disjoint real crops of *one* panel
against each other. That real-vs-real distance is the floor: it is not zero, because two finite samples of the same
material differ. The reported number is the ratio generated-vs-real / real-vs-real; 1 means "as close to real as real
is to itself".

**The four statistics.**
- *S2(r)*, the two-point correlation: the probability that two points a distance r apart are both pore. At r = 0 it equals
  φ; how fast it decays says how big pores are and how they cluster. Computed by FFT autocorrelation on 128³ windows;
  distance between curves = Wasserstein-1 (W1: the area between the two curves, roughly).
- *Pore-size distribution (PSD)*: equivalent diameter (6V/π)^(1/3) of every connected pore (6-connectivity); W1 between the
  two diameter distributions.
- *Ripley's K(r)*: the expected number of other pore centroids within r of a pore, border-corrected; says whether pores
  cluster (K above a random pattern) or repel. Distance = mean |log(K_gen / K_real)|.
- *FID*: Fréchet distance between Inception features of 64×64 grey crops, per axis. A standard image-similarity score;
  any FID computed before 2026-09-08 used unnormalised inputs and is void.

Levels φ = 0.01, 0.03, 0.06 at 192³ and 200 steps, three seeds; φ = 0.06 does not exist in two of the three test panels,
so its real reference is the closest available (recorded as `phi_miss`).
""")
    code(r'''
MS_ = manifests("microstructure")
if len(MS_) == 0:
    unavailable("microstructure volumes", "eval_v4 generate microstructure")
else:
    display(MS_[["case", "requested_global_phi", "ddim_steps", "seed", "gs_actual_label_porosity", "gs_seam_xct_ratio", "wall_time_s"]].round(4))
''')
    md("""
**Pore-size histograms, recomputed (derived view).** Equivalent diameters of the connected pores of each generated
192³ volume against the real micro reference crops at the same φ. Same shape of histogram = same pore population;
a generated histogram shifted to larger diameters means merged or inflated pores (a decoder blur symptom), to smaller
means fragmented pores.
""")
    code(r'''
from scipy import ndimage
def pore_diameters(lab):
    cc, n = ndimage.label(lab == 1, structure=ndimage.generate_binary_structure(3, 1))
    if n == 0: return np.array([])
    vols = ndimage.sum(np.ones_like(cc, dtype=np.uint8), cc, range(1, n + 1))
    return (6 * np.asarray(vols) / np.pi) ** (1 / 3)

if len(MS_):
    fig = make_subplots(rows=1, cols=3, subplot_titles=[f"φ = {lvl}" for lvl in (0.01, 0.03, 0.06)])
    for j, lvl in enumerate((0.01, 0.03, 0.06), 1):
        gen = MS_[np.isclose(MS_["requested_global_phi"], lvl)]
        for _, r in gen.iterrows():
            d_ = pore_diameters(read_volume(Path(r["dir"]), "label"))
            fig.add_trace(go.Histogram(x=d_, xbins=dict(start=0, end=40, size=1), histnorm="probability density", opacity=0.45, name=f"gen {r['case']}", legendgroup="gen", showlegend=(j == 1)), row=1, col=j)
        for d in case_dirs("real_floor"):
            if d.name.startswith(f"micro_phi{lvl}__"):
                d_ = pore_diameters(read_volume(d, "label"))
                fig.add_trace(go.Histogram(x=d_, xbins=dict(start=0, end=40, size=1), histnorm="probability density", opacity=0.35, name=f"real {d.name[-10:]}", marker_color="black", legendgroup="real", showlegend=(j == 1)), row=1, col=j)
    fig.update_layout(barmode="overlay", height=380, title="Equivalent pore diameter (voxels), generated (colour) vs real reference crops (black)"); fig.update_xaxes(title="diameter (vox)"); fig.show()
''')
    md("""
**S2(r) recomputed (derived view).** The radial two-point correlation of the pore phase on the central 128³ window,
generated versus real reference, per level. Curves that decay at the same rate mean the same pore size and spacing.
""")
    code(r'''
def s2_radial(binary, rmax=40):
    b = binary.astype(np.float32); phi = b.mean(); f = np.fft.fftn(b - phi)
    ac = np.real(np.fft.ifftn(f * np.conj(f))) / b.size + phi ** 2
    ac = np.fft.fftshift(ac); c = np.array(ac.shape) // 2
    zz, yy, xx = np.indices(ac.shape); rr = np.sqrt((zz - c[0]) ** 2 + (yy - c[1]) ** 2 + (xx - c[2]) ** 2).astype(int)
    out = ndimage.mean(ac, rr, range(0, rmax)); return np.arange(rmax), np.asarray(out)

if len(MS_):
    fig = make_subplots(rows=1, cols=3, subplot_titles=[f"φ = {lvl}" for lvl in (0.01, 0.03, 0.06)])
    for j, lvl in enumerate((0.01, 0.03, 0.06), 1):
        for _, r in MS_[np.isclose(MS_["requested_global_phi"], lvl)].iterrows():
            lab = read_volume(Path(r["dir"]), "label")[32:160, 32:160, 32:160]
            rr, s2 = s2_radial(lab == 1); fig.add_trace(go.Scatter(x=rr, y=s2 / max(s2[0], 1e-9), mode="lines", name=f"gen seed {r['seed']}", line=dict(color="#c2571a"), showlegend=(j == 1)), row=1, col=j)
        for d in case_dirs("real_floor"):
            if d.name.startswith(f"micro_phi{lvl}__"):
                lab = read_volume(d, "label")[:128, :128, :128]; rr, s2 = s2_radial(lab == 1)
                fig.add_trace(go.Scatter(x=rr, y=s2 / max(s2[0], 1e-9), mode="lines", name="real", line=dict(color="black", width=1), opacity=0.5, showlegend=(j == 1 and d.name.endswith("__a"))), row=1, col=j)
    fig.update_layout(height=380, title="S2(r) / S2(0) of the pore phase, 128³ window"); fig.update_xaxes(title="r (vox)"); fig.show()
''')
    subsection("Microstructure: measured results and paper figure")
    code(r'''
R = results("microstructure")
if R is None:
    unavailable("microstructure/results.json", "eval_v4 measure microstructure (GPU stage, runs after layup)")
else:
    rows = []
    for lvl, L in R["levels"].items():
        if not L.get("available"): rows.append({"level": lvl, "available": False}); continue
        rows.append({"level": lvl, "generated φ": ms(L["generated_phi"]), "real φ": ms(L["real_phi"]), "φ miss max": L.get("real_phi_miss_max"),
                     "S2 W1 gen/real": L["generated_vs_real"]["s2_w1"], "S2 W1 floor": L["real_vs_real"]["s2_w1"], "S2 ratio": L["ratio"]["s2_w1"],
                     "PSD W1 gen/real": L["generated_vs_real"]["psd_w1"], "PSD floor": L["real_vs_real"]["psd_w1"], "PSD ratio": L["ratio"]["psd_w1"],
                     "Ripley gen/real": L["generated_vs_real"]["ripley_log_ratio"], "Ripley floor": L["real_vs_real"]["ripley_log_ratio"], "Ripley ratio": L["ratio"]["ripley_log_ratio"],
                     "FID gen/real": (L.get("fid_generated_vs_real") or {}).get("mean"), "FID floor": (L.get("fid_real_vs_real") or {}).get("mean"), "FID ratio": L["ratio"]["fid"]})
    display(pd.DataFrame(rows).round(4))
    fig = make_subplots(rows=1, cols=len(R["levels"]), subplot_titles=[f"φ = {l}" for l in R["levels"]])
    for j, (lvl, L) in enumerate(R["levels"].items(), 1):
        if not L.get("available"): continue
        cv = L["generated_vs_real"].get("curves") or {}
        if cv: fig.add_trace(go.Scatter(x=cv["s2_r"], y=cv["s2_a"], name="generated", line=dict(color="#c2571a"), showlegend=(j == 1)), row=1, col=j); fig.add_trace(go.Scatter(x=cv["s2_r"], y=cv["s2_b"], name="real", line=dict(color="black"), showlegend=(j == 1)), row=1, col=j)
    fig.update_layout(height=350, title="Official S2 curves, generated vs real"); fig.show()
    show_findings("microstructure")
''')
    subsection("Memorisation: is any generated patch a copy of a training patch?")
    md("""
**The test.** Every 64³ window of the generated sampler, porosity_global, multichunk and assembly-mode volumes is
encoded to the latent space and compared with *all 219,580* non-overlapping training patches (the full store at stride 64,
not a sample). For each generated window: distance to its nearest training patch, x′, and to its second nearest, x″.
The *Favero ratio* ‖x − x′‖ / ‖x − x″‖ is near 1 when the window is equally far from many training patches (novel), and
near 0 when one training patch is much closer than every other (a copy). Threshold: below 1/3 = memorised.

**The floor.** The same search for 512 *real validation* patches, which are by construction not in the training set;
their ratio distribution is what "novel" looks like. In grey space the floor is the val patch decoded by the same VAE
(round-trip), so both sides carry the decoder error; `grey_raw` is the raw scan floor shown for completeness.
Breakdowns: by requested φ (a copy is likelier at rare, high porosity) and by whether the window was denoised with
some neighbour UNKNOWN (only in multi-chunk volumes).

Until the full pass exists the cell shows the smoke test (one 384³ volume), then the full result from `results.json`.
""")
    code(r'''
def memo_table(M):
    rows = []
    for name, blk in (("generated", M["generated"]), ("real val floor", M["real_val_floor"])):
        for space in ("latent", "grey", "grey_raw"):
            s = blk.get(space)
            if s: rows.append({"set": name, "space": space, "n": s["n"], "ratio mean": s["ratio_mean"], "ratio median": s["ratio_median"], "ratio p5": s["ratio_p5"], "ratio min": s["ratio_min"], "n memorised (<1/3)": s["n_memorised"], "frac": s["frac_memorised"]})
    return pd.DataFrame(rows).round(4)

M = (results("microstructure") or {}).get("memorisation")
src = "microstructure/results.json (full pass)"
if not M or not M.get("available"):
    smoke = ROOT / "memorisation_smoke.json"
    if smoke.exists(): M = load_json(smoke); src = "memorisation_smoke.json (SMOKE TEST: one volume, not the paper number)"
    else: M = None
if M is None:
    unavailable("memorisation results", "eval_v4 measure microstructure")
else:
    note(f"source: {src}; bank {M['bank']['n_rows']} train rows at stride {M['bank']['stride']}; criterion {M['criterion']['statistic']} < {M['criterion']['threshold']:.3f}; {M['n_cases']} cases, {M['n_patches']} patches; skipped too large: {M.get('skipped_too_large')}")
    display(memo_table(M))
    bp = pd.DataFrame([{ "requested φ": k, "n": v["n_patches"], "latent ratio mean": v["latent"]["ratio_mean"], "latent min": v["latent"]["ratio_min"], "grey ratio mean": v["grey"]["ratio_mean"], "memorised": v["latent"]["n_memorised"]} for k, v in M["by_requested_phi"].items()])
    bn = pd.DataFrame([{ "neighbours": k, "n": v["n_patches"], "latent ratio mean": v["latent"]["ratio_mean"], "latent min": v["latent"]["ratio_min"], "grey ratio mean": v["grey"]["ratio_mean"], "memorised": v["latent"]["n_memorised"]} for k, v in M["by_neighbours"].items()])
    display(bp.round(4)); display(bn.round(4))
    fig = go.Figure()
    for k, v in M["per_case"].items():
        s = v["spaces"]["latent"]
        fig.add_scatter(x=[k], y=[s["ratio_min"]], mode="markers", name=k, marker=dict(size=9), error_y=dict(type="data", symmetric=False, array=[s["ratio_mean"] - s["ratio_min"]], arrayminus=[0]))
    fig.add_hline(y=1 / 3, line_dash="dash", line_color="red", annotation_text="memorised below 1/3")
    fig.update_layout(height=380, title="Per case: minimum (marker) to mean (bar top) latent Favero ratio", yaxis=dict(range=[0, 1.05]), showlegend=False); fig.show()
''')


def build_field_stats() -> None:
    section("Field statistics: does the delivered porosity field look like the real one?", "marginal distribution and per-axis correlation length of local porosity, generated vs real")
    md("""
**What a porosity field is.** Slide a 64³ window over a volume every 32 voxels and record the porosity in each window
(windows with less than half material are dropped). That grid of numbers is the *delivered field*. Real material has a
characteristic field: most windows near zero, a tail of porous windows (the *marginal* distribution), and porous regions
that extend much further in-plane (x, y) than through-thickness (z) — the *correlation length* per axis, the lag at
which the correlation of the field with itself falls to 1/e.

**Why it matters.** The coherent local-porosity request is *built* from these real correlation lengths (campaign 01),
so the question is whether the model preserves them. This is also the direct comparison point with the field-controlled
rock LDM of Naiff, Ramos & Wang. Groups: the generated coherent-field cases (porosity_local and multichunk), the requested
fields themselves, and the real crops (small and large). Distances: W1 between marginals, with a real-vs-real floor
from splitting the real crops in halves; correlation length per axis is reported as *beyond reach* (null) when the crop is
shorter than the length.
""")
    code(r'''
FSR = results("field_stats")
if FSR is None:
    unavailable("field_stats/results.json", "eval_v4 measure field_stats")
else:
    print(FSR["geometry"]); print("T-D reference correlation lengths (vox):", FSR["t_d_reference"].get("corr_length_vox"))
    rows = []
    for g, G in FSR["groups"].items():
        m = G["marginal"]
        rows.append({"group": g, "fields": G["n_fields"], "windows": G["n_windows"], "mean": m["mean"], "sd": m["sd"], "cv": m["cv"], "q50": m["quantiles"]["0.5"], "q90": m["quantiles"]["0.9"], "q95": m["quantiles"]["0.95"],
                     **{f"corr len {ax}": G["per_axis"][ax].get("corr_length_vox") for ax in "zyx"}, **{f"reach {ax}": G["per_axis"][ax].get("reach_vox") for ax in "zyx"}})
    display(pd.DataFrame(rows).round(4))
''')
    md("""
**Marginals as quantile curves.** Each line is one group's quantiles of window porosity (x = quantile, y = φ, log scale).
Generated groups should follow the real curve of the same scale; a generated line above the real one at the top means
too many porous windows, below means the field is too flat. The requested-field lines show what was asked for.
""")
    code(r'''
if FSR is not None:
    fig = go.Figure()
    for g, G in FSR["groups"].items():
        q = G["marginal"]["quantiles"]; fig.add_scatter(x=[float(k) for k in q], y=list(q.values()), mode="lines+markers", name=g, line=dict(dash="dash" if g.startswith("requested") else "solid", color="black" if g.startswith("real") else None))
    fig.update_layout(height=400, title="Quantiles of window porosity per group", xaxis_title="quantile", yaxis_title="φ (window)", yaxis_type="log"); fig.show()
''')
    md("""
**Correlation along each axis.** The correlation of the field with itself at increasing lag, per axis. The lag where a curve
falls below 1/e (dotted line) is the correlation length. Real material: short in z (≈ 80 voxels in the reference), long
in-plane (hundreds). Generated fields shorter than real mean the model breaks up porous regions; longer means it smears them.
""")
    code(r'''
if FSR is not None:
    fig = make_subplots(rows=1, cols=3, subplot_titles=[f"axis {ax}" for ax in "zyx"])
    for j, ax in enumerate("zyx", 1):
        for g, G in FSR["groups"].items():
            pa = G["per_axis"][ax]
            fig.add_trace(go.Scatter(x=pa["lag_vox"], y=pa["correlation"], mode="lines+markers", name=g, legendgroup=g, showlegend=(j == 1), line=dict(dash="dash" if g.startswith("requested") else "solid", color="black" if g.startswith("real") else None)), row=1, col=j)
        fig.add_hline(y=1 / math.e, line_dash="dot", row=1, col=j)
    fig.update_layout(height=380, title="Field autocorrelation vs lag"); fig.update_xaxes(title="lag (vox)"); fig.show()
    comp = []
    for k, C in FSR["comparisons"].items():
        comp.append({"comparison": k, "W1": C["marginal"]["w1"], "KS": C["marginal"]["ks"], "W1 ratio (mean-normalised)": C["marginal"]["w1_ratio"], **{f"Δ corr len {ax}": (C["corr_length"].get(ax) or {}).get("difference_vox") for ax in "zyx"}, **{f"ratio {ax}": (C["corr_length"].get(ax) or {}).get("ratio") for ax in "zyx"}})
    display(pd.DataFrame(comp).round(4))
    show_findings("field_stats")
''')


# ===========================================================================
# 19. label uncertainty, 20. convergence, 21. decoder-ft, 22. downstream, 23. summary, 24. preview
# ===========================================================================

def build_label_uncertainty() -> None:
    section("Label uncertainty (campaign 13)", "how much the porosity of a real volume moves when our own labelling threshold moves")
    md("""
**Why this exists.** Every porosity number in this evaluation is defined by the labelling convention that built the
dataset: a Sauvola local threshold (k = 0.125, radius 30) for pores inside a material mask found by an Otsu threshold.
The model learned that convention. So "requested φ = 0.030, delivered 0.031" is a statement about the convention, not
about physical pore volume. Campaign 13 measures how firm the convention is: it relabels three full real test volumes
with k moved by ±20 % (0.100 and 0.150) and reports the range of φ, and the Dice between the perturbed pore labels.

**The one number to quote** is the k-only range at the production Otsu mask, per volume. The campaign also ran other
material thresholds (Yen, Isodata); Isodata is identical to Otsu to the voxel, and Yen picks a threshold of 33 instead of
123 and counts the surrounding air as material, so it is a different segmentation, not a perturbation. Those rows are an
appendix and must not be quoted; the script's top-level "summary" block is that all-variant number and is wrong to use.
This is not a segmentation-methods paper.
""")
    code(r'''
LU_R = LABEL_ROOT / "results.json"
if not LU_R.exists():
    unavailable("13-label-uncertainty/results.json", "the label_uncertainty campaign (copy campaigns/13-label-uncertainty)")
else:
    LUR = load_json(LU_R)
    rows = []
    for v in LUR["volumes"]:
        kv = {x["name"]: x["phi"] for x in v["variants"] if x["material_method"] == "otsu"}
        rows.append({"volume": v["volume_id"][-28:], "φ k=0.100": kv.get("k0.1/otsu"), "φ k=0.125 (production)": kv.get("k0.125/otsu"), "φ k=0.150": kv.get("k0.15/otsu"),
                     "k-only range": v["summary"]["phi_range_sauvola_k_only"], "range / 0.005 gate": v["summary"]["phi_range_sauvola_k_only"] / 0.005})
    KO = pd.DataFrame(rows); display(KO.round(4))
    print(f"mean k-only range {KO['k-only range'].mean():.4f} = {KO['k-only range'].mean() / 0.005:.1f}× the porosity gate")
    me = LUR.get("model_porosity_error")
    if me: print(f"model porosity error {me['por_mae']:.5f} ({me['variant']}, step {me['step']}, n={me['n_samples']}; this is the TRAINING-TIME diagnostic, to be replaced by the eval-v4 sampler number) → label range is {KO['k-only range'].mean() / me['por_mae']:.0f}× larger")
    fig = go.Figure()
    for _, r in KO.iterrows(): fig.add_scatter(x=[0.100, 0.125, 0.150], y=[r["φ k=0.100"], r["φ k=0.125 (production)"], r["φ k=0.150"]], mode="lines+markers", name=r["volume"])
    fig.update_layout(height=350, title="Porosity of a real volume vs Sauvola k (material mask fixed at Otsu)", xaxis_title="k", yaxis_title="φ"); fig.show()
''')
    md("""
**Dice between the k-perturbed pore labels.** For each volume, the 3×3 Dice matrix among k = 0.100 / 0.125 / 0.150 at
Otsu. This bounds what any pore Dice against these labels can mean: two defensible labellings of the *same scan* overlap
only this much. It is a lower bound (whole-volume Dice, so a pore that moved outside the other labelling's material
envelope counts as a disagreement).
""")
    code(r'''
if LU_R.exists():
    fig = make_subplots(rows=1, cols=len(LUR["volumes"]), subplot_titles=[v["volume_id"][-20:] for v in LUR["volumes"]])
    mins, meds = [], []
    for j, v in enumerate(LUR["volumes"], 1):
        names = [x["name"] for x in v["variants"]]; idx = [i for i, n in enumerate(names) if n.endswith("/otsu")]
        D = np.array(v["dice_matrix"])[np.ix_(idx, idx)]; lab = [names[i].split("/")[0] for i in idx]
        off = D[~np.eye(len(idx), dtype=bool)]; mins.append(off.min()); meds.append(np.median(off))
        fig.add_trace(go.Heatmap(z=D, x=lab, y=lab, zmin=0, zmax=1, colorscale="Blues", text=np.round(D, 3), texttemplate="%{text}", showscale=(j == 1)), row=1, col=j)
    fig.update_layout(height=320, title="Pore Dice between Sauvola-k variants at the Otsu mask"); fig.show()
    print(f"k-only Dice: min {min(mins):.3f}, median {np.median(meds):.3f}  (the all-variant 0.06 / 0.31 figures include Yen and are withdrawn)")
    display(Markdown((LABEL_ROOT / "findings.md").read_text())) if (LABEL_ROOT / "findings.md").exists() else None
''')


def build_convergence() -> None:
    section("ldm06 training-time diagnostics", "the gate table across training steps: a diagnostic view, not the paper evaluation")
    md("""
**What this is.** During training, every 20k steps (and at exit) the run generated 32 volumes of 192³ in four porosity
buckets and checked a few gates: porosity error, latent spread, air inside/outside an inset box, and a kill-switch
request pair (0.005 vs 0.05). The final line is step 130k. These are the numbers the "train longer?" discussion uses,
and they are *not* the eval-v4 tables: fewer volumes, DDIM-50/200 only, one shape. They are shown so the trajectory is
visible. The latent *std ratio* is the spread of the generated latent divided by the spread of sampled training latents
(reference 1.863 in μ units); 1.0 is right, well below 1 is a collapsed generator.
""")
    code(r'''
if not CONV_JSONL.exists():
    unavailable("convergence_check.jsonl", "copy the ldm06 run's convergence_check.jsonl (see setup)")
else:
    recs = []
    for line in CONV_JSONL.read_text().splitlines():
        try: recs.append(json.loads(line))
        except Exception: pass                      # a half-written last line while training
    rows = []
    for r in recs:
        for var, V in r["variants"].items():
            o = V.get("overall") or {}
            rows.append({"step": r["step"], "variant": var, "por_mae": o.get("por_mae"), "std_ratio": o.get("std_ratio"), "std_ratio_vs_sampled": (o.get("std_ratio") or 0) / SAMPLED_STD_REF, "air_mean": o.get("air_mean"), "x0_sat": o.get("x0_sat"), "degen": o.get("degen")})
    CV = pd.DataFrame(rows).sort_values(["variant", "step"])
    display(CV[CV["step"] == CV["step"].max()].round(5))
    fig = make_subplots(rows=1, cols=3, subplot_titles=["porosity MAE", "latent std / sampled reference", "mean air fraction"])
    for var, g in CV.groupby("variant"):
        fig.add_trace(go.Scatter(x=g["step"], y=g["por_mae"], mode="lines+markers", name=var), row=1, col=1)
        fig.add_trace(go.Scatter(x=g["step"], y=g["std_ratio_vs_sampled"], mode="lines+markers", name=var, showlegend=False), row=1, col=2)
        fig.add_trace(go.Scatter(x=g["step"], y=g["air_mean"], mode="lines+markers", name=var, showlegend=False), row=1, col=3)
    fig.add_hrect(y0=0.9, y1=1.1, fillcolor="green", opacity=0.1, line_width=0, row=1, col=2); fig.add_hline(y=POROSITY_GATE, line_dash="dash", row=1, col=1)
    fig.update_layout(height=380, title="Gates vs training step (32 volumes per variant per step)"); fig.show()
    last = recs[-1]
    print("inset-box surface probe at last step:", {k: last["inset_surface"].get(k) for k in ("air_fraction_outside_box", "air_fraction_inside_box")} if last.get("inset_surface") else None)
    print("kill switch:", last.get("killswitch", {}).get("ema"))
    print("conditioning alive (MAD of prediction change / noise MAD, %):", {k: round(v["pct_of_noise"], 2) for k, v in (last.get("alive_boundary") or {}).items() if isinstance(v, dict) and "pct_of_noise" in v})
''')


def build_decoder_ft() -> None:
    section("Decoder fine-tune (campaign 11)", "sharper grey from the same latents: gate table original vs fine-tuned decoder, and side-by-side slices")
    md("""
**What the decoder fine-tune is.** The diffusion model produces latents; the VAE decoder turns them into grey and
labels. The VAE was trained to *reconstruct*, which makes its output slightly blurrier than the scans. Decision D43:
after the author's visual inspection, fine-tune the decoder alone (encoder frozen, so the latent space and the 272 GB
store stay valid) with a sharpness-aware loss, then *re-decode the same eval-v4 latents*. Both decoders are reported
with pre-registered gates read at DDIM-50 and DDIM-200: porosity error must not get worse, the pore Dice against the
original decode must stay high, and the seam ratios must stay at the floor. This section fills in once campaign 11 exists.
""")
    code(r'''
if not DECODER_FT_ROOT.exists():
    unavailable("campaign 11-decoder-ft", "the author's visual go (runs/campaigns/decoder_ft_go) → decoder fine-tune → redecode")
else:
    for p in sorted(DECODER_FT_ROOT.rglob("*.md")): print("--", p.relative_to(DECODER_FT_ROOT)); display(Markdown(p.read_text()))
    for p in sorted(DECODER_FT_ROOT.rglob("*.json"))[:10]:
        try:
            J = load_json(p); print("--", p.relative_to(DECODER_FT_ROOT)); display(pd.json_normalize(J, sep=".").T.head(60))
        except Exception as exc: print(p, "unreadable:", exc)
    for p in sorted(DECODER_FT_ROOT.rglob("*.png"))[:12]: display(Image(filename=str(p)))
    _re = sorted(DECODER_FT_ROOT.glob("**/volumes/192_ddim200_seed101"))
    _or = case_dir("sampler", "192_ddim200_seed101")
    if _re and _or.exists(): compare_figure([_or, _re[0]], "z", 96, "grey", diff=True).show()
''')


def build_downstream() -> None:
    section("Downstream utility (campaign 14)", "a segmentation network trained on real, synthetic, or both, tested on real held-out scans")
    md("""
**The strongest realism test is use.** A 3-D segmentation network is trained three ways: (a) on real training volumes
with their labels, (b) on synthetic grey + label volumes only, (c) on both; each is tested on the *real* held-out test
panels, and pore and air Dice are reported with three seeds. If (b) approaches (a) the synthetic pairs carry the
information a segmenter needs; if (c) beats (a) the synthetic data adds something. This is the last stage in the queue
(it needs the GPU for training) and fills in when campaign 14 exists.
""")
    code(r'''
if not DOWNSTREAM_ROOT.exists():
    unavailable("campaign 14-downstream-utility", "downstream_utility (last GPU stage of the queue)")
else:
    for p in sorted(DOWNSTREAM_ROOT.rglob("*.md")): print("--", p.relative_to(DOWNSTREAM_ROOT)); display(Markdown(p.read_text()))
    for p in sorted(DOWNSTREAM_ROOT.rglob("results*.json"))[:5]:
        J = load_json(p); print("--", p.relative_to(DOWNSTREAM_ROOT))
        arms = J.get("arms") or J.get("per_arm") or J
        try: display(pd.json_normalize(arms, sep=".").T.head(80))
        except Exception: print(json.dumps(J, indent=1)[:3000])
    for p in sorted(DOWNSTREAM_ROOT.rglob("*.png"))[:12]: display(Image(filename=str(p)))
''')


def build_summary() -> None:
    section("Summary dashboard: claim → metric → value → floor → status", "one table that fills in as results land, and the list of what is still pending")
    md("""
**How to read it.** Each row is one claim the paper wants to make, the metric that supports it, the value from the
measured results (blank until measured), the real floor when one exists, and a status: *ok* when the value meets the
pre-registered reading, *check* when it does not, *pending* when the results file is not on disk. This table is built from
`results.json` files only, never from the sampler's own statistics, so it is empty until `eval_v4 measure` has run.
""")
    code(r'''
def _m(d, *path):
    for p in path:
        d = (d or {}).get(p) if isinstance(d, dict) else None
    return d.get("mean") if isinstance(d, dict) and "mean" in d else d

rows = []
def add(claim, metric, value, floor=None, ok=None):
    rows.append({"claim": claim, "metric": metric, "value": value, "floor": floor, "status": "pending" if value is None else ("ok" if ok else ("check" if ok is False else "—"))})

R = results("porosity_global")
v = _m(R, "dose_response", "slope") if R else None
add("global porosity control", "dose-response slope (in range)", v, "1.0", None if v is None else abs(v - 1) < 0.1)
v = (R or {}).get("dose_response", {}).get("frac_within_gate") if R else None
add("global porosity control", "fraction within ±0.005", v, "—", None if v is None else v >= 0.9)
R = results("porosity_local")
v = np.nanmean([_m(F, "within_volume_slope") for F in R["fields"].values()]) if R else None
add("local porosity control", "within-volume slope (mean over fields)", v, "1.0", None if v is None else v > 0.7)
R = results("sampler")
if R:
    k192 = next((k for k in R["cells"] if k.startswith("192x192")), None); k1024 = next((k for k in R["cells"] if k.startswith("1024") and k.endswith("ddim200")), None)
else: k192 = k1024 = None
v = _m(R["cells"][k192], "air_fraction_interior") if R and k192 else None; add("no interior air", "air fraction interior, 192³", v, "0.0000", None if v is None else v < 1e-4)
v = _m(R["cells"][k1024], "seam_chunk_xct_ratio") if R and k1024 else None
fl = _m(RF["by_shape"]["large"], "seam_chunk_xct_ratio") if RF else None
add("seam-free assembly at scale", "chunk-plane seam ratio, 1024 wide, DDIM-200", v, fl, None if v is None or fl is None else v < 1.1 * max(fl, 1.0))
R = results("assembly_modes")
if R:
    def cell(arm, sc): return next((c for c in R["cells"].values() if c["arm"] == arm and str(c["scale"]) == sc), None)
    h, j, a = cell("hybrid", "1024"), cell("joint", "1024"), cell("autoregressive", "1024")
    vh = _m(h, "volume_seam_xct_reference"); vj = _m(j, "volume_seam_xct_reference"); va = _m(a, "volume_seam_xct_reference")
    add("hybrid beats its parts", "reference-grid seam: hybrid vs joint / autoregressive (1024)", vh, f"joint {vj:.3f} / AR {va:.3f}" if vj and va else None, None if vh is None or vj is None or va is None else vh <= min(vj, va))
else: add("hybrid beats its parts", "reference-grid seam: hybrid vs joint / autoregressive", None)
R = results("geometry"); v = _m(R, "summary", "dice_air") if R else None; add("geometry control", "air Dice, notch + hole", v, "—", None if v is None else v > 0.95)
R = results("surface")
v = _m(R, "summary", "rough", "lower", "roughness_ratio_to_real_floor") if R else None; add("realistic surface", "rough request: Sa ratio to real floor (lower face)", v, "1.0", None if v is None else 0.5 <= v <= 2)
R = results("layup")
if R:
    for name, L in R["layups"].items():
        rd = L["readers"].get("pore_axes", {}); v = _m(rd, "median_abs_error_deg") if rd.get("available") else None
        add(f"layup {name} readable", "pore_axes median |error| deg", v, (rd.get("real_floor") or {}).get("median_abs_error_deg"), None if v is None else v <= 1.5 * ((rd.get("real_floor") or {}).get("median_abs_error_deg") or 4.2))
else: add("layup readable", "pore_axes median |error| deg", None, "4.2")
R = results("microstructure")
if R:
    for lvl, L in R["levels"].items():
        if L.get("available"):
            for stat in ("s2_w1", "psd_w1", "ripley_log_ratio", "fid"):
                v = L["ratio"].get(stat); add(f"microstructure realism φ={lvl}", f"{stat} ratio gen/real ÷ real/real", v, "1.0", None if v is None else v <= 2)
    Mm = R.get("memorisation") or {}
    v = _m(Mm, "generated", "latent", "frac_memorised") if Mm.get("available") else None; add("no memorisation", "fraction of patches with Favero ratio < 1/3 (latent)", v, _m(Mm, "real_val_floor", "latent", "frac_memorised") if Mm.get("available") else None, None if v is None else v == 0)
else:
    add("microstructure realism", "S2 / PSD / Ripley / FID ratios", None, "1.0"); add("no memorisation", "Favero ratio < 1/3 fraction", None, "val floor")
R = results("field_stats")
if R:
    for k, C in R["comparisons"].items():
        if "(real floor)" in k: continue
        v = C["marginal"]["w1"]; add("delivered field statistics", f"marginal W1: {k}", v, next((R["comparisons"][f]["marginal"]["w1"] for f in R["comparisons"] if f.startswith(k.split(" vs ")[1]) and "(real floor)" in f), None))
if (LABEL_ROOT / "results.json").exists():
    J = load_json(LABEL_ROOT / "results.json"); v = float(np.mean([x["summary"]["phi_range_sauvola_k_only"] for x in J["volumes"]])); add("label convention firmness (limitation)", "φ range for Sauvola k ±20 % at Otsu", v, "—", None)
SUMMARY = pd.DataFrame(rows); display(SUMMARY)
print("pending:", ", ".join(sorted({r["metric"] for r in rows if r["value"] is None})) or "nothing")
''')


def build_preview() -> None:
    section("Optional: the step-77k preview (NOT the paper)", "27 volumes generated by accident against an intermediate checkpoint; kept for the record only")
    md("""
The post-training runner once mistook a pause for the end of training and generated the sampler and microstructure cases
against step 77,000 of 130,000. They are kept under `preview_77k/` and nothing in them belongs in the paper. This cell only
lists them, so they cannot be confused with the final campaign in any table above (no other cell reads this folder).
""")
    code(r'''
pv = ROOT / "preview_77k"
if not pv.exists():
    print("preview_77k not on disk (excluded from the copy, or never generated)")
else:
    for a in ("sampler", "microstructure"):
        d = pv / a / "volumes"; print(f"PREVIEW 77k — {a}: {len(list(d.iterdir())) if d.exists() else 0} cases")
    if (pv / "README.md").exists(): display(Markdown((pv / "README.md").read_text()))
''')


# ===========================================================================
# assemble and write
# ===========================================================================

def build_all() -> None:
    build_setup()
    CELLS.append(("index", ""))          # placeholder, filled after sections are known
    build_config(); build_progress(); build_primer(); build_research_log(); build_case_reading(); build_slice_viewer(); build_compare_viewer()
    build_rungs(); build_real_floor(); build_sampler(); build_porosity_global(); build_porosity_local(); build_cfg(); build_layup()
    build_assembly(); build_geometry(); build_surface(); build_multichunk(); build_assembly_modes(); build_chunk_band(); build_rim_tests(); build_stress_geometry(); build_microstructure()
    build_field_stats(); build_label_uncertainty(); build_convergence(); build_decoder_ft(); build_downstream()
    build_summary(); build_preview()


def to_notebook() -> nbformat.NotebookNode:
    nb = new_notebook()
    nb.metadata["kernelspec"] = {"name": "python3", "display_name": "Python 3", "language": "python"}
    nb.metadata["language_info"] = {"name": "python"}
    for i, (kind, text) in enumerate(CELLS):
        if kind == "index":
            cell = new_markdown_cell(index_cell_text())
        elif kind == "md":
            cell = new_markdown_cell(text)
        else:
            cell = new_code_cell(text)
        cell["id"] = f"cell-{i:03d}"  # stable ids: a rebuild must not churn the committed copy
        nb.cells.append(cell)
    nbformat.validate(nb)
    return nb


def main() -> None:
    import argparse

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", type=Path, default=CAMPAIGN,
                    help="campaign directory to write the notebook and its "
                         "requirements into (default: the 12-eval-v4 campaign). "
                         "The committed copy under notebooks/ is written either "
                         "way, so a regenerated campaign gets its own notebook "
                         "without either copy going missing.")
    args = ap.parse_args()

    build_all()
    nb = to_notebook()
    out_campaign = args.root / OUT_CAMPAIGN.name
    out_campaign.parent.mkdir(parents=True, exist_ok=True)
    nbformat.write(nb, out_campaign)
    (args.root / OUT_REQ.name).write_text(REQUIREMENTS)
    OUT_REPO.parent.mkdir(parents=True, exist_ok=True)
    nbformat.write(nb, OUT_REPO)
    n_md = sum(1 for c in nb.cells if c.cell_type == "markdown"); n_code = len(nb.cells) - n_md
    print(f"wrote {out_campaign} and {OUT_REPO}: {len(nb.cells)} cells ({n_md} markdown, {n_code} code), {len(SECTIONS)} sections")


if __name__ == "__main__":
    main()
