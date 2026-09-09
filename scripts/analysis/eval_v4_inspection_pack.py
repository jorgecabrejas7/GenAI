"""A look-at-it pack: middle slices of the generated volumes, as PNGs.

Every other artefact in the campaign is a number. This one exists so a person
can see the volumes before the decoder fine-tune starts — a model can pass a
porosity gate and a seam gate and still be producing something obviously wrong
to a human eye, and no metric in eval v4 is designed to catch "that does not
look like a laminate".

CPU only, and deliberately cheap: it reads the middle z, y and x slice of each
selected case rather than the whole volume, so a 1024x1024x192 case costs three
slices, not 800 MB.

    python scripts/analysis/eval_v4_inspection_pack.py --root runs/campaigns/12-eval-v4
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

import matplotlib                                        # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt                          # noqa: E402
from matplotlib.colors import ListedColormap             # noqa: E402

logger = logging.getLogger("inspection_pack")

#: The eight cases the pack shows, in the order a reader should look at them.
#: The sampler triple comes first because the step count is the open question
#: on this model, and the two 1024 rows answer whether it survives scale.
SELECTION: tuple[tuple[str, str, str], ...] = (
    ("sampler", "192_ddim50_seed101", "step count: is 50 enough?"),
    ("sampler", "192_ddim100_seed101", "step count: 100"),
    ("sampler", "192_ddim200_seed101", "step count: 200"),
    ("sampler", "1024_ddim50_seed101", "full panel width at 50 steps"),
    ("sampler", "1024_ddim200_seed101", "full panel width at 200 steps"),
    ("surface", "flat_192_ddim50_seed101", "air above and below a requested box"),
    ("geometry", "notch_hole_seed101", "a notch and a drilled hole"),
    ("porosity_local", "checkerboard_ddim50_seed101", "painted porosity field"),
    ("surface", "rough_192_ddim50_seed101", "a ROUGH surface request"),
    ("geometry", "sphere_192_ddim50_seed101", "a curved specimen it never saw"),
    ("geometry", "sphere_256_ddim50_seed101", "the same sphere at 256 cubed"),
)

#: material / pore / air. Grey, red, blue — the pore class is the one a reader
#: is looking for, so it gets the colour that carries furthest.
LABEL_CMAP = ListedColormap([(0.75, 0.75, 0.75), (0.85, 0.10, 0.10), (0.20, 0.40, 0.85)])


def _mid_slices(path: Path, is_label: bool) -> dict[str, np.ndarray] | None:
    """The middle z, y and x slice of one TIFF, read without loading the volume."""
    import tifffile                                      # noqa: PLC0415

    if not path.exists():
        return None
    with tifffile.TiffFile(str(path)) as tf:
        series = tf.series[0]
        d, h, w = series.shape
        # z is one page; y and x need a column through every page, so they are
        # read page by page rather than by materialising the volume.
        z = series.asarray(key=d // 2)
        ys, xs = [], []
        for k in range(d):
            page = series.asarray(key=k)
            ys.append(page[h // 2, :])
            xs.append(page[:, w // 2])
    return {"z": np.asarray(z), "y": np.asarray(ys), "x": np.asarray(xs)}


def _title(assessment: str, case: str, why: str, manifest: dict) -> str:
    bits = [f"{assessment}/{case}", why]
    steps = manifest.get("ddim_steps")
    if steps:
        bits.append(f"DDIM-{steps}")
    phi = manifest.get("requested_global_phi")
    if phi is not None:
        bits.append(f"requested phi {phi:g}")
    notes = manifest.get("notes") or {}
    for k in ("field", "features", "z_lo", "scale"):
        if k in notes:
            bits.append(f"{k}={notes[k]}")
    shape = manifest.get("volume_shape")
    if shape:
        bits.append("x".join(str(v) for v in shape))
    seed = manifest.get("seed")
    if seed is not None:
        bits.append(f"seed {seed}")
    return "  |  ".join(str(b) for b in bits)


def render_case(case_dir: Path, assessment: str, case: str, why: str,
                out: Path) -> dict | None:
    mf_path = case_dir / "manifest.json"
    if not mf_path.exists():
        logger.warning("%s: no manifest — not generated", case_dir)
        return None
    manifest = json.loads(mf_path.read_text())
    grey = _mid_slices(case_dir / "volume.tif", is_label=False)
    label = _mid_slices(case_dir / "label.tif", is_label=True)
    if grey is None or label is None:
        logger.warning("%s: volume.tif or label.tif missing", case_dir)
        return None

    fig, axes = plt.subplots(2, 3, figsize=(15, 9))
    for col, ax_name in enumerate(("z", "y", "x")):
        # aspect="equal", not "auto". This pack exists to be LOOKED at, and a
        # stretched panel misrepresents pore shape and ply thickness — the two
        # things a reader is judging. For the 1024-wide cases the mid-y and
        # mid-x slices are genuinely 192x1024, so they render as thin strips;
        # that is the honest shape of the data, not a defect in the figure.
        axes[0, col].imshow(grey[ax_name], cmap="gray", vmin=0, vmax=255,
                            interpolation="nearest", aspect="equal")
        axes[0, col].set_title(f"grey, mid-{ax_name}")
        axes[1, col].imshow(label[ax_name], cmap=LABEL_CMAP, vmin=0, vmax=2,
                            interpolation="nearest", aspect="equal")
        axes[1, col].set_title(f"label, mid-{ax_name}")
        for r in (0, 1):
            axes[r, col].set_xticks([]); axes[r, col].set_yticks([])
    fig.suptitle(_title(assessment, case, why, manifest), fontsize=11)
    fig.text(0.5, 0.015, "label: grey = material, red = pore, blue = air",
             ha="center", fontsize=9)
    fig.tight_layout(rect=(0, 0.03, 1, 0.96))
    png = out / f"{assessment}__{case}.png"
    fig.savefig(png, dpi=110)
    plt.close(fig)
    logger.info("wrote %s", png.name)

    lab_z = label["z"]
    return {
        "assessment": assessment, "case": case, "why": why,
        "case_dir": str(case_dir), "png": str(png),
        "ddim_steps": manifest.get("ddim_steps"),
        "volume_shape": manifest.get("volume_shape"),
        "seed": manifest.get("seed"),
        "requested_global_phi": manifest.get("requested_global_phi"),
        "mid_z_pore_fraction": float((lab_z == 1).mean()),
        "mid_z_air_fraction": float((lab_z == 2).mean()),
        "tiffs": {"grey": str(case_dir / "volume.tif"),
                  "label": str(case_dir / "label.tif")},
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", default="runs/campaigns/12-eval-v4")
    ap.add_argument("--out", default=None, help="default: <root>/inspection")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    from poregen.eval_v4.io import case_dir            # noqa: PLC0415

    root = Path(args.root)
    out = Path(args.out) if args.out else root / "inspection"
    out.mkdir(parents=True, exist_ok=True)

    rows, missing = [], []
    for assessment, case, why in SELECTION:
        cd = case_dir(root, assessment, case)
        row = render_case(cd, assessment, case, why, out)
        (rows if row else missing).append(row or f"{assessment}/{case}")

    manifest = {
        "root": str(root),
        "n_rendered": len(rows),
        "n_missing": len(missing),
        "missing": missing,
        "how_to_get_the_volumes": (
            "Each entry's case_dir holds volume.tif (uint8 grey, the real "
            "scanner scale) and label.tif (uint8 0=material 1=pore 2=air). "
            "Copy a case with: scp -r <host>:<case_dir> ."
        ),
        "cases": rows,
    }
    (out / "inspection_manifest.json").write_text(json.dumps(manifest, indent=2))

    lines = ["# Inspection pack", "",
             f"{len(rows)} of {len(SELECTION)} selected cases rendered.", "",
             "| case | why it is here | steps | shape | PNG |", "|---|---|---|---|---|"]
    for r in rows:
        lines.append(f"| `{r['assessment']}/{r['case']}` | {r['why']} | "
                     f"{r['ddim_steps']} | "
                     f"{'x'.join(str(v) for v in (r['volume_shape'] or []))} | "
                     f"`{Path(r['png']).name}` |")
    if missing:
        lines += ["", "Not generated yet:", ""] + [f"- `{m}`" for m in missing]
    lines += ["", "Label colours: grey = material, red = pore, blue = air.", "",
              "The TIFFs are beside each case; see `inspection_manifest.json` "
              "for the exact paths."]
    (out / "README.md").write_text("\n".join(lines) + "\n")

    logger.info("pack: %d rendered, %d missing -> %s", len(rows), len(missing), out)
    return 0 if rows else 1


if __name__ == "__main__":
    raise SystemExit(main())
