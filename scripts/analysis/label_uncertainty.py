"""How much of the reported porosity error is label noise?  (CPU, read-only)

Every porosity number in this project is measured against labels that
``poregen.dataset.segmentation.onlypores`` produced, and ``onlypores`` has
tunable parameters.  If moving those parameters inside a defensible range moves
porosity by more than the control error we report, the control error describes
the segmenter, not the model.  This script measures that range so the
limitations section can put a number beside every control-error figure.

What it does
------------
For each of three FULL real test volumes (not 192³ crops — a crop is not the
quantity the paper reports, and a crop also breaks ``material_mask``'s global
Otsu, as ``05-eval-v3-fixed-decode/onlypores/`` documents):

* re-runs the segmentation with ``sauvola_k`` at the production 0.125 and at
  ±20 % (0.100 / 0.150), crossed with the material-threshold method (Otsu —
  production — plus Yen and isodata),
* reports, per volume, the porosity range over the resulting variants, and
* reports the pore Dice BETWEEN the variants: how far the variants agree with
  each other, which is the label-noise floor on any Dice we quote.

Porosity is pore voxels / material voxels of that variant's own material mask —
the same ratio ``eval_v4.metrics.porosity_error`` calls *delivered*.  Pore Dice
is taken over the whole volume, so a pore that one variant places outside the
other's material envelope counts as a disagreement, which it is.

How it stays inside memory
--------------------------
A test volume is ~200 × 3400 × 1600 uint8, so nine full pore masks would be
9 × 1.05 GB of bool.  Two facts keep this small:

* Sauvola is a 2-D operation on each (Z, X) plane — ``sauvola_thresholding``
  loops over Y — so a Y slab of the crop gives bit-identical output to the whole
  crop.  The pore masks are therefore built one slab at a time and only the
  voxel counts and the pairwise intersection counts are kept.
* The material mask is global (Otsu over the whole crop, then a max-projection,
  then ``fill_voids``), so it is built once per method and held bit-packed.

Both claims are asserted in ``tests/test_label_uncertainty.py`` against the
production ``onlypores`` on a synthetic volume with a known answer.

The material thresholds come from a 256-bin histogram of the crop.  For uint8
data that histogram is lossless, it is exactly the one skimage builds itself,
and it is the only way to ask for a threshold method without materialising the
float copies ``threshold_li`` and ``threshold_triangle`` need — which is why
those two methods are not offered here.

Sauvola runs on the SEQUENTIAL path on purpose.  Its output is identical to the
parallel path (asserted in the tests); using it leaves the other cores for
whatever else is on the machine.  Run the whole thing under ``nice``.

Usage
-----
    nice -n 19 python scripts/analysis/label_uncertainty.py

Writes ``runs/campaigns/13-label-uncertainty/`` (results.json, findings.md).
Reads ``data/split_v3/volumes.zarr`` only; no GPU, no writes to ``data/``.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import zarr
from skimage import filters

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))

from poregen.dataset.segmentation import (  # noqa: E402
    content_bbox,
    material_mask,
    sauvola_thresholding_nonconcurrent,
)
from poregen.eval_v4.manifest import head_commit  # noqa: E402
from poregen.eval_v4.metrics import POROSITY_GATE  # noqa: E402
from poregen.eval_v4.real_floor import test_volume_ids  # noqa: E402

DATA_ROOT = REPO / "data" / "split_v3"
ZARR_ROOT = DATA_ROOT / "volumes.zarr"
OUT_DIR = REPO / "runs" / "campaigns" / "13-label-uncertainty"

#: Production defaults — ``dataset.io.compute_mask`` calls ``onlypores`` with
#: no arguments, so these are the settings every stored label was built with.
SAUVOLA_RADIUS = 30
SAUVOLA_K = 0.125
MIN_SIZE_FILTERING = -1

#: Material-threshold methods.  Every one of them is a global threshold that
#: skimage can take from a histogram, so none needs the volume in float.
MATERIAL_METHODS = {
    "otsu": filters.threshold_otsu,
    "yen": filters.threshold_yen,
    "isodata": filters.threshold_isodata,
}

#: Y voxels per slab.  ~40 MB of uint8 per slab on a full volume; the bool
#: intermediates of nine variants on top of it stay under a gigabyte.
SLAB_Y = 128


# ---------------------------------------------------------------------------
# Material threshold from the crop's histogram
# ---------------------------------------------------------------------------

def grey_histogram(volume: np.ndarray) -> np.ndarray:
    """256-bin histogram of a uint8 volume, one count per grey level."""
    if volume.dtype != np.uint8:
        raise TypeError(f"expected uint8, got {volume.dtype}")
    return np.bincount(volume.ravel(), minlength=256).astype(np.int64)


def material_threshold(counts: np.ndarray, method: str) -> float:
    """Global material threshold for *method*, taken from a grey histogram.

    The histogram is trimmed to the occupied grey levels because that is what
    ``skimage.exposure.histogram(..., source_range='image')`` does, and skimage's
    threshold functions divide by a cumulative count — a leading run of empty
    bins would make the first class weight zero and the criterion NaN.
    """
    occupied = np.flatnonzero(counts)
    if occupied.size == 0:
        raise ValueError("empty histogram: the crop has no voxels")
    lo, hi = int(occupied[0]), int(occupied[-1])
    hist = (counts[lo:hi + 1], np.arange(lo, hi + 1))
    return float(MATERIAL_METHODS[method](hist=hist))


# ---------------------------------------------------------------------------
# The measurement
# ---------------------------------------------------------------------------

def variant_name(k: float, method: str) -> str:
    return f"k{k:g}/{method}"


def measure(
    cropped: np.ndarray,
    *,
    sauvola_k: list[float],
    material_methods: list[str],
    sauvola_radius: int = SAUVOLA_RADIUS,
    slab: int = SLAB_Y,
) -> dict:
    """Porosity and pairwise pore Dice for every (sauvola_k × material) variant.

    *cropped* is a uint8 volume already cut to its ``content_bbox`` — the array
    ``onlypores`` works on after its own crop.

    Duplicates in *sauvola_k* or *material_methods* are kept, not collapsed: two
    variants asked for with the same parameters must come back with Dice 1.0,
    and the tests ask for exactly that.
    """
    unique_k = list(dict.fromkeys(sauvola_k))
    unique_methods = list(dict.fromkeys(material_methods))

    counts = grey_histogram(cropped)
    thresholds = {m: material_threshold(counts, m) for m in unique_methods}

    # One material mask per method, held bit-packed along X (1/8 of the bool).
    packed: dict[str, np.ndarray] = {}
    errors: dict[str, str] = {}
    for m in unique_methods:
        try:
            packed[m] = np.packbits(material_mask(cropped, threshold=thresholds[m]), axis=-1)
        except ValueError as exc:  # ambiguous specimen — a finding, not a crash
            errors[m] = str(exc)

    variants = [(k, m) for k in sauvola_k for m in material_methods if m in packed]
    n = len(variants)
    n_pore = np.zeros(n, np.int64)
    inter = np.zeros((n, n), np.int64)
    n_material = {m: 0 for m in packed}

    depth, height, width = cropped.shape
    for y0 in range(0, height, slab):
        sub = cropped[:, y0:y0 + slab, :]
        # Sauvola marks MATERIAL True; a pore is dark material inside the mask.
        dark = {
            k: ~sauvola_thresholding_nonconcurrent(sub, sauvola_radius, k)
            for k in unique_k
        }
        mats = {
            m: np.unpackbits(packed[m][:, y0:y0 + slab, :], axis=-1, count=width).astype(bool)
            for m in packed
        }
        for m, mask in mats.items():
            n_material[m] += int(np.count_nonzero(mask))

        pores = [dark[k] & mats[m] for k, m in variants]
        for i in range(n):
            n_pore[i] += np.count_nonzero(pores[i])
            for j in range(i + 1, n):
                inter[i, j] += np.count_nonzero(pores[i] & pores[j])
    np.fill_diagonal(inter, n_pore)
    inter = inter + np.triu(inter, 1).T

    records = []
    for i, (k, m) in enumerate(variants):
        records.append({
            "name": variant_name(k, m),
            "sauvola_k": float(k),
            "material_method": m,
            "material_threshold": thresholds[m],
            "n_pore": int(n_pore[i]),
            "n_material": int(n_material[m]),
            "phi": float(n_pore[i]) / n_material[m] if n_material[m] else float("nan"),
        })

    pair_sum = n_pore[:, None] + n_pore[None, :]
    with np.errstate(invalid="ignore", divide="ignore"):
        dice = np.where(pair_sum > 0, 2.0 * inter / pair_sum, np.nan)

    return {
        "material_thresholds": thresholds,
        "material_errors": errors,
        "sauvola_radius": sauvola_radius,
        "slab_y": slab,
        "crop_shape": [int(v) for v in cropped.shape],
        "crop_voxels": int(cropped.size),
        "variants": records,
        "dice_matrix": dice.tolist(),
    }


# ---------------------------------------------------------------------------
# Per-volume summary
# ---------------------------------------------------------------------------

def summarise_volume(result: dict, baseline: str) -> dict:
    """Porosity spread and variant agreement, as the paper would quote them."""
    records = result["variants"]
    names = [r["name"] for r in records]
    phi = np.array([r["phi"] for r in records], float)
    dice = np.array(result["dice_matrix"], float)

    def spread(mask: np.ndarray) -> float | None:
        sel = phi[mask]
        return float(sel.max() - sel.min()) if sel.size else None

    base_k = next((r["sauvola_k"] for r in records if r["name"] == baseline), None)
    base_m = next((r["material_method"] for r in records if r["name"] == baseline), None)
    k_axis = np.array([r["material_method"] == base_m for r in records])
    m_axis = np.array([r["sauvola_k"] == base_k for r in records])

    off = ~np.eye(len(records), dtype=bool)
    pairs = dice[off]
    ib = names.index(baseline) if baseline in names else None

    return {
        "baseline_variant": baseline,
        "phi_baseline": float(phi[ib]) if ib is not None else None,
        "phi_min": float(phi.min()),
        "phi_max": float(phi.max()),
        "phi_range": float(phi.max() - phi.min()),
        "phi_range_sauvola_k_only": spread(k_axis),
        "phi_range_material_only": spread(m_axis),
        "phi_range_over_baseline": (
            float((phi.max() - phi.min()) / phi[ib]) if ib is not None and phi[ib] else None
        ),
        "dice_min": float(np.nanmin(pairs)) if pairs.size else None,
        "dice_median": float(np.nanmedian(pairs)) if pairs.size else None,
        "dice_vs_baseline_min": (
            float(np.nanmin(dice[ib][off[ib]])) if ib is not None else None
        ),
    }


def _f(value, spec: str = ".5f") -> str:
    """Format a number for the report, or an em dash when it is missing."""
    return "—" if value is None else format(value, spec)


def findings_markdown(results: dict) -> str:
    """findings.md, derived from the numbers rather than restating them."""
    s = results["summary"]
    settings = results["settings"]
    if not s["n_volumes"]:
        return ("# 13 — Label uncertainty of the reported porosity\n\n"
                "No volume was segmented; see `results.json` for the per-volume "
                "errors.\n")
    lines = [
        "# 13 — Label uncertainty of the reported porosity",
        "",
        "How much does the porosity we report move when the segmentation "
        "parameters move inside a defensible range?  The answer is the label-noise "
        "floor under every control-error figure.",
        "",
        f"- full real test volumes segmented: **{s['n_volumes']}**",
        f"- variants per volume: **{s['n_variants']}** "
        f"(sauvola_k {settings['sauvola_k_values']} × material "
        f"{settings['material_methods']})",
        f"- widest porosity range on any volume: **{_f(s['phi_range_max'])}** "
        f"({s['phi_range_max_volume']})",
        f"- mean porosity range: **{_f(s['phi_range_mean'])}**",
        f"- porosity gate the eval suite scores against: **{s['porosity_gate']}** "
        f"→ label uncertainty is **{_f(s['phi_range_max_over_gate'], '.1%')}** of it",
        f"- lowest pore Dice between any two variants: **{_f(s['dice_min'], '.3f')}**; "
        f"median **{_f(s['dice_median'], '.3f')}**",
        "",
        "## Per volume",
        "",
        "| volume | φ baseline | φ min | φ max | φ range | range, k only | "
        "range, material only | Dice min | Dice median |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for v in results["volumes"]:
        if "summary" not in v:
            lines.append(f"| {v['volume_id'][-28:]} | — | — | — | — | — | — | — | "
                         f"{v.get('error', 'skipped')} |")
            continue
        u = v["summary"]
        lines.append(
            f"| {v['volume_id'][-28:]} | {_f(u['phi_baseline'])} | {_f(u['phi_min'])} | "
            f"{_f(u['phi_max'])} | {_f(u['phi_range'])} | "
            f"{_f(u['phi_range_sauvola_k_only'])} | {_f(u['phi_range_material_only'])} | "
            f"{_f(u['dice_min'], '.3f')} | {_f(u['dice_median'], '.3f')} |"
        )
    lines += [
        "",
        "## Per variant",
        "",
        "| volume | variant | material threshold | material voxels | pore voxels | φ |",
        "|---|---|---|---|---|---|",
    ]
    for v in results["volumes"]:
        for r in v.get("variants", []):
            lines.append(
                f"| {v['volume_id'][-28:]} | {r['name']} | {r['material_threshold']:.2f} | "
                f"{r['n_material']} | {r['n_pore']} | {r['phi']:.5f} |"
            )
    lines += [
        "",
        "## Reading it",
        "",
        "`φ range` is the full spread over every variant, so it carries both "
        "axes at once.  `range, k only` holds the material method at "
        f"`{settings['material_methods'][0]}` and `range, material only` holds "
        f"`sauvola_k` at {settings['sauvola_k_base']}, which says which knob the "
        "number is sensitive to.",
        "",
        "`Dice min` is the agreement between the two variants that agree least. "
        "It is measured over the whole volume, not inside a shared material mask, "
        "so a pore that one variant places outside the other's specimen envelope "
        "counts as a disagreement.  No pore Dice we report against these labels "
        "can mean more than this number.",
        "",
        "Source: `scripts/analysis/label_uncertainty.py`, "
        f"commit `{results['commit']}`.",
        "",
    ]
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--volumes", type=int, default=3,
                    help="how many test volumes to segment (default 3)")
    ap.add_argument("--volume-ids", nargs="*", default=None,
                    help="explicit volume ids, instead of the first --volumes test ids")
    ap.add_argument("--k-base", type=float, default=SAUVOLA_K,
                    help=f"production sauvola_k (default {SAUVOLA_K})")
    ap.add_argument("--k-frac", type=float, default=0.20,
                    help="fractional perturbation of sauvola_k (default 0.20)")
    ap.add_argument("--material-methods", nargs="+", default=["otsu", "yen", "isodata"],
                    choices=sorted(MATERIAL_METHODS), help="material-threshold methods")
    ap.add_argument("--slab", type=int, default=SLAB_Y, help="Y voxels per slab")
    ap.add_argument("--out", type=Path, default=OUT_DIR)
    args = ap.parse_args()

    logging.getLogger("poregen").setLevel(logging.WARNING)

    ks = [round(args.k_base * (1 - args.k_frac), 6), args.k_base,
          round(args.k_base * (1 + args.k_frac), 6)]
    baseline = variant_name(args.k_base, args.material_methods[0])

    ids = args.volume_ids or test_volume_ids(DATA_ROOT)[:args.volumes]
    store = zarr.open(str(ZARR_ROOT), mode="r")

    volumes = []
    t_start = time.time()
    for i, vid in enumerate(ids, 1):
        t0 = time.time()
        print(f"[{i}/{len(ids)}] {vid}", flush=True)
        if vid not in store:
            volumes.append({"volume_id": vid, "error": "not in volumes.zarr"})
            continue

        xct = np.asarray(store[vid]["xct"])
        bbox = content_bbox(xct)
        if bbox is None:
            volumes.append({"volume_id": vid, "error": "volume has no non-zero voxel"})
            continue
        z0, z1, y0, y1, x0, x1 = bbox
        cropped = np.ascontiguousarray(xct[z0:z1 + 1, y0:y1 + 1, x0:x1 + 1])
        del xct

        record = {"volume_id": vid, "shape": [int(v) for v in store[vid]["xct"].shape],
                  "content_bbox": [int(v) for v in bbox]}
        record.update(measure(cropped, sauvola_k=ks,
                              material_methods=args.material_methods, slab=args.slab))
        del cropped
        record["summary"] = summarise_volume(record, baseline)
        record["elapsed_s"] = round(time.time() - t0, 1)
        volumes.append(record)

        u = record["summary"]
        print(f"    φ {_f(u['phi_min'])}–{_f(u['phi_max'])} "
              f"(range {_f(u['phi_range'])}), Dice min {_f(u['dice_min'], '.3f')}, "
              f"{record['elapsed_s']} s", flush=True)

    done = [v for v in volumes if "summary" in v]
    ranges = [v["summary"]["phi_range"] for v in done]
    dmin = [v["summary"]["dice_min"] for v in done]
    dmed = [v["summary"]["dice_median"] for v in done]
    worst = max(done, key=lambda v: v["summary"]["phi_range"]) if done else None
    summary = {
        "n_volumes": len(done),
        "n_variants": len(done[0]["variants"]) if done else 0,
        "phi_range_max": max(ranges) if ranges else None,
        "phi_range_max_volume": worst["volume_id"] if worst else None,
        "phi_range_mean": float(np.mean(ranges)) if ranges else None,
        "porosity_gate": POROSITY_GATE,
        "phi_range_max_over_gate": max(ranges) / POROSITY_GATE if ranges else None,
        "dice_min": min(dmin) if dmin else None,
        "dice_median": float(np.median(dmed)) if dmed else None,
        "elapsed_s": round(time.time() - t_start, 1),
    }

    results = {
        "generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "commit": head_commit(REPO),
        "zarr_root": str(ZARR_ROOT),
        "settings": {
            "sauvola_radius": SAUVOLA_RADIUS,
            "sauvola_k_base": args.k_base,
            "sauvola_k_frac": args.k_frac,
            "sauvola_k_values": ks,
            "material_methods": args.material_methods,
            "min_size_filtering": MIN_SIZE_FILTERING,
            "baseline_variant": baseline,
            "slab_y": args.slab,
        },
        "volumes": volumes,
        "summary": summary,
    }

    out = args.out
    out.mkdir(parents=True, exist_ok=True)
    (out / "results.json").write_text(json.dumps(results, indent=2) + "\n")
    (out / "findings.md").write_text(findings_markdown(results))
    print("\n" + json.dumps(summary, indent=2))
    print(f"\nWrote {out}")
    return 0 if done else 1


if __name__ == "__main__":
    raise SystemExit(main())
