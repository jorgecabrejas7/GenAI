#!/usr/bin/env python
"""Surface agreement of an air mask with a reference mask, per slice along A (campaign 40).

SAM2 is prompted on the middle A slice only and propagates ~960 slices each
way; this checks whether its surfaces drift away from a reference (the U-Net,
which has no key slice) with distance from that middle slice.  For bins of A
(``--bin`` slices), prints and saves the median and p95 of |surface - reference
surface| for the upper and lower surface (longest non-air run per column,
``score_air.surfaces``), and a plot against A.

    python scripts/analysis/air_drift_by_slice.py --ref <unet air.tif> --mask <sam2 air.tif> \
        --thickness-axis 1 --out <dir>
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import tifffile  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from score_air import surfaces  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ref", type=Path, required=True)
    ap.add_argument("--mask", type=Path, required=True)
    ap.add_argument("--thickness-axis", type=int, default=1)
    ap.add_argument("--bin", type=int, default=50)
    ap.add_argument("--label", default="")
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    s = [surfaces(np.moveaxis(tifffile.imread(p) > 0, a.thickness_axis, 0)) for p in (a.ref, a.mask)]
    du, dl = np.abs(s[1][0] - s[0][0]).astype(float), np.abs(s[1][1] - s[0][1]).astype(float)  # (A, B)
    A = du.shape[0]
    rows = []
    for a0 in range(0, A, a.bin):
        u, l = du[a0:a0 + a.bin], dl[a0:a0 + a.bin]
        rows.append({"a0": a0, "a1": min(A, a0 + a.bin), "upper_median": float(np.median(u)),
                     "upper_p95": float(np.percentile(u, 95)), "lower_median": float(np.median(l)),
                     "lower_p95": float(np.percentile(l, 95))})
    (a.out / "drift.json").write_text(json.dumps({"ref": str(a.ref), "mask": str(a.mask), "bins": rows}, indent=1))
    x = [(r["a0"] + r["a1"]) / 2 for r in rows]
    fig, ax = plt.subplots(figsize=(10, 3.5))
    for k, c in (("upper", "tab:red"), ("lower", "tab:green")):
        ax.plot(x, [r[f"{k}_p95"] for r in rows], c=c, label=f"{k} p95")
        ax.plot(x, [r[f"{k}_median"] for r in rows], c=c, ls="--", label=f"{k} median")
    ax.axvline(A // 2, c="k", ls=":", label="prompted slice")
    ax.set_xlabel("A slice"); ax.set_ylabel("|surface - reference| (voxels)"); ax.legend(fontsize=8, ncol=5)
    ax.set_title(a.label or a.mask.parent.name)
    fig.tight_layout(); fig.savefig(a.out / "drift.png", dpi=100)
    for r in rows:
        print(r["a0"], r["upper_median"], r["upper_p95"], r["lower_median"], r["lower_p95"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
