"""Figure for reference_onlypores_audit.py: one through-thickness (z, x) slice per specimen.

Rows: XCT; reference sample mask (preprocess_tools.onlypores.material_mask);
stored sample mask (volumes.zarr 'sample_mask'); pores of each notebook
parameter set vs stored labels.  (The published figure at GenAI 826dbbe also had
the repo-root onlypores.py column mask; that program is deleted.)

Usage: python scripts/analysis/reference_onlypores_audit_plot.py <audit out dir> Na_04_2 Na_02_2
"""

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def _run_file(d, kind, run):
    """Direct-call output if present, else the composed-path output (validated identical on Na_04_2)."""
    f = d / f"{kind}_{run}.npz"
    return f if f.exists() else d / f"{kind}_{run}_composed.npz"


def plot_profiles(out, specs):
    """Per-z pore voxels: stored labels, and each notebook parameter set's shared / label-only split."""
    fig, axes = plt.subplots(1, len(specs), figsize=(7 * len(specs), 4), squeeze=False)
    for j, sp in enumerate(specs):
        ax = axes[0, j]
        base = np.load(_run_file(out / sp, "profile", "defaults"))
        ax.plot(base["labels_pores"], color="k", lw=1.5, label="stored labels")
        for run, c, t in (("single_notebook", "tab:blue", "onlypores.ipynb 30/0.125/8"),
                          ("notebook", "tab:red", "batch 15/0.2/8")):
            p = np.load(_run_file(out / sp, "profile", run))
            ax.plot(p["ref_pores"], color=c, lw=1.2, label=f"reference {t}")
            ax.plot(p["labels_only"], color=c, lw=0.8, ls="--", label=f"stored only vs {t}")
        w = __import__("json").load(open(out / sp / "walls.json"))["walls"]
        for z in (w["frontwall"], w["backwall"]):
            ax.axvline(z, color="gray", lw=0.8, ls=":")
        ax.set_yscale("symlog", linthresh=10)
        ax.set_xlabel("z (through-thickness slice)"); ax.set_ylabel("pore voxels per slice")
        ax.set_title(f"{sp}: dotted lines = reference walls", fontsize=9)
        ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(out / "pore_profiles_z.png", dpi=110)
    print(out / "pore_profiles_z.png")


def main():
    out = Path(sys.argv[1])
    specs = sys.argv[2:]
    fig, axes = plt.subplots(5, len(specs), figsize=(9 * len(specs), 9.5), squeeze=False)
    for j, sp in enumerate(specs):
        d = dict(np.load(out / sp / "slice_base.npz"))
        d["ref_sample"] = np.load(_run_file(out / sp, "slice", "notebook"))["sample"]
        for run in ("notebook", "single_notebook", "defaults"):
            d[f"ref_pores_{run}"] = np.load(_run_file(out / sp, "slice", run))["pores"]
        rows = [
            ("XCT", d["xct"], "gray"),
            (f"reference sample mask ({d['ref_sample'].mean():.1%} of slice)", d["ref_sample"], "gray"),
            (f"stored sample_mask ({d['stored_sample'].mean():.1%})", d["stored_sample"], "gray"),
        ]
        for i, (t, im, cm) in enumerate(rows):
            axes[i, j].imshow(im, cmap=cm, aspect="auto", interpolation="nearest")
            axes[i, j].set_title(f"{sp}  y={int(d['y'])}  {t}", fontsize=9)
        lab = d["stored_pores"]
        for i, (run, t) in enumerate((("notebook", "batch notebook 15/0.2/8"),
                                      ("single_notebook", "onlypores.ipynb 30/0.125/8")), start=3):
            ref = d[f"ref_pores_{run}"]
            rgb = np.zeros(ref.shape + (3,))
            rgb[ref & lab] = (1, 1, 1)
            rgb[ref & ~lab] = (1, 0.2, 0.2)
            rgb[lab & ~ref] = (0.2, 0.6, 1)
            axes[i, j].imshow(rgb, aspect="auto", interpolation="nearest")
            axes[i, j].set_title(f"{sp} pores, {t}: white both, red reference only, blue stored only", fontsize=9)
        for a in axes[:, j]:
            a.set_xlabel("x"); a.set_ylabel("z")
    fig.tight_layout()
    fig.savefig(out / "sample_masks_zx.png", dpi=110)
    print(out / "sample_masks_zx.png")
    plot_profiles(out, specs)


if __name__ == "__main__":
    main()
