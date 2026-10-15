#!/usr/bin/env python
"""Three orthogonal cross-sections of chosen val patches, input vs each model's
reconstruction. (CPU.)

Where ``vae_recon_figure.py`` shows the central z-slice of an even-stride
sample, this one picks patches BY CONTENT from the split's val rows, so the
figure shows one patch of each kind the author asked for:

- ``low``   : no air (air_fraction == 0), porosity at the 2nd percentile of
              those rows ("almost no pores", not a single-voxel outlier),
- ``high``  : no air, porosity at the 99th percentile ("really porous"),
- ``air``   : air_fraction closest to 0.4 (specimen edge, air and material).

For each patch the three planes through the patch centre are shown (axis 0,
1 and 2), so a model that is right in one plane only is visible. Per patch
and model the title gives the L1 and the texture correlation (correlation of
reconstruction and input after each patch's own mean is removed) over the
whole 64^3 patch. The posterior is sampled with seed 0, as in the harness.

Usage:
    python scripts/analysis/vae_recon_cross_sections.py \\
        --run "rf-8=runs/...run-0014..." --run "rf-8 adv+perc=runs/...:step.ckpt" \\
        --out runs/campaigns/32-texture-fidelity/advperc_compare/cross_sections
Each --run is "label=path[:ckpt]"; ckpt defaults to best.ckpt.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F

from vae_val_l1 import load_model_and_patches  # same dir
from poregen.paths import default_split

REPO = Path(__file__).resolve().parents[2]
KINDS = ("low", "high", "air")
TITLES = {"low": "almost no pores", "high": "very porous", "air": "with air"}


def pick_rows(df) -> dict[str, int]:
    """Dataset row of each patch kind (see the module docstring)."""
    solid = df[df["air_fraction"] == 0.0]
    out = {}
    for kind, q in (("low", 0.02), ("high", 0.99)):
        target = float(solid["porosity"].quantile(q))
        out[kind] = int((solid["porosity"] - target).abs().idxmin())
    out["air"] = int((df["air_fraction"] - 0.4).abs().idxmin())
    return out


def corr(r: torch.Tensor, t: torch.Tensor) -> float:
    r = r - r.mean(); t = t - t.mean()
    return float((r * t).sum() / (r.norm() * t.norm() + 1e-8))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", action="append", required=True, help='"label=path[:ckpt]"')
    ap.add_argument("--split", default=default_split())
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    from poregen.models.vae.base import decode_xct
    from poregen.training.engine import to_device_inputs

    rows, x_ref, picked = [], None, None
    for spec in a.run:
        label, rest = spec.split("=", 1)
        parts = rest.split(":", 1)
        run, ckpt = REPO / parts[0], (parts[1] if len(parts) > 1 else "best.ckpt")
        model, ds, _, _, _ = load_model_and_patches(run, a.split, 1, 1, None, ckpt)
        if picked is None:
            picked = pick_rows(ds.df)
            for k in KINDS:
                r = ds.df.iloc[picked[k]]
                print(f"{k:<5} row {picked[k]}  {r['volume_id']}  z0 {r['z0']} y0 {r['y0']} x0 {r['x0']}"
                      f"  porosity {r['porosity']:.4f}  air {r['air_fraction']:.3f}")
        items = [ds[picked[k]] for k in KINDS]
        batch = {k: torch.stack([it[k] for it in items]) for k in items[0] if torch.is_tensor(items[0][k])}
        torch.manual_seed(0)
        batch_dev, args = to_device_inputs(model, batch, torch.device("cpu"))
        with torch.no_grad():
            recon = decode_xct(model(*args).xct_out)
        x = batch_dev["xct"]
        if x_ref is None:
            x_ref = x
        assert torch.equal(x, x_ref), "runs did not receive the same patches"
        l1 = [float(F.l1_loss(recon[i], x[i])) for i in range(len(KINDS))]
        cc = [corr(recon[i], x[i]) for i in range(len(KINDS))]
        rows.append((label, recon, l1, cc))
        print(f"{label:<24} " + "  ".join(f"{k}: L1 {l1[i]:.4f} corr {cc[i]:+.3f}" for i, k in enumerate(KINDS)))

    c = x_ref.shape[-1] // 2
    planes = [lambda v: v[c], lambda v: v[:, c], lambda v: v[:, :, c]]
    plane_names = ("axis 0", "axis 1", "axis 2")
    n_cols = len(KINDS) * 3
    fig, axes = plt.subplots(1 + len(rows), n_cols, figsize=(1.9 * n_cols, 2.1 * (1 + len(rows))))
    for p, kind in enumerate(KINDS):
        vol = x_ref[p, 0]
        vmin, vmax = float(vol.min()), float(vol.max())
        for j, (plane, pname) in enumerate(zip(planes, plane_names)):
            col = 3 * p + j
            axes[0, col].imshow(plane(vol).numpy(), cmap="gray", vmin=vmin, vmax=vmax)
            axes[0, col].set_title(f"{TITLES[kind]}\n{pname}" if j == 1 else pname, fontsize=8)
            for r, (label, recon, l1, cc) in enumerate(rows, start=1):
                axes[r, col].imshow(plane(recon[p, 0]).numpy(), cmap="gray", vmin=vmin, vmax=vmax)
                if j == 1:
                    axes[r, col].set_title(f"L1 {l1[p]:.3f}  corr {cc[p]:.2f}", fontsize=8)
    axes[0, 0].set_ylabel("input", fontsize=9)
    for r, (label, _, _, _) in enumerate(rows, start=1):
        axes[r, 0].set_ylabel(label, fontsize=9)
    for ax in axes.ravel():
        ax.set_xticks([]); ax.set_yticks([])
    fig.suptitle("Val 64³ patches, three planes through the patch centre: input vs reconstruction "
                 "(grey range per patch from the input; L1 and corr over the whole patch)", fontsize=10)
    fig.tight_layout()
    out = Path(a.out); out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out.with_suffix(".png"), dpi=200); fig.savefig(out.with_suffix(".pdf"))
    print("wrote", out.with_suffix(".png"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
