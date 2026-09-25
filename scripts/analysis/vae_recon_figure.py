#!/usr/bin/env python
"""Side-by-side reconstructions of several VAE checkpoints on the SAME held-out
patches, with each panel's L1. (CPU.)

Companion to ``vae_val_l1.py``: that script gives the number, this one shows
what the number looks like. Patches are the harness's deterministic held-out
sample (split_v3 val volumes that split_v2 never trained on), the posterior
sample is seeded, and the central z-slice of every 64^3 patch is shown in grey.

Usage:
    python scripts/analysis/vae_recon_figure.py \\
        --run r08=runs/vae/r08-run-0004-... --run "V0 (KL off)=runs/vae/vrrae-run-0001-..." \\
        --out runs/campaigns/28-vrrae-family/figures/recon_compare
Each --run is "label=path[:ckpt]"; ckpt defaults to best.ckpt.
"""
from __future__ import annotations
import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F

from vae_val_l1 import load_model_and_patches  # same dir

REPO = Path(__file__).resolve().parents[2]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", action="append", required=True, help='"label=path[:ckpt]"')
    ap.add_argument("--split", default="split_v3"); ap.add_argument("--exclude-train-of", default="split_v2")
    ap.add_argument("--n-patches", type=int, default=6)
    ap.add_argument("--out", required=True)
    ap.add_argument("--title", default="",
                    help="prepended to the caption, in capitals — use it to say when a\nfigure is NOT a result, e.g. a mid-training checkpoint on an interim basis")
    a = ap.parse_args()
    from poregen.models.vae.base import decode_xct
    from poregen.training.engine import to_device_inputs

    specs = []
    for r in a.run:
        label, rest = r.split("=", 1)
        path, ckpt = (rest.split(":", 1) + ["best.ckpt"])[:2]
        specs.append((label, REPO / path, ckpt))

    # the harness's sample: 20 batches x 32 = 640 patches; take every k-th for the figure
    rows, x_ref = [], None
    for label, run, ckpt in specs:
        model, ds, idx, cfg, _ = load_model_and_patches(run, a.split, 20, 32, a.exclude_train_of, ckpt)
        pick = idx[:: max(1, len(idx) // a.n_patches)][: a.n_patches]
        items = [ds[j] for j in pick]
        batch = {k: torch.stack([it[k] for it in items]) for k in items[0] if torch.is_tensor(items[0][k])}
        torch.manual_seed(0)
        batch_dev, args = to_device_inputs(model, batch, torch.device("cpu"))
        with torch.no_grad():
            out = model(*args)
        recon = decode_xct(out.xct_out)
        x = batch_dev["xct"]
        if x_ref is None:
            x_ref = x
        assert torch.equal(x, x_ref), "runs did not receive the same patches"
        l1 = [float(F.l1_loss(recon[i], x[i])) for i in range(x.shape[0])]
        # L1 is dominated by the patch's mean grey level and its air/material
        # boundary. Whether the TEXTURE (plies, pores) came back is a different
        # question: the correlation between reconstruction and input after each
        # patch's own mean is removed. A flat grey block scores ~0 here however
        # good its L1.
        def corr(r, t):
            r = r - r.mean(); t = t - t.mean()
            return float((r * t).sum() / (r.norm() * t.norm() + 1e-8))
        cc = [corr(recon[i], x[i]) for i in range(x.shape[0])]
        rows.append((label, recon, l1, cc))
        print(f"{label:<24} mean L1 {sum(l1) / len(l1):.4f}   mean texture corr {sum(cc) / len(cc):.3f}   on {x.shape[0]} patches")

    n = x_ref.shape[0]; z = x_ref.shape[-3] // 2
    vmin, vmax = float(x_ref.min()), float(x_ref.max())
    fig, axes = plt.subplots(1 + len(rows), n, figsize=(2.1 * n, 2.2 * (1 + len(rows))))
    for c in range(n):
        axes[0, c].imshow(x_ref[c, 0, z], cmap="gray", vmin=vmin, vmax=vmax); axes[0, c].set_title(f"patch {c + 1}", fontsize=9)
        for r, (label, recon, l1, cc) in enumerate(rows, start=1):
            axes[r, c].imshow(recon[c, 0, z], cmap="gray", vmin=vmin, vmax=vmax)
            axes[r, c].set_title(f"L1 {l1[c]:.3f}   corr {cc[c]:.2f}", fontsize=8)
    axes[0, 0].set_ylabel("input", fontsize=10)
    for r, (label, _, _, _) in enumerate(rows, start=1):
        axes[r, 0].set_ylabel(label, fontsize=10)
    for ax in axes.ravel():
        ax.set_xticks([]); ax.set_yticks([])
    caption = ("Held-out 64³ patches, central slice — input vs reconstruction "
               "(corr = texture correlation after removing the patch mean)")
    if a.title:
        caption = f"{a.title.upper()}\n{caption}"
    fig.suptitle(caption, fontsize=11)
    fig.tight_layout()
    out = Path(a.out); out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out.with_suffix(".png"), dpi=200); fig.savefig(out.with_suffix(".pdf"))
    print("wrote", out.with_suffix(".png"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
