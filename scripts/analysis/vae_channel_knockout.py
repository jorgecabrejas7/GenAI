#!/usr/bin/env python
"""Does a latent channel carry information the decoder uses? Knock it out and measure. (CPU, minutes.)

"Channels under the free-bits floor" counts channels whose MEAN KL per latent cell is below 0.1 nats.
A channel that encodes something sparse (pore positions occupy ~1.5 % of voxels) can average under the
floor and still matter. This script asks the decoder directly: on the harness's held-out patches, encode
to the posterior mean, replace ONE channel by the prior mean (zero), decode, and report how much the grey
L1 and the pore Dice move against the untouched decode. A channel whose knockout changes nothing carries
nothing the decoder uses; a channel the KL calls "collapsed" but whose knockout costs Dice is sparse, not empty.

    python scripts/analysis/vae_channel_knockout.py --run runs/vae/<run> [--ckpt best.ckpt] [--split split_v4]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "scripts" / "analysis"))


def main() -> int:
    from vae_val_l1 import load_model_and_patches  # noqa: E402
    from poregen.models.vae.base import decode_xct  # noqa: E402
    from poregen.paths import default_split  # noqa: E402

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", type=Path, required=True)
    ap.add_argument("--ckpt", default="best.ckpt")
    ap.add_argument("--split", default=default_split())
    ap.add_argument("--exclude-train-of", default="split_v2")
    ap.add_argument("--n-batches", type=int, default=10)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--out", type=Path, default=None)
    a = ap.parse_args()

    model, ds, idx, cfg, _ = load_model_and_patches(a.run, a.split, a.n_batches, a.batch_size,
                                                   a.exclude_train_of, a.ckpt)
    z_ch = int(cfg["model"]["z_channels"])

    def decode(z):
        dec = model.decoder(z)
        return decode_xct(model.xct_head(dec)), model.class_head(dec).argmax(1)

    tot = {"l1": torch.zeros(z_ch + 1), "pore_inter": torch.zeros(z_ch + 1), "pore_union": torch.zeros(z_ch + 1),
           "kl": torch.zeros(z_ch), "n": 0}
    with torch.no_grad():
        for i in range(0, len(idx), a.batch_size):
            rows = [ds[j] for j in idx[i:i + a.batch_size]]
            xct = torch.stack([r["xct"] for r in rows]).float()
            if xct.max() > 1.5:
                xct = xct / 255.0
            label = torch.stack([r["label"] for r in rows]).long()
            if label.ndim == xct.ndim:
                label = label[:, 0]
            pore_true = label == 1
            mu, logvar = model.encode_moments(xct, label)
            tot["kl"] += (0.5 * (mu.pow(2) + logvar.exp() - logvar - 1)).mean(dim=(0, 2, 3, 4))
            variants = [mu] + [torch.cat([mu[:, :c], torch.zeros_like(mu[:, c:c + 1]), mu[:, c + 1:]], 1)
                               for c in range(z_ch)]
            for k, z in enumerate(variants):
                x_hat, lab_hat = decode(z)
                tot["l1"][k] += (x_hat - xct).abs().mean().item() * xct.shape[0]
                pore_hat = lab_hat == 1
                tot["pore_inter"][k] += (pore_hat & pore_true).sum().item()
                tot["pore_union"][k] += pore_hat.sum().item() + pore_true.sum().item()
            tot["n"] += xct.shape[0]
    n = tot["n"]
    l1 = tot["l1"] / n
    dice = 2 * tot["pore_inter"] / tot["pore_union"].clamp(min=1)
    kl = tot["kl"] / (len(idx) // a.batch_size)
    rows = [{"channel": c, "kl_per_cell": round(kl[c].item(), 4), "under_floor_0.1": bool(kl[c] < 0.1),
             "l1_full": round(l1[0].item(), 5), "l1_knockout": round(l1[c + 1].item(), 5),
             "l1_increase": round((l1[c + 1] - l1[0]).item(), 5),
             "pore_dice_full": round(dice[0].item(), 4), "pore_dice_knockout": round(dice[c + 1].item(), 4),
             "pore_dice_drop": round((dice[0] - dice[c + 1]).item(), 4)} for c in range(z_ch)]
    print(f"{a.run.name[:40]}  {a.ckpt}  {n} held-out patches ({a.split}); full decode: L1 {l1[0]:.4f}, pore Dice {dice[0]:.4f}")
    print("ch  KL/cell  <floor  L1 +     pore Dice drop")
    for r in rows:
        print(f"{r['channel']:2d}  {r['kl_per_cell']:.3f}   {'yes' if r['under_floor_0.1'] else 'no ':3s}   {r['l1_increase']:+.4f}  {r['pore_dice_drop']:+.4f}")
    out = a.out or (a.run / f"channel_knockout_{a.ckpt.replace('.ckpt', '')}.json")
    out.write_text(json.dumps({"run": a.run.name, "ckpt": a.ckpt, "split": a.split, "n_patches": n,
                               "l1_full": l1[0].item(), "pore_dice_full": dice[0].item(), "channels": rows}, indent=2))
    print(f"-> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
