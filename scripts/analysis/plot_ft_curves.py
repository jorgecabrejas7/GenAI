#!/usr/bin/env python
"""Training curves of a campaign-40 fine-tune (U-Net air-only or SAM2), one PNG.

``--unet <run dir>``: metrics.jsonl of ``train_unet_air_ft`` (loss, CE, Dice
loss, hard air Dice on the training patches, lr).  ``--sam2 <run dir>``:
logs/log.txt of sam2 ``training/train.py`` (total loss every 10 steps, raw and
a 50-point moving mean, and GPU memory).  Steps on x; both panels on the
training data only (there is no validation split: all CuartoPulg is training).

    python scripts/analysis/plot_ft_curves.py --unet runs/unet_seg/uaf01-run-0001-... --out curves.png
    python scripts/analysis/plot_ft_curves.py --sam2 runs/sam2_ft/sf02-run-0001-large-cuarto-all --out curves.png
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--unet", type=Path)
    g.add_argument("--sam2", type=Path)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    if a.unet:
        rows = [json.loads(x) for x in (a.unet / "metrics.jsonl").read_text().splitlines()]
        s = [r["step"] for r in rows]
        for k in ("loss", "ce", "dice"):
            ax[0].plot(s, [r[k] for r in rows], label=k)
        ax[0].set_yscale("log"); ax[0].set_title("U-Net air fine-tune: training loss"); ax[0].legend()
        ax[1].plot(s, [r["air_dice_hard"] for r in rows], label="air Dice (train patches)")
        ax[1].set_ylim(0.8, 1.0); ax[1].set_title("hard air Dice on training patches")
        ax2 = ax[1].twinx(); ax2.plot(s, [r["lr"] for r in rows], "k:", label="lr"); ax2.set_ylabel("lr")
        ax[1].legend(loc="lower right")
        name = a.unet.name
    else:
        pat = re.compile(r"Train Epoch: \[(\d+)\]\[\s*(\d+)/(\d+)\].*Mem \(GB\): ([\d.]+).*Losses/train_all_loss: ([\d.e+-]+)")
        st, lo, mem = [], [], []
        for line in (a.sam2 / "logs/log.txt").read_text().splitlines():
            m = pat.search(line)
            if m:
                ep, i, n = int(m[1]), int(m[2]), int(m[3])
                st.append(ep * n + i); mem.append(float(m[4])); lo.append(float(m[5]))
        lo = np.array(lo)
        k = min(50, len(lo))
        ax[0].plot(st, lo, alpha=0.3, label="loss (every 10 steps)")
        ax[0].plot(st[k - 1:], np.convolve(lo, np.ones(k) / k, "valid"), label=f"moving mean ({k})")
        ax[0].set_yscale("log"); ax[0].set_title("SAM2 fine-tune: total training loss"); ax[0].legend()
        ax[1].plot(st, mem); ax[1].set_title("GPU memory (GB)")
        name = a.sam2.name
    for x in ax:
        x.set_xlabel("step")
    fig.suptitle(name)
    fig.tight_layout()
    a.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(a.out, dpi=110)
    print("wrote", a.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
