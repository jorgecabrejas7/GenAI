#!/usr/bin/env python
"""Training and validation curves of a U-Net segmenter run (campaign 39).

    python scripts/analysis/plot_unet_seg_curves.py runs/unet_seg/<run> --out <png>
"""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("run", type=Path)
ap.add_argument("--out", type=Path, required=True)
a = ap.parse_args()
rows = [json.loads(line) for line in open(a.run / "metrics.jsonl")]
tr = [r for r in rows if r["split"] == "train"]
va = [r for r in rows if r["split"] == "val"]
s = np.array([r["step"] for r in tr])
vs = [r["step"] for r in va]


def sm(x, k=10):
    return np.convolve(np.asarray(x), np.ones(k) / k, "valid")


fig, ax = plt.subplots(2, 3, figsize=(15, 8))
p = ax[0, 0]
for k in ("loss", "ce", "dice"):
    p.plot(s, [r[k] for r in tr], alpha=.25)
    p.plot(s[9:], sm([r[k] for r in tr]), label=f"{k} (smoothed)")
p.set_yscale("log"); p.set_title("train loss"); p.legend()
p = ax[0, 1]
p.plot(vs, [r["dice_pore"] for r in va], "o-", label="val pore Dice")
b = int(np.argmax([r["dice_pore"] for r in va]))
p.annotate(f"best {va[b]['dice_pore']:.3f} @ {vs[b]}", (vs[b], va[b]["dice_pore"]),
           textcoords="offset points", xytext=(-60, -18), fontsize=8)
p.axhline(0.938, ls="--", c="g", label="VAE decoder, true masks in (0.938)")
p.axhline(0.63, ls=":", c="r", label="grey-only VAE encoder ge01 (0.63)")
p.set_title("pore Dice"); p.legend(fontsize=8)
p = ax[0, 2]
p.plot(vs, [r["dice_air"] for r in va], "o-", label="air")
p.plot(vs, [r["dice_material"] for r in va], "s-", label="material")
p.set_title("air / material Dice"); p.legend()
for p, key, title in ((ax[1, 0], "porosity", "porosity error per patch (fraction)"),
                      (ax[1, 1], "air", "air fraction error per patch")):
    p.plot(vs, [r[f"{key}_mae"] for r in va], "o-", label="MAE")
    p.plot(vs, [r[f"{key}_bias"] for r in va], "s-", label="bias")
    p.axhline(0, c="k", lw=.5); p.set_title(title); p.legend()
ax[1, 2].plot(s, [r["lr"] for r in tr]); ax[1, 2].set_title("learning rate")
for p in ax.flat:
    p.set_xlabel("step")
fig.suptitle(f"{a.run.name}: 3D U-Net segmenter, split_v4 (val = fixed patch subset), step {s[-1]}")
fig.tight_layout(); fig.savefig(a.out, dpi=110)
print("wrote", a.out)
