"""What std should a correct ldm06 sample have?

The LDM trains on SAMPLED latents ``z = mu + sigma*eps`` divided by the store's
scalar ``sigma_hat`` (Rombach et al. 2022 Sec. G). ``sigma_hat`` is the std of
exactly those sampled train latents, so the reference a generated latent must
match is ~1 by construction. This script measures it from the store anyway —
overall and per channel — so a store whose metadata disagrees with its latents
shows up as a reference far from 1, and so the per-channel spread (which one
scalar does not equalise) is on record.

The diag (scripts/diag_ldm_samples.py) computes the same quantity itself with
the same function, ``poregen.diffusion.latents.sampled_std_reference``.

    python scripts/analysis/latent_std_reference.py --n 8000
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from poregen.diffusion.latents import sampled_std_reference
from poregen.paths import default_split, std_reference_path


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--store", default=f"data/{default_split()}/latents_r08z8")
    ap.add_argument("--split", default="train")
    ap.add_argument("--n", type=int, default=8000,
                    help="random rows to draw. The statistic is a std over n*4096 "
                         "voxels per channel, converged to <1e-3 by ~4000 rows; the "
                         "default is small because this reads the same latents.bin a "
                         "training dataloader may be using.")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=None,
                    help="default runs/campaigns/09-r08-latent-sweep/"
                         "latent_std_reference[-ds<v>].json, by the store's split")
    args = ap.parse_args()

    root = Path(args.store)
    out_path = Path(args.out) if args.out else std_reference_path(root)
    ref = sampled_std_reference(root, args.split, args.n, args.seed)
    ref["dataset_split"] = root.parent.name
    ref["note"] = ("std of (mu + sigma*eps) / sigma_hat, closed form "
                   "E[z^2] = E[mu^2] + E[sigma^2]; ~1 by construction on train.")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(ref, indent=2))

    print(f"{ref['rows']} {args.split} rows, sigma_hat {ref['sigma_hat']:.6f}")
    print(f"sampled std (rescaled): {ref['sampled_std']:.4f}   mean {ref['sampled_mean']:+.4f}")
    for c, v in enumerate(ref["sampled_std_per_channel"]):
        print(f"  ch{c}: {v:.4f}")
    print(f"\nwrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
