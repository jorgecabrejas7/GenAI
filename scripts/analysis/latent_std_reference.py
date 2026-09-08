"""What per-channel std should a correct ldm06 sample have?

The store's normalisation divides the posterior MEAN by ``per_channel_std``, so
mu-normalised latents have per-channel std 1 by construction. But ldm06 trains
on ``latent_mode: sampled`` — ``z = mu + sigma*eps``, drawn in that same
normalised space (``ldm_engine``: ``z = z + b["std"] * randn_like(z)``). The
sampled quantity is therefore WIDER than 1:

    std(z_sampled)^2 = std(mu_norm)^2 + E[(sigma / std_c)^2]

A generator that matches its training distribution must reproduce THAT, not 1.
Scoring generated latents against the mu-normalised reference builds the
posterior width into the error and reports a correct model as broken — which is
what a std-ratio gate of 1.0 +/- 10% does.

This measures both references from the store, per channel, so the diag can
report the ratio against the distribution the model was actually trained on.

    python scripts/analysis/latent_std_reference.py --n 50000
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--store", default="data/split_v3/latents_r08z8")
    ap.add_argument("--split", default="train")
    ap.add_argument("--n", type=int, default=8000,
                    help="random rows to draw. The statistic is a per-channel std over "
                         "n*4096 voxels, so it is converged to <1e-3 by ~4000 rows; the "
                         "default is deliberately small because this reads the same "
                         "latents.bin the training dataloader is using, and a 50k-row "
                         "pass slowed ldm06 from 1.74 to 2.43 s/step.")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="runs/campaigns/09-r08-latent-sweep/latent_std_reference.json")
    args = ap.parse_args()

    root = Path(args.store)
    meta = json.loads((root / "metadata.json").read_text())
    if meta["storage"]["pack_scheme"] != "mu_then_std":
        raise SystemExit(f"unexpected pack scheme {meta['storage']['pack_scheme']!r}")
    C, d, h, w = meta["latent_shape"]
    n_rows = meta["splits"][args.split]
    norm = meta["normalization"]
    mean_c = np.asarray(norm["per_channel_mean"], dtype=np.float32).reshape(C, 1, 1, 1)
    std_c = np.asarray(norm["per_channel_std"], dtype=np.float32).reshape(C, 1, 1, 1)

    lat = np.memmap(root / args.split / "latents.bin", dtype=np.float16, mode="r",
                    shape=(n_rows, 2 * C, d, h, w))

    rng = np.random.default_rng(args.seed)
    rows = np.sort(rng.choice(n_rows, size=min(args.n, n_rows), replace=False))

    # Streaming per-channel moments: 50k rows is 6.5 GB, and the answer is four
    # scalars per channel.
    acc = {k: np.zeros(C, dtype=np.float64) for k in
           ("mu_sum", "mu_sq", "sig_sq", "smp_sum", "smp_sq")}
    count = 0
    CHUNK = 256
    for i in range(0, len(rows), CHUNK):
        sel = rows[i:i + CHUNK]
        packed = np.asarray(lat[sel], dtype=np.float32)      # (B, 2C, d, h, w)
        mu, sigma = packed[:, :C], packed[:, C:]
        mu_n = (mu - mean_c) / std_c                          # exactly LatentDataset
        sig_n = sigma / std_c                                 # exactly LatentDataset
        # Closed form. E[z^2] = E[mu^2] + E[sigma^2] because eps is independent
        # of mu with unit variance, so drawing eps only adds Monte-Carlo noise
        # to an estimate that can be computed exactly. E[z] = E[mu].
        ax = (0, 2, 3, 4)
        acc["mu_sum"] += mu_n.sum(axis=ax)
        acc["mu_sq"] += (mu_n.astype(np.float64) ** 2).sum(axis=ax)
        acc["sig_sq"] += (sig_n.astype(np.float64) ** 2).sum(axis=ax)
        count += sel.size * d * h * w

    mu_mean = acc["mu_sum"] / count
    mu_std = np.sqrt(acc["mu_sq"] / count - mu_mean ** 2)
    sig_rms = np.sqrt(acc["sig_sq"] / count)
    smp_std = np.sqrt(acc["mu_sq"] / count + acc["sig_sq"] / count - mu_mean ** 2)
    predicted = np.sqrt(mu_std ** 2 + sig_rms ** 2)   # identical by construction

    out = {
        "store": str(root), "split": args.split,
        "rows_drawn": int(len(rows)), "voxels_per_channel": int(count),
        "mu_normalised_std": mu_std.tolist(),
        "posterior_sigma_rms_normalised": sig_rms.tolist(),
        "sampled_std": smp_std.tolist(),
        "sampled_std_predicted": predicted.tolist(),
        "mu_normalised_std_mean": float(mu_std.mean()),
        "sampled_std_mean": float(smp_std.mean()),
        "note": (
            "ldm06 trains on latent_mode='sampled', so sampled_std is the "
            "reference a generated latent must match. mu_normalised_std is ~1 "
            "by construction and is NOT the right reference."
        ),
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out, indent=2))

    # Write it back into the store, which is what the diag reads. A reference
    # kept only in a campaign file can be paired with the wrong store; one that
    # travels with the latents cannot.
    meta_path = root / "metadata.json"
    meta_full = json.loads(meta_path.read_text())
    meta_full["channel_stats"] = {
        "sampled_std": [round(float(v), 6) for v in smp_std],
        "sampled_std_mean": round(float(smp_std.mean()), 6),
        "posterior_sigma_rms_normalised": [round(float(v), 6) for v in sig_rms],
        "mu_normalised_std": [round(float(v), 6) for v in mu_std],
        "definition": (
            "Per-channel std of the SAMPLED latent z = mu + sigma*eps in "
            "NORMALISED units. This is the reference a generated latent must "
            "match when the LDM trains with data.latent_mode='sampled'. The "
            "mu-normalised std is ~1 by construction and is NOT the right "
            "reference."
        ),
        "provenance": f"{len(rows)} random {args.split} rows, seed {args.seed}, "
                      f"closed form E[z^2] = E[mu^2] + E[sigma^2] (no eps draw).",
    }
    meta_path.write_text(json.dumps(meta_full, indent=2))
    print(f"wrote channel_stats into {meta_path}")

    print(f"{len(rows)} rows, {count} voxels per channel\n")
    print(f"{'ch':>3}{'std(mu_norm)':>14}{'rms(sigma_norm)':>17}{'std(sampled)':>14}{'predicted':>11}")
    for c in range(C):
        print(f"{c:>3}{mu_std[c]:>14.4f}{sig_rms[c]:>17.4f}{smp_std[c]:>14.4f}{predicted[c]:>11.4f}")
    print(f"{'mean':>3}{mu_std.mean():>14.4f}{sig_rms.mean():>17.4f}"
          f"{smp_std.mean():>14.4f}{predicted.mean():>11.4f}")
    print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
