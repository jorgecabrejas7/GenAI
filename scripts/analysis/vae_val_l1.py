"""One VAE checkpoint's reconstruction L1 on one split's val set. (CPU.)

WHY THIS EXISTS RATHER THAN READING THE LOGS. Two runs' logged validation
numbers turned out not to be comparable, in two independent ways:

  * `vrrae04` logs `xct_loss` 0.078 and `mae` 0.172. Those are the SAME
    function on this code path — charbonnier at eps 1e-6 is L1 to six
    decimals, and the only other difference is a clamp that can only reduce
    the error — so a stable 2.18 ratio across every validation row is a
    logging artefact. Re-measuring agrees with `xct_loss` and not with `mae`.
  * `vrrae04` validated on `split_v2` while the r08 rungs use `split_v3`, and
    those sets are not equally hard: predicting the mean scores 0.153 on one
    and 0.066 on the other.

So this re-measures, on a split you name, and reports the CONSTANT-PREDICTION
BASELINE beside every number. A reconstruction error means nothing without it:
0.059 sounds close to 0.035 until you see that predicting the dataset mean
scores 0.066, at which point one model has removed 47 % of that error and the
other 11 %.

THE HARNESS IS VALIDATED BY REPRODUCING A RUN'S OWN LOGGED VALUE — run it on
an r08 rung and check the number against that run's `summary.json`.

Usage:
    python scripts/analysis/vae_val_l1.py --run runs/vae/<run> [--split split_v3]
"""

from __future__ import annotations

import argparse
import json
import statistics as st
from pathlib import Path

import torch
import torch.nn.functional as F
import yaml

REPO = Path(__file__).resolve().parents[2]


def evaluate(run: Path, split_root: str, n_batches: int, batch_size: int) -> dict:
    from poregen.experiments.train_vae import build_model
    from poregen.models.vae.base import decode_xct
    from poregen.training import build_patch_dataloaders
    from poregen.training.engine import to_device_inputs

    cfg = yaml.safe_load((run / "resolved_config.yaml").read_text())
    trained_on = cfg["data"].get("dataset_root")
    cfg["data"].update(batch_size=batch_size, num_workers=0, timeout=0,
                       persistent_workers=False, prefetch_factor=None,
                       dataset_root=split_root)
    dev = torch.device("cpu")
    model = build_model(cfg, dev)
    model.load_state_dict(torch.load(run / "best.ckpt", map_location="cpu",
                                     weights_only=False)["model"])
    model.eval()
    _, val, _ = build_patch_dataloaders(cfg, REPO / "data" / split_root)

    l1, baseline, lo, hi, n = [], [], [], [], 0
    it = iter(val)
    for _ in range(n_batches):
        batch = next(it)
        # Through the model's OWN input contract: a 3-class model takes the
        # label too, and calling it with the grey alone raises rather than
        # silently measuring something else.
        batch_dev, args = to_device_inputs(model, batch, dev)
        x = batch_dev["xct"]
        with torch.no_grad():
            out = model(*args)
        recon = decode_xct(out.xct_out)
        l1.append(float(F.l1_loss(recon, x)))
        # What predicting this batch's own mean would score. The floor any
        # reconstruction has to beat to have learned anything at all.
        baseline.append(float((x - x.mean()).abs().mean()))
        lo.append(float(out.xct_out.min())); hi.append(float(out.xct_out.max()))
        n += x.shape[0]

    mean_l1, mean_base = st.mean(l1), st.mean(baseline)
    return {
        "run": run.name,
        "trained_on": trained_on,
        "evaluated_on": split_root,
        "n_patches": n,
        "l1": mean_l1,
        "constant_prediction_baseline": mean_base,
        "fraction_of_baseline_error_removed": (
            (mean_base - mean_l1) / mean_base if mean_base else None),
        "raw_output_range": [min(lo), max(hi)],
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", type=Path, required=True)
    ap.add_argument("--split", default=None,
                    help="split root to evaluate on; default is the run's own")
    ap.add_argument("--n-batches", type=int, default=10)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    run = args.run if args.run.is_absolute() else REPO / args.run
    split = args.split or yaml.safe_load(
        (run / "resolved_config.yaml").read_text())["data"]["dataset_root"]
    res = evaluate(run, split, args.n_batches, args.batch_size)

    print(f"{res['run'][:60]}")
    print(f"  trained on {res['trained_on']}, evaluated on {res['evaluated_on']}, "
          f"{res['n_patches']} patches")
    print(f"  L1                                 {res['l1']:.5f}")
    print(f"  predicting the mean would score    {res['constant_prediction_baseline']:.5f}")
    print(f"  fraction of that error removed     "
          f"{res['fraction_of_baseline_error_removed']:.1%}")
    print(f"  raw decoder output range           "
          f"[{res['raw_output_range'][0]:.3f}, {res['raw_output_range'][1]:.3f}]")
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(res, indent=2) + "\n")
        print(f"  -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
