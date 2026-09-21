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


def held_out_volumes(eval_split: str, exclude_train_of: str | None) -> set[str] | None:
    """Volumes of ``eval_split`` that ``exclude_train_of`` did NOT train on.

    A model trained on one split and evaluated on another is not necessarily
    being evaluated on held-out data: SIX of split_v3's eleven validation
    volumes are in split_v2's TRAIN set. Scoring a split_v2-trained model on
    all eleven therefore flatters it. This names the subset it never saw.
    """
    if not exclude_train_of:
        return None
    ev = json.loads((REPO / "data" / eval_split / "splits.json").read_text())["volumes"]
    tr = json.loads((REPO / "data" / exclude_train_of / "splits.json").read_text())["volumes"]
    trained = {k for k, v in tr.items() if v == "train"}
    return {k for k, v in ev.items() if v == "val"} - trained


def evaluate(run: Path, split_root: str, n_batches: int, batch_size: int,
             exclude_train_of: str | None = None) -> dict:
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
    state = torch.load(run / "best.ckpt", map_location="cpu",
                       weights_only=False)["model"]
    # THE SVD BOTTLENECK'S inference_basis IS NOT A REGISTERED BUFFER — it is
    # assigned at finalisation, so a freshly built model has no slot for it and
    # a strict load refuses the checkpoint. It cannot simply be dropped: in
    # eval mode RRLayer PROJECTS ONTO IT, so a model loaded without it is a
    # different model and would be measured as one.
    basis_key = "bottleneck.rr.inference_basis"
    basis = state.pop(basis_key, None)
    missing, unexpected = model.load_state_dict(state, strict=False)
    if unexpected:
        raise RuntimeError(f"{run.name}: unexpected keys {list(unexpected)}")
    if basis is not None:
        model.bottleneck.rr.inference_basis = basis
    elif getattr(getattr(model, "bottleneck", None), "rr", None) is not None:
        raise RuntimeError(
            f"{run.name} has an RR bottleneck but its checkpoint carries no "
            f"{basis_key}; eval-mode projection would be undefined.")
    model.eval()
    _, val, _ = build_patch_dataloaders(cfg, REPO / "data" / split_root)

    # THE VAL LOADER SHUFFLES. Two runs of this script on one checkpoint gave
    # 0.0586 and 0.0626 for that reason alone, which is larger than some of the
    # differences the study wants to report. The dataset is iterated on a fixed
    # even stride instead, so the sample is the same patches every time.
    ds = val.dataset
    keep = held_out_volumes(split_root, exclude_train_of)
    rows_all = list(range(len(ds)))
    if keep is not None:
        vol = ds.df["volume_id"].to_numpy() if hasattr(ds, "df") else None
        if vol is None:
            raise RuntimeError("the dataset exposes no volume_id column to filter on")
        rows_all = [i for i in rows_all if vol[i] in keep]
        if not rows_all:
            raise RuntimeError(
                f"no {split_root} val patch is outside {exclude_train_of}'s train "
                "split, so there is no held-out subset to measure on")
    stride = max(1, len(rows_all) // (n_batches * batch_size))
    idx = rows_all[::stride][: n_batches * batch_size]

    def batches():
        for i in range(0, len(idx), batch_size):
            rows = [ds[j] for j in idx[i:i + batch_size]]
            yield {k: torch.stack([r[k] for r in rows])
                   for k in rows[0] if torch.is_tensor(rows[0][k])}

    # A VAE SAMPLES ITS POSTERIOR, and `model.eval()` does not stop it: z is
    # mu + sigma*eps on every forward pass. Fixing which patches are measured
    # is therefore only half of making this repeatable — two runs on the same
    # checkpoint and the same patches still moved vrrae04 from 13.5 % to 14.5 %
    # of baseline error removed. Seeding closes the other half.
    torch.manual_seed(0)

    l1, baseline, lo, hi, n = [], [], [], [], 0
    for batch in batches():
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

    if not l1:
        raise RuntimeError(f"{run.name}: no validation batches were built")
    mean_l1, mean_base = st.mean(l1), st.mean(baseline)
    return {
        "run": run.name,
        "trained_on": trained_on,
        "evaluated_on": split_root,
        "restricted_to_volumes_unseen_by": exclude_train_of,
        "n_volumes_used": (len(keep) if keep is not None else None),
        "n_patches": n,
        "deterministic_sample": True,
        "torch_seed": 0,
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
    ap.add_argument("--exclude-train-of", default=None,
                   help="restrict the val set to volumes this split did NOT train "
                        "on; use when the run was trained on a different split")
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    run = args.run if args.run.is_absolute() else REPO / args.run
    split = args.split or yaml.safe_load(
        (run / "resolved_config.yaml").read_text())["data"]["dataset_root"]
    res = evaluate(run, split, args.n_batches, args.batch_size,
                   args.exclude_train_of)

    print(f"{res['run'][:60]}")
    print(f"  trained on {res['trained_on']}, evaluated on {res['evaluated_on']}, "
          f"{res['n_patches']} patches")
    if res["restricted_to_volumes_unseen_by"]:
        print(f"  restricted to the {res['n_volumes_used']} volumes "
              f"{res['restricted_to_volumes_unseen_by']} did not train on")
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
