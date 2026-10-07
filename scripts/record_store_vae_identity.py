#!/usr/bin/env python
"""Record the sha256 and step of the VAE checkpoint a latent store names.

Stores built before ``build_latent_dataset.py`` recorded the checkpoint's
identity carry only its path, and ``train_ldm`` now refuses a store without
it. This writes ``vae_checkpoint_sha256`` and ``vae_checkpoint_step`` into
``<store>/metadata.json`` from the file the store names — correct ONLY while
that file still holds the weights that built the store. It refuses if the
checkpoint's run has been resumed since the store was created.

    python scripts/record_store_vae_identity.py --store data/split_v4/latents_r08z8

``--ldm-run <run dir>`` instead records, in that LDM run's run_metadata.json,
the sha256 its store holds now — for runs trained before LDM runs recorded it
themselves, and only while the store is still the one the run trained on.
"""
from __future__ import annotations

import argparse
import datetime
import json
import re
from pathlib import Path

from poregen.training.checkpoint import checkpoint_identity


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--store", type=Path)
    g.add_argument("--ldm-run", type=Path)
    args = ap.parse_args()
    if args.ldm_run is not None:
        return record_ldm_run(args.ldm_run)
    store = args.store
    meta_path = store / "metadata.json"
    meta = json.loads(meta_path.read_text())
    ckpt = Path(meta["vae_checkpoint"])
    created = datetime.datetime.fromisoformat(meta["created"])
    run_meta = json.loads((ckpt.parent / "run_metadata.json").read_text())
    if "resume_step" in run_meta and ckpt.stat().st_mtime > created.timestamp():
        raise SystemExit(f"{ckpt} was rewritten by a resume after the store was "
                         "built; its weights are not the store's.")
    ident = checkpoint_identity(ckpt)
    meta["vae_checkpoint_sha256"] = ident["sha256"]
    meta["vae_checkpoint_step"] = ident["step"]
    meta_path.write_text(json.dumps(meta, indent=2))
    print(f"{store}: {ckpt.name} step {ident['step']} sha256 {ident['sha256']}")
    return 0


def record_ldm_run(run_dir: Path) -> int:
    import yaml

    cfg = yaml.safe_load((run_dir / "resolved_config.yaml").read_text())
    store = Path(cfg["data"]["latents_root"])
    meta = json.loads((store / "metadata.json").read_text())
    created = datetime.datetime.fromisoformat(meta["created"])
    run_meta_path = run_dir / "run_metadata.json"
    run_meta = json.loads(run_meta_path.read_text())
    # The run name carries its launch time (runtime.run_name.timestamp_format).
    stamp = re.search(r"-(\d{8}-\d{6})-", run_dir.name)
    if stamp is None:
        raise SystemExit(f"{run_dir.name} carries no launch timestamp")
    started = datetime.datetime.strptime(stamp.group(1), "%Y%m%d-%H%M%S")
    if started < created:
        raise SystemExit(f"{store} was built after {run_dir.name} started; it is "
                         "not the store the run trained on.")
    run_meta["vae_checkpoint_sha256"] = meta["vae_checkpoint_sha256"]
    run_meta_path.write_text(json.dumps(run_meta, indent=2))
    print(f"{run_dir.name}: vae_checkpoint_sha256 {meta['vae_checkpoint_sha256']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
