# Running the r08 rung sweep on a second GB10

Purpose: the five remaining r08 rungs on split_v4 (rf-2, rf-4, rf-16, rf-32, rf-64) on a second
machine identical to this one (GB10, 121 GB unified memory, Ubuntu 24.04 aarch64), so the paper's
six-rung table lands while this machine trains the LDM. Author's instruction 2026-10-08.
What goes over: the code and the split_v4 memmaps only. No latent store, no zarr, no runs.

Replace `REMOTE` below with the machine's address (192.168.8.x) and `USER` with the login there.

## A. On the remote: environment (once)

```bash
# miniforge, Python 3.13 — same as here (python 3.13.12, base env)
curl -L -o Miniforge3.sh https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-aarch64.sh
bash Miniforge3.sh -b -p ~/miniforge3 && ~/miniforge3/bin/conda init bash && exec bash
conda install -y python=3.13
# PyTorch for the GB10 (CUDA 13.0 build, as here: torch 2.12.1+cu130)
pip install torch==2.12.1 --index-url https://download.pytorch.org/whl/cu130
# torchvision: the matching cu130 wheel; see docs/DEVELOPMENT.md ("torchvision") for the exact command
# the rest, pinned to this machine's versions (preprocess_tools is NOT needed for VAE training)
mkdir -p ~/Dev && cd ~/Dev && git clone git@github.com:jorgecabrejas7/GenAI.git GenAI && cd GenAI
pip install -r requirements-lock-gb10.txt
pip install -e . --no-deps
python -c "import torch, poregen; print(torch.__version__, torch.cuda.get_device_name(0))"
```

## B. From this machine: copy the dataset (~1.35 TB; ~3.5 h at 1 Gb/s, ~25 min at 10 Gb/s)

```bash
cd /home/jorgecabrejas/Dev/GenAI
ssh USER@REMOTE 'mkdir -p ~/Dev/GenAI/data/split_v4'
rsync -avP --partial data/split_v4/patches_meta.json data/split_v4/patch_index.parquet \
  data/split_v4/class_weights.json data/split_v4/splits.json data/split_v4/index_report.json \
  data/split_v4/labels.json data/split_v4/build_report.md data/split_v4/holes.json \
  data/split_v4/orientation_field.json USER@REMOTE:~/Dev/GenAI/data/split_v4/
rsync -avP --partial data/split_v4/holes USER@REMOTE:~/Dev/GenAI/data/split_v4/
rsync -avP --partial data/split_v4/patches_xct.bin data/split_v4/patches_label.bin USER@REMOTE:~/Dev/GenAI/data/split_v4/
# the label store AND the grey arrays it links to (-L dereferences the per-volume xct symlinks into split_v1's zarr,
# so the remote needs no split_v1): the calibration probe reads it. ~290 GB.
rsync -avP --partial -L data/split_v4/volumes.zarr USER@REMOTE:~/Dev/GenAI/data/split_v4/
```

Not copied, on purpose: `latents_r08z8/` (the LDM's store), `raw_data/`, `runs/`, `data/split_v1` and `split_v3`
(the `-L` copy above carries the grey arrays the probe needs). The training loader uses the memmaps when the bins
are present and never opens the zarr.

## C. On the remote: verify before launching

```bash
cd ~/Dev/GenAI
ls -l data/split_v4/patches_xct.bin data/split_v4/patches_label.bin     # 564806287360 bytes each... (v4: 570691420160)
python - <<'PY'
import json, hashlib, pathlib
root = pathlib.Path("data/split_v4"); meta = json.loads((root/"patches_meta.json").read_text())
h = hashlib.sha256((root/"patch_index.parquet").read_bytes()).hexdigest()
print("parquet sha matches meta:", h == meta["parquet_sha256"], "| N:", meta["N"])
for f in ("patches_xct.bin", "patches_label.bin"):
    print(f, (root/f).stat().st_size == meta["N"] * meta["patch_size"]**3)
PY
CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=4 python -m pytest -q tests/test_split_switch.py
```

## C2. Same stack, same numbering

- Driver 580.142 with CUDA 13.0 here; a different driver or torch makes the rows "same config, different stack".
  `nvidia-smi --query-gpu=driver_version --format=csv,noheader` must say 580.142 (or record the difference).
- Set `vm.swappiness=10` as here (`sudo sysctl vm.swappiness=10`).
- Run numbers: the remote's `runs/vae` starts empty, so its first run is `r08-run-0001-…-dsv4`, which collides
  by number with this machine's. Pre-seed before launch so the numbers continue from here:
  `cd ~/Dev/GenAI && mkdir -p runs/vae && for i in $(seq -f "%04g" 1 13); do mkdir -p runs/vae/r08-run-$i-seed-placeholder; done`
  (the chain ignores these empty directories; the real runs become 0014 onward).
- Start `scripts/analysis/host_pressure_log.sh` beside each trainer, as here (the chain does this).

## D. On the remote: launch (tmux; one CUDA job)

```bash
cd ~/Dev/GenAI
tmux new-session -d -s rungs "POREGEN_SPLIT=split_v4 bash scripts/chain_rungs_remote.sh 2>&1 | tee runs/campaigns/chain_rungs.log"
tmux new-session -d -s tb "tensorboard --logdir runs/vae --port 6006 --bind_all"
```

`scripts/chain_rungs_remote.sh` runs the five rungs in order with the smoke test, the four checks
(rung report, calibration probe, harness, recon figure) into `-dsv4` directories, pin_memory on, and
honours `runs/campaigns/STOP_CHAIN`.

## E. From this machine: read the results without a Claude session there

Once, so the pull needs no password: `ssh-copy-id USER@REMOTE`. Then, at any time:

```bash
scripts/pull_remote_results.sh USER@REMOTE     # -> runs/remote/REMOTE/{vae,campaigns,chain.log}
```

TensorBoard on the remote is reachable at http://REMOTE:6006 from this network.
