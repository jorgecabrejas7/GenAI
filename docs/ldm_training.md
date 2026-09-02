# LDM Training

All commands assume you are at the repo root `/home/jorgecabrejas/Dev/GenAI` and the `poregen` mamba environment is active.

```bash
cd /home/jorgecabrejas/Dev/GenAI
mamba activate poregen
```

---

## New training run

Create a detached tmux session and launch training inside it.

```bash
tmux new-session -d -s ldm-train
tmux send-keys -t ldm-train "cd /home/jorgecabrejas/Dev/GenAI && mamba activate poregen && python scripts/train_ldm.py run ldm04/base" Enter
```

Attach to watch the output:

```bash
tmux attach -t ldm-train
# detach without killing: Ctrl-b  d
```

---

## Resume from latest checkpoint

Replace `<run_dir>` with the actual run directory under `runs/ldm/`, e.g. `runs/ldm/ldm01/20240527-143012-0001-z16-c128-s32-bs16-lr1e-4`.

```bash
tmux new-session -d -s ldm-resume
tmux send-keys -t ldm-resume "cd /home/jorgecabrejas/Dev/GenAI && mamba activate poregen && python scripts/train_ldm.py resume <run_dir>" Enter
```

Resume from a specific checkpoint:

```bash
python scripts/train_ldm.py resume <run_dir> checkpoints/ldm_step00050000.ckpt
```

---

## TensorBoard

```bash
tmux new-session -d -s ldm-tb
tmux send-keys -t ldm-tb "cd /home/jorgecabrejas/Dev/GenAI && mamba activate poregen && tensorboard --logdir runs/ldm --port 6006 --bind_all" Enter
# open http://localhost:6006
```

---

## List runs

```bash
ls runs/ldm/
```

---

## Build the latent dataset (if re-encoding is needed)

Requires a trained VAE checkpoint. The store records the checkpoint and the
train-split per-channel normalisation stats in `metadata.json`.

```bash
python scripts/build_latent_dataset.py \
    --checkpoint runs/vae/<vae_run>/best.ckpt \
    --output data/split_v2/latents_r07z4 \
    --batch-size 256
```
