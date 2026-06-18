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
tmux send-keys -t ldm-train "cd /home/jorgecabrejas/Dev/GenAI && mamba activate poregen && python scripts/train_ldm.py ldm01/base" Enter
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
ls runs/ldm/ldm01/
```

---

## Encode latents (if re-encoding is needed)

Requires a trained VAE checkpoint. Replace `<best_run>` with the VAE run directory.

```bash
python scripts/encode_latents.py \
    --experiment r05/base \
    --checkpoint runs/vae/r05/<best_run>/checkpoints/best.ckpt \
    --data-root data/split_v2 \
    --output data/split_v2/latents_s64 \
    --stride 32 \
    --batch-size 64
```
