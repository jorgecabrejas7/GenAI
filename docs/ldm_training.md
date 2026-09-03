# LDM Training

All commands assume you are at the repo root `/home/jorgecabrejas/Dev/GenAI` and the `poregen` mamba environment is active.

```bash
cd /home/jorgecabrejas/Dev/GenAI
mamba activate poregen
```

For what the model conditions on and how a volume is generated, see
[ARCHITECTURE.md](ARCHITECTURE.md).

---

## Before the first ldm06 run

The store and the sidecar are built in that order, once, and the run then reads
both. Both steps need a trained r08 checkpoint.

```bash
# 1. Encode split_v3 with the r08 3-class VAE.  Writes latents.bin,
#    index.parquet, material.bin and air.bin per split, in one pass.
python scripts/build_latent_dataset.py \
    --checkpoint runs/vae/<r08 run>/best.ckpt \
    --output data/split_v3/latents_r08z4 \
    --batch-size 256

# 2. Per-patch conditioning: cond.parquet (cond_depth, six cond_dist6_*,
#    cond_por_raw) + the `conditioning` metadata block.  Reuses
#    data/split_v2/orientation_field.json and checks every volume has a record.
python scripts/build_conditioning.py --store data/split_v3/latents_r08z4

# 3. Point the experiment at that exact checkpoint.  ldm06/base ships a
#    placeholder; the launcher compares it with the store's own record and
#    refuses to start on a mismatch, because decoding with a different VAE than
#    the one that encoded the latents is silently wrong.
$EDITOR configs/experiments/ldm06/base.yaml     # vae.checkpoint
```

---

## New training run

Create a detached tmux session named after the experiment and launch inside it.

```bash
tmux new-session -d -s ldm06
tmux send-keys -t ldm06 "cd /home/jorgecabrejas/Dev/GenAI && mamba activate poregen && python scripts/train_ldm.py run ldm06/base" Enter
```

The rungs available today, all on the same store and the same denoiser:

| Experiment | What it changes | Compare against |
|---|---|---|
| `ldm06/base` | the ε baseline | — |
| `ldm06/aux` | adds the decoded-space auxiliary loss through the frozen r08 VAE | `ldm06/base`, same seed and budget |
| `ldm06/eps` | ablation: ε objective instead of v | `ldm06/base`, same seed and budget |

Each is a separate rung with its own tmux session named after it; a new
capability gets a new `ldmNN`, logged in the vault.

Attach to watch the output:

```bash
tmux attach -t ldm06
# detach without killing: Ctrl-b  d
```

---

## Resume from latest checkpoint

Replace `<run_dir>` with the actual run directory under `runs/ldm/`.

```bash
tmux new-session -d -s ldm06-resume
tmux send-keys -t ldm06-resume "cd /home/jorgecabrejas/Dev/GenAI && mamba activate poregen && python scripts/train_ldm.py resume <run_dir>" Enter
```

Resume from a specific checkpoint:

```bash
python scripts/train_ldm.py resume <run_dir> checkpoints/ldm_step00050000.ckpt
```

---

## Watching a run

```bash
# Convergence health-check: per-neighbour-context bucket latent moments and
# decoded pore/air fraction, the conditioning-alive probe (porosity /
# orientation / material / neighbours), the phi kill-switch, and a
# CONVERGING / STALLED / REGRESSING verdict against the previous check.
python scripts/diag_ldm_samples.py --run-dir runs/ldm/<run> --n 8
```

What to watch in TensorBoard:

| Scalar | Healthy | Meaning |
|---|---|---|
| `val/loss` | falling | ε-MSE with every neighbour at the target timestep — the situation the sampler runs in |
| `gen/std_ratio_avg` | → 1.0 | generated latents' per-channel std vs the real train distribution |
| `gen/por_cond_mae` | falling | conditional adherence: delivered vs requested φ on real val conditioning |
| `gen/x0_clamp_sat_frac` | → 0 | fraction of x̂₀ elements hitting the ±10 clamp; a rising value means off-manifold sampling. Under the ε objective this has a FLOOR: at the terminal step `sqrt(ᾱ_T) = 6.12e-17` makes the ε form of x̂₀ diverge and the whole first step saturates. `ldm06/base` trains on v, which removes that floor; compare it against the `ldm06/eps` ablation on this scalar first |
| `samples/seam_xct_ratio` | → 1.0 | window-period (64-voxel) seam vs the interior slice-to-slice baseline |
| `samples/seam_chunk_xct_ratio` | → 1.0 | chunk-period seam; a gap between the two says the chunk boundary is the problem, not the window overlap |

With `ldm06/aux` the decoded auxiliary loss adds one scalar per term, plus the
latent loss on its own so the two contributions stay separable:

| Scalar | Healthy | Meaning |
|---|---|---|
| `train/latent_loss` | falling | the ε/v objective alone, without the auxiliary terms |
| `train/aux_air_outside_material` | → 0 | p(air) where `cond_material` says specimen |
| `train/aux_pore_dice` | falling | 1 − Dice of the decoded pores against the real ones |
| `train/aux_porosity_consistency` | → 0 | delivered φ vs the conditioned φ, differentiable |
| `train/aux_grey_agreement` | → 0 | bright voxels called pore, dark voxels called material |
| `train/aux_items` | = `decoded_max_items` | items decoded this step; persistently below the cap means `t_max_frac` is too tight for the batch size |
| `train/aux_ramp` | → 1.0 | the linear warm-up factor on the whole block |

`aux_items` is worth a glance early: at `batch_size` 256 and `t_max_frac` 0.25
about 64 items qualify per step, so the 32-item cap should bind on nearly every
step. A value that keeps landing below it says the schedule window and the
batch size disagree.

The sample volume logged every `sample_every` steps is one chunk of
`generation.chunk_tiles` tiles generated by the production sampler, so those
seam numbers are the ones a generation run would report.

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

## Generate volumes from a checkpoint

```bash
python scripts/generate_volumes.py \
    --checkpoint runs/ldm/<run>/checkpoints/best.ckpt \
    --latents-root data/split_v3/latents_r08z4 \
    --ddim-steps 50 \
    --chunk-tiles 3 3 3
```

Omitted sampler flags fall back to the run's own `generation` block, so a
generation run reproduces the geometry the run was babysat with. Output is
`volume.tif` (uint8 raw-scan grey) + `label.tif` (uint8 0 material / 1 pore /
2 air) + `generation_stats.json` per sweep cell. Neither array is rescaled —
measuring a generated volume with the tools built for real ones depends on it.
