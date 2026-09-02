# Development

How to install, run and test the project. For what the pieces are, see
[ARCHITECTURE.md](ARCHITECTURE.md); for where they live, see [CODEMAP.md](CODEMAP.md).

## Commands

```bash
# Install (environment assumed to be set up with mamba/conda + PyTorch)
pip install -r requirements.txt
pip install -e ".[dev]"

# Tests
pytest tests/                          # full suite
pytest tests/test_losses_smoke.py      # single file
pytest tests/ -k "test_vae_output"     # filter by name

# Dataset construction
build_dataset --help                   # CLI entry point

# Training
python scripts/train_vae.py r05/base              # launch by experiment id
python scripts/train_ldm.py run ldm06/base        # LDM training
python scripts/train_ldm.py resume runs/ldm/<run_name> checkpoints/<ckpt>
experiments list                                   # list all defined experiments
experiments clone r05/base r06/my_variant         # create new experiment extending r05

# TensorBoard
tensorboard --logdir runs/vae/
```

## Known pre-existing test failures

Do not fix unless explicitly asked:

- `tests/test_losses_smoke.py::TestLossesSmoke::test_all_components_present`
- `tests/test_latent_metrics.py::test_active_units_counts_collapsed_channels`
- `tests/test_recon_metrics.py` (3 failures)

A clean run is therefore **264 passed, 5 failed**.

## Deployment target

**NVIDIA GB10 GPU (DGX Spark, 128 GB unified memory).** Code runs on that
machine, not locally.

- `torch.compile(mode="max-autotune-no-cudagraphs", dynamic=False)` — shapes are
  static (64³ patches); CUDA-graph trees break on the train/eval mode switch (BN
  buffers become graph outputs) and on the twice-per-step discriminator, so graph
  replay stays off
- `autocast_dtype` is `bfloat16` (Blackwell supports it natively)
- `num_workers=4` in `configs/machines/dgx_spark.yaml` — more workers cause heap
  copies in GB10 unified memory
- Discriminator intentionally runs in `float32` (spectral norm power iteration is
  less accurate in bfloat16) — do not wrap it in autocast

## Running analysis alongside training

The GPU is shared. Analysis and diagnostic scripts must set
`PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`, use small batches, and retry
on OOM — a training run's validation passes can otherwise starve them.
Long jobs belong in a tmux session, not in a foreground shell.
