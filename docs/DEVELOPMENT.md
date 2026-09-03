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
python scripts/train_vae.py run r05/base          # launch by experiment id
python scripts/train_vae.py resume runs/vae/<run_name> checkpoints/<ckpt>
python scripts/train_ldm.py run ldm06/base        # LDM training
python scripts/train_ldm.py resume runs/ldm/<run_name> checkpoints/<ckpt>
experiments list                                   # list all defined experiments
experiments clone r05/base r06/my_variant         # create new experiment extending r05

# Generated-volume evaluation (eval v4) - see docs/eval_methodology.md
eval_v4 real-floor --root runs/campaigns/10-eval-v4/          # run this FIRST
                                                              # (cuts the matched-porosity
                                                              #  reference pairs too)
eval_v4 generate sampler --model runs/ldm/<run> --ckpt 130000 \
                 --out runs/campaigns/10-eval-v4/             # GPU, hours
eval_v4 generate sampler --model runs/ldm/<run> --ckpt best --dry-run
eval_v4 measure sampler --root runs/campaigns/10-eval-v4/     # CPU, repeatable
eval_v4 measure microstructure --root runs/campaigns/10-eval-v4/   # needs the micro crops
eval_v4 report --root runs/campaigns/10-eval-v4/
eval_v4 manifest-check --root runs/campaigns/10-eval-v4/      # exit 1 on a fault

# TensorBoard
tensorboard --logdir runs/vae/
```

## Known pre-existing test failures

Do not fix unless explicitly asked:

- `tests/test_losses_smoke.py::TestLossesSmoke::test_all_components_present`
- `tests/test_latent_metrics.py::test_active_units_counts_collapsed_channels`
- `tests/test_recon_metrics.py` (3 failures)

A clean run is therefore **448 passed, 1 skipped, 5 failed**. The skip is the
end-to-end FID test, which needs torchvision.

## Testing inside a git worktree

`pip install -e .` points the `poregen` package at the **main checkout's**
`src/`. Running `pytest` inside a worktree therefore tests the main tree's code,
not the worktree's — silently, and with a plausible-looking result. Two things
are needed to test the branch you actually checked out:

```bash
W=.claude/worktrees/<worktree>
ln -sfn "$PWD/runs" "$W/runs"       # tests read campaign artefacts from runs/
ln -sfn "$PWD/data" "$W/data"       # and calibration inputs from data/
cd "$W" && PYTHONPATH="$PWD/src" python -m pytest tests/ -q
```

Without `PYTHONPATH` you get collection errors from whichever tree happens to be
installed. Without the two symlinks you get spurious `FileNotFoundError`s in
`tests/test_porosity_field.py` and anything else that reads a committed
campaign result — `/runs/` and `/data/` are git-ignored, so a worktree has
neither. Confirm you are testing what you think with:

```bash
PYTHONPATH="$PWD/src" python -c "import poregen; print(poregen.__file__)"
```

Both symlinks point at the real directories, so a test that writes through one
writes into the live tree. The suite writes to `tmp_path`, but check any new
test before adding it.

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
