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
eval_v4 measure microstructure --root runs/campaigns/10-eval-v4/   # needs the micro crops,
                                                              # AND the sampler +
                                                              # porosity_global volumes and
                                                              # the latent store: the
                                                              # memorisation search is a GPU
                                                              # pass over the whole train store
eval_v4 report --root runs/campaigns/10-eval-v4/
eval_v4 manifest-check --root runs/campaigns/10-eval-v4/      # exit 1 on a fault

# TensorBoard
tensorboard --logdir runs/vae/
```

## Known pre-existing test failures

Exactly these five tests fail. **Everything else must pass.**

```
tests/test_losses_smoke.py::TestLossesSmoke::test_all_components_present
tests/test_latent_metrics.py::test_active_units_counts_collapsed_channels
tests/test_recon_metrics.py::test_run_eval_logs_sharpness_on_post_sigmoid_xct
tests/test_recon_metrics.py::test_run_eval_aggregates_active_units_across_eval_window
tests/test_recon_metrics.py::test_train_loop_runs_final_full_eval_for_val_and_test
```

Do not fix them unless explicitly asked. The gate is this list of ids, not a
total count — the count moves every time a branch adds a test, so it is a
liability rather than a baseline. A new failure outside this list is a
regression, whatever the totals say.

Check the five cheaply, without running the whole suite:

```bash
CUDA_VISIBLE_DEVICES= PYTHONPATH="$PWD/src" python -m pytest \
    tests/test_losses_smoke.py tests/test_latent_metrics.py \
    tests/test_recon_metrics.py -q
```

Nothing is expected to skip. The end-to-end FID test
(`tests/test_eval_v4_microstructure.py::TestFid::test_identical_crop_sets_have_zero_fid`)
used to skip for want of `torchvision`; it no longer does. `torchvision` is
installed on this machine and the Inception weights are cached, so the test
**runs** — see [torchvision (FID only)](#torchvision-fid-only) for the exact
version pair. It is now the slowest test in the suite: a real Inception forward
pass on CPU, about 3 minutes with `CUDA_VISIBLE_DEVICES=` set.

**FID numbers are only valid after fix F11** (Inception input normalisation,
merged). Before F11 the features came from unnormalised input, so every FID
value recorded earlier is void. Do not compare a new FID against one from an
older campaign report unless that report post-dates F11.

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

## torchvision (FID only)

**Installed pair on this machine: `torch 2.12.1+cu130` with
`torchvision 0.27.1+cu130`.** Confirm before an eval run:

```bash
CUDA_VISIBLE_DEVICES= python -c "import torch, torchvision; print(torch.__version__, torchvision.__version__)"
# 2.12.1+cu130 0.27.1+cu130
```

If `torch` reports anything else, a `torchvision` install has replaced the
working build — stop and repair it before trusting any result.

`torchvision` is NOT an installable line in `requirements.txt` (only a
recorded comment there, so `pip install -r requirements.txt` cannot pull a
torch with it) and nothing but FID needs it —
`eval_v4.microstructure` uses `inception_v3` for the 2048-d feature. It must be
installed **from the PyTorch cu130 index, version-matched, and with
`--no-deps`**, or pip resolves a torchvision whose `Requires-Dist` pins a
different torch and silently replaces the working build:

```bash
pip install --no-deps torchvision==0.27.1+cu130 \
    --index-url https://download.pytorch.org/whl/cu130
```

0.27.1 declares `torch (==2.12.1)` — exactly what is installed. The version
pairing steps with torch (0.25 <-> 2.10, 0.26 <-> 2.11, 0.27 <-> 2.12,
0.28 <-> 2.13, 0.29 <-> 2.14), so the plain `pip install torchvision` that the
index offers today (0.29.0+cu130, for torch 2.14) is the wrong build. Check the
wheel's `Requires-Dist: torch` before installing a different one.

The ImageNet weights are fetched on first use and cached in `~/.cache/torch/`;
prefetch them off the GPU rather than discovering the download inside an eval
run. On this machine they are already cached
(`~/.cache/torch/hub/checkpoints/inception_v3_google-0cc3c7bd.pth`), which is
why the end-to-end FID test runs instead of skipping.

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

## GB10 unified memory: never stream the store beside a CUDA job

**A streaming read of the latent store while a CUDA job runs can kill the CUDA
job.** Host and device share one 121 GB pool. A pass over `latents.bin`
(195 GiB) fills the page cache in minutes, and a CUDA allocation does not wait
for the kernel to reclaim that cache — it fails. The reason this is worth a
section of its own is that the symptom never points at the cause: `free`
reports tens of GB "available" throughout, because page cache *is* reclaimable
in principle, and the job that dies is the one that did nothing wrong. It has
happened twice. During ldm06 training, external CUDA jobs failed repeatedly at
the 60k diagnostics (ldm06 run note, incident 3). On 2026-09-11 the memorisation
smoke test was re-run beside `eval_v4 generate assembly_modes` and was killed
for low memory at about nine minutes, with 97 GB reported available.

Two rules follow, and both are enforced in code rather than left to a runbook,
because the second incident was a hand-run that ignored the first:

- **Store-streaming jobs get their own queue slot, with the card idle.**
  `poregen.eval_v4.memorisation.gpu_jobs_other_than` names any other CUDA
  process; the memorisation search skips itself with a reason when the card is
  busy, and `scripts/analysis/memorisation_smoke.py` refuses to start. Both take
  `--allow-busy-gpu`, which is correct only where host and device memory are
  separate pools.
- **Readers release pages behind them.** Each bank chunk is dropped with
  `madvise(MADV_DONTNEED)` and then `posix_fadvise(POSIX_FADV_DONTNEED)` once it
  has been scored — fadvise alone cannot free a page a mapping still holds — so
  the resident file-backed set stays bounded by the chunk (~3.9 GB for latents,
  ~1 GB for grey) instead of growing to the size of the store.

## Running the tests while a job holds the GPU

Hide the card: `CUDA_VISIBLE_DEVICES= python -m pytest tests/ -q`.

Several tests allocate CUDA memory, and with a training run resident they fail
with `torch.AcceleratorError: CUDA error: out of memory` rather than anything
about the code. The symptom is confusing because it depends on what else is
running: a file passes on its own and fails in a group, so it reads as a test
ordering bug. It is not — those tests do not need the GPU to be meaningful.

## Running analysis alongside training

The GPU is shared. Analysis and diagnostic scripts must set
`PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`, use small batches, and retry
on OOM — a training run's validation passes can otherwise starve them.
Long jobs belong in a tmux session, not in a foreground shell.
