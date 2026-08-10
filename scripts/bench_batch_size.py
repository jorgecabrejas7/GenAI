"""Measure R05 throughput vs batch size to pick the fastest wall-clock setting.

For each batch size it reports, using the real model / loss / discriminator and
the real memmap DataLoader:

  * loader-only samples/s (pure I/O + collate, no GPU)
  * end-to-end train samples/s (loader + train_step, what training actually sees)
  * eval samples/s (eval_step, used for the val/test cadence cost)
  * peak GPU memory

Run:
    python scripts/bench_batch_size.py --batch-sizes 128 256 384 512 768
    python scripts/bench_batch_size.py --batch-sizes 256 512 --compile
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch

from poregen.configuration import resolve_experiment
from poregen.experiments.base import find_repo_root
from poregen.experiments.train_vae import (
    _make_loss_fn,
    build_discriminator,
    build_model,
    build_optimizer,
    resolve_data_root,
)
from poregen.training import (
    get_autocast_dtype,
    make_scaler,
    seed_everything,
    select_device,
)
from poregen.training.data import build_patch_dataloaders
from poregen.training.engine import eval_step, train_step


def _timed_loop(fn, n: int) -> float:
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(n):
        fn()
    torch.cuda.synchronize()
    return time.perf_counter() - t0


def bench_one(
    cfg: dict,
    data_root: Path,
    bs: int,
    *,
    device: torch.device,
    autocast_dtype: torch.dtype,
    warmup: int,
    iters: int,
    compile_model: bool,
) -> dict:
    cfg = json.loads(json.dumps(cfg))  # deep copy of the plain-dict config
    cfg["data"]["batch_size"] = bs

    seed_everything(int(cfg["training"]["seed"]), deterministic=False)
    train_loader, val_loader, _ = build_patch_dataloaders(cfg, data_root)

    model = build_model(cfg, device)
    optimizer = build_optimizer(cfg, model)
    loss_fn = _make_loss_fn(cfg)
    disc, disc_optimizer, disc_weight = build_discriminator(cfg, device)
    scaler = make_scaler(device)

    compile_seconds = 0.0
    if compile_model:
        model = torch.compile(model, mode="max-autotune", dynamic=False)
        if disc is not None:
            disc = torch.compile(disc, mode="max-autotune", dynamic=False)
        loss_fn = torch.compile(loss_fn, mode="max-autotune")

    torch.cuda.reset_peak_memory_stats()
    it = iter(train_loader)

    # ── loader only ───────────────────────────────────────────────────────
    for _ in range(warmup):
        next(it)
    t0 = time.perf_counter()
    for _ in range(iters):
        next(it)
    loader_s = time.perf_counter() - t0

    # ── end-to-end train step (loader + GPU) ──────────────────────────────
    step = 0
    t_compile0 = time.perf_counter()
    for _ in range(warmup):
        train_step(
            model, next(it), optimizer, scaler, loss_fn, step, device,
            autocast_dtype=autocast_dtype, max_grad_norm=cfg["training"]["max_grad_norm"],
            discriminator=disc, disc_optimizer=disc_optimizer, disc_weight=disc_weight,
        )
        step += 1
    torch.cuda.synchronize()
    compile_seconds = time.perf_counter() - t_compile0 if compile_model else 0.0

    t0 = time.perf_counter()
    for _ in range(iters):
        train_step(
            model, next(it), optimizer, scaler, loss_fn, step, device,
            autocast_dtype=autocast_dtype, max_grad_norm=cfg["training"]["max_grad_norm"],
            discriminator=disc, disc_optimizer=disc_optimizer, disc_weight=disc_weight,
        )
        step += 1
    torch.cuda.synchronize()
    train_s = time.perf_counter() - t0
    peak_gb = torch.cuda.max_memory_allocated() / 1e9

    # ── eval step ─────────────────────────────────────────────────────────
    vit = iter(val_loader)
    with torch.no_grad():
        for _ in range(max(2, warmup // 2)):
            eval_step(model, next(vit), loss_fn, step, device, autocast_dtype=autocast_dtype)
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        n_eval = max(4, iters // 2)
        for _ in range(n_eval):
            eval_step(model, next(vit), loss_fn, step, device, autocast_dtype=autocast_dtype)
        torch.cuda.synchronize()
        eval_s = time.perf_counter() - t0

    res = {
        "batch_size": bs,
        "loader_samples_per_s": bs * iters / loader_s,
        "train_samples_per_s": bs * iters / train_s,
        "train_s_per_step": train_s / iters,
        "eval_samples_per_s": bs * n_eval / eval_s,
        "eval_s_per_step": eval_s / n_eval,
        "peak_gpu_gb": peak_gb,
        "compile_warmup_s": compile_seconds,
    }

    del it, vit, model, optimizer, disc, disc_optimizer, train_loader, val_loader
    torch.cuda.empty_cache()
    return res


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--experiment", default="r05/base")
    ap.add_argument("--batch-sizes", type=int, nargs="+", default=[128, 256, 384, 512, 768])
    ap.add_argument("--warmup", type=int, default=8)
    ap.add_argument("--iters", type=int, default=20)
    ap.add_argument("--compile", action="store_true")
    ap.add_argument("--out", default="eval_results/bs_sweep.json")
    args = ap.parse_args()

    repo_root = find_repo_root()
    resolved = resolve_experiment(args.experiment, repo_root=repo_root)
    cfg = resolved.cfg
    device = select_device(None)
    autocast_dtype = get_autocast_dtype(device)
    data_root = resolve_data_root(cfg, resolved.repo_root)

    results = []
    for bs in args.batch_sizes:
        try:
            r = bench_one(
                cfg, data_root, bs,
                device=device, autocast_dtype=autocast_dtype,
                warmup=args.warmup, iters=args.iters, compile_model=args.compile,
            )
        except torch.OutOfMemoryError as exc:  # noqa: PERF203
            print(f"bs={bs}: OOM ({exc})", flush=True)
            torch.cuda.empty_cache()
            break
        results.append(r)
        print(json.dumps(r), flush=True)

    out = Path(repo_root) / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"compile": args.compile, "results": results}, indent=2))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
