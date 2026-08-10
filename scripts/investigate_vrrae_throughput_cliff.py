"""THROWAWAY diagnostic (read-only): reproduce/verify the VRRAE batch-size
throughput cliff on the GB10, and confirm the encoder/decoder symmetry fix.

Background
----------
The original ``v2.vrrae`` architecture was fast and linear-scaling at
batch_size<=192, but showed a sharp, non-linear slowdown at batch_size>=256
(measured: bs=192 ~2.3s/step, bs=256 ~8.7s/step, bs=512 ~17s/step --
backward pass dominates). ROOT CAUSE (confirmed via isolated microbenchmark
of a bare nn.Conv3d, independent of this model/repo): cuDNN's 3-D
convolution *backward* kernel falls back from 32-bit to 64-bit indexing once
a tensor reaches 2**31 elements, ~4x slower per element past that point. The
old decoder's asymmetric final stage (Upsample -> Conv3d(32,32,k3) at full
64^3 resolution -- the encoder never touched 64^3 above 1 channel) crossed
that threshold at exactly batch_size=256.

FIX: the model was rebuilt with a genuinely symmetric encoder/decoder (see
src/poregen/models/vae/v2/vrrae.py module docstring and
src/poregen/models/nn/blocks.py's up_block_v2_mirror) -- ConvTranspose3d
replaces Upsample+Conv so no decoder tensor ever exceeds encoder.0's
32ch@32^3 (well under 2**31 up to batch_size~2048). ``--mode sweep`` below
should now show flat, linear ms/sample scaling with no cliff.

A separate, confirmed torch.compile recompile storm (RRLayer's stateful
_basis_bank triggering guard-based recompiles) was found and fixed
independently (see VRRAEBottleneck.forward's @torch._dynamo.disable()
decorator in src/poregen/models/vae/v2/vrrae_bottleneck.py) -- that fix is
real, orthogonal to the indexing cliff, and unaffected by this change.

Hardware: NVIDIA GB10 (DGX Spark), 128GB TRUE UNIFIED memory (nvidia-smi
reports FB Memory Usage as N/A entirely -- no dedicated VRAM query works on
this platform, confirming genuine unified addressing, not just a shared-pool
label). `poregen` conda env. Production training uses
torch.compile(mode="max-autotune", dynamic=False) per CLAUDE.md; this
reproduces with compile OFF too (eager mode), since kernel selection is a
function of tensor shape only, not of compile/eager mode.

Usage
-----
    conda activate poregen
    cd /home/jorgecabrejas/Dev/GenAI
    python scripts/investigate_vrrae_throughput_cliff.py --mode sweep
    python scripts/investigate_vrrae_throughput_cliff.py --mode stages --batch-size 256
    python scripts/investigate_vrrae_throughput_cliff.py --mode kl-isolation
    python scripts/investigate_vrrae_throughput_cliff.py --mode profiler --batch-size 512
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

REPO = Path("/home/jorgecabrejas/Dev/GenAI")
sys.path.insert(0, str(REPO / "src"))

import torch

from poregen.models.vae import build_vae
from poregen.losses import compute_total_loss
from poregen.training import get_autocast_dtype, make_scaler, select_device
from poregen.training.engine import train_step

MODEL_KW = dict(
    name="v2.vrrae", in_channels=1, base_channels=32, n_blocks=5, patch_size=64,
    vrrae_dim=2048, vrrae_rank=300, vrrae_basis_history_size=20,
)
LOSS_CFG = {"loss": {"xct_loss_type": "charbonnier", "xct_weight": 1.0,
                     "kl_free_bits": 0.1, "kl_warmup_steps": 0, "kl_max_beta": 0.05}}


def make_batch(bs: int, device: torch.device) -> dict[str, torch.Tensor]:
    xct = torch.rand(bs, 1, 64, 64, 64, device=device)
    mask = (torch.rand(bs, 1, 64, 64, 64, device=device) < 0.05).float()
    return {"xct": xct, "mask": mask}


def run_sweep(batch_sizes: list[int], n_steps: int = 4) -> None:
    """Real train_step() timing across batch sizes -- reproduces the cliff."""
    device = select_device(None)
    autocast_dtype = get_autocast_dtype(device)
    scaler = make_scaler(device)
    loss_fn = lambda output, batch, step: compute_total_loss(output, batch, step, LOSS_CFG)

    for bs in batch_sizes:
        model = build_vae(**MODEL_KW).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=2e-4, weight_decay=0.01)
        batch = make_batch(bs, device)
        times = []
        for s in range(n_steps):
            t0 = time.time()
            train_step(model, batch, optimizer, scaler, loss_fn, step=s, device=device,
                       autocast_dtype=autocast_dtype, max_grad_norm=None)
            torch.cuda.synchronize()
            times.append(time.time() - t0)
        print(f"bs={bs:4d}  steps={[f'{t:.2f}s' for t in times]}")
        del model, optimizer
        torch.cuda.empty_cache()


def run_stages(bs: int, n_steps: int = 4) -> None:
    """Per-submodule forward + full backward timing at a fixed batch size."""
    device = select_device(None)
    autocast_dtype = get_autocast_dtype(device)
    model = build_vae(**MODEL_KW).to(device)
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-4)
    xct = torch.rand(bs, 1, 64, 64, 64, device=device)

    def sync_time(label, fn):
        torch.cuda.synchronize()
        t0 = time.time()
        out = fn()
        torch.cuda.synchronize()
        print(f"{label:30s} {time.time()-t0:8.3f}s")
        return out

    for step in range(n_steps):
        print(f"--- step {step} (bs={bs}) ---")
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type=device.type, dtype=autocast_dtype):
            h = sync_time("encoder", lambda: model.encoder(xct))
            h_flat = h.flatten(1)
            fc_out = sync_time("bottleneck.fc_in", lambda: model.bottleneck.fc_in(h_flat))
            def rr_call():
                with torch.autocast(device_type=device.type, enabled=False):
                    return model.bottleneck.rr(fc_out.float(), return_factors=True)
            rr_out = sync_time("bottleneck.rr (SVD)", rr_call)
            mu = rr_out[2].transpose(0, 1)
            logvar = sync_time("bottleneck.logvar_head", lambda: model.bottleneck.logvar_head(fc_out))
            from poregen.models.nn.blocks import reparameterize
            z = reparameterize(mu, logvar)
            ch_last = model.cfg.channel_schedule()[model.cfg.n_blocks - 1]
            ls = model.cfg.latent_spatial
            # dec_a was removed: the (rank -> vrrae_dim) leg is a multiply by
            # the RR layer's own basis (rr_out[1]), not a learned Linear.
            basis = rr_out[1].to(z.dtype)
            dec_in = sync_time(
                "basis_matmul+dec_b",
                lambda: model.dec_b(z @ basis.transpose(0, 1)).view(
                    z.shape[0], ch_last, ls, ls, ls
                ),
            )
            xct_logits = sync_time("decoder", lambda: model.decoder(dec_in))
            loss = xct_logits.pow(2).mean()
        sync_time("backward", lambda: (loss.backward(), None)[1])
        sync_time("optimizer.step", lambda: (optimizer.step(), None)[1])


def run_kl_isolation(bs: int = 512, n_repeats: int = 5) -> None:
    """Repeatedly time xct-only vs xct+KL(mu,logvar) backward -- check for
    run-to-run inconsistency (this was the key unresolved finding)."""
    device = select_device(None)
    autocast_dtype = get_autocast_dtype(device)
    model = build_vae(**MODEL_KW).to(device)
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-4)
    xct = torch.rand(bs, 1, 64, 64, 64, device=device)

    for rep in range(n_repeats):
        for label, use_kl in [("xct-only", False), ("xct+KL", True)]:
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, dtype=autocast_dtype):
                out = model(xct)
                loss = out.xct_logits.pow(2).mean()
                if use_kl:
                    kl = 0.5 * (out.mu.pow(2) + out.logvar.exp() - out.logvar - 1.0).sum()
                    loss = loss + 0.05 * kl
            torch.cuda.synchronize()
            t0 = time.time()
            loss.backward()
            torch.cuda.synchronize()
            print(f"rep {rep} {label:10s} backward: {time.time()-t0:.2f}s")
            optimizer.step()


def run_torch_profiler(bs: int = 512) -> None:
    """torch.profiler breakdown of a real train_step -- shows
    aten::_local_scalar_dense dominating (99%+), but stack traces for it
    come back empty. A next step here would be nsys/Nsight Systems for
    real GPU-kernel-level tracing, which isn't available via plain
    torch.profiler on this platform."""
    from torch.profiler import profile, ProfilerActivity

    device = select_device(None)
    autocast_dtype = get_autocast_dtype(device)
    scaler = make_scaler(device)
    model = build_vae(**MODEL_KW).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-4, weight_decay=0.01)
    loss_fn = lambda output, batch, step: compute_total_loss(output, batch, step, LOSS_CFG)
    batch = make_batch(bs, device)

    for s in range(3):
        t0 = time.time()
        train_step(model, batch, optimizer, scaler, loss_fn, step=s, device=device,
                   autocast_dtype=autocast_dtype, max_grad_norm=None)
        torch.cuda.synchronize()
        print(f"warmup step {s}: {time.time()-t0:.2f}s")

    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
                 record_shapes=False, with_stack=True) as prof:
        t0 = time.time()
        train_step(model, batch, optimizer, scaler, loss_fn, step=99, device=device,
                   autocast_dtype=autocast_dtype, max_grad_norm=None)
        torch.cuda.synchronize()
        print(f"profiled step: {time.time()-t0:.2f}s")

    print(prof.key_averages().table(sort_by="self_cpu_time_total", row_limit=20))
    # Export a chrome trace for visual/manual inspection -- open in
    # chrome://tracing or https://ui.perfetto.dev/
    trace_path = "/tmp/vrrae_trace.json"
    prof.export_chrome_trace(trace_path)
    print(f"\nChrome trace exported to {trace_path}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["sweep", "stages", "kl-isolation", "profiler"], default="sweep")
    ap.add_argument("--batch-size", type=int, default=512)
    ap.add_argument("--n-steps", type=int, default=4)
    args = ap.parse_args()

    if args.mode == "sweep":
        # vrrae_rank=300 requires batch_size >= 300 (RRLayer clips the
        # effective rank otherwise -- see VAEConfig.vrrae_rank docstring).
        run_sweep([320, 384, 448, 512], n_steps=args.n_steps)
    elif args.mode == "stages":
        run_stages(args.batch_size, n_steps=args.n_steps)
    elif args.mode == "kl-isolation":
        run_kl_isolation(args.batch_size)
    elif args.mode == "profiler":
        run_torch_profiler(args.batch_size)
