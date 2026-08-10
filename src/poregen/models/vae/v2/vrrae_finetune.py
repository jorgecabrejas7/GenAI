"""Fixed-basis inference extraction + decoder-only fine-tune hook for VRRAE.

The RR layer (``RR_layer.RRLayer``, vendored in ``third_party/RR_layer``)
already implements the core fixed-basis mechanism internally:

- During training it banks the last ``basis_history_size`` per-batch SVD
  bases (``self._basis_bank``, a ``deque``).
- On the first ``train() -> eval()`` transition it auto-computes a common
  basis ``Uf`` (``self.inference_basis``, a registered buffer) from the
  banked bases via ``finalize_basis()``, and at eval time projects onto
  ``Uf`` instead of recomputing a fresh per-batch SVD.

**Why a fresh per-batch SVD at inference would be wrong** (not optional to
avoid): RRLayer's SVD is computed *across the batch dimension* — the basis
depends on which other samples happen to share the batch.  A different
inference batch (or, in the limit, single-sample generation with batch=1)
would get a *different* basis, so the same latent coordinates would decode
to different content depending on what else was in the batch. That makes
the latent space unusable for downstream (e.g. diffusion) consumers that
need a stable coordinate system. Freezing one basis at the end of training
gives every future encode/decode call the same coordinate system.

This module adds three things RRLayer does NOT give you for free:

1. ``refinalize_basis`` — RRLayer only auto-finalizes ONCE, on the first
   ``train()->eval()`` transition. In this codebase's training loop
   (``poregen.training.engine``), ``model.train()``/``model.eval()`` are
   called every step (train_step / eval_step alternate constantly during
   normal periodic validation) — so the very FIRST periodic eval, early in
   training, silently and permanently freezes ``inference_basis`` from
   whatever was banked by that point (a handful of steps in), and never
   updates it again (``finalize_basis`` guard: ``if ... inference_basis is
   None``). That is almost certainly not what you want for the final
   checkpoint's fixed basis. Call ``refinalize_basis(model)`` deliberately
   at the very end of training (or right before the optional fine-tune
   stage below) to recompute ``Uf`` from whatever is in the basis bank at
   that point (i.e. a basis representative of the LAST
   ``basis_history_size`` batches of training, not the first).

2. ``load_vrrae_state_dict`` — a checkpoint-loading workaround for a real
   bug in the vendored library: ``RRLayer`` registers ``inference_basis``
   as a buffer with initial value ``None``
   (``register_buffer("inference_basis", None, persistent=True)``). A
   *fresh* module's ``state_dict()`` silently omits a still-``None``
   buffer, so PyTorch's default ``load_state_dict(strict=True)`` raises
   ``RuntimeError: Unexpected key(s) in state_dict: "...inference_basis"``
   when loading a checkpoint saved from a module where the buffer WAS
   populated, into a fresh module where it wasn't yet. Verified directly:
   a bare ``RRLayer.load_state_dict(trained_sd)`` on a fresh instance fails
   with exactly this error. Workaround (verified to work): manually
   pre-assign the buffer with a correctly-shaped tensor before calling
   ``load_state_dict`` — since ``inference_basis`` is already a registered
   buffer name, ``module.inference_basis = tensor`` is routed through
   ``nn.Module.__setattr__`` into ``self._buffers`` in place, after which
   the standard ``load_state_dict`` succeeds and reproduces identical
   eval-mode output to the source module.

3. ``finetune_decoder_on_fixed_basis`` — an optional short training-loop
   mode: freeze everything except the decoder (+ heads), force the
   bottleneck's RR layer into eval mode (fixed-basis projection, no
   per-batch SVD), and run a short optimization loop on decoder params
   only against ``Uf``-projected latents. Implemented as a standalone
   callable + a config flag, not wired into the main
   ``poregen.training.engine.train_loop`` — deliberately out of scope to
   touch that shared code path for this additive variant.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Iterator

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from poregen.models.vae.v2.vrrae_bottleneck import VRRAEBottleneck

# Above this bottleneck width, finalize_basis_from_dataloader switches from
# exact O(dim^2) covariance accumulation to the O(dim * rank) sketched path.
_EXACT_COVARIANCE_DIM_LIMIT = 8192


def _full_dataset_loader(loader: DataLoader) -> DataLoader:
    """Clone a training loader for one deterministic, non-dropping pass."""
    kwargs: dict[str, Any] = {
        "batch_size": loader.batch_size,
        "shuffle": False,
        "drop_last": False,
        "num_workers": loader.num_workers,
        "collate_fn": loader.collate_fn,
        "pin_memory": loader.pin_memory,
        "worker_init_fn": loader.worker_init_fn,
        "timeout": loader.timeout,
    }
    if loader.num_workers > 0:
        kwargs["persistent_workers"] = False
        if loader.prefetch_factor is not None:
            kwargs["prefetch_factor"] = loader.prefetch_factor
    return DataLoader(loader.dataset, **kwargs)


@torch.no_grad()
def finalize_basis_from_dataloader(
    model: nn.Module,
    train_loader: DataLoader,
    *,
    device: torch.device,
    autocast_dtype: torch.dtype,
) -> dict[str, int]:
    """Construct the JAX-equivalent U_f from every training-set batch.

    The reference concatenates all per-batch bases and takes the leading
    left singular vectors. If W = [U_1 ... U_n], those vectors are the
    leading eigenvectors of W W.T = sum_b U_b U_b.T. Accumulating that
    covariance is mathematically equivalent while avoiding a multi-gigabyte
    concatenated matrix at production dimensions.
    """
    base_model = getattr(model, "_orig_mod", model)
    bottleneck = getattr(base_model, "bottleneck", None)
    encoder = getattr(base_model, "encoder", None)
    if not isinstance(bottleneck, VRRAEBottleneck) or encoder is None:
        raise TypeError("Model is not a supported VRRAE encoder/bottleneck.")

    model.eval()
    dim = bottleneck.vrrae_dim
    rank = bottleneck.rank

    # Exact covariance accumulation is O(dim^2) memory: 16 MB at the default
    # dim=2048, but 17 GB at vrrae03's dim=65536 — plus an equal-sized matmul
    # temp per batch and an eigh workspace on top, which OOM-killed the
    # vrrae03 finalization pass on the 128 GB unified-memory GB10. Past this
    # threshold, switch to a single-pass sketched Nyström eigendecomposition
    # (Tropp et al. 2017, fixed-rank PSD approximation): accumulate only
    # Y = C @ Omega batch-by-batch (dim x s, ~160 MB at s=628) and recover
    # the leading eigenvectors from the sketch afterwards. With s = rank+128
    # oversampling, the recovered subspace captures the top-rank eigenspace
    # to within noise for the sharply-decaying spectra these projector sums
    # have (see test_full_dataset_finalization_sketched_path).
    use_sketch = dim > _EXACT_COVARIANCE_DIM_LIMIT
    if use_sketch:
        s = min(dim, rank + 128)
        gen = torch.Generator(device=device)
        gen.manual_seed(0)  # deterministic Omega -> reproducible U_f
        omega = torch.randn(dim, s, dtype=torch.float32, device=device, generator=gen)
        sketch = torch.zeros(dim, s, dtype=torch.float32, device=device)
    else:
        covariance = torch.zeros(dim, dim, dtype=torch.float32, device=device)
    n_batches = 0
    n_samples = 0

    full_loader = _full_dataset_loader(train_loader)
    for batch in full_loader:
        xct = batch["xct"].to(device, non_blocking=True)
        with torch.autocast(
            device_type=device.type,
            dtype=autocast_dtype,
            enabled=autocast_dtype in (torch.float16, torch.bfloat16),
        ):
            h = encoder(xct).flatten(1)
            y = bottleneck.projected_features(h)

        with torch.autocast(device_type=device.type, enabled=False):
            u, _, _ = torch.linalg.svd(
                y.float().transpose(0, 1),
                full_matrices=False,
            )
            r = min(rank, u.shape[1])
            batch_basis = u[:, :r]
            if use_sketch:
                # C @ Omega contribution: U_b (U_b^T Omega) — never forms U_b U_b^T
                sketch.add_(batch_basis @ (batch_basis.transpose(0, 1) @ omega))
            else:
                covariance.add_(batch_basis @ batch_basis.transpose(0, 1))

        n_batches += 1
        n_samples += xct.shape[0]

    if n_batches == 0:
        raise RuntimeError("Cannot finalize VRRAE basis from an empty dataset.")

    if use_sketch:
        # Nyström recovery: C ~= Y (Omega^T Y)^-1 Y^T, eigenvectors via the
        # shifted-Cholesky route for numerical stability.
        nu = (
            torch.finfo(torch.float32).eps
            * torch.linalg.matrix_norm(sketch)
            * (dim ** 0.5)
        )
        y_shifted = sketch + nu * omega
        b_mat = omega.transpose(0, 1) @ y_shifted
        b_mat = 0.5 * (b_mat + b_mat.transpose(0, 1))
        try:
            chol = torch.linalg.cholesky(b_mat)
            f_mat = torch.linalg.solve_triangular(
                chol, y_shifted.transpose(0, 1), upper=False
            ).transpose(0, 1)
        except torch.linalg.LinAlgError:
            # B lost positive-definiteness to roundoff — equivalent route via
            # its eigendecomposition with clamped eigenvalues.
            w, v = torch.linalg.eigh(b_mat)
            w = torch.clamp(w, min=torch.finfo(torch.float32).tiny)
            f_mat = y_shifted @ (v * w.rsqrt().unsqueeze(0))
        u_f, _, _ = torch.linalg.svd(f_mat, full_matrices=False)
        basis = u_f[:, :rank].contiguous()
    else:
        _, eigenvectors = torch.linalg.eigh(covariance)
        basis = eigenvectors[:, -rank:].flip(1).contiguous()
    if basis.shape != (bottleneck.vrrae_dim, bottleneck.rank):
        raise RuntimeError(
            "Aggregated training bases cannot span the requested VRRAE rank."
        )
    bottleneck.set_finalized_basis(basis)
    model.eval()
    return {"n_batches": n_batches, "n_samples": n_samples}


# ---------------------------------------------------------------------------
# 1. Deliberate re-finalization at end of training
# ---------------------------------------------------------------------------

def refinalize_basis(model: nn.Module) -> int:
    """Recompute every ``VRRAEBottleneck.rr.inference_basis`` in *model* from
    whatever is currently banked, overriding any earlier auto-finalization.

    Call this deliberately at the end of full training (or right before
    :func:`finetune_decoder_on_fixed_basis`) so the frozen basis reflects the
    LAST ``basis_history_size`` training batches, not whatever happened to
    be banked the first time ``model.eval()`` was called mid-training.

    Returns the number of ``RRLayer`` submodules re-finalized.  Raises
    ``RuntimeError`` (propagated from ``RRLayer.finalize_basis``) if a
    bottleneck's basis bank is empty (i.e. the model was never run in train
    mode).
    """
    n = 0
    for module in model.modules():
        if isinstance(module, VRRAEBottleneck):
            module.rr.inference_basis = None  # force re-finalization
            module.rr.finalize_basis()
            module.set_finalized_basis(module.rr.inference_basis)
            n += 1
    return n


# ---------------------------------------------------------------------------
# 2. Checkpoint round-trip workaround
# ---------------------------------------------------------------------------

def load_vrrae_state_dict(
    model: nn.Module,
    state_dict: dict[str, torch.Tensor],
    *,
    strict: bool = True,
) -> Any:
    """``model.load_state_dict(state_dict)`` with a workaround for RRLayer's
    None-initialized ``inference_basis`` buffer (see module docstring, item
    2, for why the plain call fails on a fresh model).

    Pre-assigns every ``*.inference_basis`` key present in *state_dict* onto
    the corresponding submodule BEFORE calling the standard
    ``load_state_dict``, so the buffer is a real tensor (not ``None``) by
    the time PyTorch's own state-dict key matching runs.
    """
    model_named_modules = dict(model.named_modules())
    compatible_state = dict(state_dict)
    for module_path, module in model_named_modules.items():
        if not isinstance(module, VRRAEBottleneck):
            continue
        marker_key = f"{module_path}._basis_finalized" if module_path else "_basis_finalized"
        # Checkpoints created before full-dataset finalization contained an
        # early rolling-bank basis. Preserve load compatibility but mark that
        # basis unfinalized so evaluation will not silently use it.
        compatible_state.setdefault(
            marker_key,
            torch.tensor(False, dtype=torch.bool),
        )

    for key, tensor in compatible_state.items():
        if key.endswith("inference_basis"):
            module_path = key[: -len(".inference_basis")]
            submodule = model_named_modules.get(module_path)
            if submodule is not None and getattr(submodule, "inference_basis", "MISSING") is None:
                submodule.inference_basis = tensor.clone()
    return model.load_state_dict(compatible_state, strict=strict)


# ---------------------------------------------------------------------------
# 3. Decoder-only fine-tune against the fixed basis
# ---------------------------------------------------------------------------

@dataclass
class FinetuneConfig:
    """Config flag/hook for the optional decoder-only fine-tune stage.

    Not wired into ``poregen.training.engine.train_loop`` — this is a
    standalone mechanism callers (e.g. a short post-training script) invoke
    explicitly. A corresponding YAML flag would live under
    ``training.vrrae_finetune_decoder: {enabled, steps, lr}`` in an
    experiment config; this dataclass is the typed equivalent for direct
    Python use.
    """

    enabled: bool = False
    steps: int = 50
    lr: float = 1.0e-4


def _freeze_non_decoder(model: nn.Module) -> None:
    """Freeze encoder + bottleneck; leave decoder trainable.

    Matches the ``ConvVAE3DVRRAEV2`` submodule names
    (``encoder``, ``bottleneck``, ``dec_a``, ``dec_b``, ``decoder`` — there is
    no separate ``xct_head``; it is absorbed into the decoder's final stage)
    but is written defensively via getattr so it degrades gracefully if
    called on a differently-named model.
    """
    for name in ("encoder", "bottleneck"):
        submodule = getattr(model, name, None)
        if submodule is not None:
            submodule.requires_grad_(False)
    for name in ("dec_a", "dec_b", "decoder"):
        submodule = getattr(model, name, None)
        if submodule is not None:
            submodule.requires_grad_(True)


def finetune_decoder_on_fixed_basis(
    model: nn.Module,
    data_iter: Iterator[dict[str, torch.Tensor]],
    loss_fn: Callable[..., dict[str, Any]],
    *,
    steps: int = 50,
    lr: float = 1.0e-4,
    device: torch.device = torch.device("cpu"),
) -> list[float]:
    """Short decoder-only fine-tune against ``Uf``-projected (fixed-basis)
    latents.

    Preconditions: every ``VRRAEBottleneck.rr.inference_basis`` in *model*
    must already be populated (call :func:`refinalize_basis` first, or load
    a checkpoint that already has it via :func:`load_vrrae_state_dict`).

    Mechanics:
      - Freezes encoder + bottleneck params (``requires_grad_(False)``);
        the optimizer is constructed over decoder-only params so no
        encoder/bottleneck gradients are ever applied even if some were
        computed.
      - ``requires_grad_(False)`` alone does NOT stop BatchNorm's running
        mean/var buffers from drifting on every forward call while the
        module is in train mode (buffer updates aren't gradient-gated) — so
        we start from ``model.eval()`` (freezes BatchNorm/Dropout etc.
        everywhere, and leaves the bottleneck's RR layer in eval / fixed-Uf
        mode as a side effect of standard recursive ``nn.Module.eval()``),
        then explicitly re-enable ``.train()`` on ONLY the decoder-side
        submodules so their own BatchNorm continues adapting during the
        fine-tune (the RR layer and encoder BatchNorm stay frozen).
      - Runs *steps* iterations of forward + loss + backward + step.

    Returns the list of per-step scalar total losses (for a smoke-test
    sanity check — should be finite throughout, and typically decreasing).
    """
    decoder_submodules = []
    for name in ("dec_a", "dec_b", "decoder"):
        submodule = getattr(model, name, None)
        if submodule is not None:
            decoder_submodules.append(submodule)
    if not decoder_submodules:
        raise ValueError("Model has no dec_a/dec_b/decoder submodules to fine-tune.")
    decoder_params = [p for m in decoder_submodules for p in m.parameters()]

    _freeze_non_decoder(model)
    optimizer = torch.optim.AdamW(decoder_params, lr=lr)

    # Freeze everything (BatchNorm/Dropout/RR-layer basis all included),
    # then selectively re-enable train mode on decoder-side submodules only.
    model.eval()
    for module in model.modules():
        if isinstance(module, VRRAEBottleneck) and not module.basis_finalized:
            raise RuntimeError(
                "VRRAE decoder fine-tuning requires an explicitly finalized "
                "U_f. Run finalize_basis_from_dataloader (preferred) or "
                "refinalize_basis before fine-tuning."
            )
    for submodule in decoder_submodules:
        submodule.train()

    losses: list[float] = []
    for step in range(steps):
        batch = next(data_iter)
        batch = {k: (v.to(device) if isinstance(v, torch.Tensor) else v) for k, v in batch.items()}
        optimizer.zero_grad(set_to_none=True)

        output = model(batch["xct"], batch.get("mask"))
        loss_dict = loss_fn(output, batch, step)
        total = loss_dict["total"]
        total.backward()
        optimizer.step()

        losses.append(float(total.detach().item()))

    return losses
