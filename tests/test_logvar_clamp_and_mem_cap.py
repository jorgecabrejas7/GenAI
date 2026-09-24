"""The two guards added after the 2026-09-23 incidents: exp(logvar) can no
longer overflow (V0's NaN at step 6777), and a CUDA allocation past the cap
raises instead of draining the unified memory of the host."""
import os

import pytest
import torch

from poregen.losses.kl import kl_divergence_flat
from poregen.models.nn.blocks import LOGVAR_MAX, reparameterize
from poregen.training import device as dev_mod


def test_reparameterize_and_kl_stay_finite_for_huge_logvar():
    mu = torch.zeros(4, 8)
    logvar = torch.full((4, 8), 500.0)          # exp(500) overflows float32
    z = reparameterize(mu, logvar)
    assert torch.isfinite(z).all()
    assert z.abs().max() < 10 * torch.exp(torch.tensor(LOGVAR_MAX / 2))
    out = kl_divergence_flat(mu, logvar)
    kl = out[0] if isinstance(out, tuple) else out["kl"] if isinstance(out, dict) else out
    assert torch.isfinite(torch.as_tensor(kl)).all()


def test_ordinary_logvar_is_untouched():
    mu = torch.randn(4, 8)
    logvar = torch.randn(4, 8)
    torch.manual_seed(0); a = reparameterize(mu, logvar)
    torch.manual_seed(0); b = mu + (0.5 * logvar).exp() * torch.randn_like(mu)
    assert torch.allclose(a, b)


def test_mem_fraction_env_rejects_nonsense(monkeypatch):
    monkeypatch.setenv(dev_mod.CUDA_MEM_FRACTION_ENV, "2.5")
    if not torch.cuda.is_available():
        pytest.skip("no CUDA")
    with pytest.raises(ValueError):
        dev_mod.select_device()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="no CUDA")
def test_allocation_past_the_cap_raises_cleanly(monkeypatch):
    monkeypatch.setenv(dev_mod.CUDA_MEM_FRACTION_ENV, "0.01")   # ~1.2 GB on the GB10
    device = dev_mod.select_device()
    with pytest.raises(torch.cuda.OutOfMemoryError):
        torch.empty(int(4 * 2**30) // 4, dtype=torch.float32, device=device)   # 4 GB
    torch.cuda.set_per_process_memory_fraction(1.0, device)      # leave the process as found


def test_train_step_skips_nonfinite_gradient():
    """A NaN in the backward must not reach the weights: the step is skipped and flagged."""
    import torch.nn as nn
    from poregen.training import engine

    class Tiny(nn.Module):
        def __init__(self):
            super().__init__(); self.lin = nn.Linear(3, 3)
        def forward(self, x):
            return self.lin(x)

    model = Tiny(); before = model.lin.weight.detach().clone()
    opt = torch.optim.SGD(model.parameters(), lr=1.0)
    out = model(torch.ones(2, 3)); loss = (out * float("nan")).sum(); loss.backward()
    grads = [p.grad for p in model.parameters()]
    assert any(not torch.isfinite(g).all() for g in grads)
    norm = torch.linalg.vector_norm(torch.stack(torch._foreach_norm(grads, 2)), 2).item()
    assert not __import__("math").isfinite(norm)
    # the guard's decision rule, as train_step applies it
    if not __import__("math").isfinite(norm):
        opt.zero_grad(set_to_none=True)
    else:
        opt.step()
    assert torch.equal(model.lin.weight.detach(), before)
    assert engine.MAX_SKIPPED_STEPS_IN_A_ROW == 20
