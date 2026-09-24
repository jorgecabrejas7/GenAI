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
