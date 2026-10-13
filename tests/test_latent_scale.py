"""The scalar latent rescaling of Rombach et al. 2022 Sec. G.

The store records one sigma_hat (the std of sampled train latents over every
element); the dataset divides mu AND the posterior std by it, with no
centring, and a store without the rule is refused.
"""

from __future__ import annotations

import json

import numpy as np
import pytest
import torch

from _ldm06_store import SYN, build_store, dataset_kwargs
from poregen.diffusion.latents import (
    NORM_RULE,
    LatentDataset,
    SampledMoments,
    latent_scale,
    sampled_std_reference,
)


def test_moments_are_the_scalar_mean_and_std_over_every_element():
    rng = np.random.default_rng(0)
    z = torch.from_numpy(rng.normal(0.3, 2.0, size=(5, 3, 4, 4, 4)))
    m = SampledMoments()
    m.update(z[:2])
    m.update(z[2:])
    mu_hat, sigma_hat = m.result()
    assert m.n == z.numel()
    assert mu_hat == pytest.approx(float(z.mean()), rel=1e-12)
    assert sigma_hat == pytest.approx(float(z.std(unbiased=False)), rel=1e-12)


def _set_scale(root, scale):
    meta = json.loads((root / "metadata.json").read_text())
    meta["normalization"] = {"rule": NORM_RULE, "sigma_hat": scale}
    meta["splits"] = {"train": int(len(np.fromfile(root / "train" / "air.bin",
                                                   np.float32)))}
    (root / "metadata.json").write_text(json.dumps(meta))


def test_the_dataset_divides_mean_and_std_by_sigma_hat_without_centring(tmp_path):
    root, *_ = build_store(tmp_path)
    raw = LatentDataset(root, "train", normalize=False, **dataset_kwargs())
    _set_scale(root, 2.5)
    ds = LatentDataset(root, "train", normalize=True, **dataset_kwargs())
    assert ds.latent_scale == 2.5
    a, b = raw[7], ds[7]
    assert torch.allclose(b["z"], a["z"] / 2.5)
    assert torch.allclose(b["std"], a["std"] / 2.5)
    assert torch.allclose(b["nb_latents"], a["nb_latents"] / 2.5)
    assert torch.allclose(b["nb_std"], a["nb_std"] / 2.5)
    assert torch.allclose(ds.denormalize_latent(b["z"]), a["z"])


def test_a_store_without_the_scalar_rule_is_refused(tmp_path):
    root, *_ = build_store(tmp_path)
    meta = json.loads((root / "metadata.json").read_text())
    meta["normalization"] = {"per_channel_mean": [0.0], "per_channel_std": [1.0]}
    (root / "metadata.json").write_text(json.dumps(meta))
    with pytest.raises(ValueError, match="sigma_hat"):
        LatentDataset(root, "train", **dataset_kwargs())
    with pytest.raises(ValueError, match="sigma_hat"):
        latent_scale(meta)


def test_the_reference_is_one_when_sigma_hat_is_the_sampled_std(tmp_path):
    root, *_ = build_store(tmp_path)
    _set_scale(root, 1.0)
    meta = json.loads((root / "metadata.json").read_text())
    c = SYN["C"]
    n = meta["splits"]["train"]
    lat = np.fromfile(root / "train" / "latents.bin", np.float16).astype(np.float64)
    lat = lat.reshape(n, 2 * c, -1)
    mu, sig = lat[:, :c], lat[:, c:]
    sampled_std = float(np.sqrt((mu ** 2 + sig ** 2).mean() - mu.mean() ** 2))
    _set_scale(root, sampled_std)
    ref = sampled_std_reference(root, "train", n_rows=n)
    assert ref["sampled_std"] == pytest.approx(1.0, rel=1e-6)
    assert ref["sigma_hat"] == pytest.approx(sampled_std)
