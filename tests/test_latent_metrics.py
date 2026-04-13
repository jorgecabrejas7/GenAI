import torch

from poregen.metrics.latent import (
    active_units,
    active_units_from_moments,
    latent_channel_moments,
    merge_latent_channel_moments,
)


def test_active_units_from_moments_matches_direct_metric():
    mu_a = torch.tensor(
        [
            [
                [[[0.0, 1.0]]],
                [[[0.0, 0.0]]],
            ],
            [
                [[[2.0, 3.0]]],
                [[[0.0, 0.0]]],
            ],
        ],
        dtype=torch.float32,
    )
    mu_b = torch.tensor(
        [
            [
                [[[4.0, 5.0]]],
                [[[0.0, 0.0]]],
            ],
        ],
        dtype=torch.float32,
    )

    combined = torch.cat([mu_a, mu_b], dim=0)
    direct = active_units(combined, threshold=0.01)

    merged = merge_latent_channel_moments(
        [latent_channel_moments(mu_a), latent_channel_moments(mu_b)]
    )
    aggregated = active_units_from_moments(
        merged["count"],
        merged["sum"],
        merged["sum_sq"],
        threshold=0.01,
    )

    assert aggregated == direct


def test_active_units_counts_collapsed_channels():
    mu = torch.zeros(2, 3, 1, 1, 2, dtype=torch.float32)
    mu[:, 0, ...] = torch.tensor([0.0, 3.0]).view(2, 1, 1, 1)
    mu[:, 1, ...] = 0.1
    mu[:, 2, ...] = torch.tensor([0.0, 0.5]).view(2, 1, 1, 1)

    metrics = active_units(mu, threshold=0.01)

    assert metrics["n_total"] == 3
    assert metrics["n_active"] == 2
    assert metrics["active_fraction"] == 2 / 3
