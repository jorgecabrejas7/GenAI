"""Verifies VolumeGenerator's two-phase checkerboard patch ordering.

The LDM denoiser is trained on a strict two-state neighbor-conditioning
scheme (see latent_dataset.py): parity-0 ("anchor") patches see only
OOB/UNKNOWN neighbors, parity-1 ("non-anchor") patches see only OOB/EXISTS
neighbors. No patch should ever see a mix of EXISTS and UNKNOWN — that
combination never occurs during training and would be an out-of-distribution
conditioning input.
"""

import torch

from poregen.diffusion.sampler import (
    _NB_EXISTS,
    _NB_OOB,
    _NB_UNKNOWN,
    VolumeGenerator,
)


class _FakeModelCfg:
    z_channels = 2


class _FakeModel:
    cfg = _FakeModelCfg()


class _FakeSampler:
    """Stub sampler that records per-patch nb_avail tensors via sample_batch."""

    def __init__(self) -> None:
        self.model = _FakeModel()
        self.nb_avail_calls: list[torch.Tensor] = []

    def sample_patch(
        self,
        nb_latents: torch.Tensor,
        nb_avail: torch.Tensor,
        pos_frac: torch.Tensor,
        global_por: float,
        local_por: float,
        autocast_dtype: torch.dtype = torch.bfloat16,
        return_intermediates: bool = False,
    ) -> torch.Tensor:
        # Kept for back-compat; _generate_latents uses sample_batch instead.
        self.nb_avail_calls.append(nb_avail.clone())
        C, D = nb_latents.shape[1], nb_latents.shape[2]
        return torch.zeros(C, D, D, D)

    def sample_batch(
        self,
        nb_latents: torch.Tensor,
        nb_avail: torch.Tensor,
        pos_frac: torch.Tensor,
        global_por: torch.Tensor,
        local_por: torch.Tensor,
        autocast_dtype: torch.dtype = torch.bfloat16,
    ) -> torch.Tensor:
        # Record one nb_avail row per patch in the batch (in order).
        B = nb_latents.shape[0]
        for i in range(B):
            self.nb_avail_calls.append(nb_avail[i].clone())
        C, D = nb_latents.shape[2], nb_latents.shape[3]
        return torch.zeros(B, C, D, D, D)


def _parity(gi: tuple[int, int, int]) -> int:
    return (gi[0] + gi[1] + gi[2]) % 2


def test_checkerboard_neighbor_states_never_mix() -> None:
    """Anchors (parity 0) see only OOB/UNKNOWN; non-anchors only OOB/EXISTS."""
    sampler = _FakeSampler()
    generator = VolumeGenerator(
        sampler=sampler,
        vae=None,  # not exercised — _generate_latents doesn't decode
        device=torch.device("cpu"),
        patch_size=64,
        patch_stride=64,
        latent_size=4,
    )

    grid_n = 4
    volume_shape = (grid_n * 64, grid_n * 64, grid_n * 64)

    generated, grid_origins = generator._generate_latents(volume_shape=volume_shape)

    assert len(generated) == grid_n ** 3
    assert len(sampler.nb_avail_calls) == grid_n ** 3

    # _generate_latents processes phase-0 (sorted) patches first, then
    # phase-1 (sorted) patches — mirror that order to associate each
    # recorded nb_avail call with the grid index that produced it.
    phase0 = sorted(gi for gi in grid_origins if _parity(gi) == 0)
    phase1 = sorted(gi for gi in grid_origins if _parity(gi) == 1)
    ordered_gis = phase0 + phase1

    for gi, nb_avail in zip(ordered_gis, sampler.nb_avail_calls):
        states = set(nb_avail.tolist())
        # Core invariant: never both EXISTS and UNKNOWN in the same patch.
        assert not ({_NB_EXISTS, _NB_UNKNOWN} <= states), (
            f"patch {gi} saw a mix of EXISTS and UNKNOWN: {states}"
        )
        if _parity(gi) == 0:
            assert states <= {_NB_OOB, _NB_UNKNOWN}, f"anchor {gi} saw states {states}"
        else:
            assert states <= {_NB_OOB, _NB_EXISTS}, f"non-anchor {gi} saw states {states}"
