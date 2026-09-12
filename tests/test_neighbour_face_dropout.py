"""Per-face neighbour dropout, and the warm start the fine-tune needs.

ldm06 drops all six neighbours of an item together, so the only "one face
missing" it ever trained on had OOB as the missing face — the specimen edge.
At a chunk frontier the sampler presents the same shape with UNKNOWN instead,
which the model has never seen, and it applies the rule it has: put the pores
away from that face. Per-face dropout is what puts that combination in the
training distribution.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from poregen.diffusion.conditioning import NB_EXISTS, NB_OOB, NB_UNKNOWN
from poregen.diffusion.noise_schedule import DDPMSchedule
from poregen.training.ldm_engine import noise_neighbours


def avail(rows):
    return torch.tensor(rows, dtype=torch.long)


def run(nb_avail, *, drop_nb=0.0, drop_face=0.0, seed=0, t=500):
    torch.manual_seed(seed)
    B, K = nb_avail.shape
    sch = DDPMSchedule(T=1000, device=torch.device("cpu"))
    nb = torch.zeros(B, K, 2, 2, 2, 2)
    tt = torch.full((B,), t, dtype=torch.long)
    _, out_avail, nb_t = noise_neighbours(
        sch, nb, nb_avail, tt, nb_t_mix=1.0,
        drop_nb_p=drop_nb, drop_nb_face_p=drop_face)
    return out_avail, nb_t


class TestPerFaceDropout:

    def test_off_by_default_nothing_is_dropped(self):
        a = avail([[NB_EXISTS] * 6] * 64)
        out, _ = run(a)
        assert torch.equal(out, a)

    def test_it_produces_mixed_sets_which_the_all_six_drop_cannot(self):
        """The point of the change, as a count.

        With the all-six drop an item is either untouched or entirely null, so
        a set with some EXISTS and some UNKNOWN never occurs.
        """
        a = avail([[NB_EXISTS] * 6] * 4096)
        only_all_six, _ = run(a, drop_nb=0.5, seed=1)
        mixed = ((only_all_six == NB_EXISTS).any(1)
                 & (only_all_six == NB_UNKNOWN).any(1))
        assert mixed.sum() == 0

        per_face, _ = run(a, drop_face=0.1, seed=1)
        mixed = ((per_face == NB_EXISTS).any(1) & (per_face == NB_UNKNOWN).any(1))
        # 1 - 0.9^6 - 0.1^6 = 0.4686 for a fully-EXISTS item.
        assert mixed.float().mean() == pytest.approx(0.469, abs=0.03)

    def test_it_never_turns_an_edge_into_a_neighbour(self):
        """OOB must survive: the specimen really does end there.

        Marking an edge UNKNOWN would tell the model the volume continues past
        a face where it does not — trading the artefact being fixed for one at
        every specimen surface.
        """
        a = avail([[NB_OOB] * 6] * 512)
        out, _ = run(a, drop_face=0.9, seed=2)
        assert torch.equal(out, a)

    def test_a_mixed_store_row_keeps_its_oob_faces(self):
        a = avail([[NB_EXISTS] * 5 + [NB_OOB]] * 512)
        out, _ = run(a, drop_face=0.5, seed=3)
        assert (out[:, 5] == NB_OOB).all()
        assert (out[:, :5] != NB_OOB).all()

    def test_a_dropped_face_carries_no_timestep(self):
        """nb_t must be zero wherever the neighbour is not EXISTS."""
        a = avail([[NB_EXISTS] * 6] * 256)
        out, nb_t = run(a, drop_face=0.5, seed=4)
        assert (nb_t[out != NB_EXISTS] == 0).all()
        assert (nb_t[out == NB_EXISTS] > 0).all()

    def test_the_two_dropouts_compose(self):
        """all-six keeps the pure null arm trained; per-face adds mixed sets."""
        a = avail([[NB_EXISTS] * 6] * 4096)
        out, _ = run(a, drop_nb=0.05, drop_face=0.1, seed=5)
        all_null = (out == NB_UNKNOWN).all(1)
        mixed = (out == NB_EXISTS).any(1) & (out == NB_UNKNOWN).any(1)
        assert all_null.float().mean() > 0.04          # the null arm survives
        assert mixed.float().mean() == pytest.approx(0.45, abs=0.04)
