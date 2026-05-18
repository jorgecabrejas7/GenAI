"""Conditioning utilities re-exported for external use.

The primary conditioning logic lives inside :class:`UNet3DDenoiser` itself
(time MLP, position MLP, porosity MLP, availability embedding).  This module
provides a standalone :class:`NeighborAvailabilityEmbedding` for callers that
need to inspect or manipulate availability embeddings outside the model, and
documents the availability state convention used across the LDM pipeline.

Neighbor availability states
-----------------------------
0 — OOB (out of bounds): the neighbor position is outside the volume.
1 — EXISTS: the neighbor latent is known and has been provided.
2 — UNKNOWN: the neighbor has not been generated yet (inference) or has
    been randomly masked during training to simulate the inference regime.
"""

from __future__ import annotations

import torch
import torch.nn as nn

NB_OOB     = 0
NB_EXISTS  = 1
NB_UNKNOWN = 2

_N_NEIGHBORS = 6


class NeighborAvailabilityEmbedding(nn.Module):
    """Learned embedding for 6-connected neighbor availability states.

    Wraps :class:`torch.nn.Embedding` with per-neighbor position offsets so
    that the embedding for "EXISTS at neighbor 0" is distinct from "EXISTS at
    neighbor 3", giving the model spatial awareness of which direction a known
    neighbor came from.

    Parameters
    ----------
    embed_dim : int
        Dimensionality of the embedding per neighbor.
    """

    def __init__(self, embed_dim: int = 8) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        # 6 neighbors × 3 states = 18 entries; each neighbor gets its own offset
        self.embedding = nn.Embedding(_N_NEIGHBORS * 3, embed_dim)

    def forward(self, nb_avail: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        nb_avail : (B, 6) long — availability states {0, 1, 2}

        Returns
        -------
        (B, 6 * embed_dim) — flattened per-neighbor embeddings
        """
        B = nb_avail.shape[0]
        offsets   = torch.arange(_N_NEIGHBORS, device=nb_avail.device).unsqueeze(0) * 3
        avail_idx = (nb_avail + offsets).long()           # (B, 6)
        emb       = self.embedding(avail_idx)             # (B, 6, embed_dim)
        return emb.view(B, _N_NEIGHBORS * self.embed_dim) # (B, 6*embed_dim)
