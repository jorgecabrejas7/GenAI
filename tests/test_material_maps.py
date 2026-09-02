"""The ldm06 material map: pooling, uint8 storage, and the specimen envelope.

The map itself is written by ``scripts/build_latent_dataset.py`` in the same
pass that encodes the latents, from the split_v3 voxel label; what is unit
tested here is the arithmetic that pass performs.  ``LatentDataset`` serving of
``cond_material`` is covered in ``test_ldm06_conditioning.py``.
"""

from __future__ import annotations

import itertools

import numpy as np
import pytest

from poregen.dataset.material import (
    decode_material_u8,
    encode_material_u8,
    pool_material_fractions,
)
from poregen.models.vae.base import CLASS_AIR, CLASS_MATERIAL, CLASS_PORE

FACTOR = 4
PATCH = 16
L = PATCH // FACTOR


class TestPooling:

    def test_block_mean_matches_brute_force(self):
        rng = np.random.default_rng(3)
        mask = rng.random((24, 20, 16)) < 0.4
        pooled = pool_material_fractions(mask, FACTOR)
        assert pooled.shape == (6, 5, 4)
        assert pooled.dtype == np.float32
        for i, j, k in itertools.product(range(6), range(5), range(4)):
            block = mask[4 * i:4 * i + 4, 4 * j:4 * j + 4, 4 * k:4 * k + 4]
            assert pooled[i, j, k] == pytest.approx(block.mean(), abs=1e-7)

    def test_remainder_is_cropped(self):
        assert pool_material_fractions(np.ones((10, 9, 11), bool), FACTOR).shape \
            == (2, 2, 2)

    def test_a_batch_of_patches_pools_per_patch(self):
        """The builder pools a whole DataLoader batch in one call."""
        rng = np.random.default_rng(4)
        batch = rng.random((7, PATCH, PATCH, PATCH)) < 0.5
        pooled = pool_material_fractions(batch, FACTOR)
        assert pooled.shape == (7, L, L, L)
        for i in range(7):
            one = pool_material_fractions(batch[i], FACTOR)
            np.testing.assert_allclose(pooled[i], one, atol=1e-7)

    def test_a_2d_input_is_refused(self):
        with pytest.raises(ValueError, match="at least 3 dimensions"):
            pool_material_fractions(np.ones((4, 4), bool), FACTOR)

    def test_factor_must_be_positive(self):
        with pytest.raises(ValueError, match="positive"):
            pool_material_fractions(np.ones((4, 4, 4), bool), 0)


class TestLabelDerivedFractions:
    """material, pore and air are three disjoint classes of the same label."""

    def test_the_three_fractions_sum_to_one(self):
        rng = np.random.default_rng(5)
        label = rng.integers(0, 3, size=(PATCH, PATCH, PATCH))
        mat = pool_material_fractions(label == CLASS_MATERIAL, FACTOR).mean()
        pore = pool_material_fractions(label == CLASS_PORE, FACTOR).mean()
        air = pool_material_fractions(label == CLASS_AIR, FACTOR).mean()
        assert float(mat + pore + air) == pytest.approx(1.0, abs=1e-6)

    def test_air_is_not_one_minus_material(self):
        """Pores are not material, so the ldm05 identity no longer holds."""
        label = np.full((PATCH, PATCH, PATCH), CLASS_MATERIAL)
        label[:4] = CLASS_PORE
        label[4:8] = CLASS_AIR
        mat = float(pool_material_fractions(label == CLASS_MATERIAL, FACTOR).mean())
        air = float((label == CLASS_AIR).mean())
        assert air == pytest.approx(0.25)
        assert mat == pytest.approx(0.5)
        assert air != pytest.approx(1.0 - mat)


class TestEncoding:

    def test_u8_round_trip_error_bound(self):
        rng = np.random.default_rng(6)
        cells = rng.random((L, L, L)).astype(np.float32)
        back = decode_material_u8(encode_material_u8(cells))
        assert np.abs(back - cells).max() <= 0.5 / 255 + 1e-7

    def test_extremes_are_exact(self):
        assert encode_material_u8(np.array([0.0, 1.0])).tolist() == [0, 255]
        assert decode_material_u8(np.array([0, 255], np.uint8)).tolist() == [0.0, 1.0]

    def test_out_of_range_values_are_clipped(self):
        assert encode_material_u8(np.array([-0.5, 1.5])).tolist() == [0, 255]


def test_compute_sample_mask_fills_internal_voids():
    """The specimen envelope still comes from the XCT for dataset building."""
    from poregen.dataset.segmentation import compute_sample_mask

    xct = np.zeros((40, 40, 40), np.uint8)
    xct[8:32, 8:32, 8:32] = 200          # bright specimen cube
    xct[18:22, 18:22, 18:22] = 10        # dark internal pore
    sm = compute_sample_mask(xct)
    assert sm is not None
    assert sm[20, 20, 20]                # internal pore is filled: material envelope
    assert sm[16:24, 16:24, 16:24].all()
    assert not sm[2, 2, 2]               # exterior air stays out
    assert not sm[:6].any()
