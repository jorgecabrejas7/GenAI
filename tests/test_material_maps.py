"""The ldm06 material map: pooling, uint8 storage, and the specimen envelope.

The map is written by ``scripts/build_latent_dataset.py`` in the same pass that
encodes the latents, from the split_v3 voxel label; what is unit tested here is
the arithmetic that pass performs.  ``LatentDataset`` serving of
``cond_material`` is covered in ``test_ldm06_conditioning.py``.

The definition under test is ``label != 2`` — the specimen ENVELOPE, pores
included — not ``label == 0``.  The envelope is 1 throughout the interior and
carries information only at the outer surface and the drilled holes; the solid
fraction would be ``1 - pore fraction`` at 4³-voxel cells, i.e. the pore mask
at 100 µm handed to the denoiser as an input for it to upsample.
"""

from __future__ import annotations

import itertools

import numpy as np
import pytest

from poregen.dataset.material import (
    air_fraction,
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


def _envelope(label: np.ndarray) -> np.ndarray:
    """What the builder pools: the specimen envelope, pores included."""
    return pool_material_fractions(label != CLASS_AIR, FACTOR)


class TestEnvelopeDefinition:

    def test_the_map_is_one_wherever_the_specimen_is(self):
        """A patch entirely inside the specimen is featureless — by design."""
        rng = np.random.default_rng(5)
        label = np.where(rng.random((PATCH, PATCH, PATCH)) < 0.2,
                         CLASS_PORE, CLASS_MATERIAL)
        assert np.all(_envelope(label) == 1.0)

    def test_pores_do_not_show_up_in_the_map(self):
        """The whole point: cond_material must not be a pore mask.

        Two patches with the SAME envelope and very different porosity must
        produce byte-identical maps, or the model can read the pore field off
        an input instead of generating it.
        """
        solid = np.full((PATCH, PATCH, PATCH), CLASS_MATERIAL)
        porous = solid.copy()
        porous[::2, ::2, ::2] = CLASS_PORE
        assert float((porous == CLASS_PORE).mean()) > 0.1
        np.testing.assert_array_equal(_envelope(solid), _envelope(porous))
        np.testing.assert_array_equal(encode_material_u8(_envelope(solid)),
                                      encode_material_u8(_envelope(porous)))

    def test_the_solid_fraction_would_have_leaked_the_pores(self):
        """Pinned as the reason for the definition, not as behaviour."""
        solid = np.full((PATCH, PATCH, PATCH), CLASS_MATERIAL)
        porous = solid.copy()
        porous[::2, ::2, ::2] = CLASS_PORE
        leaky_solid = pool_material_fractions(solid == CLASS_MATERIAL, FACTOR)
        leaky_porous = pool_material_fractions(porous == CLASS_MATERIAL, FACTOR)
        assert not np.array_equal(leaky_solid, leaky_porous)

    def test_only_air_drops_the_map_below_one(self):
        label = np.full((PATCH, PATCH, PATCH), CLASS_MATERIAL)
        label[:4] = CLASS_PORE          # inside the specimen
        label[4:8] = CLASS_AIR          # outside it
        cells = _envelope(label)
        assert float(cells[0].mean()) == pytest.approx(1.0)   # the pore plane
        assert float(cells[1].mean()) == pytest.approx(0.0)   # the air plane
        assert float(cells[2:].mean()) == pytest.approx(1.0)

    def test_a_partly_cut_cell_is_fractional(self):
        """The surface is the only place the map carries a gradient."""
        label = np.full((PATCH, PATCH, PATCH), CLASS_MATERIAL)
        label[:2] = CLASS_AIR           # half of the first 4-voxel block
        cells = _envelope(label)
        assert float(cells[0].mean()) == pytest.approx(0.5)
        assert float(cells[1:].mean()) == pytest.approx(1.0)


class TestAirFraction:

    def test_air_is_exactly_one_minus_the_envelope_mean(self):
        rng = np.random.default_rng(7)
        label = rng.integers(0, 3, size=(PATCH, PATCH, PATCH))
        cells = _envelope(label)
        air = float(air_fraction(cells))
        assert air == pytest.approx(1.0 - float(cells.mean(dtype=np.float64)),
                                    abs=1e-12)
        assert air == pytest.approx(float((label == CLASS_AIR).mean()), abs=1e-12)

    def test_it_is_batched_like_the_pooling(self):
        rng = np.random.default_rng(8)
        batch = rng.integers(0, 3, size=(5, PATCH, PATCH, PATCH))
        cells = _envelope(batch)
        air = air_fraction(cells)
        assert air.shape == (5,) and air.dtype == np.float32
        np.testing.assert_allclose(
            air, (batch == CLASS_AIR).mean(axis=(1, 2, 3)), atol=1e-7)

    def test_the_extremes(self):
        full = np.full((PATCH, PATCH, PATCH), CLASS_MATERIAL)
        empty = np.full((PATCH, PATCH, PATCH), CLASS_AIR)
        assert float(air_fraction(_envelope(full))) == pytest.approx(0.0)
        assert float(air_fraction(_envelope(empty))) == pytest.approx(1.0)


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
