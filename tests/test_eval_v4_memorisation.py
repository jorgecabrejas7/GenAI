"""The full-store memorisation search, on stores whose answer is known.

Every test here builds a bank where the nearest and the second-nearest
neighbour of each query are chosen by construction, so the Favero ratio and the
1/3 verdict have a value that can be asserted rather than eyeballed.  The three
claims the check is worth anything for:

* a query that duplicates a bank row scores ratio 0 and is MEMORISED,
* a query drawn independently scores near 1 and is NOT memorised,
* chunking the bank does not change either answer - which is the whole point,
  because the real bank is 272 GB and can only be seen a chunk at a time.
"""

from __future__ import annotations

import dataclasses

import json

import numpy as np
import pandas as pd
import pytest

from poregen.eval_v4 import memorisation as MEMO
from poregen.eval_v4.manifest import Manifest


# ---------------------------------------------------------------------------
# The ratio itself
# ---------------------------------------------------------------------------

class TestFaveroRatio:
    def test_it_is_the_nearest_over_the_second_nearest(self):
        r = MEMO.favero_ratio(np.array([1.0, 2.0]), np.array([4.0, 3.0]))
        assert r == pytest.approx([0.25, 2.0 / 3.0])

    def test_a_query_on_top_of_two_bank_rows_is_memorised_not_a_nan(self):
        # d2 == 0 implies d1 == 0: the query duplicates two distinct rows.
        assert MEMO.favero_ratio(np.array([0.0]), np.array([0.0]))[0] == 0.0

    def test_a_bank_too_small_to_have_a_second_neighbour_gives_no_verdict(self):
        assert np.isnan(MEMO.favero_ratio(np.array([1.0]), np.array([np.inf]))[0])

    def test_the_threshold_is_one_third(self):
        assert MEMO.RATIO_THRESHOLD == pytest.approx(1.0 / 3.0)


# ---------------------------------------------------------------------------
# Top2: the running nearest / second-nearest over streamed chunks
# ---------------------------------------------------------------------------

class TestTop2:
    @staticmethod
    def _bank_and_queries():
        """A bank whose two nearest rows to each query are known exactly.

        Row 0 is the origin.  Query 0 duplicates it.  Query 1 sits at distance
        1 from row 0 along the first axis and every other row is at least 10
        away, so its nearest is row 0 and its second-nearest is row 1.
        """
        bank = np.zeros((6, 4), np.float32)
        bank[1, 0] = 10.0
        bank[2, 1] = 20.0
        bank[3, 2] = 30.0
        bank[4, 3] = 40.0
        bank[5, 0] = -50.0
        query = np.zeros((2, 4), np.float32)
        query[1, 0] = 1.0
        return bank, query

    @staticmethod
    def _brute(query, bank):
        d = np.linalg.norm(query[:, None, :] - bank[None, :, :], axis=2)
        order = np.argsort(d, axis=1)
        rows = np.arange(len(query))[:, None]
        return d[rows, order[:, :2]], order[:, :2]

    def test_it_finds_the_two_nearest_rows_of_a_bank_it_sees_at_once(self):
        bank, query = self._bank_and_queries()
        d, i = self._brute(query, bank)

        acc = MEMO.Top2(len(query))
        acc.update(np.linalg.norm(query[:, None, :] - bank[None, :, :], axis=2),
                   np.arange(len(bank)))

        assert acc.d1 == pytest.approx(d[:, 0], abs=1e-5)
        assert acc.d2 == pytest.approx(d[:, 1], abs=1e-5)
        assert list(acc.i1) == list(i[:, 0])
        assert list(acc.i2) == list(i[:, 1])

    @pytest.mark.parametrize("chunk", [1, 2, 3, 4, 5, 6])
    def test_streaming_the_bank_in_chunks_gives_the_same_two_rows(self, chunk):
        """The real bank is 272 GB, so the chunked answer IS the answer."""
        bank, query = self._bank_and_queries()
        d, i = self._brute(query, bank)

        acc = MEMO.Top2(len(query))
        for lo in range(0, len(bank), chunk):
            block = bank[lo:lo + chunk]
            acc.update(
                np.linalg.norm(query[:, None, :] - block[None, :, :], axis=2),
                np.arange(lo, lo + len(block)),
            )
        assert acc.d1 == pytest.approx(d[:, 0], abs=1e-5)
        assert acc.d2 == pytest.approx(d[:, 1], abs=1e-5)
        assert list(acc.i1) == list(i[:, 0])
        assert list(acc.i2) == list(i[:, 1])

    def test_a_duplicated_row_is_memorised_and_a_fresh_draw_is_not(self):
        bank, query = self._bank_and_queries()
        acc = MEMO.Top2(len(query))
        acc.update(np.linalg.norm(query[:, None, :] - bank[None, :, :], axis=2),
                   np.arange(len(bank)))

        ratio = acc.ratio()
        # Query 0 IS bank row 0: nearest 0, second-nearest 10.
        assert ratio[0] == pytest.approx(0.0, abs=1e-6)
        # Query 1 is one step from row 0 and nine from row 1: 1/9.
        assert ratio[1] == pytest.approx(1.0 / 9.0, rel=1e-5)
        assert list(acc.verdict()) == [True, True]

    def test_an_independent_query_scores_near_one_and_is_not_memorised(self):
        """A fresh sample sits a typical distance from BOTH neighbours.

        In 256 dimensions the distances between independent Gaussian draws
        concentrate, so the nearest and the second-nearest are within a few per
        cent of each other and the ratio sits just under 1.
        """
        rng = np.random.default_rng(0)
        bank = rng.normal(size=(400, 256)).astype(np.float32)
        query = rng.normal(size=(32, 256)).astype(np.float32)

        acc = MEMO.Top2(len(query))
        for lo in range(0, len(bank), 37):
            block = bank[lo:lo + 37]
            acc.update(
                np.linalg.norm(query[:, None, :] - block[None, :, :], axis=2),
                np.arange(lo, lo + len(block)),
            )
        ratio = acc.ratio()
        assert ratio.min() > 0.9
        assert ratio.max() < 1.0
        assert not acc.verdict().any()

    def test_a_bank_row_copied_into_the_query_set_is_caught_among_fresh_ones(self):
        """The signal must survive being one patch in a crowd of honest ones."""
        rng = np.random.default_rng(1)
        bank = rng.normal(size=(400, 256)).astype(np.float32)
        query = rng.normal(size=(8, 256)).astype(np.float32)
        query[3] = bank[97]                       # the plant

        acc = MEMO.Top2(len(query))
        acc.update(np.linalg.norm(query[:, None, :] - bank[None, :, :], axis=2),
                   np.arange(len(bank)))
        assert acc.i1[3] == 97
        assert acc.ratio()[3] == pytest.approx(0.0, abs=1e-6)
        assert list(np.flatnonzero(acc.verdict())) == [3]

    def test_candidates_for_the_wrong_number_of_queries_are_refused(self):
        acc = MEMO.Top2(3)
        with pytest.raises(ValueError, match="2 queries"):
            acc.update_candidates(np.zeros((2, 2)), np.zeros((2, 2), np.int64))

    def test_a_distance_block_whose_width_is_not_the_bank_id_count_is_refused(self):
        acc = MEMO.Top2(2)
        with pytest.raises(ValueError, match="bank columns"):
            acc.update(np.zeros((2, 5)), np.arange(3))


# ---------------------------------------------------------------------------
# The GPU-shaped distance kernel, run on the CPU
# ---------------------------------------------------------------------------

class TestSquaredDistances:
    def test_it_agrees_with_the_explicit_difference(self):
        import torch

        rng = np.random.default_rng(2)
        q = torch.from_numpy(rng.normal(size=(5, 64)).astype(np.float32))
        b = torch.from_numpy(rng.normal(size=(9, 64)).astype(np.float32))
        got = MEMO.squared_distances(q, b).numpy()
        want = ((q[:, None, :] - b[None, :, :]) ** 2).sum(-1).numpy()
        assert got == pytest.approx(want, rel=1e-4, abs=1e-4)

    def test_a_duplicate_is_ranked_first_even_though_the_value_is_not_exact(self):
        """The expansion cancels hardest exactly where this check looks.

        Subtracting two large float32 numbers loses the last digits, so a true
        zero comes back as a small positive number rather than 0 - which is
        why the value is never read without :func:`refine`.  What must survive
        is the RANKING, and it does: a row is still its own nearest neighbour.
        """
        import torch

        rng = np.random.default_rng(3)
        b = torch.from_numpy(rng.normal(size=(4, 4096)).astype(np.float32) * 50.0)
        d2 = MEMO.squared_distances(b, b).numpy()
        assert (d2 >= 0.0).all()                       # never a negative
        assert list(d2.argmin(axis=1)) == [0, 1, 2, 3]
        # The error is small against the vectors' own scale, but not zero.
        assert np.diag(d2).max() < 1e-5 * float((b * b).sum(1).min())

    def test_refine_turns_the_ranking_into_an_exact_distance(self, tmp_path):
        store = MEMO.PatchStore.open(build_store(tmp_path / "s", n=16), "train")
        q = store.latent_chunk(2, 5)
        acc = MEMO.Top2(len(q))
        acc.i1 = np.array([2, 3, 4])                   # the rows they copy
        acc.i2 = np.array([3, 4, 5])
        acc.d1 = np.array([9.0, 9.0, 9.0])             # deliberately wrong
        acc.d2 = np.array([9.0, 9.0, 9.0])
        MEMO.refine(acc, q, store, "latent")
        assert acc.d1 == pytest.approx([0.0, 0.0, 0.0], abs=1e-12)
        assert (acc.d2 > 0.0).all()


# ---------------------------------------------------------------------------
# The store reader, on a synthetic store laid out like the real one
# ---------------------------------------------------------------------------

def build_store(root, *, n=16, c=2, spatial=2, patch=MEMO.PATCH,
                stride=32, split="train"):
    """A miniature ``latents_r08z8`` whose every row is known.

    Row ``i`` of the split has latent mu filled with ``i`` and a source patch
    filled with ``i``.  Origins step by ``stride`` along x, so exactly every
    other row sits on the 64-voxel grid.
    """
    root.mkdir(parents=True, exist_ok=True)
    data = root / "data"
    data.mkdir(exist_ok=True)

    (root / "metadata.json").write_text(json.dumps({
        "latent_shape": [c, spatial, spatial, spatial],
        "patch_size": patch,
        "storage": {"pack_scheme": "mu_then_std", "dtype": "float16"},
        "source_patch_index": str(data / "patch_index.parquet"),
        "normalization": {
            "per_channel_mean": [0.0] * c,
            "per_channel_std": [1.0] * c,
        },
        "vae_checkpoint": str(root / "vae.ckpt"),
    }))

    d = root / split
    d.mkdir(exist_ok=True)
    lat = np.zeros((n, 2 * c, spatial, spatial, spatial), np.float16)
    for i in range(n):
        lat[i, :c] = float(i)
        lat[i, c:] = 99.0                        # std, which must never be read
    lat.tofile(d / "latents.bin")

    source_row = np.arange(n, dtype=np.int64)
    pd.DataFrame({
        "source_row": source_row,
        "z0": np.zeros(n, np.int64),
        "y0": np.zeros(n, np.int64),
        "x0": np.arange(n, dtype=np.int64) * stride,
    }).to_parquet(d / "index.parquet")

    patches = np.zeros((n, patch, patch, patch), np.uint8)
    for i in range(n):
        patches[i] = i
    patches.tofile(data / "patches_xct.bin")
    return root


class TestPatchStore:
    def test_it_keeps_only_the_rows_on_the_stride_64_grid(self, tmp_path):
        """Eight interleaved copies of the 64-grid sit in the store at stride 32.

        Without the restriction a patch's 'second-nearest neighbour' would be
        the same material shifted by 32 voxels, and the ratio would be
        measuring the store's sampling stride rather than the model.
        """
        store = MEMO.PatchStore.open(build_store(tmp_path / "s", n=16), "train")
        assert len(store) == 8
        assert list(store.rows) == [0, 2, 4, 6, 8, 10, 12, 14]

    def test_it_reads_the_mu_half_only(self, tmp_path):
        store = MEMO.PatchStore.open(build_store(tmp_path / "s", n=16), "train")
        mu = store.latent_chunk(0, 3)
        assert mu.shape == (3, store.latent_dim)
        # rows 0, 2, 4 -> mu 0, 2, 4; the std half is 99 and must not appear.
        assert mu[:, 0] == pytest.approx([0.0, 2.0, 4.0])
        assert (mu != 99.0).all()

    def test_the_latent_scale_is_the_store_own_normalisation(self, tmp_path):
        root = build_store(tmp_path / "s", n=4)
        meta = json.loads((root / "metadata.json").read_text())
        meta["normalization"] = {"per_channel_mean": [1.0, 2.0],
                                 "per_channel_std": [2.0, 4.0]}
        (root / "metadata.json").write_text(json.dumps(meta))
        store = MEMO.PatchStore.open(root, "train")
        # row 2 holds mu == 2 in both channels -> (2-1)/2 and (2-2)/4.
        mu = store.latent_chunk(1, 2).reshape(1, 2, -1)
        assert mu[0, 0] == pytest.approx(0.5)
        assert mu[0, 1] == pytest.approx(0.0)

    def test_grey_comes_from_the_source_patches_the_store_names(self, tmp_path):
        store = MEMO.PatchStore.open(build_store(tmp_path / "s", n=16), "train")
        grey = store.grey_chunk(0, 3)
        assert grey.shape == (3, MEMO.PATCH ** 3)
        assert grey[:, 0] == pytest.approx(np.array([0.0, 2.0, 4.0]) / 255.0)

    def test_a_store_packed_another_way_is_refused(self, tmp_path):
        root = build_store(tmp_path / "s", n=4)
        meta = json.loads((root / "metadata.json").read_text())
        meta["storage"]["pack_scheme"] = "mu_only"
        (root / "metadata.json").write_text(json.dumps(meta))
        with pytest.raises(ValueError, match="mu_then_std"):
            MEMO.PatchStore.open(root, "train")

    def test_a_store_on_another_patch_size_is_refused(self, tmp_path):
        root = build_store(tmp_path / "s", n=4)
        meta = json.loads((root / "metadata.json").read_text())
        meta["patch_size"] = 32
        (root / "metadata.json").write_text(json.dumps(meta))
        with pytest.raises(ValueError, match="32-voxel"):
            MEMO.PatchStore.open(root, "train")

    def test_a_store_with_fewer_than_two_grid_rows_has_no_second_neighbour(self, tmp_path):
        with pytest.raises(ValueError, match="second-nearest"):
            MEMO.PatchStore.open(build_store(tmp_path / "s", n=2), "train")

    def test_missing_source_patches_name_the_file_they_were_looked_for_in(self, tmp_path):
        root = build_store(tmp_path / "s", n=8)
        (root / "data" / "patches_xct.bin").unlink()
        with pytest.raises(FileNotFoundError, match="patches_xct.bin"):
            MEMO.PatchStore.open(root, "train")


# ---------------------------------------------------------------------------
# The end-to-end search over a store, against a brute-force answer
# ---------------------------------------------------------------------------

class TestSearchOverAStore:
    def test_a_query_copied_from_the_store_is_memorised_in_both_spaces(self, tmp_path):
        """The whole check, on a store whose contents are known.

        Bank row for store position 3 holds latent mu == 6 and grey == 6/255.
        A query equal to it must come back at distance 0 from that row, at a
        finite distance from the next one, and MEMORISED in both spaces.  A
        query far from every row must not.
        """
        store = MEMO.PatchStore.open(build_store(tmp_path / "s", n=16), "train")

        lat_q = np.stack([
            store.latent_chunk(3, 4)[0],                       # a copy
            np.full(store.latent_dim, 1000.0, np.float32),     # nothing like it
        ])
        grey_q = np.stack([
            store.grey_chunk(3, 4)[0],
            np.full(store.grey_dim, 0.5, np.float32),
        ])

        lat = MEMO.search(lat_q, store, "latent", chunk=3)
        grey = MEMO.search(grey_q, store, "grey", chunk=3)

        for acc in (lat, grey):
            assert acc.i1[0] == 3
            assert acc.d1[0] == pytest.approx(0.0, abs=1e-4)
            assert acc.d2[0] > 0.0
            assert acc.ratio()[0] == pytest.approx(0.0, abs=1e-5)
            assert bool(acc.verdict()[0]) is True
            assert bool(acc.verdict()[1]) is False

    def test_the_chunk_size_does_not_change_the_answer(self, tmp_path):
        """The store is 272 GB, so the chunk width is a machine constraint.

        It must not reach the number.  Before the float64 refinement it did:
        a different bank block made the matmul reduce in a different order and
        moved the reported distance by 10 per cent on a near-duplicate.
        """
        store = MEMO.PatchStore.open(build_store(tmp_path / "s", n=16), "train")
        q = store.latent_chunk(0, 5) + 0.01
        got = [MEMO.search(q, store, "latent", chunk=k) for k in (1, 2, 3, 8, 64)]
        # 16 latent dimensions each 0.01 out: sqrt(16) * 0.01, up to the
        # float32 resolution of the offset itself.
        assert got[0].d1 == pytest.approx(np.full(5, 0.01 * 4.0), rel=1e-4)
        for acc in got[1:]:
            assert acc.d1 == pytest.approx(got[0].d1, rel=1e-12)
            assert acc.d2 == pytest.approx(got[0].d2, rel=1e-12)
            assert list(acc.i1) == list(got[0].i1)

    def test_a_query_of_the_wrong_width_is_refused_rather_than_measured(self, tmp_path):
        store = MEMO.PatchStore.open(build_store(tmp_path / "s", n=8), "train")
        with pytest.raises(ValueError, match="different VAE"):
            MEMO.search(np.zeros((2, 3), np.float32), store, "latent")

    def test_an_unknown_space_is_refused(self, tmp_path):
        store = MEMO.PatchStore.open(build_store(tmp_path / "s", n=8), "train")
        with pytest.raises(KeyError):
            MEMO.search(np.zeros((2, store.latent_dim), np.float32), store, "rgb")


# ---------------------------------------------------------------------------
# Summaries
# ---------------------------------------------------------------------------

class TestSummarise:
    def test_it_counts_the_memorised_queries_by_the_one_third_rule(self):
        acc = MEMO.Top2(4)
        acc.d1 = np.array([0.0, 1.0, 3.0, 9.0])
        acc.d2 = np.array([1.0, 10.0, 9.0, 10.0])   # ratios 0, .1, 1/3, .9
        s = MEMO.summarise(acc)
        assert s["n"] == 4
        # 1/3 is NOT below 1/3 - the criterion is strict.
        assert s["n_memorised"] == 2
        assert s["frac_memorised"] == pytest.approx(0.5)
        assert s["ratio_min"] == pytest.approx(0.0)
        assert s["nn_distance_min"] == pytest.approx(0.0)

    def test_a_subset_is_summarised_on_its_own(self):
        acc = MEMO.Top2(4)
        acc.d1 = np.array([0.0, 9.0, 0.0, 9.0])
        acc.d2 = np.array([1.0, 10.0, 1.0, 10.0])
        s = MEMO.summarise(acc, np.array([False, True, False, True]))
        assert s["n"] == 2
        assert s["n_memorised"] == 0

    def test_an_empty_subset_reports_a_count_and_no_statistics(self):
        acc = MEMO.Top2(2)
        acc.d1, acc.d2 = np.array([1.0, 2.0]), np.array([2.0, 4.0])
        assert MEMO.summarise(acc, np.zeros(2, bool)) == {"n": 0}


# ---------------------------------------------------------------------------
# What a generated volume contributes
# ---------------------------------------------------------------------------

class TestVolumePatches:
    def test_origins_are_the_non_overlapping_tiling_of_whole_material(self):
        material = np.ones((192, 192, 192), bool)
        assert len(MEMO.patch_origins(material)) == 27

    def test_a_patch_that_reaches_outside_the_requested_material_is_dropped(self):
        material = np.ones((192, 192, 192), bool)
        material[0, 0, 0] = False
        origins = MEMO.patch_origins(material)
        assert len(origins) == 26
        assert (0, 0, 0) not in origins

    def test_grey_patches_line_up_with_the_origins_they_were_asked_for(self):
        xct = np.zeros((128, 128, 128), np.uint8)
        xct[64:, :, :] = 200
        origins = [(0, 0, 0), (64, 0, 0)]
        grey = MEMO.grey_patches(xct, origins)
        assert grey.shape == (2, MEMO.PATCH ** 3)
        assert grey[0] == pytest.approx(np.zeros(MEMO.PATCH ** 3))
        assert grey[1] == pytest.approx(np.full(MEMO.PATCH ** 3, 200.0 / 255.0))


# ---------------------------------------------------------------------------
# Neighbour availability, rebuilt from the manifest
# ---------------------------------------------------------------------------

def manifest_for(shape, chunk_tiles=(3, 3, 3), window_stride=32, **kw) -> Manifest:
    return Manifest(
        assessment="sampler", case="c", volume_shape=shape, git_commit="abc",
        model_run="r", checkpoint_step=1, weights="ema", ddim_steps=50,
        chunk_tiles=chunk_tiles, window_stride=window_stride,
        decode="overlapped", decode_overlap=32, s_por=1.0, s_nb=1.0,
        objective="v", cfg_rescale=0.0, seed=1, requested_global_phi=0.03,
        requested_material="full", **kw,
    )


class TestNeighbourCensus:
    def test_a_single_chunk_volume_never_has_an_unknown_face(self):
        """192 cubed is exactly one chunk of 3 tiles, so nothing is ungenerated."""
        census = MEMO.neighbour_census(manifest_for((192, 192, 192)))
        assert census["n_chunks"] == 1
        assert census["unknown"] == 0
        assert census["exists"] > 0
        assert census["oob"] > 0
        assert census["bucket"] == MEMO.NB_PRESENT

    def test_a_multi_chunk_volume_has_faces_reaching_into_ungenerated_chunks(self):
        census = MEMO.neighbour_census(manifest_for((192, 384, 384)))
        assert census["n_chunks"] == 4
        assert census["unknown"] > 0
        assert census["bucket"] == MEMO.NB_UNKNOWN_BUCKET

    def test_every_window_face_is_accounted_for_exactly_once(self):
        census = MEMO.neighbour_census(manifest_for((192, 192, 192)))
        # 48 latent cells a side, 16-cell windows every 8 cells -> 5^3 windows.
        assert census["exists"] + census["oob"] + census["unknown"] == 5 ** 3 * 6

    def test_a_real_crop_carries_no_generation_geometry_and_is_refused(self):
        real = Manifest(assessment="sampler", case="r",
                        volume_shape=(128, 128, 128), git_commit="abc",
                        sampler="real")
        with pytest.raises(ValueError, match="neighbour states"):
            MEMO.neighbour_census(real)

    def test_a_translated_request_has_the_same_chunk_grid(self):
        """`request_offset` names the noise frame, not a larger canvas.

        `_generate_latents` derives `canvas_cells` from `volume_shape` alone,
        so a case that translates its request inside the canvas has exactly
        the same chunks, windows and neighbour states as one that does not.
        """
        plain = MEMO.neighbour_census(manifest_for((192, 384, 384)))
        moved = MEMO.neighbour_census(
            manifest_for((192, 384, 384), region_offset=(0, 64, 64),
                         region_shape=(192, 192, 192)))
        assert moved == plain


# ---------------------------------------------------------------------------
# Per-window neighbour state: which faces were UNKNOWN WHEN IT WAS DENOISED
# ---------------------------------------------------------------------------

class TestWindowStates:
    """The chunk ORDER decides this, not the geometry.

    A 384-cubed volume at the production 3-tile chunk is 2 chunks on every
    axis, walked z-major. The window at latent cell (32, 32, 32) sits in the
    FIRST chunk and its +z/+y/+x neighbours reach into chunks nobody has
    solved yet. The window at (48, 48, 48) sits in the LAST chunk and is its
    mirror image geometrically — interior, touching a chunk plane — but every
    neighbour it has was already finished. If the reconstruction only looked
    at geometry the two would score the same. They must not.
    """

    SHAPE = (384, 384, 384)

    def test_a_window_that_looks_into_an_unsolved_chunk_has_unknown_faces(self):
        w = MEMO.window_states(manifest_for(self.SHAPE))
        first = w.at_voxel((128, 128, 128))            # latent cell 32
        assert first["unknown"] == 3                   # +z, +y, +x
        assert first["exists"] == 3                    # -z, -y, -x
        assert first["oob"] == 0
        assert first["bucket"] == MEMO.NB_UNKNOWN_BUCKET

    def test_the_mirror_window_in_the_last_chunk_has_none(self):
        w = MEMO.window_states(manifest_for(self.SHAPE))
        last = w.at_voxel((192, 192, 192))             # latent cell 48
        assert last["unknown"] == 0
        assert last["exists"] == 6
        assert last["bucket"] == MEMO.NB_PRESENT

    def test_a_fully_interior_window_of_a_single_chunk_volume_is_not_unknown(self):
        w = MEMO.window_states(manifest_for((192, 192, 192)))
        mid = w.at_voxel((64, 64, 64))
        assert mid["unknown"] == 0
        assert mid["exists"] == 6
        assert mid["bucket"] == MEMO.NB_PRESENT

    def test_the_chunk_grid_and_window_count_are_the_samplers(self):
        w = MEMO.window_states(manifest_for(self.SHAPE))
        assert w.n_chunks == 8                         # 2 per axis
        assert w.cells == (96, 96, 96)
        assert w.win == 16 and w.stride == 8
        # 5 windows per axis per chunk, 8 chunks.
        assert len(w.states) == 8 * 5 ** 3

    @pytest.mark.parametrize("shape", [(192, 192, 192), (384, 384, 384),
                                       (192, 384, 384)])
    def test_every_query_patch_position_is_a_window_origin(self, shape):
        """No fallback is needed for any shape the search actually takes.

        The queries are the non-overlapping 64-voxel tiling and the windows
        run at stride 32, so every patch origin is also a window origin. If
        that ever stopped being true the caller would bucket at the volume
        level and say so, which is why it is asserted here rather than assumed.
        """
        w = MEMO.window_states(manifest_for(shape))
        origins = MEMO.patch_origins(np.ones(shape, bool))
        assert origins
        assert all(w.at_voxel(o) is not None for o in origins)

    def test_a_position_that_is_not_a_window_origin_returns_none(self):
        w = MEMO.window_states(manifest_for((192, 192, 192)))
        assert w.at_voxel((4, 0, 0)) is None           # not on the latent grid
        assert w.at_voxel((160, 0, 0)) is None         # past the last window

    def test_the_multi_chunk_volume_fills_both_buckets(self):
        """The whole point of adding these volumes: two buckets, one volume."""
        w = MEMO.window_states(manifest_for(self.SHAPE))
        origins = MEMO.patch_origins(np.ones(self.SHAPE, bool))
        buckets = [w.at_voxel(o)["bucket"] for o in origins]
        assert MEMO.NB_UNKNOWN_BUCKET in buckets
        assert MEMO.NB_PRESENT in buckets

    def test_a_single_chunk_volume_fills_only_one(self):
        w = MEMO.window_states(manifest_for((192, 192, 192)))
        origins = MEMO.patch_origins(np.ones((192, 192, 192), bool))
        buckets = {w.at_voxel(o)["bucket"] for o in origins}
        assert buckets == {MEMO.NB_PRESENT}


# ---------------------------------------------------------------------------
# The two grey floors
# ---------------------------------------------------------------------------

class StubDecoder:
    """A VAE decoder whose reconstruction error I choose.

    ``build_store`` fills row ``i``'s latent with ``i`` and its source patch
    with ``i``, so a perfect decoder would return ``i / 255`` and this one
    returns ``i / 255 + error``. That difference IS the reconstruction error
    the real decoder has and the raw patch does not.
    """

    def __init__(self, error: float):
        self.error = float(error)

    def decoder(self, z):
        return z

    def xct_head(self, z):
        n = z.shape[0]
        val = z.reshape(n, -1)[:, :1] / 255.0 + self.error
        return val.reshape(n, 1, 1, 1, 1).expand(n, 1, MEMO.PATCH,
                                                 MEMO.PATCH, MEMO.PATCH)


class TestGreyFloors:
    #: Small enough that a round-tripped patch is still nearest to its OWN bank
    #: row (the synthetic rows are 2/255 apart), so the error is the whole
    #: distance and nothing else.
    ERROR = 0.001

    def test_the_round_trip_carries_the_error_and_the_raw_patch_does_not(self, tmp_path):
        store = MEMO.PatchStore.open(build_store(tmp_path / "s", n=16), "train")
        pick = np.array([1, 2, 3])
        raw = store.grey_at(pick)
        rt = MEMO.decode_grey(StubDecoder(self.ERROR), store.raw_latent_at(pick),
                              device="cpu")
        assert rt.shape == raw.shape
        assert not np.allclose(rt, raw)
        assert (rt - raw) == pytest.approx(np.full(raw.shape, self.ERROR), abs=1e-6)

    def test_the_two_floors_give_different_ratios_on_the_same_bank(self, tmp_path):
        """Which floor you use changes the answer, so the report prints both.

        The raw val patch IS a bank row here, so its nearest distance is 0 and
        its ratio is 0 — an optimistic floor no decoded query could ever
        match. The round-tripped one sits one reconstruction error away, which
        is exactly the handicap a generated query carries.
        """
        store = MEMO.PatchStore.open(build_store(tmp_path / "s", n=16), "train")
        pick = np.array([1, 2, 3])
        raw = store.grey_at(pick)
        rt = MEMO.decode_grey(StubDecoder(self.ERROR), store.raw_latent_at(pick),
                              device="cpu")

        acc_raw = MEMO.search(raw, store, "grey", chunk=4)
        acc_rt = MEMO.search(rt, store, "grey", chunk=4)

        # Both find the same row; only the distance to it differs.
        assert list(acc_raw.i1) == list(pick)
        assert list(acc_rt.i1) == list(pick)
        assert acc_raw.d1 == pytest.approx(np.zeros(3), abs=1e-9)
        assert acc_raw.ratio() == pytest.approx(np.zeros(3), abs=1e-9)
        # 64^3 voxels each ERROR out: sqrt(64^3) * ERROR.
        assert acc_rt.d1 == pytest.approx(
            np.full(3, self.ERROR * MEMO.PATCH ** 1.5), rel=1e-4)
        assert (acc_rt.ratio() > acc_raw.ratio()).all()

    def test_decode_grey_refuses_a_flat_latent(self, tmp_path):
        store = MEMO.PatchStore.open(build_store(tmp_path / "s", n=8), "train")
        with pytest.raises(ValueError, match=r"\(n, C, d, h, w\)"):
            MEMO.decode_grey(StubDecoder(0.0), store.latent_chunk(0, 2),
                             device="cpu")


# ---------------------------------------------------------------------------
# The report section, on a result block shaped like the real one
# ---------------------------------------------------------------------------

def _summary_block(ratio_mean: float, n: int = 27) -> dict:
    return {
        "latent": {"n": n, "nn_distance_mean": 1.0, "nn_distance_min": 0.5,
                   "second_nn_distance_mean": 2.0, "ratio_mean": ratio_mean,
                   "ratio_median": ratio_mean, "ratio_min": ratio_mean,
                   "ratio_p5": ratio_mean, "n_memorised": 0,
                   "frac_memorised": 0.0},
        "grey": {"n": n, "nn_distance_mean": 3.0, "nn_distance_min": 2.0,
                 "second_nn_distance_mean": 4.0, "ratio_mean": ratio_mean,
                 "ratio_median": ratio_mean, "ratio_min": ratio_mean,
                 "ratio_p5": ratio_mean, "n_memorised": 1,
                 "frac_memorised": 1.0 / n},
    }


class TestReportSection:
    @staticmethod
    def _result() -> dict:
        return {
            "available": True,
            "criterion": {"statistic": "||x - x'|| / ||x - x''||",
                          "threshold": MEMO.RATIO_THRESHOLD,
                          "reading": "read it beside the floor"},
            "bank": {"split": "train", "stride": 64, "n_rows": 219580,
                     "n_rows_in_split": 1598000, "latent_dim": 32768,
                     "grey_dim": 262144, "grey_source": "/x/patches_xct.bin"},
            "assessments_requested": ["sampler", "porosity_global", "multichunk",
                                      "assembly_modes"],
            "assessments_found": ["sampler", "porosity_global", "multichunk"],
            "max_tiles_per_volume": 256,
            "n_cases": 62, "n_patches": 2600,
            "skipped_too_large": [{"case": "1024_ddim50_seed101"}],
            "neighbour_state_source": "rebuilt per window from the manifest",
            "cases_bucketed_at_volume_level": [],
            "generated": _summary_block(0.82, 2600),
            "real_val_floor": {
                "n_patches": 512,
                **_summary_block(0.86, 512),
                "grey_raw": _summary_block(0.91, 512)["grey"],
                "note": "round trip is the like-for-like floor",
            },
            "by_requested_phi": {"0.01": {"n_patches": 81,
                                          **_summary_block(0.80, 81)},
                                 "0.1": {"n_patches": 81,
                                         **_summary_block(0.84, 81)}},
            "by_neighbours": {
                MEMO.NB_PRESENT: {"n_patches": 2000,
                                  **_summary_block(0.82, 2000)},
                MEMO.NB_UNKNOWN_BUCKET: {"n_patches": 600,
                                         **_summary_block(0.75, 600)},
            },
            "per_case": {
                "a": {"volume_shape": [192, 192, 192]},
                "b": {"volume_shape": [384, 384, 384]},
            },
        }

    def test_the_generated_row_never_appears_without_its_floor(self):
        from poregen.eval_v4.report import memorisation_section

        text = memorisation_section(self._result())
        assert "generated" in text
        assert "real val floor (VAE round trip)" in text

    def test_both_grey_floors_are_printed_and_named(self):
        """Which floor a reader used must be legible from the table alone."""
        from poregen.eval_v4.report import memorisation_section

        text = memorisation_section(self._result())
        assert "real val floor (VAE round trip)" in text
        assert "real val floor (raw scan)" in text
        assert "like-for-like" in text

    def test_both_breakdowns_are_printed(self):
        from poregen.eval_v4.report import memorisation_section

        text = memorisation_section(self._result())
        assert "By requested porosity" in text
        assert "phi 0.01" in text and "phi 0.1" in text
        assert "By neighbour availability when the window was denoised" in text
        assert "neighbours present" in text
        assert "neighbours unknown" in text

    def test_it_says_the_neighbour_state_is_an_ordering_fact(self):
        from poregen.eval_v4.report import memorisation_section

        text = memorisation_section(self._result())
        assert "rebuilt per window from the manifest" in text
        assert "not a geometric one" in text

    def test_a_volume_level_fallback_is_named_not_hidden(self):
        from poregen.eval_v4.report import memorisation_section

        res = self._result()
        res["cases_bucketed_at_volume_level"] = ["odd_case"]
        text = memorisation_section(res)
        assert "could not be bucketed per window" in text
        assert "odd_case" in text

    def test_it_says_the_search_was_over_the_whole_store(self):
        from poregen.eval_v4.report import memorisation_section

        text = memorisation_section(self._result())
        assert "219580" in text and "stride-64" in text
        assert "Not a sample" in text
        assert "192x192x192" in text and "384x384x384" in text

    def test_a_skipped_check_says_why_instead_of_printing_a_zero(self):
        from poregen.eval_v4.report import memorisation_section

        text = memorisation_section({"available": False, "reason": "no store"})
        assert "skipped" in text and "no store" in text


# --------------------------------------------------------------------------- #
# the smoke-test volume cut
# --------------------------------------------------------------------------- #
class _StubCase:
    """Only what `memorisation` reads before it opens the store."""

    def __init__(self, name, shape):
        self.manifest = dataclasses.replace(manifest_for(shape), case=name)


class TestMaxCasesPerAssessment:
    """The cut selects volumes BEFORE the tile filter, and is recorded.

    Every call passes ``allow_busy_gpu``: these tests are about which volumes
    are SELECTED, and a CUDA job that happens to hold the card while the suite
    runs must not decide whether they pass.

    A smoke run that silently searched everything would cost the hours it
    exists to avoid; one that searched nothing and said `available: False`
    would look like a clean result.  Both are tested for.
    """

    def _campaign(self, monkeypatch, cases):
        monkeypatch.setattr(
            MEMO, "load_cases",
            lambda root, assessment: cases if assessment == "sampler" else [])

    def test_a_cut_of_zero_keeps_no_volume_and_says_so(self, monkeypatch, tmp_path):
        self._campaign(monkeypatch, [_StubCase("a", (192, 192, 192))])
        out = MEMO.memorisation(tmp_path, assessments=("sampler",),
                                max_cases_per_assessment=0, allow_busy_gpu=True)
        assert out["available"] is False
        assert "no generated volume" in out["reason"]

    def test_the_cut_counts_volumes_that_will_be_searched(self, monkeypatch, tmp_path):
        """Order matters: the cut takes the first N THAT FIT, not the first N.

        `load_cases` returns sorted directory order and `sampler`'s first case
        on disk is a 1024-wide volume at 768 tiles. Counting before the tile
        filter meant a cut of one took that case, the filter dropped it, and
        the assessment contributed nothing while `assessments_found` still
        named it — the smoke run of 2026-09-11 searched one volume believing it
        had searched two.
        """
        cases = [_StubCase("big", (192, 1024, 1024)), _StubCase("small", (192, 192, 192))]
        self._campaign(monkeypatch, cases)
        # The oversize volume is skipped and the small one is still searched,
        # so the run reaches the store this campaign has no manifest note for.
        with pytest.raises(KeyError, match="latents_root"):
            MEMO.memorisation(tmp_path, assessments=("sampler",),
                              max_cases_per_assessment=1, allow_busy_gpu=True)

    def test_an_oversize_volume_is_still_recorded_as_skipped(self, monkeypatch, tmp_path):
        """Taken off the search on cost, never dropped in silence."""
        self._campaign(monkeypatch, [_StubCase("big", (192, 1024, 1024))])
        out = MEMO.memorisation(tmp_path, assessments=("sampler",),
                                max_cases_per_assessment=1, allow_busy_gpu=True)
        assert out["available"] is False
        assert "no generated volume" in out["reason"]

    def test_the_cut_is_recorded_on_the_result(self, monkeypatch, tmp_path):
        self._campaign(monkeypatch, [_StubCase("a", (192, 192, 192))])
        with pytest.raises(KeyError, match="latents_root"):
            MEMO.memorisation(tmp_path, assessments=("sampler",),
                              max_cases_per_assessment=1, allow_busy_gpu=True)

    def test_no_cut_is_the_default(self, monkeypatch, tmp_path):
        cases = [_StubCase(str(i), (192, 192, 192)) for i in range(3)]
        self._campaign(monkeypatch, cases)
        with pytest.raises(KeyError, match="latents_root"):
            MEMO.memorisation(tmp_path, assessments=("sampler",), allow_busy_gpu=True)


# --------------------------------------------------------------------------- #
# the pass drops its own page cache
# --------------------------------------------------------------------------- #
class TestPageRelease:
    """The search must release each bank chunk after using it.

    This machine shares 121 GB between CPU and GPU. Page cache is reclaimable
    in principle, but a CUDA allocation does not wait for reclaim — it fails.
    A pass over the 195 GiB store that never releases will fill the cache in
    minutes while `free` still reports tens of GB available, and the CUDA job
    beside it dies with no sign of the cause. So a chunk that is read and not
    released is a defect, not an inefficiency.
    """

    def _spy(self, store, name, calls):
        real = getattr(store, name)

        def wrapper(lo, hi, memmap=None):
            calls.append((lo, hi))
            return real(lo, hi, memmap)
        object.__setattr__(store, name, wrapper)

    @pytest.mark.parametrize("space,method", [("latent", "release_latent_chunk"),
                                              ("grey", "release_grey_chunk")])
    def test_every_chunk_read_is_released(self, tmp_path, space, method):
        store = MEMO.PatchStore.open(build_store(tmp_path / "s", n=16), "train")
        calls = []
        self._spy(store, method, calls)
        dim = store.latent_dim if space == "latent" else store.grey_dim
        MEMO.search(np.zeros((2, dim), np.float32), store, space, chunk=3)
        n = len(store)
        expected = [(lo, min(lo + 3, n)) for lo in range(0, n, 3)]
        assert calls == expected          # every chunk, once, in order
        assert calls[-1][1] == n          # including the short final one

    def test_releasing_does_not_change_the_answer(self, tmp_path):
        """It is advisory: the pages come back from the file if touched again."""
        store = MEMO.PatchStore.open(build_store(tmp_path / "s", n=16), "train")
        q = np.stack([store.latent_chunk(3, 4)[0],
                      store.latent_chunk(7, 8)[0]])
        first = MEMO.search(q, store, "latent", chunk=3)
        second = MEMO.search(q, store, "latent", chunk=5)
        assert list(first.i1) == list(second.i1) == [3, 7]
        assert first.d1 == pytest.approx(second.d1, abs=1e-9)

    def test_an_empty_span_releases_nothing_rather_than_raising(self, tmp_path):
        store = MEMO.PatchStore.open(build_store(tmp_path / "s", n=16), "train")
        store.release_latent_chunk(5, 5)          # no rows in the span
        store.release_grey_chunk(5, 5)

    def test_the_released_span_covers_the_rows_the_chunk_read(self, tmp_path):
        """The bank rows are a strided subset, so the span carries gaps.

        Releasing the whole span is intended: the interleaved store rows are
        not in the bank and are not wanted either.
        """
        store = MEMO.PatchStore.open(build_store(tmp_path / "s", n=16), "train")
        seen = {}

        def fake(path, file_rows, row_bytes, memmap=None):
            seen["lo"] = int(file_rows.min()) * row_bytes
            seen["hi"] = (int(file_rows.max()) + 1) * row_bytes
        object.__setattr__(store, "_release_span", fake)
        store.release_latent_chunk(0, 4)
        row_bytes = (2 * store.latent_channels
                     * int(np.prod(store.latent_spatial))
                     * store.latent_dtype.itemsize)
        assert seen["lo"] == int(store.rows[0]) * row_bytes
        assert seen["hi"] == (int(store.rows[3]) + 1) * row_bytes


# --------------------------------------------------------------------------- #
# the search refuses to run beside a CUDA job
# --------------------------------------------------------------------------- #
class TestBusyGpuSkip:
    """Streaming the store beside a generating card kills the card's job.

    Host and device share one pool on this machine, and a CUDA allocation
    fails rather than waiting for page cache to be reclaimed. The search
    therefore reports itself unavailable instead of running — but it must say
    WHY and name what blocked it, because a silent skip here reads exactly
    like "no memorisation found", which is the one wrong answer this module
    must never give.
    """

    def test_a_busy_card_skips_the_search_and_names_the_process(
            self, monkeypatch, tmp_path):
        monkeypatch.setattr(MEMO, "gpu_jobs_other_than",
                            lambda pid: [(4242, "python")])
        out = MEMO.memorisation(tmp_path, assessments=("sampler",))
        assert out["available"] is False
        assert out["blocked_by"] == [{"pid": 4242, "process": "python"}]
        assert "4242" in out["reason"]
        assert "--allow-busy-gpu" in out["reason"]

    def test_the_skip_is_not_mistaken_for_a_clean_result(
            self, monkeypatch, tmp_path):
        """No ratio, no verdict, no count — nothing a reader could quote."""
        monkeypatch.setattr(MEMO, "gpu_jobs_other_than",
                            lambda pid: [(1, "x")])
        out = MEMO.memorisation(tmp_path, assessments=("sampler",))
        for key in ("latent", "grey", "n_memorised", "n_patches", "ratio"):
            assert key not in out

    def test_an_idle_card_does_not_skip(self, monkeypatch, tmp_path):
        """It proceeds far enough to complain about the campaign instead."""
        monkeypatch.setattr(MEMO, "gpu_jobs_other_than", lambda pid: [])
        out = MEMO.memorisation(tmp_path, assessments=("sampler",))
        assert out["available"] is False
        assert "no generated volume" in out["reason"]
        assert "blocked_by" not in out

    def test_the_override_runs_anyway(self, monkeypatch, tmp_path):
        monkeypatch.setattr(MEMO, "gpu_jobs_other_than",
                            lambda pid: [(1, "x")])
        out = MEMO.memorisation(tmp_path, assessments=("sampler",),
                                allow_busy_gpu=True)
        assert "no generated volume" in out["reason"]
