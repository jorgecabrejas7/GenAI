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
        with pytest.raises(ValueError, match="neighbour census"):
            MEMO.neighbour_census(real)

    def test_a_translated_request_is_refused_rather_than_measured_wrong(self):
        m = manifest_for((192, 192, 192), region_offset=(0, 0, 0),
                         region_shape=(64, 64, 64))
        with pytest.raises(ValueError, match="region_offset"):
            MEMO.neighbour_census(m)


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
            "assessments": ["sampler", "porosity_global"],
            "query_shape": [192, 192, 192],
            "n_cases": 57, "n_patches": 1539,
            "generated": _summary_block(0.82, 1539),
            "real_val_floor": {"n_patches": 512, **_summary_block(0.86, 512)},
            "by_requested_phi": {"0.01": {"n_patches": 81,
                                          **_summary_block(0.80, 81)},
                                 "0.1": {"n_patches": 81,
                                         **_summary_block(0.84, 81)}},
            "by_neighbours": {MEMO.NB_PRESENT: {"n_patches": 1539,
                                                **_summary_block(0.82, 1539)}},
        }

    def test_the_generated_row_never_appears_without_its_floor(self):
        from poregen.eval_v4.report import memorisation_section

        text = memorisation_section(self._result())
        assert "generated" in text
        assert "real val floor" in text

    def test_both_breakdowns_are_printed(self):
        from poregen.eval_v4.report import memorisation_section

        text = memorisation_section(self._result())
        assert "By requested porosity" in text
        assert "phi 0.01" in text and "phi 0.1" in text
        assert "By neighbour availability" in text
        assert "neighbours present" in text

    def test_it_says_the_search_was_over_the_whole_store(self):
        from poregen.eval_v4.report import memorisation_section

        text = memorisation_section(self._result())
        assert "219580" in text and "stride-64" in text
        assert "Not a sample" in text

    def test_a_skipped_check_says_why_instead_of_printing_a_zero(self):
        from poregen.eval_v4.report import memorisation_section

        text = memorisation_section({"available": False, "reason": "no store"})
        assert "skipped" in text and "no store" in text
