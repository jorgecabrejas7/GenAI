"""The microstructure statistics, against structures whose answer is known.

Every test here builds a phantom whose value can be written down from theory
before the code runs — a periodic array of spheres, whose S2(r) is the sphere
self-overlap law; spheres of a chosen radius, whose equivalent diameter is
twice it; a Poisson point process, whose Ripley K is the volume of a ball; and
a set compared with itself, whose FID is zero.  A failure therefore names the
statistic that is wrong rather than the model.
"""

from __future__ import annotations

import numpy as np
import pytest

from poregen.eval_v4 import microstructure as MS
from poregen.eval_v4.io import LABEL_PORE
from poregen.eval_v4.manifest import Manifest

COMMIT = "0" * 40


def manifest_for(shape) -> Manifest:
    return Manifest(
        assessment="unit", case="phantom", volume_shape=tuple(shape),
        git_commit=COMMIT, sampler="real", requested_material="full",
    )


# ---------------------------------------------------------------------------
# Phantoms
# ---------------------------------------------------------------------------

def sphere_lattice(shape, radius: float, spacing: int) -> np.ndarray:
    """Spheres of ``radius`` centred on a cubic lattice of pitch ``spacing``."""
    if spacing < 2 * radius + 2:
        raise ValueError("the spheres would touch; the phantom needs them separate.")
    zz, yy, xx = np.indices(shape)
    cz = (zz % spacing) - spacing // 2
    cy = (yy % spacing) - spacing // 2
    cx = (xx % spacing) - spacing // 2
    return (cz ** 2 + cy ** 2 + cx ** 2) <= radius ** 2


def sphere_overlap_s2(r, radius: float, phi: float) -> np.ndarray:
    """S2(r) of a dilute dispersion of spheres, for r below the sphere diameter.

    ``S2(r) = phi * (1 - 3r/(4R) + r^3/(16R^3))`` — the fraction of a sphere
    that still overlaps itself after a shift of r.  It holds while r is under
    2R and under the gap between neighbouring spheres, which is where the test
    reads it.
    """
    r = np.asarray(r, np.float64)
    return phi * (1.0 - 3.0 * r / (4.0 * radius) + r ** 3 / (16.0 * radius ** 3))


def poisson_points(n: int, shape, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return np.column_stack([rng.uniform(0, s, n) for s in shape])


def scattered_pores(shape, n: int, seed: int, pitch: int = 4) -> np.ndarray:
    """``n`` single-voxel pores at random sites of a coarse grid.

    The grid guarantees the pores stay ``pitch`` voxels apart, so every one is
    its own connected component and the pattern has pairs at every radius the
    K estimator looks at — which a lattice of well-separated spheres does not.
    """
    rng = np.random.default_rng(seed)
    sites = np.stack(np.indices([s // pitch for s in shape]), -1).reshape(-1, 3) * pitch
    chosen = sites[rng.choice(len(sites), size=n, replace=False)]
    pore = np.zeros(shape, bool)
    pore[tuple(chosen.T)] = True
    return pore


def label_from(pore: np.ndarray) -> np.ndarray:
    return (pore.astype(np.uint8) * LABEL_PORE)


# ---------------------------------------------------------------------------
# S2
# ---------------------------------------------------------------------------

#: A periodic array of spheres.  ``SPACING > 4 * RADIUS`` on purpose: it leaves
#: a band of radii that no pair of pore voxels can span — further than across
#: one sphere, closer than the gap to the next — where S2 must be exactly 0.
S2_RADIUS, S2_SPACING = 6.0, 32
S2_SHAPE = (MS.S2_WINDOW,) * 3


@pytest.fixture(scope="module")
def sphere_curve():
    pore = sphere_lattice(S2_SHAPE, S2_RADIUS, S2_SPACING)
    return pore, *MS.s2_radial(pore)


class TestTwoPointCorrelation:
    """S2 of a periodic array of spheres follows the self-overlap law."""

    RADIUS, SPACING = S2_RADIUS, S2_SPACING
    SHAPE = S2_SHAPE

    def test_the_zero_lag_value_is_the_phase_fraction(self, sphere_curve):
        """S2(0) = phi is the whole normalisation, so it is checked first."""
        pore, _, s2 = sphere_curve
        assert s2[0] == pytest.approx(float(pore.mean()), rel=0.06)

    def test_the_curve_follows_the_sphere_overlap_law(self, sphere_curve):
        pore, r, s2 = sphere_curve
        phi = float(pore.mean())
        near = r < 2.0 * self.RADIUS - 1.0
        expected = sphere_overlap_s2(r[near], self.RADIUS, phi)
        # 12% of phi: the sphere is rasterised on a voxel grid and the radial
        # bins average over a shell, so the law is approached, not hit.
        assert np.abs(s2[near] - expected).max() < 0.12 * phi

    def test_the_curve_falls_to_zero_between_the_spheres(self, sphere_curve):
        """No pair of pore voxels is 2R to (spacing - 2R) apart in this lattice."""
        pore, r, s2 = sphere_curve
        phi = float(pore.mean())
        gap = (r > 2.0 * self.RADIUS + 1.0) & (r < self.SPACING - 2.0 * self.RADIUS - 1.0)
        assert gap.any()
        assert np.abs(s2[gap]).max() < 0.05 * phi

    def test_a_denser_array_is_a_different_curve(self):
        """Guards the distance: two different structures must not score 0."""
        a = sphere_lattice(self.SHAPE, 6.0, 32)
        b = sphere_lattice(self.SHAPE, 3.0, 32)
        r, s2_a = MS.s2_radial(a)
        _, s2_b = MS.s2_radial(b)
        assert MS.curve_w1(r, s2_a, s2_a) == pytest.approx(0.0, abs=1e-9)
        assert MS.curve_w1(r, s2_a, s2_b) > 1.0

    def test_a_non_cubic_window_is_refused(self):
        with pytest.raises(ValueError, match="cubic window"):
            MS.s2_radial(np.zeros((8, 8, 16), bool))


class TestCurveW1:
    def test_two_shifted_spikes_are_their_separation_apart(self):
        """W1 between two point masses is the distance between them, in voxels."""
        r = np.arange(0.0, 50.0)
        a = np.zeros_like(r); a[10] = 1.0
        b = np.zeros_like(r); b[17] = 1.0
        assert MS.curve_w1(r, a, b) == pytest.approx(7.0)

    def test_an_empty_curve_has_no_distance(self):
        r = np.arange(5.0)
        assert np.isnan(MS.curve_w1(r, np.zeros(5), np.ones(5)))


# ---------------------------------------------------------------------------
# Pore size distribution
# ---------------------------------------------------------------------------

class TestPoreSizeDistribution:
    def test_spheres_of_a_known_radius_measure_twice_it(self):
        radius = 6.0
        pore = sphere_lattice((96, 96, 96), radius, 32)
        d = MS.pore_diameters(pore)
        assert d.size == 27                      # 3x3x3 lattice sites
        assert d.mean() == pytest.approx(2.0 * radius, rel=0.02)
        assert d.std() < 1e-9                    # every sphere is the same size

    def test_a_single_voxel_pore_is_the_floor_of_the_scale(self):
        pore = np.zeros((16, 16, 16), bool)
        pore[4, 4, 4] = True
        pore[12, 12, 12] = True
        d = MS.pore_diameters(pore)
        assert d.size == 2
        assert d[0] == pytest.approx((6.0 / np.pi) ** (1.0 / 3.0))

    def test_pores_meeting_at_a_corner_stay_two_pores(self):
        """6-connectivity, as the module docstring commits to."""
        pore = np.zeros((8, 8, 8), bool)
        pore[2, 2, 2] = True
        pore[3, 3, 3] = True                     # touches only at a corner
        assert MS.pore_diameters(pore).size == 2

    def test_the_w1_of_a_sample_with_itself_is_zero(self):
        d = MS.pore_diameters(sphere_lattice((96, 96, 96), 6.0, 32))
        assert MS.psd_w1(d, d) == pytest.approx(0.0)
        assert MS.psd_w1(d, d * 2.0) == pytest.approx(float(d.mean()), rel=0.02)


# ---------------------------------------------------------------------------
# Ripley's K
# ---------------------------------------------------------------------------

class TestRipleysK:
    """A Poisson process has K(r) = (4/3) pi r^3 — the volume of a ball.

    ``pi r^2`` is the two-dimensional value; these are point patterns in three
    dimensions, so the ball volume is the reference the estimator must hit.
    """

    SHAPE = (128, 128, 128)

    def test_a_poisson_process_recovers_the_csr_value(self):
        pts = poisson_points(4000, self.SHAPE, seed=7)
        r, k = MS.ripleys_k(pts, self.SHAPE, r_max=16)
        expected = MS.csr_k(r)
        keep = r >= 4                            # small r is dominated by counts of 0-3
        assert np.isfinite(k[keep]).all()
        assert np.abs(k[keep] / expected[keep] - 1.0).max() < 0.15

    def test_the_estimate_is_not_biased_low_by_the_box_edges(self):
        """The reason the border correction is here: r near the box size.

        An uncorrected count-in-radius estimator loses every neighbour that
        falls outside the box, so its K drops further below CSR as r grows.
        The border estimator must not.
        """
        pts = poisson_points(4000, self.SHAPE, seed=11)
        r, k = MS.ripleys_k(pts, self.SHAPE, r_max=24)
        ratio = k / MS.csr_k(r)
        assert abs(ratio[-1] - 1.0) < 0.15
        assert abs(ratio[-1] - 1.0) < 3.0 * abs(ratio[3] - 1.0) + 0.1

    def test_a_clustered_pattern_scores_above_csr(self):
        rng = np.random.default_rng(3)
        parents = poisson_points(120, self.SHAPE, seed=5)
        pts = np.clip(
            np.repeat(parents, 30, axis=0) + rng.normal(0, 3.0, (120 * 30, 3)),
            0, np.asarray(self.SHAPE) - 1e-6,
        )
        r, k = MS.ripleys_k(pts, self.SHAPE, r_max=16)
        assert (k[r >= 4] / MS.csr_k(r[r >= 4])).min() > 1.5

    def test_too_few_points_give_nan_and_not_a_number(self):
        r, k = MS.ripleys_k(np.zeros((1, 3)), self.SHAPE, r_max=4)
        assert np.isnan(k).all()

    def test_the_log_ratio_distance_of_a_curve_with_itself_is_zero(self):
        pts = poisson_points(2000, self.SHAPE, seed=13)
        _, k = MS.ripleys_k(pts, self.SHAPE, r_max=12)
        assert MS.log_ratio_distance(k, k) == pytest.approx(0.0)
        assert MS.log_ratio_distance(k, 2.0 * k) == pytest.approx(np.log(2.0))


# ---------------------------------------------------------------------------
# FID
# ---------------------------------------------------------------------------

class TestFid:
    def test_the_frechet_distance_of_a_feature_set_with_itself_is_zero(self):
        """The algebra, without Inception: FID(X, X) = 0 for any X."""
        rng = np.random.default_rng(0)
        f = rng.normal(size=(600, 48))
        assert MS.frechet_distance(f, f) == pytest.approx(0.0, abs=1e-6)
        shifted = f + 1.0
        assert MS.frechet_distance(f, shifted) == pytest.approx(48.0, rel=1e-3)

    def test_identical_crop_sets_have_zero_fid(self):
        """End to end through the real extractor, when torchvision is installed."""
        pytest.importorskip("torchvision", reason="FID needs the torchvision Inception")
        rng = np.random.default_rng(0)
        crops = rng.random((64, MS.FID_CROP, MS.FID_CROP)).astype(np.float32)
        feats = MS.inception_features(crops)
        assert feats.shape[1] == 2048
        assert MS.frechet_distance(feats, feats) == pytest.approx(0.0, abs=1e-3)

    def test_crops_come_only_from_inside_the_requested_material(self):
        shape = (64, 192, 192)
        xct = np.full(shape, 200, np.uint8)
        material = np.zeros(shape, bool)
        material[:, :128, :128] = True
        xct[~material] = 0                        # exterior is black
        crops = MS.fid_crops(xct, material, manifest=manifest_for(shape),
                             axis="axial", n_crops=40, seed=1)
        assert crops.shape == (40, MS.FID_CROP, MS.FID_CROP)
        assert (crops > 0).all()

    def test_the_same_seed_draws_the_same_crops(self):
        shape = (64, 128, 128)
        xct = np.arange(np.prod(shape), dtype=np.uint8).reshape(shape)
        material = np.ones(shape, bool)
        m = manifest_for(shape)
        a = MS.fid_crops(xct, material, manifest=m, axis="coronal", n_crops=16, seed=5)
        b = MS.fid_crops(xct, material, manifest=m, axis="coronal", n_crops=16, seed=5)
        np.testing.assert_array_equal(a, b)


# ---------------------------------------------------------------------------
# Memorisation
# ---------------------------------------------------------------------------

class TestMemorisation:
    def test_a_missing_store_is_an_explicit_failure_to_read(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            MS.train_latent_mu(tmp_path / "latents_that_do_not_exist")

    def test_a_store_with_another_pack_scheme_is_refused(self, tmp_path):
        import json

        (tmp_path / "metadata.json").write_text(json.dumps({
            "latent_shape": [4, 16, 16, 16],
            "storage": {"pack_scheme": "mu_only", "dtype": "float16"},
        }))
        with pytest.raises(ValueError, match="mu_then_std"):
            MS.train_latent_mu(tmp_path)

    def test_the_store_reader_returns_the_mu_half_only(self, tmp_path):
        import json

        import pandas as pd

        c, sp, n = 2, 4, 6
        (tmp_path / "metadata.json").write_text(json.dumps({
            "latent_shape": [c, sp, sp, sp],
            "storage": {"pack_scheme": "mu_then_std", "dtype": "float16"},
        }))
        split = tmp_path / "train"
        split.mkdir()
        arr = np.zeros((n, 2 * c, sp, sp, sp), np.float16)
        arr[:, :c] = 1.0                          # mu
        arr[:, c:] = 9.0                          # std, which must not be read
        arr.tofile(split / "latents.bin")
        pd.DataFrame({"source_row": np.arange(n)}).to_parquet(split / "index.parquet")

        mu = MS.train_latent_mu(tmp_path, n_sample=4)
        assert mu.shape == (4, c * sp ** 3)
        assert (mu == 1.0).all()

    def test_a_latent_of_the_wrong_width_is_refused(self):
        with pytest.raises(ValueError, match="different VAE"):
            MS.nearest_neighbour_distance(np.zeros((2, 8), np.float32),
                                          np.zeros((3, 16), np.float32))

    def test_a_patch_identical_to_a_stored_one_has_distance_zero(self):
        rng = np.random.default_rng(0)
        bank = rng.normal(size=(50, 12)).astype(np.float32)
        query = np.vstack([bank[7], bank[7] + 100.0])
        d = MS.nearest_neighbour_distance(query, bank)
        assert d[0] == pytest.approx(0.0, abs=1e-4)
        assert d[1] > 100.0


# ---------------------------------------------------------------------------
# The per-volume profile and the set comparison
# ---------------------------------------------------------------------------

class TestProfilesAndComparison:
    @staticmethod
    def _profile(pore, name, group=""):
        shape = pore.shape
        m = manifest_for(shape)
        label = label_from(pore)
        material = np.ones(shape, bool)
        s2 = MS.s2_profile(label, material, manifest=m)
        psd = MS.psd_profile(label, material, manifest=m)
        rip = MS.ripley_profile(label, material, manifest=m)
        return MS.VolumeProfile(
            case=name, group=group, phi=float(pore.mean()),
            s2_r=np.asarray(s2["r"]), s2=np.asarray(s2["s2"]),
            diameters=np.asarray(psd["diameters"]),
            ripley_r=np.asarray(rip["r"]), ripley_k=np.asarray(rip["k"]),
            n_pores=int(psd["n_pores"]),
        )

    def test_a_set_compared_with_itself_scores_zero_on_every_statistic(self):
        p = self._profile(scattered_pores((128, 128, 128), 3000, seed=2), "a")
        out = MS.compare_sets([p], [p])
        assert out["s2_w1"] == pytest.approx(0.0, abs=1e-9)
        assert out["psd_w1"] == pytest.approx(0.0)
        assert out["ripley_log_ratio"] == pytest.approx(0.0)

    def test_two_different_structures_score_above_zero(self):
        a = self._profile(sphere_lattice((128, 128, 128), 6.0, 32), "a")
        b = self._profile(sphere_lattice((128, 128, 128), 3.0, 32), "b")
        out = MS.compare_sets([a], [b])
        assert out["s2_w1"] > 0.5
        assert out["psd_w1"] > 1.0
        assert out["psd_median_a"] == pytest.approx(12.0, rel=0.03)

    def test_a_volume_larger_than_the_window_uses_several_windows(self):
        pore = sphere_lattice((192, 192, 192), 6.0, 32)
        m = manifest_for(pore.shape)
        out = MS.s2_profile(label_from(pore), np.ones(pore.shape, bool), manifest=m)
        assert out["n_windows"] == 8            # 2x2x2 at half-window stride
        assert out["window"] == MS.S2_WINDOW

    def test_a_window_that_is_mostly_exterior_is_not_measured(self):
        shape = (128, 128, 128)
        pore = sphere_lattice(shape, 6.0, 32)
        material = np.zeros(shape, bool)
        material[:64] = True                     # half the box is exterior air
        with pytest.raises(ValueError, match="requested material"):
            MS.s2_profile(label_from(pore), material, manifest=manifest_for(shape))

    def test_a_volume_smaller_than_the_analysis_window_is_refused(self):
        with pytest.raises(ValueError, match="analysis window"):
            MS.analysis_windows((64, 64, 64), MS.S2_WINDOW, MS.S2_STRIDE)


class TestRatio:
    def test_a_ratio_of_one_means_as_close_as_real_material_gets(self):
        assert MS.ratio(0.4, 0.4) == pytest.approx(1.0)
        assert MS.ratio(0.8, 0.4) == pytest.approx(2.0)

    def test_a_zero_or_missing_floor_has_no_ratio(self):
        assert MS.ratio(0.4, 0.0) is None
        assert MS.ratio(0.4, None) is None
        assert MS.ratio(None, 0.4) is None
        assert MS.ratio(float("nan"), 0.4) is None
