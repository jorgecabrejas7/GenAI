"""The coherent field must deliver the correlation length it was asked for.

Smoothing white noise with a Gaussian kernel of standard deviation s gives an
autocorrelation that is itself Gaussian with standard deviation s*sqrt(2):

    rho(r) = exp(-r^2 / (4 s^2))

so rho reaches 1/e at r = 2s, NOT at r = s. `build_porosity_field` used
sigma = L/stride and therefore delivered 2L. These tests pin the relation
itself, on synthetic noise where the answer is analytic, and then on the real
generator.
"""

from __future__ import annotations

import numpy as np
import pytest
from scipy.ndimage import gaussian_filter

from poregen.diffusion.porosity_field import (
    PHI_MAX,
    PHI_MIN,
    build_porosity_field,
)

STRIDE = 64


def one_over_e(rho: np.ndarray, lags: np.ndarray) -> float:
    below = np.flatnonzero(rho < np.exp(-1.0))
    if not below.size or below[0] == 0:
        return float("nan")
    i = below[0]
    t = (rho[i - 1] - np.exp(-1.0)) / (rho[i - 1] - rho[i])
    return float(lags[i - 1] + t)


def axis_length(field: np.ndarray, axis: int) -> float:
    f = field - field.mean()
    n = field.shape[axis]
    lags = np.arange(0, min(n - 1, 40))
    rho = []
    for lag in lags:
        a = np.take(f, np.arange(0, n - lag), axis=axis)
        b = np.take(f, np.arange(lag, n), axis=axis)
        sa, sb = a.std(), b.std()
        rho.append(float((a * b).mean() / (sa * sb)) if sa > 0 and sb > 0 else np.nan)
    return one_over_e(np.asarray(rho), lags) * STRIDE


class TestTheAnalyticRelation:
    """rho(r) = exp(-r^2/4s^2), so the 1/e length is 2s and not s."""

    @pytest.mark.parametrize("sigma", [2.0, 4.0, 8.0])
    def test_smoothed_white_noise_decorrelates_at_twice_sigma(self, sigma):
        rng = np.random.default_rng(0)
        x = gaussian_filter(rng.standard_normal(200_000), sigma=sigma, mode="wrap")
        x = (x - x.mean()) / x.std()
        lags = np.arange(0, 60)
        ac = np.correlate(x, x, "full")[len(x) - 1:len(x) + len(lags) - 1] / len(x)
        assert one_over_e(ac / ac[0], lags) == pytest.approx(2 * sigma, rel=0.05)


class TestTheGenerator:

    @staticmethod
    @pytest.fixture(scope="class")
    def sampler_and_corr():
        from poregen.diffusion.porosity_field import (
            DEFAULT_TD_RESULTS,
            DEFAULT_TE_RESULTS,
            load_corr_lengths_voxels,
            load_sampler,
        )
        return load_sampler(DEFAULT_TE_RESULTS), load_corr_lengths_voxels(DEFAULT_TD_RESULTS)

    def test_it_delivers_every_WELL_DEFINED_requested_length(self, sampler_and_corr):
        """A grid big enough that the box is not the limit.

        Only the axes whose T-D curve actually decays to 1/e are checked. On
        the train refit the in-plane y curve does not: it plateaus near 0.45
        and dips to 0.359 once against a 1/e of 0.3679, so its "length" of 2521
        voxels is where noise crossed a line. Asserting the generator delivers
        that would be asserting it reproduces an artefact.
        """
        import json

        from poregen.diffusion.porosity_field import (
            DEFAULT_TD_RESULTS,
            corr_length_is_well_defined,
        )

        sampler, corr = sampler_and_corr
        patch_level = json.loads(DEFAULT_TD_RESULTS.read_text())["patch_level"]
        grid = (24, 96, 160)
        got = []
        for seed in (101, 202, 303):
            f = build_porosity_field(grid, 0.03, sampler, corr,
                                     stride_voxels=STRIDE, seed=seed)
            got.append([axis_length(f, a) for a in range(3)])
        mean = np.nanmean(got, axis=0)
        checked = 0
        for a, name in enumerate(("z", "y", "x")):
            if not corr_length_is_well_defined(patch_level[f"{name}_volume_mean_removed"]):
                continue
            checked += 1
            assert mean[a] == pytest.approx(corr[a], rel=0.12), (
                f"{name}: asked {corr[a]:.0f}, delivered {mean[a]:.0f}")
        assert checked >= 2, "at least two axes should have a usable length"

    def test_the_old_sigma_delivered_twice_the_requested_length(
            self, sampler_and_corr):
        """Guarding the fix: the previous formula must FAIL this."""
        from poregen.diffusion.porosity_field import _draw_marginal

        sampler, corr = sampler_and_corr
        grid = (24, 96, 160)
        rng = np.random.default_rng(101)
        f = _draw_marginal(0.03, sampler, rng.uniform(0, 1, size=grid))
        # sigma = L/stride, as it was.
        f = gaussian_filter(f, sigma=tuple(c / STRIDE for c in corr), mode="nearest")
        f *= 0.03 / f.mean()
        f = np.clip(f, PHI_MIN, PHI_MAX)
        f *= 0.03 / f.mean()
        f = np.clip(f, PHI_MIN, PHI_MAX)
        assert axis_length(f, 0) == pytest.approx(2 * corr[0], rel=0.15)

    def test_a_length_longer_than_the_canvas_is_capped_to_it(self, sampler_and_corr):
        """Asking for more than the canvas can express gives a constant field.

        Real in-plane porosity correlation is longer than any coupon-scale
        canvas — campaign 21 found the in-plane porosity of all 80 volumes flat
        to within 6 % — so the fitted in-plane length is not something a 1024
        canvas can deliver. The cap makes that explicit instead of smoothing
        with 20 grid steps on a 16-step grid and calling the result a
        measurement.
        """
        sampler, _ = sampler_and_corr
        grid = (3, 16, 16)                       # 192 x 1024 x 1024 voxels
        huge = (50.0, 100_000.0, 100_000.0)
        at_cap = (50.0, 16 * STRIDE, 16 * STRIDE)
        a = build_porosity_field(grid, 0.03, sampler, huge, stride_voxels=STRIDE, seed=1)
        b = build_porosity_field(grid, 0.03, sampler, at_cap, stride_voxels=STRIDE, seed=1)
        # A request of 100 000 voxels and one of exactly the canvas extent must
        # produce the SAME field: everything above the cap is the same request.
        assert np.allclose(a, b)

    def test_the_cap_does_not_touch_a_length_the_canvas_can_hold(
            self, sampler_and_corr):
        sampler, _ = sampler_and_corr
        grid = (24, 96, 160)
        short = (50.0, 200.0, 300.0)             # all far inside the extents
        a = build_porosity_field(grid, 0.03, sampler, short, stride_voxels=STRIDE, seed=1)
        b = build_porosity_field(grid, 0.03, sampler,
                                 tuple(min(c, g * STRIDE) for c, g in zip(short, grid)),
                                 stride_voxels=STRIDE, seed=1)
        assert np.array_equal(a, b)

    def test_the_mean_is_still_delivered_exactly(self, sampler_and_corr):
        sampler, corr = sampler_and_corr
        for target in (0.01, 0.03, 0.06):
            f = build_porosity_field((24, 96, 160), target, sampler, corr,
                                     stride_voxels=STRIDE, seed=7)
            assert float(f.mean()) == pytest.approx(target, rel=0.01)

    def test_smoothing_narrows_the_marginal_and_that_is_reported_not_hidden(
            self, sampler_and_corr):
        """NOT a defect — it is what smoothing does — but it is a property of
        the delivered field, so a test states it rather than leaving it to be
        discovered from a downstream R-squared."""
        from poregen.diffusion.porosity_field import _draw_marginal

        sampler, corr = sampler_and_corr
        raw = _draw_marginal(0.03, sampler,
                             np.random.default_rng(0).uniform(0, 1, size=100_000))
        f = build_porosity_field((24, 96, 160), 0.03, sampler, corr,
                                 stride_voxels=STRIDE, seed=7)
        assert f.std() < 0.2 * raw.std(), (
            "the delivered field should be much narrower than the fitted "
            f"marginal: got {f.std():.4f} against {raw.std():.4f}")
