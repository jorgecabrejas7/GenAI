"""The label-uncertainty measurement, on a volume whose answer is known.

``scripts/analysis/label_uncertainty.py`` re-runs the dataset segmentation with
perturbed parameters and reports how far porosity moves.  A number like that is
worth nothing unless the machinery that produced it is shown to be faithful, so
these tests fix a synthetic volume where the right answer can be written down
in closed form, and check four separate claims:

1. the streaming, slab-by-slab pipeline reproduces the production ``onlypores``
   exactly — same pore voxels, same material voxels;
2. raising ``sauvola_k`` lowers porosity, by the amount the Sauvola formula
   predicts for this volume and no other amount;
3. two variants asked for with identical parameters agree at Dice exactly 1.0,
   and the pairwise Dice the script accumulates is the same number
   ``eval_v4.metrics.pore_dice`` computes on the whole arrays;
4. the histogram shortcut the script uses for the material threshold returns
   exactly what skimage returns from the volume itself.

The synthetic volume
--------------------
A uniform block of intensity 255 in air, holding eight 6³ cubes of graded
intensity.  Every cube sits at the same (z, x) and is separated from the others
along y, and Sauvola runs on each (z, x) plane independently, so each cube is
alone in its own plane.  Every cube voxel therefore sees the SAME 31×31 window:
36 cube pixels and 925 block pixels.  The local mean and standard deviation of
that window are arithmetic, so the Sauvola threshold

    T = m (1 + k (s / 128 - 1))

is arithmetic too, and a cube is pore exactly when its intensity is at or below
its own T.  Grading the cube intensities across the values T takes at
k = 0.100 / 0.125 / 0.150 makes the number of detected cubes — and therefore the
porosity — a known step function of k.

The cubes are kept at least 16 voxels from every crop face in z and x, so no
window reaches the reflect-padded border, and the closest cube intensity is
2.6 grey levels from a threshold, which is far outside the integral-image
arithmetic's error.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[1]
# The analysis scripts are not an installable package.  Importing the script
# also puts REPO/src first on the path, which is what makes a worktree test the
# worktree's own poregen — see docs/DEVELOPMENT.md, "Testing inside a worktree".
sys.path.insert(0, str(REPO / "scripts" / "analysis"))
sys.path.insert(0, str(REPO / "src"))

import label_uncertainty as LU  # noqa: E402
from poregen.dataset.loader import build_label  # noqa: E402
from poregen.dataset.segmentation import (  # noqa: E402
    content_bbox,
    material_mask,
    onlypores,
    sauvola_thresholding_concurrent,
    sauvola_thresholding_nonconcurrent,
)
from poregen.eval_v4.metrics import pore_dice  # noqa: E402

BLOCK_VALUE = 255
CUBE = 6                       # side of every pore cube
WINDOW = LU.SAUVOLA_RADIUS + 1  # the odd window sauvola_thresholding uses
#: Graded so that the detected set is 7 / 6 / 5 cubes at k = 0.100 / 0.125 / 0.150.
PORE_VALUES = (40, 120, 180, 210, 214, 221, 227, 235)
#: Block extent inside the volume: 48 x 90 x 80 voxels.
BLOCK = (slice(8, 56), slice(10, 100), slice(10, 90))
BLOCK_VOXELS = 48 * 90 * 80
CUBE_VOXELS = CUBE ** 3


def synthetic_volume() -> np.ndarray:
    """The block, in air, with one graded cube per y band."""
    volume = np.zeros((64, 110, 100), np.uint8)
    volume[BLOCK] = BLOCK_VALUE
    for i, value in enumerate(PORE_VALUES):
        y0 = 14 + 10 * i
        volume[29:29 + CUBE, y0:y0 + CUBE, 48:48 + CUBE] = value
    return volume


def cropped_volume() -> np.ndarray:
    """The volume as ``onlypores`` sees it — cut to its content bounding box."""
    volume = synthetic_volume()
    z0, z1, y0, y1, x0, x1 = content_bbox(volume)
    return np.ascontiguousarray(volume[z0:z1 + 1, y0:y1 + 1, x0:x1 + 1])


def sauvola_threshold_on_a_cube(value: int, k: float) -> float:
    """The Sauvola threshold every voxel of a cube of *value* sees.

    Closed form, from the window composition alone: nothing here calls skimage.
    """
    n_window = WINDOW ** 2
    f = CUBE ** 2 / n_window
    mean = (1 - f) * BLOCK_VALUE + f * value
    second = (1 - f) * BLOCK_VALUE ** 2 + f * value ** 2
    std = np.sqrt(second - mean ** 2)
    return mean * (1 + k * (std / 128.0 - 1))


def predicted_porosity(k: float) -> float:
    """Known answer: cubes at or below their own threshold, over the block."""
    detected = [v for v in PORE_VALUES if v <= sauvola_threshold_on_a_cube(v, k)]
    return len(detected) * CUBE_VOXELS / BLOCK_VOXELS


def variant(result: dict, name: str) -> dict:
    return next(r for r in result["variants"] if r["name"] == name)


# ---------------------------------------------------------------------------


def test_streaming_reproduces_production_onlypores():
    """The slab-by-slab pipeline is the production pipeline, voxel for voxel.

    A slab width that divides nothing evenly is used on purpose: if Sauvola were
    not independent per (z, x) plane, the seams would show up as a count
    difference.
    """
    volume = synthetic_volume()
    pore_mask, sample_mask, _ = onlypores(
        volume, sauvola_radius=LU.SAUVOLA_RADIUS, sauvola_k=LU.SAUVOLA_K
    )

    result = LU.measure(
        cropped_volume(), sauvola_k=[LU.SAUVOLA_K], material_methods=["otsu"], slab=17
    )
    measured = variant(result, f"k{LU.SAUVOLA_K:g}/otsu")

    assert measured["n_pore"] == int(pore_mask.sum())
    assert measured["n_material"] == int(sample_mask.sum())
    assert measured["n_material"] == BLOCK_VOXELS   # and that is the right block


def test_sequential_and_parallel_sauvola_agree():
    """Why the script may take the polite path: the two give the same bits."""
    rng = np.random.default_rng(0)
    volume = rng.integers(30, 240, size=(20, 30, 40), dtype=np.uint8)
    parallel = sauvola_thresholding_concurrent(volume, LU.SAUVOLA_RADIUS, LU.SAUVOLA_K)
    sequential = sauvola_thresholding_nonconcurrent(volume, LU.SAUVOLA_RADIUS, LU.SAUVOLA_K)
    assert np.array_equal(parallel, sequential)


@pytest.fixture(scope="module")
def result():
    """The three-k sweep, measured once for the whole class below."""
    return LU.measure(
        cropped_volume(),
        sauvola_k=[0.100, 0.125, 0.150],
        material_methods=["otsu"],
        slab=17,
    )


class TestPorosityRespondsToTheThreshold:

    def test_the_predicted_cubes_are_the_detected_cubes(self, result):
        """The known answer, exactly — not approximately."""
        for k in (0.100, 0.125, 0.150):
            measured = variant(result, f"k{k:g}/otsu")
            assert measured["n_material"] == BLOCK_VOXELS
            assert measured["phi"] == pytest.approx(predicted_porosity(k), rel=1e-12)
            # and the pore voxels are whole cubes, so nothing partial crept in
            assert measured["n_pore"] % CUBE_VOXELS == 0

    def test_raising_k_lowers_porosity(self, result):
        """Sauvola lowers its threshold as k rises, so fewer voxels are pore."""
        phi = [variant(result, f"k{k:g}/otsu")["phi"] for k in (0.100, 0.125, 0.150)]
        assert phi[0] > phi[1] > phi[2]

    def test_the_step_is_one_cube_per_twenty_percent_of_k(self, result):
        """The size of the move, not only its sign."""
        phi = {k: variant(result, f"k{k:g}/otsu")["phi"] for k in (0.100, 0.125, 0.150)}
        one_cube = CUBE_VOXELS / BLOCK_VOXELS
        assert phi[0.100] - phi[0.125] == pytest.approx(one_cube, rel=1e-12)
        assert phi[0.125] - phi[0.150] == pytest.approx(one_cube, rel=1e-12)
        # ±20 % of k is worth a sixth of the baseline porosity on this volume
        assert phi[0.100] - phi[0.150] == pytest.approx(2 * one_cube, rel=1e-12)

    def test_variants_disagree_by_exactly_the_cubes_they_differ_on(self, result):
        """Dice between k variants is the Dice of nested pore sets."""
        dice = np.array(result["dice_matrix"])
        n = [variant(result, f"k{k:g}/otsu")["n_pore"] for k in (0.100, 0.125, 0.150)]
        # the pore sets are nested, so the intersection is the smaller set
        assert dice[0, 1] == pytest.approx(2 * n[1] / (n[0] + n[1]), rel=1e-12)
        assert dice[0, 2] == pytest.approx(2 * n[2] / (n[0] + n[2]), rel=1e-12)
        assert dice[0, 2] < dice[0, 1] < 1.0


class TestVariantAgreement:

    def test_identical_parameters_give_dice_one(self):
        """The control: a variant compared with its own twin must score 1.0."""
        result = LU.measure(
            cropped_volume(),
            sauvola_k=[LU.SAUVOLA_K, LU.SAUVOLA_K],
            material_methods=["otsu", "otsu"],
            slab=17,
        )
        dice = np.array(result["dice_matrix"])
        assert dice.shape == (4, 4)
        assert np.all(dice == 1.0)
        assert len({r["n_pore"] for r in result["variants"]}) == 1

    def test_dice_is_the_eval_suite_s_pore_dice(self):
        """The streamed pairwise Dice equals the metric the eval suite reports."""
        cropped = cropped_volume()
        result = LU.measure(
            cropped, sauvola_k=[0.100, 0.150], material_methods=["otsu"], slab=17
        )

        sample = material_mask(cropped)
        labels = [
            build_label(~sauvola_thresholding_nonconcurrent(cropped, LU.SAUVOLA_RADIUS, k)
                        & sample, sample)
            for k in (0.100, 0.150)
        ]
        assert result["dice_matrix"][0][1] == pytest.approx(
            pore_dice(labels[0], labels[1]), rel=1e-12
        )


def test_the_report_carries_the_measured_numbers(result):
    """The summary and findings.md are what the paper quotes, so render them."""
    baseline = f"k{LU.SAUVOLA_K:g}/otsu"
    summary = LU.summarise_volume(result, baseline)
    one_cube = CUBE_VOXELS / BLOCK_VOXELS

    assert summary["phi_baseline"] == pytest.approx(predicted_porosity(LU.SAUVOLA_K))
    assert summary["phi_range"] == pytest.approx(2 * one_cube, rel=1e-12)
    assert summary["phi_range_sauvola_k_only"] == pytest.approx(2 * one_cube, rel=1e-12)
    assert summary["phi_range_material_only"] == 0.0   # one method in this sweep
    assert summary["dice_min"] < 1.0

    volume = {"volume_id": "synthetic", "summary": summary, **result}
    report = LU.findings_markdown({
        "commit": "test",
        "settings": {
            "sauvola_k_values": [0.100, 0.125, 0.150],
            "sauvola_k_base": LU.SAUVOLA_K,
            "material_methods": ["otsu"],
        },
        "volumes": [volume],
        "summary": {
            "n_volumes": 1,
            "n_variants": len(result["variants"]),
            "phi_range_max": summary["phi_range"],
            "phi_range_max_volume": "synthetic",
            "phi_range_mean": summary["phi_range"],
            "porosity_gate": LU.POROSITY_GATE,
            "phi_range_max_over_gate": summary["phi_range"] / LU.POROSITY_GATE,
            "dice_min": summary["dice_min"],
            "dice_median": summary["dice_median"],
        },
    })
    assert f"{summary['phi_range']:.5f}" in report
    assert f"{summary['dice_min']:.3f}" in report
    assert "k0.125/otsu" in report


class TestMaterialThreshold:

    @pytest.mark.parametrize("method", sorted(LU.MATERIAL_METHODS))
    def test_histogram_threshold_matches_skimage_on_the_volume(self, method):
        """The 256-bin shortcut is lossless for uint8, so it must be exact."""
        rng = np.random.default_rng(1)
        volumes = [
            cropped_volume(),
            rng.integers(30, 240, size=(20, 30, 40), dtype=np.uint8),
            np.clip(rng.normal(180, 40, (12, 20, 25)), 0, 255).astype(np.uint8),
        ]
        for volume in volumes:
            from_hist = LU.material_threshold(LU.grey_histogram(volume), method)
            assert from_hist == float(LU.MATERIAL_METHODS[method](volume))

    def test_the_material_method_moves_the_material_mask(self):
        """The second axis of the sweep has to do something, or it is not a test.

        On the synthetic block Otsu and Yen land either side of the darkest
        cubes, so the two masks agree on the block but the thresholds differ —
        which is what the script records per volume.
        """
        counts = LU.grey_histogram(cropped_volume())
        thresholds = {m: LU.material_threshold(counts, m) for m in LU.MATERIAL_METHODS}
        assert len(set(thresholds.values())) > 1
