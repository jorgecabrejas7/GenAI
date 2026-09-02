"""split_v3: panel-level splits and drilled-hole removal.

Two invariants are worth a test. A panel must never appear in two splits —
that is the whole point of the v3 split — and the hole detector must find the
three through-holes without picking up ordinary internal voids, which is what
the z-minimum projection buys over a z-maximum one.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pytest

from poregen.dataset.holes import detect_holes, patches_touching_holes

REPO = Path(__file__).resolve().parents[1]


def _load_builder():
    path = REPO / "scripts" / "build_split_v3.py"
    spec = importlib.util.spec_from_file_location("build_split_v3", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


build = _load_builder()

VOLUMES = [
    "MedidasDB__Airbus_Panel_Pegaso_probetas_1_10_volume_eq_aligned",
    "MedidasDB__Airbus_Panel_Pegaso_probetas_1_26_volumen_eq_aligned",
    "MedidasDB__Fabricacion_Nacho_05_Probetas_Nacho_2025_probetas_Na_05_1_volume_eq_aligned",
    "MedidasDB__Fabricacion_Nacho_05_Probetas_Nacho_2025_probetas_Na_05_4_volume_eq_aligned",
    "MedidasDB__Fabricacion_Nacho_05_Probetas_Nacho_2025_probetas_Na_08_2_volume_eq_aligned",
    "MedidasDB__Fabricacion_Nacho_05_Probetas_Nacho_2025_probetas_Na_01_3_volume_eq_aligned",
    "MedidasDB__Juan_Ignacio_probetas_8_volume_eq_aligned",
    "MedidasDB__Juan_Ignacio_probetas_12_volume_eq_aligned",
    "MedidasDB__Juan_Ignacio_probetas_4_volume_eq_aligned",
]


# ---------------------------------------------------------------------------
# Panels
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("vid,expected", [
    (VOLUMES[0], "Pegaso_1"),
    (VOLUMES[1], "Pegaso_1"),
    (VOLUMES[2], "Na_05"),
    (VOLUMES[4], "Na_08"),
    (VOLUMES[6], "JI_8"),
])
def test_panel_id(vid, expected):
    assert build.panel_id(vid) == expected


def test_panel_id_rejects_unknown():
    with pytest.raises(ValueError):
        build.panel_id("MedidasDB__something_else")


def test_no_panel_crosses_a_split():
    by_panel: dict[str, set[str]] = {}
    for v in VOLUMES:
        p = build.panel_id(v)
        by_panel.setdefault(p, set()).add(build.split_of(p))
    assert all(len(s) == 1 for s in by_panel.values()), by_panel


def test_requested_panels_land_in_the_right_split():
    assert build.split_of("Na_05") == "test"
    assert build.split_of("JI_8") == "test"
    assert build.split_of("Na_08") == "val"
    assert build.split_of("JI_12") == "val"
    # Pegaso is one panel, so it can only be train.
    assert build.split_of("Pegaso_1") == "train"


# ---------------------------------------------------------------------------
# Holes
# ---------------------------------------------------------------------------

@pytest.fixture()
def synthetic_volume():
    """A slab with three through-holes and one single-slice internal void."""
    D, H, W = 40, 300, 300
    sm = np.zeros((D, H, W), dtype=np.uint8)
    sm[4:36, 20:280, 20:280] = 1                       # the specimen
    yy, xx = np.mgrid[0:H, 0:W]
    centres = [(80, 80), (80, 200), (200, 140)]
    for cy, cx in centres:
        disc = (yy - cy) ** 2 + (xx - cx) ** 2 <= 20 ** 2
        sm[:, disc] = 0                                # drilled right through
    void = (yy - 150) ** 2 + (xx - 60) ** 2 <= 25 ** 2  # bigger than a hole...
    sm[20, void] = 0                                    # ...but one slice deep
    return sm, centres


def test_detect_holes_finds_the_through_holes(synthetic_volume):
    sm, centres = synthetic_volume
    res = detect_holes(sm, dilate_vox=0)
    assert res["n_holes"] == 3
    found = sorted(tuple(round(c) for c in h["centre_yx"]) for h in res["holes"])
    assert found == sorted(centres)


def test_detect_holes_ignores_a_single_slice_void(synthetic_volume):
    """The void is larger in area than a hole but is not a through-hole.

    A z-maximum projection would report it; the z-minimum projection does not.
    """
    sm, _ = synthetic_volume
    res = detect_holes(sm, dilate_vox=0)
    assert not res["mask"][150, 60]


def test_dilation_grows_the_footprint(synthetic_volume):
    sm, _ = synthetic_volume
    tight = detect_holes(sm, dilate_vox=0)["mask"]
    grown = detect_holes(sm, dilate_vox=16)["mask"]
    assert grown[tight].all()
    assert grown.sum() > tight.sum()
    # A point 10 voxels outside a hole edge is caught only after dilation.
    assert grown[80, 80 + 25] and not tight[80, 80 + 25]


def test_patches_touching_holes():
    hole = np.zeros((64, 64), dtype=bool)
    hole[40:44, 40:44] = True
    y0 = np.array([0, 0, 32, 32])
    x0 = np.array([0, 32, 0, 32])
    got = patches_touching_holes(hole, y0, x0, 32)
    # Only the (32, 32) patch covers rows/cols 40-43.
    assert got.tolist() == [False, False, False, True]
