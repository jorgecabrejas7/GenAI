"""Known-answer checks for poregen.metrics.pore_morphology."""
import numpy as np
import pytest

from poregen.metrics.pore_morphology import measure


def ellipsoid(shape, centre, semi):  # semi-axes along (z, y, x)
    zz, yy, xx = np.indices(shape, dtype=np.float64)
    r = sum(((g - c) / a) ** 2 for g, c, a in zip((zz, yy, xx), centre, semi))
    return r <= 1.0


def test_ellipsoid_tensors_and_descriptors():
    lab = np.zeros((80, 50, 40), np.uint8)
    lab[ellipsoid(lab.shape, (40, 25, 20), (30, 15, 10))] = 1  # long axis = z
    df, s = measure(lab, voxel_um=1.0)
    assert s["n_pores"] == 1
    r = df.iloc[0]
    for k, a in zip((1, 2, 3), (30, 15, 10)):
        assert r[f"cov_eig{k}_um2"] == pytest.approx(a * a / 5, rel=0.03)
    assert r["cov_elongation"] == pytest.approx(0.25, rel=0.05)
    assert r["inertia_elongation"] == pytest.approx(1000 / 1125, rel=0.03)
    assert r["inertia_eig1_um2"] == pytest.approx((900 + 225) / 5, rel=0.03)
    assert r["cov_phi_deg"] < 2.0          # covariance v1 = long axis = z
    assert r["inertia_phi_deg"] > 88.0     # inertia v1 = short axis, in the xy plane
    assert r["volume_um3"] == pytest.approx(4 / 3 * np.pi * 30 * 15 * 10, rel=0.02)
    assert not r["touches_edge"] and np.isnan(r["dist_surface_um"])


def test_sphere_sphericity_and_scale():
    lab = np.zeros((60, 60, 60), np.uint8)
    lab[ellipsoid(lab.shape, (30, 30, 30), (20, 20, 20))] = 1
    r1 = measure(lab, 1.0)[0].iloc[0]
    r25 = measure(lab, 25.0)[0].iloc[0]
    # the repo's 4/pi factor over-reads a sphere's area by ~1.9x; 2/3 is right
    assert r1["sphericity"] == pytest.approx(0.52, abs=0.03)
    assert 0.93 < r1["sphericity_corr"] <= 1.0
    assert r1["area_corr_um2"] == pytest.approx(4 * np.pi * 400, rel=0.08)
    assert r1["area_um2"] == pytest.approx(r1["area_corr_um2"] * (4 / np.pi) / (2 / 3))
    assert r25["volume_um3"] == pytest.approx(r1["volume_um3"] * 25 ** 3)
    assert r25["sphericity"] == pytest.approx(r1["sphericity"])


def test_distance_air_edge_and_components():
    lab = np.zeros((40, 40, 40), np.uint8)
    lab[:5] = 2                                  # air slab z < 5
    lab[20:23, 10:13, 10:13] = 1                 # cube, nearest voxel z = 20
    lab[5, 30, 30] = 1                           # single voxel against the air
    lab[39, 0, 0] = 1                            # corner voxel: touches the edge
    df, s = measure(lab, 1.0)
    assert s["n_pores"] == 3
    cube = df[df.n_voxels == 27].iloc[0]
    assert cube.dist_surface_um == pytest.approx(16.0)   # z 20 -> z 4
    assert not cube.touches_air
    near = df[(df.n_voxels == 1) & (df.bary_z_um == 5)].iloc[0]
    assert near.touches_air and near.dist_surface_um == pytest.approx(1.0)
    assert np.isnan(near.cov_eig1_um2)                   # N = 1
    assert df[df.bary_z_um == 39].iloc[0].touches_edge
    assert s["porosity"] == pytest.approx(29 / (40 * 40 * 35))


def test_diagonal_neighbours_join():
    lab = np.zeros((10, 10, 10), np.uint8)
    lab[2, 2, 2] = lab[3, 3, 3] = 1              # corner-touching: one 26-component
    assert measure(lab, 1.0)[1]["n_pores"] == 1


def test_blocked_distance_matches_full_edt():
    from scipy import ndimage
    from poregen.metrics.pore_morphology import _dist_to_air
    rng = np.random.default_rng(0)
    air = np.zeros((60, 700, 80), bool)
    air[:4] = air[-5:] = True
    air[:, :, :3] = rng.random((60, 700, 3)) < 0.5
    zz, yy, xx = np.nonzero(rng.random(air.shape) < 0.01)
    full = ndimage.distance_transform_edt(~air, sampling=25.0)[zz, yy, xx]
    blk = _dist_to_air(air, 25.0, zz, yy, xx, block=128, margin=64)
    np.testing.assert_allclose(blk, full, atol=1e-3)
