"""The periodic validation subset (vault D58 / OQ-5).

The periodic val.loss used the first 50 batches of an unshuffled loader, all
from one volume. It now scores a fixed subset of the same size: an equal share
per val volume, spread over the porosity bins.
"""

from __future__ import annotations

import numpy as np

from poregen.diffusion.latents import VAL_POROSITY_BINS, stratified_val_rows

N_VOL = 11


def _val_index(seed=0):
    rng = np.random.default_rng(seed)
    vols, phi = [], []
    for v in range(N_VOL):
        n = 3000 if v else 40                      # volume 0 is tiny
        vols += [f"vol{v:02d}"] * n
        # mostly low porosity, a thin high tail, like the real val panels
        phi += list(np.where(rng.random(n) < 0.8, rng.uniform(0, 0.02, n),
                             rng.uniform(0.02, 0.12, n)))
    order = np.argsort(np.array(vols), kind="stable")   # contiguous, like the store
    return np.array(vols)[order], np.array(phi)[order]


def test_the_subset_covers_every_val_volume_and_keeps_the_size():
    vols, phi = _val_index()
    n_rows = 50 * 32
    rows, comp = stratified_val_rows(vols, phi, n_rows)
    assert len(rows) == n_rows
    assert len(np.unique(rows)) == n_rows
    assert set(np.unique(vols[rows])) == {f"vol{v:02d}" for v in range(N_VOL)}
    assert set(comp) == set(np.unique(vols))
    # The tiny volume gives everything it has; the rest share the remainder.
    counts = {v: sum(c.values()) for v, c in comp.items()}
    assert counts["vol00"] == 40
    big = [counts[f"vol{v:02d}"] for v in range(1, N_VOL)]
    assert max(big) - min(big) <= 1


def test_every_porosity_bin_present_in_a_volume_is_sampled():
    vols, phi = _val_index()
    rows, comp = stratified_val_rows(vols, phi, 50 * 32)
    for v, c in comp.items():
        sel = vols == v
        bins = np.digitize(phi[sel], VAL_POROSITY_BINS[1:-1])
        for b, (label, n) in enumerate(c.items()):
            assert (n > 0) == bool((bins == b).any()), (v, label)


def test_the_subset_is_fixed_by_its_seed():
    vols, phi = _val_index()
    a, _ = stratified_val_rows(vols, phi, 1600)
    b, _ = stratified_val_rows(vols, phi, 1600)
    c, _ = stratified_val_rows(vols, phi, 1600, seed=1)
    assert np.array_equal(a, b)
    assert not np.array_equal(a, c)


def test_a_request_larger_than_the_split_takes_every_row():
    vols, phi = _val_index()
    rows, _ = stratified_val_rows(vols, phi, len(phi) + 10)
    assert np.array_equal(rows, np.arange(len(phi)))
