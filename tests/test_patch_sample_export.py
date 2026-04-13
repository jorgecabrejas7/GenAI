import json

import numpy as np
import tifffile

from poregen.training.sample_export import (
    convert_patch_sample_archive,
    export_patch_sample_split,
)


def _make_arrays() -> dict[str, np.ndarray]:
    base = np.arange(2 * 1 * 4 * 4 * 4, dtype=np.float32).reshape(2, 1, 4, 4, 4)
    return {
        "xct_gt": base / base.max(),
        "mask_gt": (base % 2).astype(np.float32),
        "xct_recon": np.flip(base, axis=-1) / base.max(),
        "mask_recon": np.clip(base / 10.0, 0.0, 1.0),
    }


def test_export_patch_sample_split_writes_imagej_tiffs(tmp_path):
    arrays = _make_arrays()
    metas = [{"volume_id": "vol_a"}, {"volume_id": "vol_b"}]

    split_dir = export_patch_sample_split(tmp_path / "train", arrays, metas)

    sample_dir = split_dir / "sample_000"
    with tifffile.TiffFile(sample_dir / "xct_gt.tiff") as tif:
        assert tif.is_imagej
    roundtrip = tifffile.imread(sample_dir / "xct_gt.tiff")

    assert roundtrip.shape == (4, 4, 4)
    assert np.allclose(roundtrip, arrays["xct_gt"][0, 0])

    meta = json.loads((sample_dir / "meta.json").read_text())
    assert meta["volume_id"] == "vol_a"
    assert meta["sample_id"] == "sample_000"

    manifest = json.loads((split_dir / "index.json").read_text())
    assert [entry["sample_id"] for entry in manifest] == ["sample_000", "sample_001"]


def test_convert_patch_sample_archive_replaces_legacy_npz(tmp_path):
    arrays = _make_arrays()
    npz_path = tmp_path / "val.npz"
    np.savez_compressed(npz_path, **arrays)
    (tmp_path / "val_meta.json").write_text(
        json.dumps([{"volume_id": "vol_a"}, {"volume_id": "vol_b"}], indent=2)
    )

    split_dir = convert_patch_sample_archive(npz_path)

    assert split_dir == tmp_path / "val"
    assert not npz_path.exists()
    assert not (tmp_path / "val_meta.json").exists()
    assert (split_dir / "sample_001" / "mask_recon.tiff").exists()

    roundtrip = tifffile.imread(split_dir / "sample_001" / "mask_recon.tiff")
    assert np.allclose(roundtrip, arrays["mask_recon"][1, 0])
