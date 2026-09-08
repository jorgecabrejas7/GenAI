import json

import numpy as np
import tifffile
import torch
from torch import nn

from poregen.models.vae.base import CLASS_AIR, CLASS_PORE, N_CLASSES, VAEOutput
from poregen.training.engine import _save_patch_samples
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


# ---------------------------------------------------------------------------
# 3-class head: the exported pore mask must not be empty
# ---------------------------------------------------------------------------

PS = 4
B = 2


class _ClsModel(nn.Module):
    """Stand-in for an r08 ``*_cls`` variant: class_logits, never mask_logits."""

    encoder_inputs = ("xct", "label")

    def forward(self, xct, label):
        logits = torch.full((xct.shape[0], N_CLASSES, PS, PS, PS), -10.0)
        for c in range(N_CLASSES):
            logits[:, c][label == c] = 10.0
        return VAEOutput(
            xct_out=xct.clone(),
            class_logits=logits,
            mu=torch.zeros(xct.shape[0], 1),
            logvar=torch.zeros(xct.shape[0], 1),
            z=torch.zeros(xct.shape[0], 1),
        )


class _MaskModel(nn.Module):
    """Stand-in for a binary variant, which still emits mask_logits."""

    encoder_inputs = ("xct", "mask")

    def forward(self, xct, mask):
        return VAEOutput(
            xct_out=xct.clone(),
            mask_logits=torch.full_like(xct, 10.0),
            mu=torch.zeros(xct.shape[0], 1),
            logvar=torch.zeros(xct.shape[0], 1),
            z=torch.zeros(xct.shape[0], 1),
        )


def _cls_batch() -> dict:
    label = torch.zeros(B, PS, PS, PS, dtype=torch.long)
    label[:, :2] = CLASS_PORE          # half the patch is pore
    label[:, :, :, :1] = CLASS_AIR     # and one wall of it is air
    return {
        "xct": torch.rand(B, 1, PS, PS, PS),
        "label": label,
        "mask": (label == CLASS_PORE).float().unsqueeze(1),
        "coords": torch.zeros(B, 3, dtype=torch.long),
        "porosity": torch.zeros(B),
        "volume_id": ["vol_a", "vol_b"],
        "source_group": ["synthetic", "synthetic"],
    }


def test_three_class_export_writes_a_real_pore_mask(tmp_path):
    """A 3-class head has no mask_logits; the export used to write zeros.

    ``mask_recon`` must be ``argmax(class_logits) == CLASS_PORE``, and the
    3-class label itself must be exported alongside it — a binary mask throws
    away the material/air distinction the r08 head exists to make.
    """
    batch = _cls_batch()
    _save_patch_samples(
        model=_ClsModel(),
        loaders={"val": [batch]},
        n_samples=B,
        step=7,
        run_dir=tmp_path,
        device=torch.device("cpu"),
        autocast_dtype=torch.bfloat16,
    )

    sample_dir = tmp_path / "samples" / "step_00000007" / "val" / "sample_000"
    mask_recon = tifffile.imread(sample_dir / "mask_recon.tiff")
    label_recon = tifffile.imread(sample_dir / "label_recon.tiff")

    expected_label = batch["label"][0].numpy()
    assert mask_recon.shape == (PS, PS, PS)
    assert mask_recon.any(), "the pore mask must not be empty"
    assert np.array_equal(mask_recon, (expected_label == CLASS_PORE).astype(np.float32))
    assert np.array_equal(label_recon, expected_label.astype(np.float32))
    assert set(np.unique(label_recon)) == {0.0, 1.0, 2.0}


def test_binary_head_export_has_no_label_recon(tmp_path):
    """A binary variant keeps the historic four arrays and nothing more."""
    _save_patch_samples(
        model=_MaskModel(),
        loaders={"val": [_cls_batch()]},
        n_samples=B,
        step=1,
        run_dir=tmp_path,
        device=torch.device("cpu"),
        autocast_dtype=torch.bfloat16,
    )

    sample_dir = tmp_path / "samples" / "step_00000001" / "val" / "sample_000"
    assert not (sample_dir / "label_recon.tiff").exists()
    assert np.allclose(tifffile.imread(sample_dir / "mask_recon.tiff"),
                       torch.sigmoid(torch.tensor(10.0)).item())
