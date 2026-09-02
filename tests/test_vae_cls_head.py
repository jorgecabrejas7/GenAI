"""r08: the 3-class decoder head — model, loss and metrics.

The contract these tests pin down is that a ``*_cls`` variant emits
``class_logits`` and NOT ``mask_logits``, that its encoder takes
``cat([xct, pore, air])``, and that a perfect prediction drives every class
term to zero. The older binary-head variant is left alone and is checked here
too, because the r07 checkpoint behind the latent store still depends on it.
"""

from __future__ import annotations

import pytest
import torch

from poregen.losses.total import compute_total_loss
from poregen.metrics.seg import multiclass_metrics
from poregen.models.vae import build_vae
from poregen.models.vae.base import (
    CLASS_AIR, CLASS_MATERIAL, CLASS_PORE, N_CLASSES, VAEOutput,
    decode_class_probs, decode_label,
)
from poregen.models.vae.v2.conv_noattn_dualbranch_cls import label_to_channels
from poregen.training.engine import encoder_input_keys, to_device_inputs

PS = 16
B = 2
CLS_NAME = "v2.conv_noattn_dualbranch_cls"


def make_model(**kw):
    return build_vae(CLS_NAME, in_channels=3, z_channels=4, base_channels=8,
                     n_blocks=2, patch_size=PS, **kw)


@pytest.fixture()
def batch():
    label = torch.zeros(B, PS, PS, PS, dtype=torch.long)
    label[0, :4] = CLASS_PORE
    label[1, :, :4] = CLASS_AIR
    return {"xct": torch.rand(B, 1, PS, PS, PS), "label": label}


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

def test_forward_shapes(batch):
    out = make_model()(batch["xct"], batch["label"])
    assert out.xct_out.shape == (B, 1, PS, PS, PS)
    assert out.class_logits.shape == (B, N_CLASSES, PS, PS, PS)
    assert out.mask_logits is None, "a 3-class variant must not also emit a binary mask"
    assert out.mu.shape == (B, 4, PS // 4, PS // 4, PS // 4)


def test_in_channels_is_not_a_switch():
    for bad in (1, 2, 4):
        with pytest.raises(ValueError, match="cat"):
            build_vae(CLS_NAME, in_channels=bad, patch_size=PS)


def test_declares_its_encoder_inputs():
    assert make_model().encoder_inputs == ("xct", "label")
    assert encoder_input_keys(make_model()) == ("xct", "label")
    # Everything else keeps the historic pair.
    assert encoder_input_keys(build_vae("v2.conv_noattn_dualbranch")) == ("xct", "mask")


def test_binary_variant_is_untouched():
    """r07 checkpoints are still decoded by the campaign 05 and 08 scripts."""
    m = build_vae("v2.conv_noattn_dualbranch", in_channels=2, z_channels=4,
                  base_channels=8, n_blocks=2, patch_size=PS)
    out = m(torch.rand(B, 1, PS, PS, PS), torch.rand(B, 1, PS, PS, PS))
    assert out.mask_logits.shape == (B, 1, PS, PS, PS)
    assert out.class_logits is None


def test_label_to_channels(batch):
    ch = label_to_channels(batch["label"])
    assert ch.shape == (B, 2, PS, PS, PS)
    assert torch.equal(ch[:, 0] > 0, batch["label"] == CLASS_PORE)
    assert torch.equal(ch[:, 1] > 0, batch["label"] == CLASS_AIR)
    # Material carries no plane of its own — both are zero there.
    both_zero = (ch.sum(1) == 0)
    assert torch.equal(both_zero, batch["label"] == CLASS_MATERIAL)


def test_label_to_channels_accepts_a_channel_dim(batch):
    a = label_to_channels(batch["label"])
    b = label_to_channels(batch["label"].unsqueeze(1))
    assert torch.equal(a, b)


def test_to_device_inputs_picks_the_declared_keys(batch):
    m = make_model()
    dev, args = to_device_inputs(m, batch, torch.device("cpu"))
    assert len(args) == 2
    assert torch.equal(args[0], batch["xct"])
    assert torch.equal(args[1], batch["label"])
    assert dev["label"] is args[1]


# ---------------------------------------------------------------------------
# Decode helpers
# ---------------------------------------------------------------------------

def test_decode_label_and_probs(batch):
    logits = torch.full((B, N_CLASSES, PS, PS, PS), -10.0)
    for c in range(N_CLASSES):
        logits[:, c][batch["label"] == c] = 10.0
    assert torch.equal(decode_label(logits), batch["label"])
    probs = decode_class_probs(logits)
    assert probs.shape == logits.shape
    assert torch.allclose(probs.sum(1), torch.ones(B, PS, PS, PS), atol=1e-5)


# ---------------------------------------------------------------------------
# Loss
# ---------------------------------------------------------------------------

CFG = {"loss": {"xct_loss_type": "charbonnier", "xct_weight": 1.0,
                "kl_free_bits": 0.1, "kl_warmup_steps": 0, "kl_max_beta": 0.05,
                "class_ce_weight": 1.0, "class_dice_weight": 1.0,
                "class_weights": [0.34, 12.0, 5.0]}}


def _output(class_logits, xct):
    z = torch.zeros(B, 4, 2, 2, 2)
    return VAEOutput(xct_out=xct, class_logits=class_logits, mu=z, logvar=z, z=z)


def test_loss_has_the_class_terms(batch):
    out = _output(torch.randn(B, N_CLASSES, PS, PS, PS), torch.rand(B, 1, PS, PS, PS))
    r = compute_total_loss(out, batch, 100, CFG)
    for k in ("class_ce", "class_dice", "class_dice_1", "class_dice_2", "class_total"):
        assert k in r
    assert "mask_bce" not in r, "no binary mask term without a mask head"


def test_perfect_prediction_zeroes_the_class_terms(batch):
    logits = torch.full((B, N_CLASSES, PS, PS, PS), -20.0)
    for c in range(N_CLASSES):
        logits[:, c][batch["label"] == c] = 20.0
    xct = batch["xct"]
    r = compute_total_loss(_output(logits, xct), batch, 100, CFG)
    assert float(r["class_ce"]) == pytest.approx(0.0, abs=1e-6)
    assert float(r["class_dice"]) == pytest.approx(0.0, abs=1e-6)
    assert float(r["class_total"]) == pytest.approx(0.0, abs=1e-6)


def test_class_weights_change_the_ce(batch):
    logits = torch.randn(B, N_CLASSES, PS, PS, PS)
    flat = {**CFG, "loss": {**CFG["loss"], "class_weights": [1.0, 1.0, 1.0]}}
    a = float(compute_total_loss(_output(logits, batch["xct"]), batch, 0, CFG)["class_ce"])
    b = float(compute_total_loss(_output(logits, batch["xct"]), batch, 0, flat)["class_ce"])
    assert a != pytest.approx(b)


def test_dice_scores_only_the_minority_classes(batch):
    """Material is the background; including it would dilute the signal."""
    logits = torch.randn(B, N_CLASSES, PS, PS, PS)
    r = compute_total_loss(_output(logits, batch["xct"]), batch, 0, CFG)
    assert "class_dice_0" not in r
    assert {"class_dice_1", "class_dice_2"} <= set(r)


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def test_multiclass_metrics_perfect(batch):
    logits = torch.full((B, N_CLASSES, PS, PS, PS), -10.0)
    for c in range(N_CLASSES):
        logits[:, c][batch["label"] == c] = 10.0
    m = multiclass_metrics(logits, batch["label"])
    for name in ("material", "pore", "air"):
        assert m[f"dice_{name}"] == pytest.approx(1.0)
    assert m["porosity_mae"] == pytest.approx(0.0)
    assert m["air_mae"] == pytest.approx(0.0)


def test_multiclass_metrics_all_material(batch):
    """Predicting only material: pore and air Dice collapse, porosity is 0."""
    logits = torch.full((B, N_CLASSES, PS, PS, PS), -10.0)
    logits[:, CLASS_MATERIAL] = 10.0
    m = multiclass_metrics(logits, batch["label"])
    assert m["dice_pore"] < 1.0 and m["dice_air"] < 1.0
    assert m["porosity_pred_mean"] == pytest.approx(0.0)
    true_por = (batch["label"] == CLASS_PORE).float().flatten(1).mean(1)
    assert m["porosity_bias"] == pytest.approx(float(-true_por.mean()), abs=1e-6)


def test_absent_class_scores_one_not_zero():
    """A patch with neither prediction nor truth for a class agrees perfectly."""
    label = torch.zeros(1, 4, 4, 4, dtype=torch.long)      # all material
    logits = torch.full((1, N_CLASSES, 4, 4, 4), -10.0)
    logits[:, CLASS_MATERIAL] = 10.0
    m = multiclass_metrics(logits, label)
    assert m["dice_pore"] == pytest.approx(1.0)
    assert m["dice_air"] == pytest.approx(1.0)
