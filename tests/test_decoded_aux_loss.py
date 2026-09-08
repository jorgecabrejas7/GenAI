"""The decoded-space auxiliary loss (ldm06/aux).

Each term is a claim about voxels, so each is tested the same way: build a
tensor that satisfies the claim and require ~0, then break exactly one thing
and require the value to rise.  A term that is always positive, or always
zero, would look identical to a working one in a training curve.

The other two things pinned here are the ones that would be expensive to
discover on the GPU: the sub-batch selection (wrong items decoded = the loss
measures the schedule), and the frozen VAE (a gradient reaching the decoder
would let the LDM move the target it is scored against, and a decode in train
mode would quietly rewrite the r08 BatchNorm running statistics).
"""

from __future__ import annotations

import numpy as np
import pytest
import torch
import torch.nn as nn

from poregen.diffusion.noise_schedule import DDPMSchedule
from poregen.losses.decoded import (
    AIR_GREY_THRESHOLD,
    DecodedAuxLoss,
    DecodedLossConfig,
    air_outside_material,
    grey_agreement,
    pore_dice,
    porosity_consistency,
    ramp_weight,
    select_decoded_items,
)
from poregen.models.vae.base import CLASS_AIR, CLASS_MATERIAL, CLASS_PORE

B, D, L = 2, 8, 2          # 8³ voxels per item, 2³ latent cells (downsample 4)
DARK, BRIGHT = 0.2, 0.9    # either side of AIR_GREY_THRESHOLD = 182/255


def _probs(material: float, pore: float, air: float) -> torch.Tensor:
    """(B, 3, D, D, D) constant class probabilities."""
    p = torch.zeros(B, 3, D, D, D)
    p[:, CLASS_MATERIAL] = material
    p[:, CLASS_PORE] = pore
    p[:, CLASS_AIR] = air
    return p


# ── term (a): air outside the specimen envelope ───────────────────────────────

class TestAirOutsideMaterial:

    def test_no_air_inside_the_envelope_scores_zero(self):
        p_air = torch.zeros(B, D, D, D)
        material = torch.ones(B, 1, L, L, L)
        assert float(air_outside_material(p_air, material)) == pytest.approx(0.0)

    def test_air_inside_the_envelope_is_penalised(self):
        p_air = torch.full((B, D, D, D), 0.4)
        material = torch.ones(B, 1, L, L, L)
        assert float(air_outside_material(p_air, material)) == pytest.approx(0.4, abs=1e-6)

    def test_air_outside_the_envelope_is_free(self):
        """Exterior air is correct — the envelope said the specimen ends."""
        p_air = torch.ones(B, D, D, D)
        material = torch.zeros(B, 1, L, L, L)
        assert float(air_outside_material(p_air, material)) == pytest.approx(0.0)

    def test_a_partially_filled_cell_is_not_scored(self):
        """0.5 envelope = a surface or drilled-hole cell; its air is legitimate."""
        p_air = torch.ones(B, D, D, D)
        material = torch.full((B, 1, L, L, L), 0.5)
        assert float(air_outside_material(p_air, material)) == pytest.approx(0.0)

    def test_the_envelope_is_upsampled_to_the_voxel_grid(self):
        """Half the latent cells full => the penalty covers half the voxels."""
        p_air = torch.ones(B, D, D, D)
        material = torch.zeros(B, 1, L, L, L)
        material[:, :, 0] = 1.0                       # one of two z-slabs
        assert float(air_outside_material(p_air, material)) == pytest.approx(1.0)

    def test_it_is_differentiable_in_p_air(self):
        p_air = torch.full((B, D, D, D), 0.3, requires_grad=True)
        air_outside_material(p_air, torch.ones(B, 1, L, L, L)).backward()
        assert p_air.grad is not None and float(p_air.grad.abs().sum()) > 0.0


# ── term (b): pore Dice ───────────────────────────────────────────────────────

class TestPoreDice:

    def test_a_perfect_pore_prediction_scores_near_zero(self):
        label = torch.full((B, D, D, D), CLASS_MATERIAL, dtype=torch.uint8)
        label[:, :4] = CLASS_PORE
        p_pore = (label == CLASS_PORE).float()
        assert float(pore_dice(p_pore, label)) < 1e-3

    def test_missing_every_pore_scores_near_one(self):
        label = torch.full((B, D, D, D), CLASS_MATERIAL, dtype=torch.uint8)
        label[:, :4] = CLASS_PORE
        assert float(pore_dice(torch.zeros(B, D, D, D), label)) > 0.99

    def test_pores_in_the_wrong_place_are_worse_than_pores_in_the_right_place(self):
        label = torch.full((B, D, D, D), CLASS_MATERIAL, dtype=torch.uint8)
        label[:, :4] = CLASS_PORE
        right = (label == CLASS_PORE).float()
        wrong = (label != CLASS_PORE).float()
        assert float(pore_dice(wrong, label)) > float(pore_dice(right, label))

    def test_air_is_not_counted_as_pore(self):
        """Term (a) owns air; scoring it here too would double-count."""
        label = torch.full((B, D, D, D), CLASS_AIR, dtype=torch.uint8)
        assert float(pore_dice(torch.zeros(B, D, D, D), label)) < 1e-3


# ── term (c): porosity consistency ────────────────────────────────────────────

class TestPorosityConsistency:

    def test_a_delivered_porosity_that_matches_scores_zero(self):
        p_pore = torch.zeros(B, D, D, D)
        p_pore[:, 0] = 1.0                            # 1/8 of the voxels
        phi = torch.full((B, 1), 0.125)
        assert float(porosity_consistency(p_pore, phi)) == pytest.approx(0.0, abs=1e-6)

    def test_under_delivery_is_penalised_by_the_gap(self):
        p_pore = torch.zeros(B, D, D, D)
        phi = torch.full((B, 1), 0.05)
        assert float(porosity_consistency(p_pore, phi)) == pytest.approx(0.05, abs=1e-6)

    def test_over_delivery_is_penalised_the_same(self):
        p_pore = torch.full((B, D, D, D), 0.10)
        phi = torch.full((B, 1), 0.05)
        assert float(porosity_consistency(p_pore, phi)) == pytest.approx(0.05, abs=1e-6)

    def test_it_is_measured_per_item(self):
        p_pore = torch.zeros(B, D, D, D)
        p_pore[0] = 0.10
        phi = torch.tensor([[0.10], [0.02]])
        # item 0 exact, item 1 off by 0.02 -> mean 0.01
        assert float(porosity_consistency(p_pore, phi)) == pytest.approx(0.01, abs=1e-6)


# ── term (d): grey / label agreement ──────────────────────────────────────────

class TestGreyAgreement:

    def test_dark_pores_and_bright_material_score_zero(self):
        grey = torch.full((B, D, D, D), BRIGHT)
        grey[:, :4] = DARK
        probs = _probs(1.0, 0.0, 0.0)
        probs[:, CLASS_MATERIAL, :4] = 0.0
        probs[:, CLASS_PORE, :4] = 1.0
        assert float(grey_agreement(probs, grey)) == pytest.approx(0.0, abs=1e-6)

    def test_a_bright_voxel_called_pore_is_penalised(self):
        grey = torch.full((B, D, D, D), BRIGHT)
        value = float(grey_agreement(_probs(0.0, 1.0, 0.0), grey))
        assert value == pytest.approx(BRIGHT - AIR_GREY_THRESHOLD, abs=1e-6)

    def test_a_dark_voxel_called_material_is_penalised(self):
        grey = torch.full((B, D, D, D), DARK)
        value = float(grey_agreement(_probs(1.0, 0.0, 0.0), grey))
        assert value == pytest.approx(AIR_GREY_THRESHOLD - DARK, abs=1e-6)

    def test_dark_air_is_free(self):
        """Exterior air and the drilled holes are dark AND correct."""
        grey = torch.full((B, D, D, D), DARK)
        assert float(grey_agreement(_probs(0.0, 0.0, 1.0), grey)) == pytest.approx(0.0)

    def test_the_penalty_is_one_sided_per_class(self):
        """A dark pore and a bright material voxel are both fine."""
        assert float(grey_agreement(_probs(0.0, 1.0, 0.0),
                                    torch.full((B, D, D, D), DARK))) == pytest.approx(0.0)
        assert float(grey_agreement(_probs(1.0, 0.0, 0.0),
                                    torch.full((B, D, D, D), BRIGHT))) == pytest.approx(0.0)

    def test_it_is_differentiable_in_both_heads(self):
        probs = _probs(0.0, 1.0, 0.0).requires_grad_(True)
        grey = torch.full((B, D, D, D), BRIGHT, requires_grad=True)
        grey_agreement(probs, grey).backward()
        assert float(probs.grad.abs().sum()) > 0.0
        assert float(grey.grad.abs().sum()) > 0.0


# ── sub-batch selection and ramp ──────────────────────────────────────────────

class TestSelection:

    def test_only_the_lowest_quartile_is_eligible(self):
        t = torch.tensor([10, 200, 249, 250, 900])
        idx = select_decoded_items(t, T=1000, t_max_frac=0.25, max_items=99)
        assert sorted(t[idx].tolist()) == [10, 200, 249]

    def test_the_cap_keeps_the_lowest_t_items(self):
        t = torch.tensor([90, 10, 50, 30, 70])
        idx = select_decoded_items(t, T=1000, t_max_frac=1.0, max_items=3)
        assert t[idx].tolist() == [10, 30, 50]           # sorted, lowest first

    def test_the_cap_is_a_hard_bound(self):
        t = torch.zeros(200, dtype=torch.long)
        assert select_decoded_items(t, 1000, 1.0, 32).numel() == 32

    def test_no_eligible_item_returns_an_empty_selection(self):
        t = torch.full((8,), 900, dtype=torch.long)
        assert select_decoded_items(t, 1000, 0.25, 32).numel() == 0

    def test_the_indices_address_the_original_batch(self):
        t = torch.tensor([900, 5, 900, 3])
        idx = select_decoded_items(t, 1000, 0.25, 32)
        assert idx.tolist() == [3, 1]                    # positions, not values

    def test_a_zero_cap_decodes_nothing(self):
        t = torch.zeros(8, dtype=torch.long)
        assert select_decoded_items(t, 1000, 0.25, 0).numel() == 0


class TestRamp:

    def test_it_rises_linearly_and_saturates(self):
        assert ramp_weight(0, 100) == pytest.approx(0.01)
        assert ramp_weight(49, 100) == pytest.approx(0.50)
        assert ramp_weight(99, 100) == pytest.approx(1.0)
        assert ramp_weight(1000, 100) == 1.0

    def test_a_zero_ramp_is_full_weight(self):
        assert ramp_weight(0, 0) == 1.0


# ── the orchestrator, against a stand-in 3-class VAE ──────────────────────────

class _TinyVAE(nn.Module):
    """A decoder-shaped stand-in: latent (C, L, L, L) -> voxels (D, D, D).

    Real enough to test the plumbing — it has the three attributes the loss
    requires, a BatchNorm the eval()/train() check can observe, and parameters
    that would collect a gradient if the loss let them.
    """

    def __init__(self, z_channels: int = 2, width: int = 4) -> None:
        super().__init__()
        self.decoder = nn.Sequential(
            nn.Upsample(scale_factor=D // L, mode="nearest"),
            nn.Conv3d(z_channels, width, 3, padding=1),
            nn.BatchNorm3d(width),
        )
        self.xct_head = nn.Conv3d(width, 1, 1)
        self.class_head = nn.Conv3d(width, 3, 1)


def _frozen_vae(z_channels: int = 2) -> _TinyVAE:
    vae = _TinyVAE(z_channels)
    for p in vae.parameters():
        p.requires_grad_(False)
    return vae


def _aux(**overrides) -> DecodedAuxLoss:
    cfg = DecodedLossConfig(t_max_frac=0.25, ramp_steps=0, max_items=2,
                            **overrides)
    return DecodedAuxLoss(cfg, _frozen_vae(), latent_mean=0.0, latent_std=1.0)


def _call_batch(n: int = 4, z_channels: int = 2, t=None):
    torch.manual_seed(0)
    z_t = torch.randn(n, z_channels, L, L, L)
    model_out = torch.randn(n, z_channels, L, L, L, requires_grad=True)
    if t is None:
        t = torch.tensor([5, 300, 9, 400])[:n]
    batch = {
        "cond_material": torch.ones(n, 1, L, L, L),
        "label": torch.zeros(n, D, D, D, dtype=torch.uint8),
        "phi": torch.full((n, 1), 0.02),
    }
    return z_t, model_out, t, batch


class TestDecodedAuxLoss:

    def test_it_refuses_a_vae_that_is_not_frozen(self):
        with pytest.raises(ValueError, match="FROZEN"):
            DecodedAuxLoss(DecodedLossConfig(), _TinyVAE(), 0.0, 1.0)

    def test_it_refuses_a_vae_without_a_class_head(self):
        vae = _frozen_vae()
        del vae.class_head
        with pytest.raises(TypeError, match="class_head"):
            DecodedAuxLoss(DecodedLossConfig(), vae, 0.0, 1.0)

    def test_it_returns_one_scalar_and_a_metric_per_term(self):
        aux = _aux()
        z_t, model_out, t, batch = _call_batch()
        loss, metrics = aux(model_out=model_out, z_t=z_t, t=t,
                            schedule=DDPMSchedule(T=1000), batch=batch, step=0,
                            autocast_dtype=torch.float32)
        assert loss is not None and loss.ndim == 0 and torch.isfinite(loss)
        for key in ("aux_air_outside_material", "aux_pore_dice",
                    "aux_porosity_consistency", "aux_grey_agreement",
                    "aux_total", "aux_items", "aux_ramp"):
            assert key in metrics
        assert metrics["aux_items"] == 2.0        # the max_items cap

    def test_no_eligible_item_returns_no_loss_but_still_reports(self):
        """A skipped step must be visible in the log, not silently absent."""
        aux = _aux()
        z_t, model_out, _, batch = _call_batch()
        t = torch.full((4,), 900, dtype=torch.long)
        loss, metrics = aux(model_out=model_out, z_t=z_t, t=t,
                            schedule=DDPMSchedule(T=1000), batch=batch, step=0,
                            autocast_dtype=torch.float32)
        assert loss is None
        assert metrics["aux_items"] == 0.0

    def test_the_gradient_reaches_the_model_output(self):
        aux = _aux()
        z_t, model_out, t, batch = _call_batch()
        loss, _ = aux(model_out=model_out, z_t=z_t, t=t,
                      schedule=DDPMSchedule(T=1000), batch=batch, step=0,
                      autocast_dtype=torch.float32)
        loss.backward()
        assert model_out.grad is not None
        assert float(model_out.grad.abs().sum()) > 0.0

    def test_only_the_selected_items_receive_a_gradient(self):
        """The cap must not silently spread the signal over the whole batch."""
        aux = _aux()
        z_t, model_out, _, batch = _call_batch()
        t = torch.tensor([5, 900, 9, 900])
        loss, _ = aux(model_out=model_out, z_t=z_t, t=t,
                      schedule=DDPMSchedule(T=1000), batch=batch, step=0,
                      autocast_dtype=torch.float32)
        loss.backward()
        per_item = model_out.grad.abs().flatten(1).sum(1)
        assert float(per_item[0]) > 0.0 and float(per_item[2]) > 0.0
        assert float(per_item[1]) == 0.0 and float(per_item[3]) == 0.0

    def test_the_vae_parameters_receive_no_gradient(self):
        """The LDM must not be able to move the decoder it is scored against."""
        aux = _aux()
        z_t, model_out, t, batch = _call_batch()
        loss, _ = aux(model_out=model_out, z_t=z_t, t=t,
                      schedule=DDPMSchedule(T=1000), batch=batch, step=0,
                      autocast_dtype=torch.float32)
        loss.backward()
        for name, p in aux.vae.named_parameters():
            assert p.grad is None, f"{name} collected a gradient"

    def test_the_decode_never_updates_the_frozen_batchnorm(self):
        """eval() is load-bearing: requires_grad_(False) does not stop BN."""
        aux = _aux()
        aux.vae.train()                       # the caller left it in train mode
        bn = aux.vae.decoder[2]
        before = (bn.running_mean.clone(), bn.running_var.clone(),
                  bn.num_batches_tracked.clone())
        z_t, model_out, t, batch = _call_batch()
        aux(model_out=model_out, z_t=z_t, t=t, schedule=DDPMSchedule(T=1000),
            batch=batch, step=0, autocast_dtype=torch.float32)
        assert torch.equal(bn.running_mean, before[0])
        assert torch.equal(bn.running_var, before[1])
        assert torch.equal(bn.num_batches_tracked, before[2])

    def test_the_ramp_scales_the_total_but_not_the_reported_terms(self):
        z_t, model_out, t, batch = _call_batch()
        schedule = DDPMSchedule(T=1000)
        kw = dict(model_out=model_out, z_t=z_t, t=t, schedule=schedule,
                  batch=batch, autocast_dtype=torch.float32)

        aux = DecodedAuxLoss(
            DecodedLossConfig(t_max_frac=0.25, ramp_steps=100, max_items=2),
            _frozen_vae(), 0.0, 1.0,
        )
        early_loss, early = aux(step=0, **kw)
        late_loss, late = aux(step=1000, **kw)

        assert early["aux_ramp"] == pytest.approx(0.01)
        assert late["aux_ramp"] == 1.0
        # The per-term values are the measurement and must not move with the ramp.
        assert early["aux_pore_dice"] == pytest.approx(late["aux_pore_dice"], abs=1e-6)
        assert abs(float(early_loss.detach())) < abs(float(late_loss.detach()))

    def test_a_missing_label_is_rejected_with_a_useful_message(self):
        aux = _aux()
        z_t, model_out, t, batch = _call_batch()
        del batch["label"]
        with pytest.raises(KeyError, match="label"):
            aux(model_out=model_out, z_t=z_t, t=t, schedule=DDPMSchedule(T=1000),
                batch=batch, step=0, autocast_dtype=torch.float32)

    def test_it_works_under_the_v_objective_too(self):
        """x0 comes from schedule.predict_x0, so the objective is not its business."""
        aux = _aux()
        z_t, model_out, t, batch = _call_batch()
        schedule = DDPMSchedule(T=1000, objective="v", zero_terminal_snr=True)
        loss, metrics = aux(model_out=model_out, z_t=z_t, t=t, schedule=schedule,
                            batch=batch, step=0, autocast_dtype=torch.float32)
        assert loss is not None and torch.isfinite(loss)
        assert metrics["aux_items"] == 2.0


# ── config resolution ─────────────────────────────────────────────────────────

class TestConfig:

    def test_a_disabled_block_builds_nothing(self):
        assert DecodedLossConfig.from_cfg({}) is None
        assert DecodedLossConfig.from_cfg({"loss": {"decoded": {"enabled": False}}}) is None

    def test_ldm06_base_leaves_the_decoded_loss_off(self):
        from poregen.configuration import resolve_experiment

        cfg = resolve_experiment("ldm06/base").cfg
        assert DecodedLossConfig.from_cfg(cfg) is None

    def test_ldm06_aux_resolves_to_the_specified_block(self):
        from poregen.configuration import resolve_experiment

        cfg = resolve_experiment("ldm06/aux").cfg
        decoded = DecodedLossConfig.from_cfg(cfg)
        assert decoded is not None
        assert decoded.t_max_frac == 0.25
        assert decoded.max_items == 32
        assert decoded.ramp_steps == 10000
        assert decoded.w_air_outside_material == 1.0
        assert decoded.w_pore_dice == 1.0
        assert decoded.w_porosity_consistency == 1.0
        assert decoded.w_grey_agreement == 0.5
        # Everything else is ldm06/base — the rung differs by the loss alone.
        assert cfg["data"]["latents_root"] == (
            f"data/split_v3/latents_r08z{cfg['model']['z_channels']}")
        assert cfg["training"]["seed"] == 42
        assert cfg["training"]["total_steps"] == 130000


# ── the loader serves the label only when it is asked for ─────────────────────

def _store_with_labels(tmp_path):
    """The miniature ldm06 store plus the patch-level arrays it points at."""
    from _ldm06_store import SYN, build_store

    data_root = tmp_path / "data_root"
    data_root.mkdir()
    root, *_ = build_store(tmp_path / "store")

    import json
    import pandas as pd

    meta = json.loads((root / "metadata.json").read_text())
    n = len(pd.read_parquet(root / "train" / "index.parquet"))
    ps = SYN["PATCH"]

    # patches_label.bin is row-aligned with patch_index.parquet; the store's
    # source_row indexes into it.  Row i is filled with i % 3 so the test can
    # tell rows apart.
    label = np.zeros((n, ps, ps, ps), np.uint8)
    for i in range(n):
        label[i] = i % 3
    label.tofile(data_root / "patches_label.bin")
    (data_root / "patches_meta.json").write_text(
        json.dumps({"N": n, "patch_size": ps})
    )
    (data_root / "patch_index.parquet").write_bytes(b"")

    meta["source_patch_index"] = str(data_root / "patch_index.parquet")
    (root / "metadata.json").write_text(json.dumps(meta))
    return root, label


class TestLatentDatasetLabel:

    def test_the_label_is_absent_unless_requested(self, tmp_path):
        from _ldm06_store import dataset_kwargs
        from poregen.diffusion.latents import LatentDataset

        root, _ = _store_with_labels(tmp_path)
        ds = LatentDataset(root, "train", normalize=True, **dataset_kwargs())
        assert "label" not in ds[0]

    def test_the_label_comes_from_the_row_source_row_names(self, tmp_path):
        from _ldm06_store import SYN, dataset_kwargs
        from poregen.diffusion.latents import LatentDataset

        root, label = _store_with_labels(tmp_path)
        ds = LatentDataset(root, "train", normalize=True, with_label=True,
                           **dataset_kwargs())
        for i in (0, 5, 17):
            item = ds[i]
            assert item["label"].dtype == torch.uint8
            assert tuple(item["label"].shape) == (SYN["PATCH"],) * 3
            expected = label[item["source_row"]]
            assert np.array_equal(item["label"].numpy(), expected)

    def test_a_missing_label_file_names_the_data_root(self, tmp_path):
        from _ldm06_store import build_store, dataset_kwargs
        from poregen.diffusion.latents import LatentDataset

        root, *_ = build_store(tmp_path / "store")
        import json
        meta = json.loads((root / "metadata.json").read_text())
        meta["source_patch_index"] = str(tmp_path / "nowhere" / "patch_index.parquet")
        (root / "metadata.json").write_text(json.dumps(meta))

        with pytest.raises(FileNotFoundError, match="patches_label.bin"):
            LatentDataset(root, "train", with_label=True, **dataset_kwargs())

    @pytest.mark.parametrize("enabled", [False, True])
    def test_the_dataloader_builder_reads_the_flag_off_the_loss_block(self, tmp_path, enabled):
        """A run cannot ask for the decoded loss and get label-free batches."""
        import shutil

        from _ldm06_store import SYN
        from poregen.diffusion.latents import build_latent_dataloaders

        root, _ = _store_with_labels(tmp_path)
        shutil.copytree(root / "train", root / "val")   # the builder wants both

        cfg = {
            "training": {"batch_size": 2},
            "data": {
                "num_workers": 0,
                "sample_stride": SYN["SAMPLE_STRIDE"],
                "generation_stride": SYN["OFFSET"],
                "neighbour_offset": SYN["OFFSET"],
            },
            "loss": {"decoded": {"enabled": enabled}},
        }
        train_loader, _ = build_latent_dataloaders(cfg, root)
        assert train_loader.dataset.with_label is enabled
        assert ("label" in next(iter(train_loader))) is enabled


# ── the training step actually consumes it ────────────────────────────────────

class TestTrainStepIntegration:

    @staticmethod
    def _batch_and_model(tmp_path):
        from _ldm06_store import dataset_kwargs
        from poregen.diffusion.latents import LatentDataset
        from poregen.models.diffusion import UNet3DConfig, UNet3DDenoiser

        root, _ = _store_with_labels(tmp_path)
        ds = LatentDataset(root, "train", normalize=True, with_label=True,
                           **dataset_kwargs())
        items = [ds[i] for i in range(4)]
        keys = ("z", "std", "cond_por", "cond_depth", "cond_dist6", "cond_orient",
                "cond_material", "nb_latents", "nb_std", "nb_avail", "phi", "label")
        batch = {k: torch.stack([it[k] for it in items]) for k in keys}

        cfg = UNet3DConfig(z_channels=batch["z"].shape[1], base_channels=8,
                           channel_mult=(1, 2), n_res_blocks=1, cond_embed_dim=16)
        return batch, UNet3DDenoiser(cfg)

    @staticmethod
    def _aux_for(batch):
        z_channels = batch["z"].shape[1]
        cfg = DecodedLossConfig(t_max_frac=1.0, ramp_steps=0, max_items=2)
        return DecodedAuxLoss(cfg, _frozen_vae(z_channels), 0.0, 1.0)

    def test_the_step_reports_every_term_and_the_latent_loss_separately(self, tmp_path):
        """A single summed number cannot say WHICH defect moved."""
        from poregen.training.ldm_engine import ldm_train_step

        batch, model = self._batch_and_model(tmp_path)
        out = ldm_train_step(
            model, batch, torch.optim.AdamW(model.parameters(), lr=1e-3),
            torch.amp.GradScaler(enabled=False), DDPMSchedule(T=1000),
            step=0, device=torch.device("cpu"), autocast_dtype=torch.float32,
            decoded_aux=self._aux_for(batch),
        )
        assert np.isfinite(out["loss"]) and np.isfinite(out["grad_norm"])
        assert out["loss"] != out["latent_loss"]        # the aux was added
        for key in ("aux_air_outside_material", "aux_pore_dice",
                    "aux_porosity_consistency", "aux_grey_agreement",
                    "aux_total", "aux_items", "aux_ramp"):
            assert key in out

    def test_without_the_aux_the_step_is_unchanged(self, tmp_path):
        from poregen.training.ldm_engine import ldm_train_step

        batch, model = self._batch_and_model(tmp_path)
        out = ldm_train_step(
            model, batch, torch.optim.AdamW(model.parameters(), lr=1e-3),
            torch.amp.GradScaler(enabled=False), DDPMSchedule(T=1000),
            step=0, device=torch.device("cpu"), autocast_dtype=torch.float32,
        )
        assert set(out) == {"loss", "grad_norm"}

    def test_the_decoder_stays_out_of_the_optimiser_step(self, tmp_path):
        """The optimiser only ever sees denoiser parameters."""
        from poregen.training.ldm_engine import ldm_train_step

        batch, model = self._batch_and_model(tmp_path)
        aux = self._aux_for(batch)
        before = {k: v.clone() for k, v in aux.vae.state_dict().items()}
        ldm_train_step(
            model, batch, torch.optim.AdamW(model.parameters(), lr=1e-1),
            torch.amp.GradScaler(enabled=False), DDPMSchedule(T=1000),
            step=0, device=torch.device("cpu"), autocast_dtype=torch.float32,
            decoded_aux=aux,
        )
        for k, v in aux.vae.state_dict().items():
            assert torch.equal(v, before[k]), f"the frozen VAE's {k} moved"
        for name, p in aux.vae.named_parameters():
            assert p.grad is None, f"{name} collected a gradient"
