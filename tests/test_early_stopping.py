from __future__ import annotations

import torch

from poregen.models.vae.base import VAEOutput
from poregen.training.engine import train_loop


class _Dataset(torch.utils.data.Dataset):
    def __len__(self):
        return 2

    def __getitem__(self, idx):
        xct = torch.zeros(1, 2, 2, 2)
        return {"xct": xct, "mask": torch.zeros_like(xct)}


class _Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.bias = torch.nn.Parameter(torch.tensor(0.0))

    def forward(self, xct, mask):
        mu = torch.zeros(xct.shape[0], 2, 1, 1, 1, device=xct.device)
        return VAEOutput(
            xct_out=xct * 0.0 + self.bias,
            mask_logits=None,
            mu=mu,
            logvar=mu,
            z=mu,
        )


def test_train_loop_stops_after_validation_patience(tmp_path):
    model = _Model()
    loader = torch.utils.data.DataLoader(_Dataset(), batch_size=1)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    scaler = torch.amp.GradScaler(enabled=False)

    def loss_fn(output, batch, step):
        total = output.xct_out.mean()
        return {
            "total": total,
            "xct_loss": total.detach() * 0.0,
            "kl_per_channel": torch.zeros(2),
        }

    history = train_loop(
        model,
        loader,
        loader,
        optimizer,
        scaler,
        loss_fn,
        total_steps=10,
        eval_every=1,
        val_batches=1,
        save_every=100,
        image_log_every=0,
        montecarlo_every=0,
        montecarlo_batch_size=0,
        sample_every=0,
        run_dir=tmp_path,
        device=torch.device("cpu"),
        autocast_dtype=torch.bfloat16,
        early_stopping_patience=2,
        early_stopping_metric="val.xct_loss",
    )

    assert len([r for r in history if r["split"] == "train"]) == 3
    assert history[-1]["event"] == "early_stopping"
    assert (tmp_path / f"{tmp_path.name}_step00000003.ckpt").exists()


# ---------------------------------------------------------------------------
# Patience in steps rather than eval checks
# ---------------------------------------------------------------------------

class TestPatienceInSteps:
    """Patience counted in eval checks silently depends on the data.

    ``eval_every`` is derived from the split sizes, so the split_v3 re-split —
    which grew val 89 % — halved it from 264 to 120 and with it the effective
    patience, 3168 training steps down to 1440, without any config changing.
    Expressing patience in STEPS makes the protocol independent of the data.
    """

    @staticmethod
    def _cfg(**training):
        return {"training": {"eval_every": 120, **training}}

    def test_steps_convert_to_checks_at_the_current_cadence(self):
        from poregen.experiments.train_vae import resolve_early_stopping_patience
        assert resolve_early_stopping_patience(
            self._cfg(early_stopping_patience_steps=3000)) == 25

    def test_the_same_step_budget_survives_a_cadence_change(self):
        """3000 steps is 25 checks at eval_every 120 and 12 at 264.

        12 checks at 264 is exactly what r08-run-0002 ran, which is the point:
        the step budget is the invariant, the check count is not.
        """
        from poregen.experiments.train_vae import resolve_early_stopping_patience
        a = resolve_early_stopping_patience(
            {"training": {"eval_every": 120, "early_stopping_patience_steps": 3000}})
        b = resolve_early_stopping_patience(
            {"training": {"eval_every": 264, "early_stopping_patience_steps": 3000}})
        assert (a, b) == (25, 12)
        assert a * 120 >= 3000 and b * 264 >= 3000

    def test_the_eval_check_field_still_works_for_historical_configs(self):
        """r03-r07, vrrae03/04 and ldm05/06 are configured in checks.

        Their runs are the record; rewriting those configs would change what a
        historical experiment id resolves to.
        """
        from poregen.experiments.train_vae import resolve_early_stopping_patience
        assert resolve_early_stopping_patience(
            self._cfg(early_stopping_patience=12)) == 12

    def test_absent_means_disabled(self):
        from poregen.experiments.train_vae import resolve_early_stopping_patience
        assert resolve_early_stopping_patience(self._cfg()) == 0

    def test_setting_both_raises_rather_than_picking_one(self):
        """They are the same knob in different units, not a fallback chain."""
        import pytest
        from poregen.experiments.train_vae import resolve_early_stopping_patience
        with pytest.raises(ValueError, match="both set"):
            resolve_early_stopping_patience(
                self._cfg(early_stopping_patience=12,
                          early_stopping_patience_steps=3000))

    def test_every_r08_rung_uses_the_step_budget(self):
        from poregen.configuration import resolve_experiment
        for variant in ("base", "reduction-factor-2", "reduction-factor-4",
                        "reduction-factor-8", "reduction-factor-32",
                        "reduction-factor-64"):
            t = resolve_experiment(f"r08/{variant}").cfg["training"]
            assert t["early_stopping_patience_steps"] == 3000, variant
            assert t["early_stopping_patience"] == 0, variant
