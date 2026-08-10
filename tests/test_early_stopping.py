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
            xct_logits=xct * 0.0 + self.bias,
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
        total = output.xct_logits.mean()
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
