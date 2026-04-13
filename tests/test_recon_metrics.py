import json

import torch

from poregen.experiments.r03 import AuxiliaryXCTDecoder
from poregen.metrics.recon import sharpness_proxy
from poregen.models.vae.base import VAEConfig, VAEOutput
from poregen.training.engine import _run_eval, train_loop


class _DummyVAE(torch.nn.Module):
    def __init__(self, xct_logits: torch.Tensor, mask_logits: torch.Tensor) -> None:
        super().__init__()
        self._xct_logits = xct_logits
        self._mask_logits = mask_logits

    def forward(self, xct: torch.Tensor, mask: torch.Tensor) -> VAEOutput:
        batch_size = xct.shape[0]
        mu = torch.zeros(batch_size, 2, 1, 1, 1, dtype=xct.dtype, device=xct.device)
        logvar = torch.zeros_like(mu)
        return VAEOutput(
            xct_logits=self._xct_logits.to(xct.device),
            mask_logits=self._mask_logits.to(mask.device),
            mu=mu,
            logvar=logvar,
            z=mu,
        )


class _MuFromInputVAE(torch.nn.Module):
    def forward(self, xct: torch.Tensor, mask: torch.Tensor) -> VAEOutput:
        batch_size = xct.shape[0]
        mu = torch.zeros(batch_size, 2, 1, 1, 1, dtype=xct.dtype, device=xct.device)
        mu[:, 0, 0, 0, 0] = xct[:, 0, 0, 0, 0]
        logvar = torch.zeros_like(mu)
        zeros = torch.zeros_like(xct)
        return VAEOutput(
            xct_logits=zeros,
            mask_logits=zeros,
            mu=mu,
            logvar=logvar,
            z=mu,
        )


class _TinyTrainVAE(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.bias = torch.nn.Parameter(torch.tensor(0.0))

    def forward(self, xct: torch.Tensor, mask: torch.Tensor) -> VAEOutput:
        batch_size = xct.shape[0]
        xct_logits = xct * 0.0 + self.bias
        mask_logits = mask * 0.0
        mu = torch.zeros(batch_size, 2, 1, 1, 1, dtype=xct.dtype, device=xct.device)
        logvar = torch.zeros_like(mu)
        return VAEOutput(
            xct_logits=xct_logits,
            mask_logits=mask_logits,
            mu=mu,
            logvar=logvar,
            z=mu,
        )


class _DummyWriter:
    def add_scalar(self, *args, **kwargs):
        return None

    def add_histogram(self, *args, **kwargs):
        return None

    def add_images(self, *args, **kwargs):
        return None


def test_run_eval_logs_sharpness_on_post_sigmoid_xct():
    xct = torch.tensor(
        [[[[[0.0, 1.0], [1.0, 0.0]], [[1.0, 0.0], [0.0, 1.0]]]]],
        dtype=torch.float32,
    )
    mask = torch.zeros_like(xct)
    xct_logits = torch.tensor(
        [[[[[-4.0, 4.0], [4.0, -4.0]], [[4.0, -4.0], [-4.0, 4.0]]]]],
        dtype=torch.float32,
    )
    mask_logits = torch.zeros_like(xct_logits)

    model = _DummyVAE(xct_logits=xct_logits, mask_logits=mask_logits)
    batch = {
        "xct": xct,
        "mask": mask,
        "volume_id": ["volume_001"],
        "coords": (
            torch.tensor([0]),
            torch.tensor([0]),
            torch.tensor([0]),
        ),
        "porosity": torch.tensor([0.0]),
        "source_group": ["synthetic"],
    }

    def loss_fn(output, batch_dev, step):
        return {
            "total": torch.tensor(0.0, device=batch_dev["xct"].device),
            "xct_loss": torch.tensor(0.0, device=batch_dev["xct"].device),
            "kl_per_channel": torch.tensor([0.0, 0.0], device=batch_dev["xct"].device),
        }

    metrics, _, _, _ = _run_eval(
        model=model,
        data_iter=iter([batch]),
        loss_fn=loss_fn,
        n_batches=1,
        step=0,
        device=torch.device("cpu"),
        autocast_dtype=torch.bfloat16,
    )

    expected_recon = sharpness_proxy(torch.sigmoid(xct_logits)).item()
    expected_gt = sharpness_proxy(xct).item()

    assert metrics["sharpness_recon"] == expected_recon
    assert metrics["sharpness_gt"] == expected_gt
    assert metrics["sharpness_recon_over_gt"] == expected_recon / expected_gt
    assert metrics["sharpness_recon"] != sharpness_proxy(xct_logits).item()


def test_auxiliary_decoder_restores_r03_patch_shape():
    cfg = VAEConfig(z_channels=4, base_channels=8, n_blocks=2, patch_size=16)
    decoder = AuxiliaryXCTDecoder(cfg)
    mu = torch.randn(3, cfg.z_channels, cfg.latent_spatial, cfg.latent_spatial, cfg.latent_spatial)

    out = decoder(mu)

    assert out.shape == (3, 1, cfg.patch_size, cfg.patch_size, cfg.patch_size)


def test_run_eval_aggregates_active_units_across_eval_window():
    model = _MuFromInputVAE()

    def make_batch(first_voxel: float) -> dict:
        xct = torch.zeros(1, 1, 2, 2, 2, dtype=torch.float32)
        xct[0, 0, 0, 0, 0] = first_voxel
        return {
            "xct": xct,
            "mask": torch.zeros_like(xct),
            "volume_id": [f"volume_{first_voxel}"],
            "coords": (
                torch.tensor([0]),
                torch.tensor([0]),
                torch.tensor([0]),
            ),
            "porosity": torch.tensor([0.0]),
            "source_group": ["synthetic"],
        }

    def loss_fn(output, batch_dev, step):
        return {
            "total": torch.tensor(0.0, device=batch_dev["xct"].device),
            "xct_loss": torch.tensor(0.0, device=batch_dev["xct"].device),
            "kl_per_channel": torch.tensor([0.0, 0.0], device=batch_dev["xct"].device),
        }

    metrics, _, _, _ = _run_eval(
        model=model,
        data_iter=iter([make_batch(0.0), make_batch(1.0)]),
        loss_fn=loss_fn,
        n_batches=2,
        step=0,
        device=torch.device("cpu"),
        autocast_dtype=torch.bfloat16,
    )

    assert metrics["n_total"] == 2
    assert metrics["n_active"] == 1
    assert metrics["active_fraction"] == 0.5


def test_train_loop_skips_image_logging_until_first_validation(tmp_path):
    class _Dataset(torch.utils.data.Dataset):
        def __len__(self):
            return 2

        def __getitem__(self, idx):
            xct = torch.zeros(1, 2, 2, 2, dtype=torch.float32)
            mask = torch.zeros_like(xct)
            return {
                "xct": xct,
                "mask": mask,
                "volume_id": f"volume_{idx}",
                "coords": (0, 0, 0),
                "porosity": 0.0,
                "source_group": "synthetic",
            }

    model = _TinyTrainVAE()
    train_loader = torch.utils.data.DataLoader(_Dataset(), batch_size=1, shuffle=False)
    val_loader = torch.utils.data.DataLoader(_Dataset(), batch_size=1, shuffle=False)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    scaler = torch.amp.GradScaler(enabled=False)

    def loss_fn(output, batch_dev, step):
        total = output.xct_logits.mean()
        return {
            "total": total,
            "xct_loss": total.detach() * 0.0,
            "kl_per_channel": torch.tensor([0.0, 0.0], device=batch_dev["xct"].device),
        }

    history = train_loop(
        model,
        train_loader,
        val_loader,
        optimizer,
        scaler,
        loss_fn,
        total_steps=1,
        log_every=1,
        eval_every=10,
        val_batches=1,
        save_every=100,
        image_log_every=1,
        sample_every=0,
        run_dir=tmp_path,
        device=torch.device("cpu"),
        autocast_dtype=torch.bfloat16,
        tb_writer=_DummyWriter(),
    )

    assert len(history) == 1
    assert history[0]["split"] == "train"


def test_train_loop_runs_montecarlo_on_fixed_cadence(monkeypatch, tmp_path):
    class _Dataset(torch.utils.data.Dataset):
        def __len__(self):
            return 2

        def __getitem__(self, idx):
            xct = torch.zeros(1, 2, 2, 2, dtype=torch.float32)
            mask = torch.zeros_like(xct)
            return {
                "xct": xct,
                "mask": mask,
                "volume_id": f"volume_{idx}",
                "coords": (0, 0, 0),
                "porosity": 0.0,
                "source_group": "synthetic",
            }

    calls: list[tuple[int, int]] = []

    def fake_run_montecarlo_eval(model, batch, step, device, writer, **kwargs):
        calls.append((step, batch["xct"].shape[0]))

    monkeypatch.setattr("poregen.training.engine.run_montecarlo_eval", fake_run_montecarlo_eval)

    model = _TinyTrainVAE()
    train_loader = torch.utils.data.DataLoader(_Dataset(), batch_size=1, shuffle=False)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    scaler = torch.amp.GradScaler(enabled=False)

    def loss_fn(output, batch_dev, step):
        total = output.xct_logits.mean()
        return {
            "total": total,
            "xct_loss": total.detach() * 0.0,
            "kl_per_channel": torch.tensor([0.0, 0.0], device=batch_dev["xct"].device),
        }

    train_loop(
        model,
        train_loader,
        None,
        optimizer,
        scaler,
        loss_fn,
        total_steps=3,
        log_every=1,
        eval_every=10,
        val_batches=1,
        save_every=100,
        image_log_every=0,
        montecarlo_every=2,
        montecarlo_batch_size=1,
        sample_every=0,
        run_dir=tmp_path,
        device=torch.device("cpu"),
        autocast_dtype=torch.bfloat16,
        tb_writer=_DummyWriter(),
    )

    assert calls == [(1, 1)]


def test_train_loop_runs_final_full_eval_for_val_and_test(tmp_path):
    class _Dataset(torch.utils.data.Dataset):
        def __len__(self):
            return 2

        def __getitem__(self, idx):
            xct = torch.zeros(1, 2, 2, 2, dtype=torch.float32)
            mask = torch.zeros_like(xct)
            return {
                "xct": xct,
                "mask": mask,
                "volume_id": f"volume_{idx}",
                "coords": (0, 0, 0),
                "porosity": 0.0,
                "source_group": "synthetic",
            }

    model = _TinyTrainVAE()
    train_loader = torch.utils.data.DataLoader(_Dataset(), batch_size=1, shuffle=False)
    val_loader = torch.utils.data.DataLoader(_Dataset(), batch_size=1, shuffle=False)
    test_loader = torch.utils.data.DataLoader(_Dataset(), batch_size=1, shuffle=False)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    scaler = torch.amp.GradScaler(enabled=False)

    def loss_fn(output, batch_dev, step):
        total = output.xct_logits.mean()
        return {
            "total": total,
            "xct_loss": total.detach() * 0.0,
            "kl_per_channel": torch.tensor([0.0, 0.0], device=batch_dev["xct"].device),
        }

    train_loop(
        model,
        train_loader,
        val_loader,
        optimizer,
        scaler,
        loss_fn,
        total_steps=1,
        log_every=1,
        eval_every=10,
        val_batches=1,
        test_loader=test_loader,
        test_every=10,
        test_batches=1,
        save_every=100,
        image_log_every=0,
        montecarlo_every=0,
        sample_every=0,
        run_dir=tmp_path,
        device=torch.device("cpu"),
        autocast_dtype=torch.bfloat16,
        tb_writer=_DummyWriter(),
        final_full_eval=True,
    )

    metrics = [
        json.loads(line)
        for line in (tmp_path / "metrics.jsonl").read_text().splitlines()
        if line.strip()
    ]

    assert len(metrics) == 2
    assert metrics[0]["split"] == "val"
    assert metrics[0]["full_eval"] is True
    assert metrics[0]["n_batches"] == len(val_loader)
    assert metrics[1]["split"] == "test"
    assert metrics[1]["full_eval"] is True
    assert metrics[1]["n_batches"] == len(test_loader)
