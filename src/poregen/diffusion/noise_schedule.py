"""DDPM cosine noise schedule (Nichol & Dhariwal 2021).

All schedule tensors are pre-computed at construction time and stored as plain
attributes so they can be moved to a target device with :meth:`DDPMSchedule.to`
without requiring the object to be an ``nn.Module``.
"""

from __future__ import annotations

import math

import torch


class DDPMSchedule:
    """Cosine-schedule DDPM with ε-prediction.

    Parameters
    ----------
    T : int
        Number of diffusion timesteps (default 1000).
    s : float
        Small offset that prevents beta from being too small near t=0
        (Nichol & Dhariwal default: 0.008).
    device : str | torch.device
        Initial device for pre-computed tensors.
    """

    def __init__(self, T: int = 1000, s: float = 0.008, device: str | torch.device = "cpu") -> None:
        self.T = T
        self.s = s

        ts = torch.linspace(0, T, T + 1, dtype=torch.float64)
        f  = torch.cos((ts / T + s) / (1.0 + s) * math.pi / 2.0) ** 2
        alphas_cumprod = (f / f[0]).float()                             # (T+1,)

        betas = 1.0 - alphas_cumprod[1:] / alphas_cumprod[:-1]         # (T,)
        betas = betas.clamp(0.0, 0.999)

        alphas = 1.0 - betas                                            # (T,)
        # alphas_cumprod[0] = 1.0 by construction (f[0]/f[0]), so [:-1] gives
        # [ᾱ_0=1, ᾱ_1, …, ᾱ_{T-1}] — the "previous" cumprod for each timestep.
        alphas_cumprod_prev = alphas_cumprod[:-1]                       # (T,)

        # Posterior variance β̃_t = β_t (1−ᾱ_{t-1}) / (1−ᾱ_t)
        posterior_variance = betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod[1:])

        self.betas                          = betas.to(device)
        self.alphas                         = alphas.to(device)
        self.alphas_cumprod                 = alphas_cumprod[1:].to(device)   # (T,) ᾱ_1..ᾱ_T
        self.alphas_cumprod_prev            = alphas_cumprod_prev.to(device)  # (T,) ᾱ_0..ᾱ_{T-1}
        self.sqrt_alphas_cumprod            = self.alphas_cumprod.sqrt()
        self.sqrt_one_minus_alphas_cumprod  = (1.0 - self.alphas_cumprod).sqrt()
        self.posterior_variance             = posterior_variance.to(device)
        self.log_posterior_variance_clipped = posterior_variance.clamp(min=1e-20).log().to(device)
        self.sqrt_recip_alphas              = (1.0 / alphas).sqrt().to(device)
        self.posterior_mean_coef1 = (
            betas * alphas_cumprod_prev.sqrt() / (1.0 - alphas_cumprod[1:])
        ).to(device)
        self.posterior_mean_coef2 = (
            (1.0 - alphas_cumprod_prev) * alphas.sqrt() / (1.0 - alphas_cumprod[1:])
        ).to(device)

    # ------------------------------------------------------------------

    def to(self, device: str | torch.device) -> "DDPMSchedule":
        """Move all schedule tensors to *device* in-place. Returns self."""
        for attr in (
            "betas", "alphas", "alphas_cumprod", "alphas_cumprod_prev",
            "sqrt_alphas_cumprod", "sqrt_one_minus_alphas_cumprod",
            "posterior_variance", "log_posterior_variance_clipped",
            "sqrt_recip_alphas", "posterior_mean_coef1", "posterior_mean_coef2",
        ):
            setattr(self, attr, getattr(self, attr).to(device))
        return self

    # ------------------------------------------------------------------
    # Forward process
    # ------------------------------------------------------------------

    def _gather(self, coeff: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """Gather schedule coefficients at timestep indices, shaped for broadcasting.

        Parameters
        ----------
        coeff : (T,) — schedule tensor
        t     : (B,) long — timestep indices

        Returns
        -------
        (B, 1, 1, 1, 1) — broadcastable over (B, C, D, H, W)
        """
        return coeff[t].float().view(-1, 1, 1, 1, 1)

    def q_sample(
        self,
        x0: torch.Tensor,
        t: torch.Tensor,
        noise: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Sample noisy latent z_t from the forward process q(z_t | z_0).

        z_t = sqrt(ᾱ_t) · z_0  +  sqrt(1 − ᾱ_t) · ε

        Parameters
        ----------
        x0    : (B, C, D, H, W) — clean latent
        t     : (B,) long — timestep indices in [0, T-1]
        noise : optional pre-sampled ε; if None, sampled from N(0,I)

        Returns
        -------
        z_t : same shape as x0
        """
        if noise is None:
            noise = torch.randn_like(x0)
        sqrt_alpha = self._gather(self.sqrt_alphas_cumprod, t).to(x0.device)
        sqrt_one_minus = self._gather(self.sqrt_one_minus_alphas_cumprod, t).to(x0.device)
        return sqrt_alpha * x0 + sqrt_one_minus * noise

    # ------------------------------------------------------------------
    # Reverse process
    # ------------------------------------------------------------------

    def predict_x0(self, x_t: torch.Tensor, t: torch.Tensor, eps_pred: torch.Tensor) -> torch.Tensor:
        """Recover the x_0 estimate from ε-prediction.

        x̂_0 = (x_t − sqrt(1−ᾱ_t) · ε_pred) / sqrt(ᾱ_t)
        """
        sqrt_alpha     = self._gather(self.sqrt_alphas_cumprod, t).to(x_t.device)
        sqrt_one_minus = self._gather(self.sqrt_one_minus_alphas_cumprod, t).to(x_t.device)
        return (x_t - sqrt_one_minus * eps_pred) / sqrt_alpha.clamp(min=1e-8)

    def p_mean_variance(
        self,
        x_t: torch.Tensor,
        t: torch.Tensor,
        eps_pred: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute posterior mean and log-variance for the reverse step p(x_{t-1} | x_t).

        Returns
        -------
        (mean, log_variance) — both same shape as x_t
        """
        x0_pred = self.predict_x0(x_t, t, eps_pred).clamp(-10.0, 10.0)

        c1 = self._gather(self.posterior_mean_coef1, t).to(x_t.device)
        c2 = self._gather(self.posterior_mean_coef2, t).to(x_t.device)
        mean = c1 * x0_pred + c2 * x_t

        log_var = self._gather(self.log_posterior_variance_clipped, t).to(x_t.device)
        log_var = log_var.expand_as(x_t)
        return mean, log_var

    def p_sample(
        self,
        x_t: torch.Tensor,
        t: torch.Tensor,
        eps_pred: torch.Tensor,
    ) -> torch.Tensor:
        """Single reverse step: sample x_{t-1} ~ p(x_{t-1} | x_t).

        Parameters
        ----------
        x_t      : (B, C, D, H, W) — current noisy latent
        t        : (B,) long — current timestep indices
        eps_pred : (B, C, D, H, W) — model's noise prediction

        Returns
        -------
        x_{t-1} : same shape as x_t
        """
        mean, log_var = self.p_mean_variance(x_t, t, eps_pred)
        noise = torch.randn_like(x_t)
        # No noise at t=0 (final step)
        nonzero = (t > 0).float().view(-1, 1, 1, 1, 1).to(x_t.device)
        return mean + nonzero * (0.5 * log_var).exp() * noise

    def ddim_step(
        self,
        x_t: torch.Tensor,
        t: torch.Tensor,
        t_prev: torch.Tensor,
        eps_pred: torch.Tensor,
    ) -> torch.Tensor:
        """Deterministic DDIM reverse step (η=0, Song et al. 2020).

        When t_prev=0, alphas_cumprod_prev[0]=ᾱ_0=1 so the formula collapses
        to x̂_0 directly — no noise, clean final sample.
        """
        at    = self._gather(self.alphas_cumprod,      t     ).to(x_t.device)
        at_m1 = self._gather(self.alphas_cumprod_prev, t_prev).to(x_t.device)
        x0_pred = (x_t - (1.0 - at).sqrt() * eps_pred) / at.sqrt().clamp(min=1e-8)
        x0_pred = x0_pred.clamp(-10.0, 10.0)
        return at_m1.sqrt() * x0_pred + (1.0 - at_m1).sqrt() * eps_pred
