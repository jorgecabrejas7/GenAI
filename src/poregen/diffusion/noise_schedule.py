"""DDPM cosine noise schedule (Nichol & Dhariwal 2021).

Two knobs decide what the schedule *is*, and both are read from
``cfg["noise_schedule"]`` by :meth:`DDPMSchedule.from_cfg` so that training,
generation and every diagnostic build the same object from the same run
config.

``objective``
    What the denoiser regresses.  ``"eps"`` is the original DDPM target;
    ``"v"`` is the velocity parameterisation of Salimans & Ho 2022
    (arXiv:2202.00512)::

        v_t = sqrt(ᾱ_t)·ε − sqrt(1−ᾱ_t)·x₀

    v is the only usable target once the terminal SNR is zero: at ᾱ_T = 0 the
    ε-parameterisation gives ``x̂₀ = (x_T − ε̂)/sqrt(ᾱ_T)``, a division by zero,
    and there is nothing in ``x_T`` for the network to regress ε against —
    ``x_T`` *is* the noise.  The v-form needs no division at all.

``zero_terminal_snr``
    Rescale ᾱ so ``sqrt(ᾱ_T) = 0`` exactly (Lin et al. 2024,
    arXiv:2305.08891, Algorithm 1).

    Measured on THIS schedule, the rescale moves ``sqrt(ᾱ)`` by at most
    ``6.12e-17`` anywhere.  That is not the usual story, and the reason is
    worth stating: the widespread cosine implementation derives ᾱ by
    ``cumprod(1 − β)`` after clamping β, which leaves a visible terminal
    signal (``sqrt(ᾱ_T) ≈ 0.068``), and *that* is what Lin et al. attack.  This
    schedule computes ᾱ directly from the cosine ``f``, where ``f(T) = cos(π/2)²``
    is already zero to float precision, so there is no signal leak to close.

    What the flag fixes here is the *exactness*, and the ε-parameterisation
    depended on the inexactness: at ``sqrt(ᾱ_T) = 6.12e-17`` the ε form of
    ``predict_x0`` divides by a ``1e-8`` clamp and returns x̂₀ of order 1e8,
    which the ±10 clamp then saturates.  Every ldm06 sampling chain therefore
    threw away its first DDIM step's x̂₀ and started from a clamp artefact.
    Making the terminal ᾱ exactly zero says plainly that x_T is pure noise;
    pairing it with the v objective makes x̂₀ at that step well defined
    (``x̂₀ = −v̂``, order 1) instead of undefined.

All schedule tensors are pre-computed at construction time and stored as plain
attributes so they can be moved to a target device with :meth:`DDPMSchedule.to`
without requiring the object to be an ``nn.Module``.
"""

from __future__ import annotations

import math
from typing import Any

import torch

OBJECTIVES = ("eps", "v")

# The x̂₀ estimate is clamped to this range inside the reverse process.  It is a
# guard against an off-manifold prediction blowing the whole chain up, not a
# modelling choice: the real normalised latents live well inside ±10, so a
# clamped element is a diagnostic (see ``return_x0_saturation``).
X0_CLAMP = 10.0


def _rescale_zero_terminal_snr(alphas_cumprod: torch.Tensor) -> torch.Tensor:
    """Lin et al. 2024, Algorithm 1 — force ``sqrt(ᾱ_T) = 0``.

    Shift ``sqrt(ᾱ)`` down so its last entry lands on zero, then scale it back
    up so its first entry is unchanged.  ``ᾱ_0 = 1`` therefore survives exactly
    and only the tail of the schedule moves.

    Parameters
    ----------
    alphas_cumprod : (T+1,) — ᾱ_0 … ᾱ_T, with ᾱ_0 = 1.
    """
    sqrt_ac = alphas_cumprod.sqrt()
    first, last = sqrt_ac[0].clone(), sqrt_ac[-1].clone()
    sqrt_ac = (sqrt_ac - last) * (first / (first - last))
    return sqrt_ac ** 2


class DDPMSchedule:
    """Cosine-schedule DDPM with a selectable prediction objective.

    Parameters
    ----------
    T : int
        Number of diffusion timesteps (default 1000).
    s : float
        Small offset that prevents beta from being too small near t=0
        (Nichol & Dhariwal default: 0.008).
    device : str | torch.device
        Initial device for pre-computed tensors.
    objective : {"eps", "v"}
        What the denoiser's output means.  Everything that converts a model
        output into x̂₀ or ε goes through :meth:`predict_x0` / :meth:`predict_eps`,
        so a caller never has to know which one is in force.
    zero_terminal_snr : bool
        Rescale ᾱ so ``sqrt(ᾱ_T) = 0`` exactly.

    Notes
    -----
    Index convention, unchanged from the ε-only version: ``alphas_cumprod[t]``
    is ᾱ_{t+1} and ``alphas_cumprod_prev[t]`` is ᾱ_t, so ``t = 0`` is the
    least-noisy step and ``t = T − 1`` the most-noisy one.  Sampling starts at
    ``t = T − 1`` — with ``zero_terminal_snr`` that is exactly ᾱ = 0.
    """

    def __init__(
        self,
        T: int = 1000,
        s: float = 0.008,
        device: str | torch.device = "cpu",
        *,
        objective: str = "eps",
        zero_terminal_snr: bool = False,
    ) -> None:
        if objective not in OBJECTIVES:
            raise ValueError(
                f"Unknown noise-schedule objective {objective!r} — use one of {OBJECTIVES}."
            )
        if zero_terminal_snr and objective != "v":
            raise ValueError(
                "zero_terminal_snr=True requires objective='v'.  At ᾱ_T = 0 the "
                "ε-parameterisation recovers x̂₀ by dividing by sqrt(ᾱ_T) = 0, and "
                "x_T carries no signal for ε to be regressed against."
            )
        self.T = T
        self.s = s
        self.objective = objective
        self.zero_terminal_snr = bool(zero_terminal_snr)

        ts = torch.linspace(0, T, T + 1, dtype=torch.float64)
        f  = torch.cos((ts / T + s) / (1.0 + s) * math.pi / 2.0) ** 2
        alphas_cumprod = f / f[0]                                       # (T+1,) float64
        if zero_terminal_snr:
            alphas_cumprod = _rescale_zero_terminal_snr(alphas_cumprod)
        alphas_cumprod = alphas_cumprod.float()

        betas = 1.0 - alphas_cumprod[1:] / alphas_cumprod[:-1]          # (T,)
        # A beta of exactly 1 is what zero terminal SNR means at the last step
        # (α_T = 0, the signal is entirely gone), so the ceiling is 1, not the
        # 0.999 an ε-only schedule used to keep 1/α finite.
        betas = betas.clamp(0.0, 1.0)

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
        self.sqrt_one_minus_alphas_cumprod  = (1.0 - self.alphas_cumprod).clamp(min=0.0).sqrt()
        self.posterior_variance             = posterior_variance.to(device)
        self.log_posterior_variance_clipped = posterior_variance.clamp(min=1e-20).log().to(device)
        self.posterior_mean_coef1 = (
            betas * alphas_cumprod_prev.sqrt() / (1.0 - alphas_cumprod[1:])
        ).to(device)
        self.posterior_mean_coef2 = (
            (1.0 - alphas_cumprod_prev) * alphas.sqrt() / (1.0 - alphas_cumprod[1:])
        ).to(device)

    # ------------------------------------------------------------------

    @classmethod
    def from_cfg(cls, cfg: dict[str, Any], device: str | torch.device = "cpu") -> "DDPMSchedule":
        """Build the schedule a resolved experiment config asks for.

        Every entry point — training, generation, the diagnostics — goes
        through this, so the objective can never be right in one place and
        wrong in another.
        """
        ns = cfg.get("noise_schedule") or {}
        return cls(
            T=int(ns.get("T", 1000)),
            s=float(ns.get("s", 0.008)),
            device=device,
            objective=str(ns.get("objective", "eps")),
            zero_terminal_snr=bool(ns.get("zero_terminal_snr", False)),
        )

    def to(self, device: str | torch.device) -> "DDPMSchedule":
        """Move all schedule tensors to *device* in-place. Returns self."""
        for attr in (
            "betas", "alphas", "alphas_cumprod", "alphas_cumprod_prev",
            "sqrt_alphas_cumprod", "sqrt_one_minus_alphas_cumprod",
            "posterior_variance", "log_posterior_variance_clipped",
            "posterior_mean_coef1", "posterior_mean_coef2",
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

    def _ab(self, t: torch.Tensor, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
        """``(sqrt(ᾱ_t), sqrt(1−ᾱ_t))`` gathered at *t* and moved to *device*."""
        return (
            self._gather(self.sqrt_alphas_cumprod, t).to(device),
            self._gather(self.sqrt_one_minus_alphas_cumprod, t).to(device),
        )

    def q_sample(
        self,
        x0: torch.Tensor,
        t: torch.Tensor,
        noise: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Sample noisy latent z_t from the forward process q(z_t | z_0).

        z_t = sqrt(ᾱ_t) · z_0  +  sqrt(1 − ᾱ_t) · ε

        The forward process does not depend on the objective — it is the same
        draw whether the network is later asked for ε or for v.

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
        sqrt_alpha, sqrt_one_minus = self._ab(t, x0.device)
        return sqrt_alpha * x0 + sqrt_one_minus * noise

    def training_target(
        self,
        x0: torch.Tensor,
        noise: torch.Tensor,
        t: torch.Tensor,
    ) -> torch.Tensor:
        """What the denoiser must regress at step *t*.

        ``eps`` → the noise itself.
        ``v``   → ``sqrt(ᾱ_t)·ε − sqrt(1−ᾱ_t)·x₀`` (Salimans & Ho 2022).
        """
        if self.objective == "eps":
            return noise
        sqrt_alpha, sqrt_one_minus = self._ab(t, x0.device)
        return sqrt_alpha * noise - sqrt_one_minus * x0

    # ------------------------------------------------------------------
    # Reverse process
    # ------------------------------------------------------------------

    def predict_x0(
        self,
        x_t: torch.Tensor,
        t: torch.Tensor,
        model_out: torch.Tensor,
    ) -> torch.Tensor:
        """Recover the x₀ estimate from the model's output.

        ``eps`` → ``x̂₀ = (x_t − sqrt(1−ᾱ_t)·ε̂) / sqrt(ᾱ_t)``
        ``v``   → ``x̂₀ = sqrt(ᾱ_t)·x_t − sqrt(1−ᾱ_t)·v̂``  (no division)
        """
        sqrt_alpha, sqrt_one_minus = self._ab(t, x_t.device)
        if self.objective == "eps":
            return (x_t - sqrt_one_minus * model_out) / sqrt_alpha.clamp(min=1e-8)
        return sqrt_alpha * x_t - sqrt_one_minus * model_out

    def predict_eps(
        self,
        x_t: torch.Tensor,
        t: torch.Tensor,
        model_out: torch.Tensor,
    ) -> torch.Tensor:
        """Recover the ε estimate from the model's output.

        ``eps`` → the output is already ε.
        ``v``   → ``ε̂ = sqrt(1−ᾱ_t)·x_t + sqrt(ᾱ_t)·v̂``
        """
        if self.objective == "eps":
            return model_out
        sqrt_alpha, sqrt_one_minus = self._ab(t, x_t.device)
        return sqrt_one_minus * x_t + sqrt_alpha * model_out

    def p_mean_variance(
        self,
        x_t: torch.Tensor,
        t: torch.Tensor,
        model_out: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute posterior mean and log-variance for the reverse step p(x_{t-1} | x_t).

        Returns
        -------
        (mean, log_variance) — both same shape as x_t
        """
        x0_pred = self.predict_x0(x_t, t, model_out).clamp(-X0_CLAMP, X0_CLAMP)

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
        model_out: torch.Tensor,
    ) -> torch.Tensor:
        """Single reverse step: sample x_{t-1} ~ p(x_{t-1} | x_t).

        Parameters
        ----------
        x_t       : (B, C, D, H, W) — current noisy latent
        t         : (B,) long — current timestep indices
        model_out : (B, C, D, H, W) — the denoiser's raw output (ε or v)

        Returns
        -------
        x_{t-1} : same shape as x_t
        """
        mean, log_var = self.p_mean_variance(x_t, t, model_out)
        noise = torch.randn_like(x_t)
        # No noise at t=0 (final step)
        nonzero = (t > 0).float().view(-1, 1, 1, 1, 1).to(x_t.device)
        return mean + nonzero * (0.5 * log_var).exp() * noise

    def ddim_step(
        self,
        x_t: torch.Tensor,
        t: torch.Tensor,
        t_prev: torch.Tensor,
        model_out: torch.Tensor,
    ) -> torch.Tensor:
        """Deterministic DDIM reverse step (η=0, Song et al. 2020).

        The step is written on (x̂₀, ε̂), which both objectives can produce, so
        there is one reverse process and not one per parameterisation.

        The state this returns is the state the NEXT network call reads, so it
        must sit at the noise level that call assumes.  A call at index k reads
        ``alphas_cumprod[k]``: that is the level ``q_sample`` builds for index k
        at training time, and the level ``predict_x0`` inverts.  The step
        therefore targets ``alphas_cumprod[t_prev]``, not
        ``alphas_cumprod_prev[t_prev]`` — the latter is ᾱ_{t_prev}, one index
        of the 1000-step ladder below ᾱ_{t_prev+1}, so it left every state at a
        slightly wrong noise level (up to 1.6e-3 in sqrt(ᾱ), and 3.9 % of
        sqrt(1−ᾱ) at the bottom rung of a 50-step ladder, where the remaining
        noise is small and the error is therefore largest in relative terms).

        ``t_prev = 0`` closes the chain — the sampler grid ends there and no
        call follows — so the target is the clean level ᾱ = 1 and the step
        returns x̂₀ itself.
        """
        x0_pred = self.predict_x0(x_t, t, model_out).clamp(-X0_CLAMP, X0_CLAMP)
        eps_pred = self.predict_eps(x_t, t, model_out)
        a_next = self._gather(self.alphas_cumprod, t_prev).to(x_t.device)
        terminal = (t_prev == 0).view(-1, 1, 1, 1, 1).to(x_t.device)
        a_next = torch.where(terminal, torch.ones_like(a_next), a_next)
        return a_next.sqrt() * x0_pred + (1.0 - a_next).clamp(min=0.0).sqrt() * eps_pred
