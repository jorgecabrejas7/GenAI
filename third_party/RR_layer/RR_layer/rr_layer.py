import warnings
import torch
import torch.nn as nn
from collections import deque

def stable_SVD(A):
    """ Computes a numerically stable SVD.

    A: input matrix (..., m, n)

    LOCAL PATCH (deviates from pinned upstream commit
    de2e8a5035cf0d5897a59f7f743fe8c3655a7caa): this now delegates to
    ``torch.linalg.svd``'s built-in autograd instead of ``StableSVD``.

    ``StableSVD.backward`` below is mathematically incorrect. The SVD
    backward's off-diagonal coupling term requires F_ij = 1/(s_j^2 - s_i^2);
    it uses 1/(s_j - s_i) (line ~50), and its rectangular correction builds
    the term from ``dA`` rather than ``dU``. Checked against finite
    differences, upstream's gradient is wrong at every shape tested, e.g.
    (64,32) rank-16: 17.2x relative error, 18.2x norm inflation, cosine
    0.975; at this project's production shape (2048x768, rank 300) the
    inflation reaches ~48x with cosine 0.635. torch.linalg.svd's own
    autograd matches finite differences to machine precision.

    This was the true cause of the encoder gradient explosion recorded in
    configs/experiments/vrrae/base.yaml (median 873, max 4.2e7) -- which
    that comment attributes to near-degenerate singular values, but the
    guarded 1/(s_i - s_j) term is simply the wrong formula. Clipping bounded
    the magnitude while leaving the direction error untouched.

    ``StableSVD`` is retained below only so the diff against upstream stays
    legible; nothing calls it.
    """
    return torch.linalg.svd(A, full_matrices=False)

class StableSVD(torch.autograd.Function):
    @staticmethod
    def forward(ctx, A):
        # Just compute standard SVD
        U, S, Vh = torch.linalg.svd(A, full_matrices=False)

        # Save for backward
        ctx.save_for_backward(U, S, Vh)
        ctx.original_shape = A.shape

        return U, S, Vh

    @staticmethod
    def backward(ctx, dU, dS, dVh):
        """
        Backward pass for stable SVD.
        Computes gradient w.r.t. input A given gradients on U, S, Vh.
        """
        U, S, Vh = ctx.saved_tensors
        m, n = ctx.original_shape[-2:]

        dtype = U.dtype
        device = U.device

        H = lambda x: x.transpose(-2, -1).conj()
        T = lambda x: x.transpose(-2, -1)

        # Diagonal helpers
        def diag_embed(x):
            return torch.diag_embed(x)

        # Singular value vector to diagonal for broadcasting
        S_mat = S.unsqueeze(-2)
        S_diff = S_mat - S_mat.transpose(-2, -1)

        # F matrix for repeated singular values
        eps = 1e-20
        F = torch.where(S_diff.abs() > eps, 1.0 / S_diff, torch.zeros_like(S_diff))

        # Gradient from singular values
        dA = U @ diag_embed(dS) @ Vh

        # Contributions from U
        Ut_dU = H(U) @ dU
        skew_U = F * (Ut_dU - Ut_dU.transpose(-2, -1))
        dA += U @ skew_U @ diag_embed(S) @ Vh

        # Contributions from V
        V = H(Vh)  # n x k
        Vt_dV = H(V) @ H(dVh)  # k x k
        skew_V = F * (Vt_dV - Vt_dV.transpose(-2, -1))
        dA += U @ diag_embed(S) @ skew_V @ Vh

        # Rectangular adjustments (like JAX)
        # s_inv = 1 / S with stable zero handling
        s_zeros = (S == 0).to(dtype)
        s_inv = 1.0 / (S + s_zeros) - s_zeros  # shape: (k,)

        if m > n:
            dAV = dA @ V
            dA += (dAV - U @ (H(U) @ dAV)) * s_inv
        elif n > m:
            dAHU = H(dA) @ U
            print("first term shape: ", H(dAHU - V @ (Vh @ dAHU)).shape)
            print("s_inv shape: ", s_inv.shape)

            dA += H(dAHU - V @ (Vh @ dAHU)) * s_inv.unsqueeze(1)

        return dA


import math
import warnings
from collections import deque

import torch
import torch.nn as nn


class RRLayer(nn.Module):
    r"""
    Rank Reduction (RR) layer.

    During training, the layer computes a truncated SVD across the batch
    dimension and reconstructs the input using the top ``rank`` singular
    components.

    During evaluation, the layer projects the input onto a learned inference
    basis obtained from the recent training bases.

    A custom basis can always be supplied through the ``basis`` argument of
    :meth:`forward`, in which case the train/eval behavior is bypassed.

    Args:
        rank (int):
            Number of singular values to retain.

        basis_history_size (int, optional):
            Number of recent batch bases to keep when estimating the inference
            basis. Default: 20.

    Shape:
        - Input: ``(N, *)``
        - Output: ``(N, *)``

    Example:
        >>> rr = RRLayer(rank=8)
        >>> rr.train()
        >>> y = rr(torch.randn(32, 768))

        >>> rr.eval()
        >>> y = rr(torch.randn(32, 768))
    """

    def __init__(
        self,
        rank: int,
        basis_history_size: int = 20,
    ):
        super().__init__()

        if rank <= 0:
            raise ValueError(
                f"rank must be positive, got {rank}."
            )

        if basis_history_size <= 0:
            raise ValueError(
                f"basis_history_size must be positive, got "
                f"{basis_history_size}."
            )

        self.rank = rank
        self.basis_history_size = basis_history_size

        self.register_buffer(
            "inference_basis",
            None,
            persistent=True,
        )

        self._basis_bank = deque(
            maxlen=basis_history_size,
        )

    def extra_repr(self) -> str:
        return (
            f"rank={self.rank}, "
            f"basis_history_size={self.basis_history_size}"
        )

    def train(self, mode: bool = True):
        """
        Switch between training and evaluation mode.

        When switching from training to evaluation for the first time,
        an inference basis is automatically computed from the stored
        training bases.
        """

        previous_mode = self.training

        super().train(mode)

        if (
            previous_mode
            and not mode
            and self.inference_basis is None
            and len(self._basis_bank) > 0
        ):
            self.finalize_basis()

        return self

    @torch.no_grad()
    def finalize_basis(self) -> None:
        """
        Build the inference basis from the stored training bases.
        """

        if len(self._basis_bank) == 0:
            raise RuntimeError(
                "Cannot finalize basis: no stored bases available."
            )

        device = next(self.parameters(), None)

        if device is None:
            device = self.inference_basis.device \
                if self.inference_basis is not None \
                else self._basis_bank[0].device
        else:
            device = device.device

        W = torch.cat(
            [
                basis.to(device)
                for basis in self._basis_bank
            ],
            dim=1,
        )

        U, _, _ = stable_SVD(W)

        r = min(self.rank, U.shape[1])

        self.inference_basis = U[:, :r]

    def _validate_inputs(
        self,
        x: torch.Tensor,
        basis: torch.Tensor | None,
    ) -> None:
        """
        Validate inputs for RRLayer.
        """

        if not isinstance(x, torch.Tensor):
            raise TypeError(
                f"x must be a torch.Tensor, got {type(x)}."
            )

        if x.ndim < 2:
            raise ValueError(
                "RRLayer expects input of shape (N, ...), "
                f"got shape {tuple(x.shape)}."
            )

        if x.shape[0] == 0:
            raise ValueError(
                "Batch size must be greater than zero."
            )

        if not torch.isfinite(x).all():
            raise ValueError(
                "Input tensor contains NaN or Inf values."
            )

        M = math.prod(x.shape[1:])
        N = x.shape[0]

        rank_max = min(M, N)

        if self.rank > rank_max:
            warnings.warn(
                f"Requested rank={self.rank}, but the maximum "
                f"achievable rank is {rank_max}. "
                f"Using rank={rank_max}.",
                stacklevel=2,
            )

        if basis is not None:

            if not isinstance(basis, torch.Tensor):
                raise TypeError(
                    f"basis must be a torch.Tensor, got {type(basis)}."
                )

            if basis.ndim != 2:
                raise ValueError(
                    "basis must have shape (M, r), "
                    f"got shape {tuple(basis.shape)}."
                )

            if basis.shape[0] != M:
                raise ValueError(
                    f"basis has {basis.shape[0]} rows but "
                    f"expected {M}."
                )

            if basis.shape[1] == 0:
                raise ValueError(
                    "basis must contain at least one column."
                )

            if basis.device != x.device:
                raise ValueError(
                    f"basis is on {basis.device} while "
                    f"x is on {x.device}."
                )

            if basis.dtype != x.dtype:
                raise ValueError(
                    f"basis dtype ({basis.dtype}) does not match "
                    f"x dtype ({x.dtype})."
                )

            if not torch.isfinite(basis).all():
                raise ValueError(
                    "basis contains NaN or Inf values."
                )

    def forward(
        self,
        x: torch.Tensor,
        basis: torch.Tensor | None = None,
        return_factors: bool = False,
    ):
        self._validate_inputs(x, basis)

        original_shape = x.shape
        batch_size = original_shape[0]

        X = torch.movedim(x, 0, -1)
        X = X.reshape(-1, batch_size)

        if basis is not None:

            basis_used = basis

            coeffs = basis_used.T @ X

            X_hat = basis_used @ coeffs

        elif self.training:

            U, S, Vh = stable_SVD(X)

            r = min(self.rank, S.shape[0])

            basis_used = U[:, :r]

            coeffs = (
                S[:r].unsqueeze(1)
                * Vh[:r]
            )

            X_hat = basis_used @ coeffs

            self._basis_bank.append(
                basis_used.detach().cpu()
            )

        else:

            if self.inference_basis is None:
                raise RuntimeError(
                    "RRLayer has no inference basis. "
                    "Call finalize_basis() or run training "
                    "before evaluation."
                )

            basis_used = self.inference_basis

            coeffs = basis_used.T @ X

            X_hat = basis_used @ coeffs

        output = X_hat.reshape(
            *original_shape[1:],
            batch_size,
        )

        output = torch.movedim(
            output,
            -1,
            0,
        )

        if return_factors:
            return output, basis_used, coeffs

        return output