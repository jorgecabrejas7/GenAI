# Local patches

The vendored package is pinned to upstream commit
`de2e8a5035cf0d5897a59f7f743fe8c3655a7caa`. One intentional source
deviation is maintained in `RR_layer/rr_layer.py`.

## `stable_SVD` uses native PyTorch autograd

Upstream dispatches `stable_SVD(A)` through the custom
`StableSVD.apply(A)`. Locally it delegates to:

```python
torch.linalg.svd(A, full_matrices=False)
```

No other vendored behavior is changed.

The upstream backward is mathematically incorrect. Its off-diagonal coupling
uses a denominator proportional to `s_j - s_i`; the SVD derivative requires
one proportional to `s_j**2 - s_i**2`. Its rectangular correction is also
formed from the partially accumulated input gradient `dA` rather than from
the singular-vector gradient `dU`.

Finite-difference and native-autograd comparisons measured:

- Shape `(64, 32)`, rank 16: relative error factor `17.2x`, gradient-norm
  inflation `18.2x`.
- Production shape `(2048, 768)`, rank 300: gradient-norm inflation
  approximately `48x`, cosine similarity to the correct gradient `0.635`.

Gradient clipping at `max_grad_norm=1.0` can bound the inflated magnitude,
but it cannot restore the incorrect direction; the production-shape cosine
measurement demonstrates that distinction.

PyTorch's native SVD backward implements the correct derivative and is the
closest equivalent to the JAX `linalg.svd` autodifferentiation used by the
authoritative repository. The local regression test compares
`stable_SVD` with `torch.linalg.svd` on identical inputs and objectives.
Because the patched function directly delegates to the native operation, the
measured relative gradient error is `0.0`.


## Vendored provenance
Upstream: https://github.com/JadM133/RR_layer @ de2e8a5035cf0d5897a59f7f743fe8c3655a7caa (nested .git removed for vendoring, 2026-08-10).
