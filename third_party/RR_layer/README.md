# RR_layer

`RR_layer` is a PyTroch library implementing a **Rank-Reduction (RR) layer**: a layer that compresses its input by projecting onto a low-rank SVD basis.

The mechanics are different between training and evaluation:

- **During training**, the SVD basis is **recomputed from scratch for every batch**. For each batch, the layer computes the SVD of the batch, truncates to the top `r` components, and uses that truncated basis to project — so the basis is always data-dependent and up to date with the current batch, and gradients flow through the SVD/projection step. The model also automatically saves the last `N=basis_history_size` bases to be later used in evaluation.
- **During evaluation**, when layer.eval() is called, the layer will find a common basis to the whole dataset `Uf`, based on the last `N=basis_history_size` that were saved. When the layer is called in evaluation mode, the SVD is replaced by a projection on that basis `Uf`.
- In both cases, the shape of the input should be (N x ...), where N is the number samples in a batch. Note that the output will have the same shape is the input. This is because the layer performs the SVD, truncates, but then returns the reconstructed latent space. So the compression happens inside eventhough the output still has the same shape.

## What's inside

```
RR_layer/
├── RR_layer/
│   ├── __init__.py
│   └── rr_layer.py        # the layer implementation(s)
├── examples/
|   ├──── autoencoder
│   |     ├── generate_data.ipynb
│   |     ├── train_baseline.ipynb
│   |     ├── train_RRAE.ipynb
│   |     └── compare.ipynb
└── tests/
    └── tests.py
```

### `RR_layer/rr_layer.py`

The library code. Contains the `RR_layer` Equinox module: a layer with two modes of operation —

- **train mode**: computes the SVD of the current batch's activations, truncates to rank `r`, and projects onto that basis (gradients flow through this).
- **eval mode**: projects onto a basis that was saved from training, rather than computing a new SVD.

### `Examples/`

The example notebooks are meant to be worked through **in order**:

1. **`generate_data.ipynb`** — generates the dataset used in the other notebooks. Run this first; it should leave behind whatever data artifacts the later notebooks load.

2. **`train_baseline.ipynb`** — trains a standard, non-RR autoencoder as a reference point. This gives you a baseline reconstruction quality and runtime to compare the RRAE against.

3. **`train_RRAE.ipynb`** — trains the Rank-Reduction Auto-Encoder, built with one or more `RR_layer` in the latent space. During training this recomputes the SVD basis per batch as described above; at the end of training it should save the basis to be reused at evaluation time.

4. **`compare.ipynb`** — loads the trained baseline and RRAE models (using the saved, frozen SVD basis for the RRAE) and compares them for tasks such as linear interpolation and random generation.

Run the notebooks in the order above — each later notebook generally depends on artifacts (data, images) produced by the earlier ones.

### `tests/tests.py`

Unit tests for the `RR_layer` module. Useful both to verify correctness after installing, and as additional usage examples of the layer API beyond the notebooks (e.g. shape checks, gradient flow through the SVD step in train mode, etc.)

## Running the tests

```bash
python -m pytest tests/tests.py
```

## Installation

```bash
pip install RR_layer
```

## Quick start

```python
import torch
from RR_layer import RRLayer

inp = torch.randn(32, 768)
rr = RRLayer(rank=8)

rr.train()
y = rr(inp) # will compute the SVD

rr.eval()
y = rr(inp) # will project onto a saved basis
```

## Background

For more information about Rank Reduction layers, please refer to:

- **Paper:** [Rank Reduction Autoencoders](https://arxiv.org/abs/2405.13980)
- **Related repo:** [JadM133/RRAEs](https://github.com/JadM133/RRAEs)
