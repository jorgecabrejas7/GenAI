"""SliceGAN (Kench & Cooper, Nat. Mach. Intell. 3, 299-305, 2021).

arXiv:2102.07708 · reference code github.com/stke9/SliceGAN

One 3-D generator trained against THREE 2-D discriminators, one per axis, on
64x64 slices. The trick the paper is about: a 3-D structure can be learned from
2-D sections alone, because a generated volume is only ever judged by its
slices.

WHAT IS FAITHFUL HERE
  - 3-D generator of transposed convolutions, 4^3 latent -> 64^3 voxels
  - k=4, s=2, p=2 throughout, the paper's choice for avoiding checkerboard
    artefacts (its Section 2.3 / Table 1)
  - three 2-D critics on 64x64 slices, one per orthogonal axis
  - WGAN-GP, lambda 10, 5 critic steps per generator step, Adam(1e-4, 0.9, 0.99)
  - fully convolutional, so a larger latent gives a proportionally larger
    volume — the paper's route to volumes bigger than the training patch

WHAT IS OURS, AND WHY
  - **Four output channels**, not one: grey (tanh) plus a 3-class label
    (softmax). The paper generates a segmentation OR a greyscale micrograph;
    this project's eval needs a PAIRED grey+label volume, and every metric from
    porosity to seams reads both. The critics see the same four channels of
    real slices, so the pairing is learned rather than assembled afterwards.
  - Real slices come from split_v3 TRAIN patches only, all three axes.

WHAT IS MISSING, AND IS NOT BOLTED ON
  **SliceGAN is unconditional.** It cannot be asked for a porosity, a layup, a
  specimen envelope or a volume shape. Every conditional assessment in eval_v4
  is therefore inapplicable to it, and this baseline is scored only on the
  request-free metrics. Adding a conditioning path would make it a different
  method and a dishonest comparison.
"""

from poregen.baselines.slicegan.networks import Critic2D, Generator3D, latent_for_shape

__all__ = ["Generator3D", "Critic2D", "latent_for_shape"]
