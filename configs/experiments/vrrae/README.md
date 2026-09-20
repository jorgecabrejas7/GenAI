# The VRRAE expressiveness study

Four runs, lowest priority, a separate thread from the paper. Queue order,
decided by the supervisor:

    V0  vrrae/beta0     vrrae04 with the KL off          ~10.7 h
    B   vrrae/b         vrrae04's tensor, NO FC          ~6 h
    A   vrrae/a         6 convs, smaller tensor, NO FC   ~8.5 h
    A0  vrrae/a0        A with the KL off                ~8.5 h

**B before A on purpose.** B removes ONLY the FC and keeps vrrae04's 4096
tensor, so it isolates the FC. A removes the FC *and* shrinks the tensor
four-fold to 1024 with no spatial extent left, so on its own it cannot say
which of the two changes did anything. A is read against B.

## The geometry, computed not assumed

| config | model | blocks | last C | spatial | flattened | vrrae_dim | rank | batch | FC |
|---|---|---|---|---|---|---|---|---|---|
| vrrae04 | linear | 5 | 512 | 2 | 4096 | 2048 | 300 | 768 | Linear(4096→2048) |
| beta0 | linear | 5 | 512 | 2 | 4096 | 2048 | 300 | 768 | Linear(4096→2048) |
| b | SVD | 5 | 512 | 2 | 4096 | 4096 | 1280 | 1536 | **Identity** |
| a | SVD | 6 | 1024 | 1 | 1024 | 1024 | 768 | 1024 | **Identity** |
| a0 | SVD | 6 | 1024 | 1 | 1024 | 1024 | 768 | 1024 | **Identity** |

"No FC" is not a new code path: `vrrae_dim == flattened width` makes `fc_in`
and its decoder mirror `nn.Identity()`, which `vrrae.py` already does.

`rank <= batch` holds everywhere, and it has to: `RRLayer` silently CLIPS the
effective rank to the batch size, so an undersized batch trains a narrower
bottleneck than the config claims and nothing says so. `b_fallback.yaml` is
B at batch 1024 / rank 768 for the OOM case — a separate file, so which one
ran stays answerable from the run name.

## Report on a re-measured L1, not on the logged numbers

`scripts/analysis/vae_val_l1.py`. Two things make the logged validation
numbers uncomparable, and both are recorded in the vrrae04 run's `NOTES.md`:

1. vrrae04's logged `mae` (0.172) is not its reconstruction error — it sits on
   the constant-prediction baseline. Its logged `xct_loss` (0.078) is right.
   The two are the same function on this code path, so the 2.18 ratio between
   them is a logging artefact.
2. vrrae04 validated on **split_v2**, the r08 rungs on **split_v3**, and those
   sets are not equally hard — predicting the mean scores 0.153 on one and
   0.066 on the other.

**These four runs inherit `split_v2` from `vrrae/base`.** V0 must, to stay a
one-variable change to vrrae04. Whether A, A0 and B should move to `split_v3`
is an open question for the supervisor: moving them makes them comparable to
the r08 rungs and incomparable to vrrae03/04.

## The comparison, on one val set, with a baseline

Both on split_v3, 320 patches, same harness — validated by reproducing r08's
own logged 0.0307.

| model | L1 | predicting the mean | error removed |
|---|---|---|---|
| r08 rf-8 | 0.0347 | 0.0656 | **47.2 %** |
| vrrae04 | 0.0586 | 0.0656 | **10.7 %** |

The raw ratio of 1.7x understates it. vrrae04 removes about a ninth of the
error that predicting the dataset mean already achieves. Its decoder output
spans [0.589, 0.840] against data spanning [0, 1] — a narrow band around the
mean, which is what its **one active latent dimension of 300** buys.
