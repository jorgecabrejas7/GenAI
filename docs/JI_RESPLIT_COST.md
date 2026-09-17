# If the six JI coupons are one physical panel: what it would cost

**Open question, escalated to the author.** `data/split_v3/splits.json` asserts
"Each Juan_Ignacio coupon is its own panel", giving 17 panel ids. The paper says
12, which is what 1 Pegaso + 10 Na + **1** JI would give. All six JI coupons
share sequence B and material `im7_m56`, and their numbering (4, 5, 7, 8, 10,
12) looks like a selection from one series — consistent with one panel and
equally with six. **The data cannot settle it; the author can.**

It matters because the split rule exists to stop one panel spanning two splits,
and today JI_8 is in **test**, JI_12 in **val**, and JI_4/5/7/10 in **train**.
If they are one panel, that is exactly the leak the rule forbids.

## The two repairs are not equally expensive

| | JI all-TRAIN | JI all-TEST |
|---|---|---|
| train | 58 → 60 vol, 1 598 000 → 1 643 885 patches (**+2.9 %**) | 58 → 54 vol, → 1 505 850 (**−5.8 %**) |
| val | 11 → 10 vol, −8.4 % | 11 → 10 vol, −8.4 % |
| test | 11 → 10 vol, −8.1 % | 11 → 16 vol, **+39.5 %** |
| φ ≥ 6 % still judgeable on val/test | yes (test 5800, val 1178) | yes (test 6427, val 1178) |
| **models need retraining** | **NO** | **YES — all of them** |

**That asymmetry is the whole decision.** Every current model — the r08 VAE,
ldm06, facedrop, SliceGAN, DDPM3D, and the φ-only baseline — was trained on
today's train split.

* **JI all-train**: the models were trained on a *subset* of the new train
  split. No model ever saw a test panel. Nothing is contaminated, so nothing
  needs retraining; the cost is re-measurement.
* **JI all-test**: every model was trained on four volumes that are now test.
  That is direct contamination and **every model must be retrained** — the VAE
  first, then the latent store rebuilt, then ldm06, facedrop and all three
  baselines, then every campaign regenerated on top. On measured times that is
  ~16 h (ldm06) + ~11 h (SliceGAN) + 24 h (DDPM) + the VAE rungs + the whole
  evaluation queue again: **weeks, not hours.**

## What JI all-train would invalidate (the cheap case, and it is not cheap)

**The real floor, and therefore every ratio read against it.** JI_8 supplies
**7 of the 39 floor crops (18 %)** and is one of only **three panels** in the
matched-porosity micro reference — 6 of its 18 crops, **a third of every
porosity level's floor**. No level loses its floor entirely (all three panels
appear at 0.01, 0.03 and 0.06), but every level loses a third of it. The floor
must be recut and every generated-vs-real ratio in campaigns **12, 18 and 22**
re-read against the new one.

**Downstream test scores.** JI_8 is **8.1 % of the test patches** the segmenter
arms are scored on, so every arm's number moves. Campaign 14's headline — that
synthetic data *hurts* — rests on differences of about 0.02 pore Dice, and an
8 % change of test set is not obviously small against that. **Campaign 14 must
be re-scored, not merely re-read.**

**The val floor inside the memorisation statistic**, which loses JI_12.

**Refits, all cheap and all mechanical**: `T-D_v3` / `T-E_v3` (train changed),
`class_weights.json`, and the rough-surface request once it is refitted from
train. The latent store's normalisation was computed on the old train split; in
this case it stays *valid* (a constant computed on a subset of the new train,
with no test material in it) and should be **documented rather than changed** —
changing it would invalidate every trained model for no gain.

**Not affected**: campaign 21 (all 80 volumes, split-independent), campaign 13
(its three volumes are Na_05, no JI), and the air-detector calibration (8
volumes, none of them JI).

## Recommendation

If the author confirms one panel, **put JI in train**. It is the only repair
that does not contaminate every existing model, it keeps the φ ≥ 6 % bin
judgeable on both held-out splits, and its cost is a floor recut plus a
re-scored campaign 14 rather than a full retrain. The price is a test split of
10 volumes drawn from two panels instead of three, which narrows the panel
diversity the floor rests on and should be stated as a limitation.

Until the author answers, **hold the panel-count wording in the paper**.
