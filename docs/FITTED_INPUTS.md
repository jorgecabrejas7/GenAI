# Fitted inputs to generation and evaluation, and what each was fitted on

Anything **fitted from data and then read at generation time** is a channel by
which held-out material can reach the generator. This file lists every one,
what it was fitted on, and whether that is a leak. Audited 2026-09-17.

A fit used only to SCORE a result is not a leak — a floor should be measured on
held-out data. A fit used to BUILD a request is, because the request is then
derived from material the model was not supposed to see.

| input | read by | fitted on | verdict |
|---|---|---|---|
| **T-D correlation lengths** | `porosity_field.build_porosity_field` (generation) | was: all 80 volumes, `split_v2` index, 2 274 623 patches | **LEAK — fixed**, now `T-D_v3`: train only, `split_v3`, 58 vol / 1 598 000 patches |
| **T-E porosity marginal** | `porosity_field._draw_marginal` (generation) | was: same 80 volumes | **LEAK — fixed**, now `T-E_v3` on the same train split |
| **Rough-surface Sa + correlation length** | `eval_v4.cases.real_surface_target` → the rough surface REQUEST | `12-eval-v4/real_floor/surface_floor.json`, which is cut from the 11 **TEST** volumes | **LEAK — capability added, refit queued.** The same file is also the FLOOR the result is scored against, which is legitimate; it is building the REQUEST from it that is not |
| Latent normalisation (per-channel mean/std) | every encode and decode | store metadata records `computed_over: "train"` | **clean** |
| `POR_MIN` / `POR_MAX` = 0.002 / 0.107 | the conditioning clamp | quoted from an all-volume EDA | **clean — verified.** The volume-level porosity range is identical on all 80 and on train alone, `[0.00215, 0.10938]`, because both extreme volumes are train volumes. The number would not change if refitted |
| Downstream class weights | `downstream_utility.load_class_weights` | `class_weights.json` records `computed_from: split == train` | **clean** |
| Air-detector threshold `T_abs = 182` | `_eval_v3`, every air measurement | 8 calibration volumes, of which **2 are val** and 6 train; none are test | **minor, and it is an instrument.** It calibrates a MEASUREMENT, not a request, and touches no test material. Worth stating in the paper rather than fixing |
| Layup ground truth | the layup requests and their scoring | manually extracted by the domain expert | **not fitted from the scans** — no leakage channel |
| Default material map | `VolumeGenerator._default_material_map` | geometric — the specimen box, not fitted | **clean** |
| Orientation field (`split_v2/orientation_field.json`) | `build_conditioning` | derived from the T-I fit + expert ground truth | **to audit** — it is a conditioning input built from a fit, and the T-I fit's split has not been checked here |

## What the air detector is validated against, and what it is not

`scripts/analysis/air_audit_v2.py` computes its Dice, precision and recall
against the **stored `mask` and `sample_mask` arrays** in the zarr — the
dataset's own algorithmic segmentation. It is **not** validated against expert
annotation, and no expert voxel annotation exists anywhere in this repository.
The only expert-provided data is `data/layup_ground_truth.json`, which is ply
sequences, not masks.

Any text describing the air-detector reference as "expert masks" is wrong and
should read "the dataset's stored algorithmic segmentation". What the number
means is agreement with the existing pipeline, not agreement with a human.

## The correlation lengths are not all well-defined

Refitting T-D on train changed the three lengths by −26 %, +510 % and +24 %.
The middle one is not a real change: the in-plane **y** correlation does not
decay to 1/e at all on the train split — it plateaus near 0.45 and makes one
excursion to 0.359 against a 1/e of 0.3679, so the "length" is where a noisy
plateau happens to dip. `porosity_field.corr_length_is_well_defined` now tests
monotonicity up to the crossing and warns per axis; z and x pass in both fits,
y passes only in the old one.

**A number that moves by 6× on a curve that never decays should not be used as
a smoothing kernel without a decision.** See `runs/campaigns/26-field-validation`.
