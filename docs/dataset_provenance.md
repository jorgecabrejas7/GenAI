# Dataset provenance

How each dataset root was produced, and how to rebuild it. This file is the
canonical record: `/data/` is git-ignored by directory rule, so a `BUILD.md`
sitting in a data root would vanish with the disk. Those files are short
pointers back here.

Roots, newest first:

| Root | State | Used by |
|---|---|---|
| `data/split_v3` | **current** | r08 VAE, ldm06 |
| `data/split_v2` | superseded for training; still holds the orientation field and the ldm05 latent store | campaigns 02-08, ldm05 |
| `data/split_v1` | owns the only real `volumes.zarr` (207 GB) | everything, through symlinks |

## Deletion checklist

Before removing anything from a derived data root. Written for the planned
removal of `data/split_v2/patches_xct.bin` and `patches_mask.bin` (1.19 TB),
but the hazards apply to any cleanup here.

**1. Never delete through the symlink.** `volumes.zarr` is a symlink in both
v2 and v3, chained `split_v3 -> split_v2 -> split_v1`, and only
`data/split_v1/volumes.zarr` (207 GB) is real. A recursive delete of a derived
root that follows symlinks destroys every root at once. Name the files to
remove explicitly; never `rm -rf` a root. Check first:

```bash
ls -l data/split_v2/volumes.zarr data/split_v3/volumes.zarr   # both -> a link
readlink -f data/split_v3/volumes.zarr                        # split_v1's store
```

and confirm afterwards that both still resolve.

**2. The memmap layout is one-way.** Rebuilding `split_v2`'s memmaps with the
extractor as it stands produces the 3-class `patches_label.bin`, not the
original `patches_xct.bin` + `patches_mask.bin`. The binary-mask layout cannot
be recreated from the current tree. This is acceptable only because `split_v3`
supersedes that root for training.

**3. Keep everything in the survival table** of the root's section below —
`patch_index.parquet`, `splits.json`, `volume_stats.json`,
`patches_meta.json` (the only surviving record of N, the strides and the
parquet checksum of the deleted memmaps), `orientation_field.json`, and
`latents_r07z4/`, which campaigns 02-08 and ldm05 all resolve through.

**4. Loading still works afterwards.** `build_patch_dataloaders` checks for the
files the memmap backend actually reads, so a root whose `.bin` files are gone
falls back to the Zarr backend rather than failing on a missing file. Confirm
with a one-batch load before considering the cleanup done.

**5. Gates.** The `split_v2` memmap removal was gated on the r08 VAE passing
its acceptance gates AND the r08 latent store being built and verified. **That
second condition turned out to be circular** and was dropped on 2026-09-07: the
z=8 store needs 291 GB, the build guard wants twice that, and only 340 GB was
free — so the store could not be built until the memmaps went, and the memmaps
were not to go until the store was built. What actually gated the deletion was
the first condition alone: the sweep finished, `reduction-factor-8` was chosen
on the four-rung table, and the store build was already queued against that
checkpoint. If you write a gate like this again, check it can be satisfied in
the order it names.
`latents_r07z4/` is NOT part of that removal; it goes only after ldm06 is
evaluated.

---

## `data/split_v3`

Built 2026-09-02. This is the root the r08 VAE and ldm06 use. Everything below
is reproducible from committed code — `scripts/build_split_v3.py` and
`scripts/extract_patches_memmap.py`.

**`volumes.zarr` here is a symlink**, not data:

```
data/split_v3/volumes.zarr -> data/split_v2/volumes.zarr -> data/split_v1/volumes.zarr
```

Nothing is copied and `data/split_v2` is never written. See
`data/split_v2/BUILD.md` for how the store itself was produced.

---

## Rebuild, in one command

```bash
python scripts/build_split_v3.py --stage all
python scripts/extract_patches_memmap.py --data-root data/split_v3 --verify
```

Then, once an r08 VAE exists, the LDM store on top of it:

```bash
python scripts/build_latent_dataset.py \
    --checkpoint runs/vae/<r08 run>/best.ckpt \
    --output data/split_v3/latents_r08z8
python scripts/build_conditioning.py --store data/split_v3/latents_r08z8
```

`build_latent_dataset.py` writes `latents.bin`, `index.parquet`,
`material.bin` (uint8 specimen-ENVELOPE fraction per latent cell, `label != 2`) and
`air.bin` (float32, voxel label 2) per split in ONE pass over the label memmap,
so the four files cannot fall out of alignment. `build_conditioning.py` then
adds `cond.parquet` (`cond_depth`, six `cond_dist6_*`, `cond_por_raw`) and the
`conditioning` metadata block, reusing `data/split_v2/orientation_field.json`
after checking every volume in the store has a record and a foreground extent.

`--stage all` runs `holes → splits → index → weights → report`; each stage is
idempotent and can be run alone with `--stage <name>`. No stage takes a
parameter: every constant lives in the script or in
`src/poregen/dataset/holes.py`, so the rebuild cannot drift from the record.

Wall time on the DGX Spark: holes 537 s, patch index 4747 s, weights and report
seconds, memmap extraction ~1 h.

## What each stage does

### 1. `holes` → `holes/<volume_id>.npy`, `holes.json`

Every coupon carries three ~200-voxel drilled registration through-holes. They
are `False` in `sample_mask` and `0` in `mask`, so they read as porosity exactly
0 while being ~95 % air, and they are the only interior source of large air in
the dataset.

`poregen.dataset.holes.detect_holes`, defaults `z_fraction 0.6`,
`min_area_px 400`, `dilate_vox 32`:

1. take the middle 60 % of the slices;
2. bound the specimen in-plane from "inside the mask on at least one slice";
3. inside that box, project `~sample_mask` along z with a **minimum** — a pixel
   is marked only when *every* slice is outside the material there, which is
   the definition of a through-hole;
4. drop components that touch the box border (the exterior) or are ≤ 400 px;
5. dilate the rest by 32 voxels (exact Euclidean).

> The minimum projection is load-bearing. A **maximum** projection marks a pixel
> outside on any single slice, which on the Airbus_Panel_Pegaso coupons connects
> the holes to the exterior — they are then discarded as border-touching, and
> ~23 one-slice internal voids are kept instead. Measured on three volumes:
> maximum gives 26 / 4 / 3 components, minimum gives 3 / 3 / 3 at 201–204 px.

Result: 80/80 volumes at exactly 3 holes, 192–243 px equivalent diameter,
dilated footprint 2.9–3.4 % of a slice.

### 2. `splits` → `splits.json`

By **panel**, never by coupon — `split_v2` scattered the five coupons of a panel
across train/val/test, so val and test shared a panel with train.

- test = every coupon of panels `Na_05` and `Na_09` + `Juan_Ignacio_probetas_8`
- val = every coupon of panels `Na_08` and `Na_01` + `Juan_Ignacio_probetas_12`
- train = everything else, 11 panels
- excluded: `MedidasDB__Juan_Ignacio_probetas_11_volume_eq_aligned`

The 24 Airbus_Panel_Pegaso coupons are one single panel, so they can only ever
be train (`pegaso_single_panel_train_only: true`); holding them out would remove
a whole material family. Each Juan_Ignacio coupon is its own panel.

Counts: train 58 / val 11 / test 11 volumes.

**Why Na_09 and Na_01 are in val/test (re-split of 2026-09-03).** The first
split — `Na_05 + JI_8` / `Na_08 + JI_12` — left only 263 and 42 patches at
φ ≥ 6 %, so that porosity bin could not be judged on either split. It was not a
sampling accident: both panels are low-porosity.

They were NOT chosen as the panels with the most high-porosity material, and
that matters, because the high-porosity regime is extremely concentrated:

| panel | volumes | patches | mean φ | n ≥6 % | n 3–6 % | n 1–3 % |
|---|---|---|---|---|---|---|
| Na_02 | 5 | 138736 | 0.09948 | 130376 | 8136 | 223 |
| Na_10 | 5 | 181950 | 0.03320 | 25110 | 71969 | 43738 |
| Na_09 | 5 | 138114 | 0.01775 | 5792 | 26163 | 28996 |
| Pegaso_1 | 24 | 677023 | 0.01224 | 5428 | 77648 | 186033 |
| Na_01 | 5 | 125525 | 0.02340 | 1178 | 34309 | 72429 |
| Na_07 | 5 | 132077 | 0.01522 | 306 | 16617 | 58657 |
| JI_8 | 1 | 23435 | 0.01105 | 255 | 2073 | 6430 |
| Na_06 | 5 | 126722 | 0.00695 | 142 | 3948 | 16965 |
| JI_5 | 1 | 22915 | 0.00975 | 138 | 1522 | 6273 |
| JI_7 | 1 | 23895 | 0.00621 | 100 | 926 | 3694 |
| JI_4 | 1 | 22930 | 0.00771 | 53 | 1141 | 5036 |
| Na_03 | 5 | 118791 | 0.00577 | 52 | 2817 | 10341 |
| JI_12 | 1 | 22450 | 0.01364 | 42 | 1637 | 11088 |
| JI_10 | 1 | 22410 | 0.01473 | 39 | 1676 | 12806 |
| Na_05 | 5 | 128897 | 0.00587 | 8 | 1536 | 17390 |
| Na_04 | 5 | 130551 | 0.00433 | 1 | 450 | 8493 |
| Na_08 | 5 | 118144 | 0.00487 | 0 | 783 | 9283 |

**`Na_02` alone holds 77 % of the train patches at φ ≥ 6 %, and `Na_02` +
`Na_10` hold 92 %.** Moving those two out — which is what "largest ≥6 % count
to test, second-largest to val" would have done — would have left train with
8 % of its high-porosity data and its mean φ down from 0.0207 to 0.0121, then
tested a regime the model had barely seen. `Na_09` + `Na_01` instead give 6055
and 1220 patches in that bin while train keeps 96 % of it and its mean φ is
unchanged.

Resulting patches per bin:

| split | volumes | patches | mean φ | φ<1 % | 1–3 % | 3–6 % | ≥6 % |
|---|---|---|---|---|---|---|---|
| train | 58 | 1598000 | 0.02074 | 897146 | 352259 | 186850 | 161745 |
| val | 11 | 266119 | 0.01435 | 135370 | 92800 | 36729 | 1220 |
| test | 11 | 290446 | 0.01194 | 201803 | 52816 | 29772 | 6055 |

> **Training support for φ ≥ 6 % rests on two panels.** `Na_02` (77 %) and
> `Na_10` (15 %). A reader judging any high-porosity result needs to know the
> training support for it is not spread across the dataset, and that losing
> either panel from train would change what the model can represent there.

The re-split was applied in place with
`python scripts/build_split_v3.py --stage resplit`, which rewrites only the
`split` COLUMN of the parquet. Row order is asserted unchanged, so
`patches_xct.bin` and `patches_label.bin` stay row-aligned and were not
re-extracted. `--stage all` from scratch produces the same assignment.

### 3. `index` → `patch_index.parquet`, `index_report.json`

64³ patches at stride 32, the `split_v2` columns plus `panel_id` and
`air_fraction` (fraction of `sample_mask == 0` in the patch). Porosity and air
fraction both come from `poregen.dataset.patch_index.patch_fractions`, one
integral volume each, so they are exact voxel counts.

Any patch whose `(y, x)` footprint intersects a dilated hole is dropped.

Result: 2 274 623 → **2 154 565** patches, 120 058 dropped (5.28 %).
Fully-interior air patches (`air_fraction > 0.5`, ≥ 64 voxels from every face of
the specimen bounding box) fall from 7 833 (0.344 %) to 50 (0.002 %).

### 4. `weights` → `class_weights.json`

Train-split voxel frequencies, read straight off the parquet (every patch is
64³, so a mean over patches *is* the voxel frequency):
material 0.907569 / pore 0.020702 / air 0.071730 over 1 861 639 patches.

`w_c = 1 / (n_classes · f_c)`, so `Σ_c f_c w_c = 1` and the weighted
cross-entropy keeps the magnitude of an unweighted one:
**[0.367282, 16.101824, 4.647067]**. Pasted into
`configs/experiments/r08/base.yaml`; `compute_total_loss` refuses to run a
3-class head without them.

### 5. `report` → `build_report.md`

Per-split counts before/after hole removal, mean porosity and air fraction, the
interior-air check, and the hole count and diameters per volume.

## 6. Memmaps

```bash
python scripts/extract_patches_memmap.py --data-root data/split_v3 --verify
```

`patches_xct.bin` (uint8 grey) and `patches_label.bin` (uint8 3-class), each
`N × 64³` = 564.8 GB, row-aligned with the parquet. The label is
**2 = air** (`sample_mask == 0`, exterior or a drilled hole), **1 = pore**
(`mask != 0`), **0 = material**; air takes precedence over pore.
`--verify` re-reads 128 random patches from the zarr and compares.

`patches_meta.json` records N, patch size, stride, splits, the class rule, the
measured per-class voxel fractions, the hole and split rules, and the parquet
SHA-256.

## Loading

`build_patch_dataloaders` picks `MemmapPatchDataset` when `patches_meta.json`,
`patches_xct.bin` and `patches_label.bin` are all present, and falls back to the
Zarr `PatchDataset` otherwise — which builds the identical 3-class label from
the store's `mask` and `sample_mask`. Both return `label` (int64, no channel
dim, ready for cross-entropy) and the derived binary pore `mask` (`label == 1`).

## Not built yet

Latents and conditioning for ldm06 come after the r08 VAE is trained and its
rung chosen. Nothing in this root depends on them.


---

## `data/split_v2`

> **DONE 2026-09-07 — the memmaps are gone.** `patches_xct.bin` and
> `patches_mask.bin` (596 GB each, 1.19 TB together) were unlinked on the
> user's instruction once `r08/reduction-factor-8` was chosen as the production
> rung. Freed the space the z=8 latent store needed: 340 GB free before,
> 1.5 TB after. Everything in the survival table below is intact, and
> `volumes.zarr` is still a symlink resolving to `data/split_v1/volumes.zarr`.
>
> **Consequence: no r03–r07 config can run on the memmap backend any more.**
> `build_patch_dataloaders` checks for the files the memmap backend actually
> reads and falls back to Zarr when they are absent, so those configs still
> load — slower, off `volumes.zarr`, not broken. Anything that needs the old
> binary-mask memmap layout has to rebuild it, and the extractor as it stands
> emits the 3-class `patches_label.bin` instead, so that layout cannot be
> recreated from this tree at all.

Written 2026-09-02, before the memmaps were deleted, so the root can be rebuilt
from raw data without reading the code archaeology again.

**`volumes.zarr` here is a symlink**, not data:

```
data/split_v3/volumes.zarr -> data/split_v2/volumes.zarr -> data/split_v1/volumes.zarr
```

The real store is `data/split_v1/volumes.zarr` (207 GB). Deleting anything in
`split_v2` must leave that symlink alone, or `split_v3` loses its data too.

---

## The chain

### 1. Raw TIFF → `data/split_v1`

```bash
build_dataset --raw_root <raw TIFF root> --out_root data/processed \
              --patch_size 64 --stride 32 --chunk_size 64,64,64
```

(`build_dataset` is the console entry point for
`poregen.dataset.build_dataset:main`.) This wrote `volumes.zarr`,
`patch_index.parquet`, `volume_stats.json` and `splits.json`.

What the artefacts record: `data/split_v1/splits.json` has `seed: 123` — the
CLI default — and counts train 65 / val 8 / test 8, so 81 volumes. Chunks are
64³, aligned with the patch size (commit `9f3b06c`).

**Not verifiable from the artefacts:** the exact `--raw_root`, and whether
`--n_train` / `--n_val` / `--n_test` were passed or left at `None`. The counts
65/8/8 are consistent with the deterministic `assign_volume_splits` path.

### 2. `data/split_v1` → `data/split_v2`

Not a rebuild — a re-assignment. `split_v2` re-splits the *same* patch index
stratified by volume porosity, and shares the zarr by symlink:

```bash
python -c "from poregen.dataset.splits import materialize_split_roots; \
           materialize_split_roots('data', seed=42)"
```

`materialize_split_roots` renames `data/processed` → `data/split_v1` if it is
still under the old name, then calls `materialize_split_v2`, which:

- computes each volume's median patch porosity from `split_v1`'s parquet;
- bins the volumes on `SPLIT_V2_BIN_EDGES` `(0, 0.005, 0.01, 0.02, 0.06, 0.12)`
  with labels `<0.5% / 0.5-1% / 1-2% / 2-6% / 6-12%`;
- draws train/val/test per bin to `SPLIT_V2_TARGET_COUNTS` with `seed=42`
  (`SPLIT_V2_SEED`);
- excludes `SPLIT_V2_EXCLUDED_VOLUME_IDS` =
  `MedidasDB__Juan_Ignacio_probetas_11_volume_eq_aligned`;
- **fails** if any bin has no val or test volume.

Every constant lives in `src/poregen/dataset/splits.py`; none is passed on the
command line. Result: `data/split_v2/splits.json`, `seed: 42`, train 64 / val 8
/ test 8 = 80 volumes, and a `patch_index.parquet` of 2 274 623 rows.

### 3. Patch index → memmaps

```bash
python scripts/extract_patches_memmap.py --data-root data/split_v2
```

`patches_meta.json` records what came out: `N = 2274623`, `patch_size 64`,
`stride 32`, dtype uint8 for both arrays, splits `{train: 1822599, val: 229195,
test: 222829}`, `voxel_size_um 25.0`, and `parquet_sha256
a3f3f2f7cc65f90f772c4f8a62d66d07d429fdb5395c98b95c762f3724eda1d1`.

> The extractor in the tree today writes `patches_xct.bin` +
> `patches_label.bin` (3-class). The `split_v2` memmaps are the older
> `patches_xct.bin` + `patches_mask.bin` (binary pore mask). Rebuilding
> `split_v2`'s memmaps with the current script would produce the 3-class
> layout, not the original one. There is no reason to: `split_v3` supersedes
> this root for training.

### 4. Added later, not part of the build

- `sample_mask` arrays inside `volumes.zarr` — written by
  `poregen.dataset.io.save_volume_zarr` during the split_v1 build, compressed
  beside `xct`/`mask`. Offline builders read them back rather than recomputing
  the specimen envelope. The ldm06 material map no longer comes from here: it
  is pooled from the 3-class voxel label by `scripts/build_latent_dataset.py`.
- `orientation_field.json` — `scripts/build_conditioning.py --rebuild-orientation`.
  Per-VOLUME (θ(z) from the nominal layup plus each volume's foreground extent),
  so `split_v3` reuses it unchanged.
- `latents_r07z4/` — `scripts/build_latent_dataset.py`, encoded with
  `runs/vae/r07-run-0006-…-z4-c32-…/best.ckpt` (experiment
  `r07/reduction-factor-16`). 🕰 the ldm05 store; the current loader cannot read
  it (one `cond_dist`, no material sidecar). Superseded by
  `data/split_v3/latents_r08z8`.

---

## Rebuilding from scratch, in one go

```bash
build_dataset --raw_root <raw TIFF root> --out_root data/processed \
              --patch_size 64 --stride 32 --chunk_size 64,64,64
python -c "from poregen.dataset.splits import materialize_split_roots; \
           materialize_split_roots('data', seed=42)"
python scripts/extract_patches_memmap.py --data-root data/split_v2
```

Step 1 is the expensive one (raw TIFF decode + segmentation over 81 volumes).
Steps 2 and 3 are cheap given the zarr.

## What must survive a cleanup

| Path | Why |
|---|---|
| `volumes.zarr` (symlink) | `split_v3` resolves its data through it |
| `patch_index.parquet` | the only record of the v2 patch grid and split column |
| `splits.json` | the v2 volume assignment (seed 42) |
| `volume_stats.json` | per-volume intensity statistics |
| `patches_meta.json` | N, strides and the parquet checksum of the deleted memmaps |
| `orientation_field.json` | read at training AND sampling time by `split_v3` too; rebuilding it needs the T-I artefacts |
| `latents_r07z4/` | campaigns 02–08 and ldm05 all resolve through it |

`patches_xct.bin` and `patches_mask.bin` (596 GB each) were the only entries
the cleanup removed, on 2026-09-07. `build_patch_dataloaders` checks for the
files it actually reads, so this root now falls back to the Zarr backend and
still loads. Verified after the deletion: both `volumes.zarr` symlinks still
resolve to `data/split_v1/volumes.zarr` (209 GB) and `latents_r07z4/` (148 GB)
is untouched.
