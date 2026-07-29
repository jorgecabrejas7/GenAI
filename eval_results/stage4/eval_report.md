# PoreGen Evaluation Report

- **Real dir**: `data/real_test_volumes`
- **Generated dir**: `inference/ldm03-run-0001-20260622-093226-z16-c128-s32-bs128-lr1e-04/20260624-112335-ddim200-lstd0.9985-spor1.00-snb1.00`
- **Volumes**: 8 real, 11 generated
- **Elapsed**: 897.2 s
- **FID crops per volume**: 500

## FID — 64×64 native crops → bilinear 299×299 → InceptionV3 pool3

| Set | Axial | Coronal | Sagittal | Mean |
|:----|------:|--------:|---------:|-----:|
| Generated | n/a | n/a | n/a | n/a |

> **Lower FID is better.**  Crops are 64×64 at native voxel resolution (median pore diameter ~1.79 voxels),  bilinearly upsampled to 299×299.  Full-slice resize was rejected: the ~10× downscale shrinks pores to ~0.17 px.

**FID caveats (copy-paste for paper methods):** FID is computed on 64×64 crops at native voxel resolution, resized to 299×299 for InceptionV3 feature extraction. Full-slice resize to 299×299 was rejected because PoreGen volumes are ~3179×1759 voxels; the resulting ~10.6× downscale reduces the median pore diameter (1.79 voxels) to ~0.17 pixels, destroying pore-scale information before feature extraction. FID values reported here are internally consistent (all baselines evaluated with the same protocol) but are not numerically comparable to Naiff 2025, He 2024, or Pinaya 2022, which apply full-slice resize to volumes of 256³ or smaller.

## Summary Metrics

| Metric | Generated | Real |
|:-------|----------:|-----:|
| Porosity mean | 0.0307 | 0.0188 |
| Porosity std | 0.0004 | 0.0246 |
| Porosity W1 | 0.0239 | — |
| Porosity MAE (paired) | n/a | — |
| PSD W1 (diameters, vox) | 0.9011 | — |
| PSD median diam (vox) | 1.7894 | 1.2407 |
| PSD P90 diam (vox) | 4.8300 | 2.2545 |
| S2(r) RMSE | 0.0001 | — |
| S2(r) per-vol W1 mean | 5.1032 | — |
| Ripley K W1 | 6930.4582 | — |
| Ripley K at mean spacing (gen) | 54.6693 | 128.7007 |
| Ripley K / CSR at mean spacing | 0.8082 | — |
| FID mean | n/a | — |
| FID axial | n/a | — |
| FID coronal | n/a | — |
| FID sagittal | n/a | — |
| Boundary XCT seam ratio | 1.0248 | 0.8309 |
| Boundary mask seam ratio | 1.5245 | 0.8437 |
| Morph sphericity mean | 0.3108 | 0.5065 |
| Morph sphericity P50 | 0.2973 | 0.4734 |
| Morph aspect ratio mean | 2.2408 | 12.6958 |
| Diversity phi std | 0.0004 | 0.0246 |
| Diversity PSD W1 std | 0.0024 | 0.0458 |
| Diversity Ripley K std | 18.7037 | 1928.5196 |
| Memorisation NN dist (mean) | n/a | — |
| Memorisation NN dist (std) | n/a | — |

## Metric Comparability

| Metric | Comparable to | Notes |
|:-------|:-------------|:------|
| Porosity MAE | Naiff 2025 | Identical definition |
| W1-PSD | Naiff 2025 | Identical definition |
| S₂(r) RMSE | Gayon-Lombardo 2020, SliceGAN 2021, Naiff 2025 | Raw curve RMSE; prior work reports curves visually |
| Ripley's K | Novel for this domain | No prior porous media paper reports this |
| FID | Internal only | See FID section |
| Boundary inconsistency | Novel | No prior work reports this |
| Diversity (φ std, PSD W1 std) | Gayon-Lombardo 2020 (mode collapse analysis) | Conceptually comparable, not numerically |

## Sanity Check Results

| Check | Actual | Expected | Status |
|:------|-------:|---------:|:-------|
| phi_mean | 0.0307 | 0.0550 | PASS |
| psd_median_diam_vox | 1.7894 | 1.7900 | PASS |
| psd_p90_diam_vox | 4.8300 | 2.9200 | PASS |
| S2(r=30)/expected | 0.0040 | 1.0000 | FAIL |

## Sanity Warnings

- WARN [generated] S2(r=30)/expected = 0.004 (expected near 1.0; EDA coeff = 4.73)

## Figures

- [FID table](figures/fid_table.png)
- [S₂(r) curves](figures/s2_curves.png)
- [PSD histogram](figures/psd_histogram.png)
- [Ripley K(r)](figures/ripley_k.png)

