#!/usr/bin/env bash
# Build the full dataset from raw TIFF volumes.
# Adjust --n_train / --n_val / --n_test to your volume count.
set -euo pipefail

build_dataset \
    --raw_root ./raw_data \
    --out_root ./data/split_v1 \
    --patch_size 64 \
    --stride 32 \
    --chunk_size "32,32,32" \
    --clevel 3 \
    --n_train 40 \
    --n_val 5 \
    --n_test 5 \
    --seed 123
