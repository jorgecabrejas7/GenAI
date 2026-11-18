#!/bin/bash
# Campaign 39, resumed: the author gave the GPU to campaign 40 first (2026-11-18).
# Waits for "C40GPU DONE" in campaign 40's GPU log, then scores best.ckpt on test
# and val, segments the Teijin volume with the final model, and paints the limits.
cd /home/jorgecabrejas/Dev/GenAI
export POREGEN_SPLIT=split_v4
C=runs/campaigns/39-unet-segmenter
O=data/incoming/teijin/v05_1006-1230_unet_seg
until grep -q "^C40GPU DONE" runs/campaigns/40-teijin-air/c40_gpu.log 2>/dev/null; do sleep 60; done
until [ -z "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader)" ]; do sleep 60; done
CK=$(ls runs/unet_seg/us01-run-0001-*/best.ckpt)
echo "C39R start $(date) ckpt $CK" >> $C/c39.log
python scripts/analysis/unet_seg_eval.py --ckpt "$CK" --splits test val --out $C >> $C/c39.log 2>&1
echo "C39R eval rc=$?" >> $C/c39.log
python scripts/analysis/unet_segment_volume.py --ckpt "$CK" \
  --tif data/incoming/teijin/01_GL_Teijin_MediaPulg_2_8b_1570x222x225.tif --thickness-axis 1 \
  --out $O >> $C/c39.log 2>&1
echo "C39R teijin rc=$?" >> $C/c39.log
python scripts/analysis/paint_material_limits.py \
  --grey data/incoming/teijin/01_GL_Teijin_MediaPulg_2_8b_1570x222x225.tif \
  --label $O/label.tif --thickness-axis 1 --out $O/material_limits_rgb.tif >> $C/c39.log 2>&1
echo "C39R paint rc=$? $(date)" >> $C/c39.log
echo "C39R DONE $(date)" >> $C/c39.log
