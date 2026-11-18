#!/bin/bash
# Campaign 39: after the Teijin U-Net segmentation, paint the material limits
# (upper red, lower green) on the original grey. Waits on the unique "C39 DONE" line.
cd /home/jorgecabrejas/Dev/GenAI
C=runs/campaigns/39-unet-segmenter
O=data/incoming/teijin/v05_1006-1230_unet_seg
until grep -q "^C39 DONE" $C/c39.log 2>/dev/null; do sleep 60; done
python scripts/analysis/paint_material_limits.py \
  --grey data/incoming/teijin/01_GL_Teijin_MediaPulg_2_8b_1570x222x225.tif \
  --label $O/label.tif --thickness-axis 1 --out $O/material_limits_rgb.tif >> $C/c39.log 2>&1
echo "C39 PAINT rc=$? $(date)" >> $C/c39.log
