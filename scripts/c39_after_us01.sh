#!/bin/bash
# Campaign 39: once us01 ends, score best.ckpt on the full test and val splits,
# then segment the Teijin volume. Waits on the unique "US01 EXIT" line.
cd /home/jorgecabrejas/Dev/GenAI
export POREGEN_SPLIT=split_v4
C=runs/campaigns/39-unet-segmenter
until grep -q "^US01 EXIT" runs/unet_seg/us01.log; do sleep 60; done
CK=$(ls runs/unet_seg/us01-run-0001-*/best.ckpt)
echo "C39 start $(date) ckpt $CK" >> $C/c39.log
python scripts/analysis/unet_seg_eval.py --ckpt "$CK" --splits test val --out $C >> $C/c39.log 2>&1
echo "C39 eval rc=$?" >> $C/c39.log
python scripts/analysis/unet_segment_volume.py --ckpt "$CK" \
  --tif data/incoming/teijin/01_GL_Teijin_MediaPulg_2_8b_1570x222x225.tif --thickness-axis 1 \
  --out data/incoming/teijin/v05_1006-1230_unet_seg >> $C/c39.log 2>&1
echo "C39 teijin rc=$?" >> $C/c39.log
echo "C39 DONE $(date)" >> $C/c39.log
