#!/bin/bash
# Campaign 40: GPU inference tests of air methods, after campaign 39 ends (one CUDA job per machine).
# Waits on the unique "C39 DONE" line. Inference and linear probes only; no network is trained.
cd /home/jorgecabrejas/Dev/GenAI
C=runs/campaigns/40-teijin-air
K=$C/crops
E=/home/jorgecabrejas/miniforge3/envs/segfm/bin/python
P=/home/jorgecabrejas/miniforge3/envs/poregen/bin/python
L=$C/c40_gpu.log
until grep -q "^C39 DONE" runs/campaigns/39-unet-segmenter/c39.log 2>/dev/null; do sleep 60; done
echo "C40GPU start $(date)" >> $L

# 1. Current pipeline (envelope + grey map + U-Net) on the labelled CuartoPulg crop.
CK=$(ls runs/unet_seg/us01-run-0001-*/best.ckpt)
$P scripts/analysis/unet_segment_volume.py --ckpt "$CK" --tif $K/cuarto_800-1100/grey.tif --thickness-axis 0 \
  --out $C/unet/cuarto_800-1100 >> $L 2>&1; echo "unet cuarto rc=$?" >> $L

# 2. SAM2.1-large, zero per-volume input, each time axis.
for c in mediapulg cuarto_800-1100; do for t in A B T; do
  $E scripts/analysis/fm_sam2_air.py --model large --tif $K/$c/grey.tif --thickness-axis 0 --time-axis $t \
    --out $C/sam2/${c}_$t >> $L 2>&1; echo "sam2 $c $t rc=$?" >> $L
done; done
for c in Na_05_1 Na_09_2 probetas_8; do for t in A B; do
  $E scripts/analysis/fm_sam2_air.py --model large --tif $K/$c/grey.tif --thickness-axis 0 --time-axis $t \
    --out $C/sam2/${c}_$t >> $L 2>&1; echo "sam2 $c $t rc=$?" >> $L
done; done

# 3. DINOv2-base + linear head from 2 slices, three training sources, applied across scans.
APPLY="--apply cuarto=tif=$K/cuarto_800-1100/grey.tif,axis=0 --apply mediapulg=tif=$K/mediapulg/grey.tif,axis=0"
for c in Na_05_1 Na_09_2 probetas_8; do APPLY="$APPLY --apply $c=tif=$K/$c/grey.tif,axis=0"; done
$E scripts/analysis/fm_dino_air.py --train cuarto=tif=$K/cuarto_200-500/grey.tif,axis=0,label=$K/cuarto_200-500/air.tif \
  $APPLY --out $C/dino/train_cuarto >> $L 2>&1; echo "dino train_cuarto rc=$?" >> $L
$E scripts/analysis/fm_dino_air.py --train na04=tif=$K/Na_04_2/grey.tif,axis=0,label=$K/Na_04_2/air.tif \
  $APPLY --out $C/dino/train_na04 >> $L 2>&1; echo "dino train_na04 rc=$?" >> $L
$E scripts/analysis/fm_dino_air.py --train cuarto=tif=$K/cuarto_200-500/grey.tif,axis=0,label=$K/cuarto_200-500/air.tif \
  --train na04=tif=$K/Na_04_2/grey.tif,axis=0,label=$K/Na_04_2/air.tif \
  $APPLY --out $C/dino/train_both >> $L 2>&1; echo "dino train_both rc=$?" >> $L

# 4. Scores against each crop's air (reference for coupons and CuartoPulg; the envelope for MediaPulg).
for c in mediapulg cuarto_800-1100 Na_05_1 Na_09_2 probetas_8; do
  d=$c; [ $c = cuarto_800-1100 ] && d=cuarto
  args=""
  for t in A B T; do [ -f $C/sam2/${c}_$t/air.tif ] && args="$args --pred sam2_$t=$C/sam2/${c}_$t/air.tif"; done
  for tr in train_cuarto train_na04 train_both; do for m in air air_face; do
    [ -f $C/dino/$tr/$d/$m.tif ] && args="$args --pred dino_${tr}_$m=$C/dino/$tr/$d/$m.tif"; done; done
  [ -f $C/unet/$c/air.tif ] && args="$args --pred unet_c39=$C/unet/$c/air.tif"
  $P scripts/analysis/score_air.py --ref $K/$c/air.tif --thickness-axis 0 $args --out $C/scores/$c.json >> $L 2>&1
  echo "score $c rc=$?" >> $L
done
echo "C40GPU DONE $(date)" >> $L
