#!/bin/bash
# Campaign 40: GPU inference tests of air methods on Teijin only (author 2026-11-18: no split_v4),
# after campaign 39 ends (one CUDA job per machine). Waits on the unique "C39 DONE" line.
# Inference and linear probes only; no network is trained.
# Scored on the author's CuartoPulg surface lines, crops A 800:1100 and 1500:1800 (never used to fit);
# DINOv2 heads are fitted on CuartoPulg A 200:500. MediaPulg has no label: compared with the envelope.
cd /home/jorgecabrejas/Dev/GenAI
C=runs/campaigns/40-teijin-air
K=$C/crops
E=/home/jorgecabrejas/miniforge3/envs/segfm/bin/python
P=/home/jorgecabrejas/miniforge3/envs/poregen/bin/python
L=$C/c40_gpu.log
until grep -q "^C39 DONE" runs/campaigns/39-unet-segmenter/c39.log 2>/dev/null; do sleep 60; done
echo "C40GPU start $(date)" >> $L
nvidia-smi --query-compute-apps=pid,process_name --format=csv >> $L
TEST="cuarto_800-1100 cuarto_1500-1800"

# 1. Current pipeline (envelope + grey map + U-Net) on the labelled CuartoPulg crops.
CK=$(ls runs/unet_seg/us01-run-0001-*/best.ckpt)
for c in $TEST; do
  $P scripts/analysis/unet_segment_volume.py --ckpt "$CK" --tif $K/$c/grey.tif --thickness-axis 0 \
    --out $C/unet/$c >> $L 2>&1; echo "unet $c rc=$?" >> $L
done

# 2. SAM2.1-large, zero per-volume input, each time axis.
for c in $TEST mediapulg; do for t in A B T; do
  $E scripts/analysis/fm_sam2_air.py --model large --tif $K/$c/grey.tif --thickness-axis 0 --time-axis $t \
    --out $C/sam2/${c}_$t >> $L 2>&1; echo "sam2 $c $t rc=$?" >> $L
done; done

# 3. DINOv2 + linear head fitted on 2 and on 8 CuartoPulg slices (A 200:500).
APPLY=""
for c in $TEST mediapulg; do APPLY="$APPLY --apply $c=tif=$K/$c/grey.tif,axis=0"; done
for n in 2 8; do for m in base large; do
  $E scripts/analysis/fm_dino_air.py --model facebook/dinov2-$m --train-slices $n \
    --train cuarto=tif=$K/cuarto_200-500/grey.tif,axis=0,label=$K/cuarto_200-500/air.tif \
    $APPLY --out $C/dino/${m}_n$n >> $L 2>&1; echo "dino $m n$n rc=$?" >> $L
done; done

# 4. Scores against each crop's air (author's lines for CuartoPulg; the envelope for MediaPulg).
for c in $TEST mediapulg; do
  args="--pred envelope=$K/${c}_envelope/air.tif"
  [ $c = mediapulg ] && args=""
  for t in A B T; do [ -f $C/sam2/${c}_$t/air.tif ] && args="$args --pred sam2_$t=$C/sam2/${c}_$t/air.tif"; done
  for d in $C/dino/*_n*; do for m in air air_face; do
    [ -f $d/$c/$m.tif ] && args="$args --pred dino_$(basename $d)_$m=$d/$c/$m.tif"; done; done
  [ -f $C/unet/$c/air.tif ] && args="$args --pred unet_c39=$C/unet/$c/air.tif"
  $P scripts/analysis/score_air.py --ref $K/$c/air.tif --thickness-axis 0 $args --out $C/scores/$c.json >> $L 2>&1
  echo "score $c rc=$?" >> $L
done
echo "C40GPU DONE $(date)" >> $L
