#!/bin/bash
# Campaign 40 GPU queue (author 2026-11-18). Target = MediaPulg (no label: figures only).
# Ground truth = the author's red/green lines on CuartoPulg: SAM2.1 base+ is fine-tuned on CuartoPulg
# A 0:750 + 1150:1450 only, and every method is scored on the held-out CuartoPulg A 800:1100 and 1500:1800.
# Ends with "SF01 DONE" in c40_sam2_ft.log (campaign 39 waits on it), even if a step fails.
cd /home/jorgecabrejas/Dev/GenAI
C=runs/campaigns/40-teijin-air
K=$C/crops
E=/home/jorgecabrejas/miniforge3/envs/segfm/bin/python
P=/home/jorgecabrejas/miniforge3/envs/poregen/bin/python
R=runs/sam2_ft/sf01-run-0001-cuarto-v0
L=$C/c40_sam2_ft.log
O=$C/mediapulg
H="cuarto_800-1100 cuarto_1500-1800"
out() { [ $1 = mediapulg ] && echo $O/$2 || echo $C/heldout/$1/$2; }
echo "C40M start (v2, held-out scoring) $(date)" >> $L
nvidia-smi --query-compute-apps=pid,process_name --format=csv >> $L

# 1. SAM2.1 base+ fine-tune on the train part of CuartoPulg.
(cd /home/jorgecabrejas/ext/sam2 && $E training/train.py -c configs/sam2.1_training/sam2.1_hiera_b+_teijin_air.yaml \
  --use-cluster 0 --num-gpus 1) >> $L 2>&1
echo "SF01 train rc=$? $(date)" >> $L

# 2. Fine-tuned base+ and zero-shot base+ (same size, fair comparison), time axes A and B.
for c in $H mediapulg; do for t in A B; do
  $E scripts/analysis/fm_sam2_air.py --model base+ --ckpt $R/checkpoints/checkpoint.pt --tif $K/$c/grey.tif \
    --thickness-axis 0 --time-axis $t --out $(out $c sam2bp_ft_$t) >> $L 2>&1; echo "sf01 $c $t rc=$?" >> $L
  $E scripts/analysis/fm_sam2_air.py --model base+ --tif $K/$c/grey.tif --thickness-axis 0 --time-axis $t \
    --out $(out $c sam2bp_zs_$t) >> $L 2>&1; echo "sam2bp zs $c $t rc=$?" >> $L
done; done

# 3. SAM2.1-large zero-shot, time axis A (best zero-shot axis so far) and B.
for c in $H mediapulg; do for t in A B; do
  d=$(out $c sam2L_zs_$t)
  [ $c = cuarto_800-1100 ] && [ $t = A ] && [ -f $C/sam2/cuarto_800-1100_A/air.tif ] && { mkdir -p $d; cp $C/sam2/cuarto_800-1100_A/* $d/; continue; }
  $E scripts/analysis/fm_sam2_air.py --model large --tif $K/$c/grey.tif --thickness-axis 0 --time-axis $t \
    --out $d >> $L 2>&1; echo "sam2L zs $c $t rc=$?" >> $L
done; done

# 4. Scores on the held-out CuartoPulg parts against the author's lines.
for c in $H; do
  args="--pred envelope=$K/${c}_envelope/air.tif --pred unet_c39=$C/unet/$c/air.tif"
  for m in sam2bp_ft sam2bp_zs sam2L_zs; do for t in A B; do
    f=$C/heldout/$c/${m}_$t/air.tif; [ -f $f ] && args="$args --pred ${m}_$t=$f"; done; done
  $P scripts/analysis/score_air.py --ref $K/$c/air.tif --thickness-axis 0 $args --out $C/scores/$c.json >> $L 2>&1
  echo "score $c rc=$?" >> $L
done
echo "SF01 DONE $(date)" >> $L
