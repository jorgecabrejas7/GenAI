#!/bin/bash
# Campaign 40, author-approved (2026-11-18): fine-tune SAM2.1 base+ on the author's CuartoPulg surface lines
# (videos from A 0:750 and 1150:1450, data/sam2_vos/cuarto_v0), then infer on the held-out CuartoPulg crops
# (A 800:1100, 1500:1800) and on MediaPulg, beside zero-shot base+, and score.
# Waits on the unique "C40GPU DONE" line (one CUDA job per machine).
cd /home/jorgecabrejas/Dev/GenAI
C=runs/campaigns/40-teijin-air
K=$C/crops
E=/home/jorgecabrejas/miniforge3/envs/segfm/bin/python
P=/home/jorgecabrejas/miniforge3/envs/poregen/bin/python
R=runs/sam2_ft/sf01-run-0001-cuarto-v0
L=$C/c40_sam2_ft.log
until grep -q "^C40GPU DONE" $C/c40_gpu.log 2>/dev/null; do sleep 60; done
echo "SF01 start $(date)" >> $L
nvidia-smi --query-compute-apps=pid,process_name --format=csv >> $L
(cd /home/jorgecabrejas/ext/sam2 && $E training/train.py -c configs/sam2.1_training/sam2.1_hiera_b+_teijin_air.yaml \
  --use-cluster 0 --num-gpus 1) >> $L 2>&1
echo "SF01 train rc=$? $(date)" >> $L
CK=$R/checkpoints/checkpoint.pt
for c in cuarto_800-1100 cuarto_1500-1800 mediapulg; do for t in A B; do
  $E scripts/analysis/fm_sam2_air.py --model base+ --ckpt $CK --tif $K/$c/grey.tif --thickness-axis 0 --time-axis $t \
    --out $C/sam2_ft/sf01_${c}_$t >> $L 2>&1; echo "sf01 $c $t rc=$?" >> $L
  $E scripts/analysis/fm_sam2_air.py --model base+ --tif $K/$c/grey.tif --thickness-axis 0 --time-axis $t \
    --out $C/sam2_ft/zs_bplus_${c}_$t >> $L 2>&1; echo "zs base+ $c $t rc=$?" >> $L
done; done
for c in cuarto_800-1100 cuarto_1500-1800 mediapulg; do
  args=""
  [ $c != mediapulg ] && args="--pred envelope=$K/${c}_envelope/air.tif"
  for t in A B; do
    args="$args --pred sf01_$t=$C/sam2_ft/sf01_${c}_$t/air.tif --pred zs_bplus_$t=$C/sam2_ft/zs_bplus_${c}_$t/air.tif"
  done
  $P scripts/analysis/score_air.py --ref $K/$c/air.tif --thickness-axis 0 $args --out $C/scores_ft/$c.json >> $L 2>&1
  echo "score_ft $c rc=$?" >> $L
done
echo "SF01 DONE $(date)" >> $L
