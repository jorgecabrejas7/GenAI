#!/bin/bash
# Campaign 40 GPU queue (author 2026-11-18): CuartoPulg and its red/green lines are training help only;
# the test is MediaPulg alone (no label: figures). SAM2.1 base+ is fine-tuned on all of CuartoPulg,
# then run on MediaPulg beside zero-shot base+ and zero-shot large (U-Net c39 MediaPulg output exists).
# Ends with "SF01 DONE" in c40_sam2_ft.log (campaign 39 waits on it), even if a step fails.
cd /home/jorgecabrejas/Dev/GenAI
C=runs/campaigns/40-teijin-air
M=$C/crops/mediapulg/grey.tif
E=/home/jorgecabrejas/miniforge3/envs/segfm/bin/python
R=runs/sam2_ft/sf01-run-0002-cuarto-all
L=$C/c40_sam2_ft.log
O=$C/mediapulg
echo "C40M start (v3, train on all CuartoPulg, test MediaPulg) $(date)" >> $L
nvidia-smi --query-compute-apps=pid,process_name --format=csv >> $L

(cd /home/jorgecabrejas/ext/sam2 && $E training/train.py -c configs/sam2.1_training/sam2.1_hiera_b+_teijin_air.yaml \
  --use-cluster 0 --num-gpus 1) >> $L 2>&1
echo "SF01 train rc=$? $(date)" >> $L

for t in A B; do
  $E scripts/analysis/fm_sam2_air.py --model base+ --ckpt $R/checkpoints/checkpoint.pt --tif $M --thickness-axis 0 \
    --time-axis $t --out $O/sam2bp_ft_$t >> $L 2>&1; echo "sf01 mediapulg $t rc=$?" >> $L
done
for t in A B; do
  $E scripts/analysis/fm_sam2_air.py --model base+ --tif $M --thickness-axis 0 --time-axis $t \
    --out $O/sam2bp_zs_$t >> $L 2>&1; echo "sam2bp zs mediapulg $t rc=$?" >> $L
  $E scripts/analysis/fm_sam2_air.py --model large --tif $M --thickness-axis 0 --time-axis $t \
    --out $O/sam2L_zs_$t >> $L 2>&1; echo "sam2L zs mediapulg $t rc=$?" >> $L
done
echo "SF01 DONE $(date)" >> $L
