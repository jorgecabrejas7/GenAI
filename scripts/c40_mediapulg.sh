#!/bin/bash
# Campaign 40 GPU queue, MediaPulg only (author 2026-11-18: test only on the original volume;
# CuartoPulg and its red/green lines are only training help). Replaces c40_gpu_tests.sh and c40_sam2_ft.sh.
# Ends with "SF01 DONE" in c40_sam2_ft.log (campaign 39 waits on it), even if a step fails.
cd /home/jorgecabrejas/Dev/GenAI
C=runs/campaigns/40-teijin-air
M=$C/crops/mediapulg/grey.tif          # MediaPulg, thickness-first (T, A, B)
Q=$C/crops
E=/home/jorgecabrejas/miniforge3/envs/segfm/bin/python
P=/home/jorgecabrejas/miniforge3/envs/poregen/bin/python
R=runs/sam2_ft/sf01-run-0001-cuarto-all
L=$C/c40_sam2_ft.log
O=$C/mediapulg
echo "C40M start $(date)" >> $L
nvidia-smi --query-compute-apps=pid,process_name --format=csv >> $L

# 1. Current pipeline with the final U-Net (best.ckpt) on MediaPulg.
$P scripts/analysis/unet_segment_volume.py --ckpt runs/unet_seg/us01-run-0001-20261118-090407/best.ckpt \
  --tif $M --thickness-axis 0 --out $O/unet_c39 >> $L 2>&1; echo "unet rc=$?" >> $L

# 2. SAM2.1-large zero-shot (prompts from the rule faces = air, mid-plane = specimen), three time axes.
for t in A B T; do
  $E scripts/analysis/fm_sam2_air.py --model large --tif $M --thickness-axis 0 --time-axis $t \
    --out $O/sam2L_zs_$t >> $L 2>&1; echo "sam2L zs $t rc=$?" >> $L
done

# 3. DINOv2-large + linear head fitted on 8 CuartoPulg slices (author's lines), applied to MediaPulg.
$E scripts/analysis/fm_dino_air.py --model facebook/dinov2-large --train-slices 8 \
  --train cuarto=tif=$Q/cuarto_200-500/grey.tif,axis=0,label=$Q/cuarto_200-500/air.tif \
  --apply mediapulg=tif=$M,axis=0 --out $O/dinoL_n8 >> $L 2>&1; echo "dino rc=$?" >> $L

# 4. SAM2.1 base+ fine-tuned on all of CuartoPulg (author's lines), then MediaPulg, beside zero-shot base+.
until [ -f data/sam2_vos/cuarto_all/export.json ]; do sleep 30; done
(cd /home/jorgecabrejas/ext/sam2 && $E training/train.py -c configs/sam2.1_training/sam2.1_hiera_b+_teijin_air.yaml \
  --use-cluster 0 --num-gpus 1) >> $L 2>&1
echo "SF01 train rc=$? $(date)" >> $L
for t in A B; do
  $E scripts/analysis/fm_sam2_air.py --model base+ --ckpt $R/checkpoints/checkpoint.pt --tif $M --thickness-axis 0 \
    --time-axis $t --out $O/sam2bp_ft_$t >> $L 2>&1; echo "sf01 $t rc=$?" >> $L
  $E scripts/analysis/fm_sam2_air.py --model base+ --tif $M --thickness-axis 0 --time-axis $t \
    --out $O/sam2bp_zs_$t >> $L 2>&1; echo "sam2bp zs $t rc=$?" >> $L
done
echo "SF01 DONE $(date)" >> $L
