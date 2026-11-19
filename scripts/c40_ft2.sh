#!/bin/bash
# Campaign 40 GPU queue 2 (author approved 2026-11-19):
#  1. U-Net air-only fine-tune from us01 on all of CuartoPulg, raw grey input (no envelope, no grey map),
#     then MediaPulg air + RGB line volume.
#  2. SAM2.1 LARGE fine-tune on all of CuartoPulg (same data and settings as the base+ run, option c),
#     then MediaPulg along A and B at tile 256, plus a tile-size test (192, 384) on both axes.
# Every MediaPulg mask is also painted as an RGB volume (grey + red upper / green lower surface).
# One job at a time on the GPU. Ends with "C40FT2 DONE" in the log, even if a step fails.
cd /home/jorgecabrejas/Dev/GenAI
C=runs/campaigns/40-teijin-air
L=$C/c40_ft2.log
O=$C/mediapulg
SRC=data/incoming/teijin/01_GL_Teijin_MediaPulg_2_8b_1570x222x225.tif
S=data/incoming/teijin/cuarto_overlay_split
E=/home/jorgecabrejas/miniforge3/envs/segfm/bin/python
R=runs/sam2_ft/sf02-run-0001-large-cuarto-all
echo "C40FT2 start $(date)" >> $L
nvidia-smi --query-compute-apps=pid,process_name --format=csv >> $L

paint() {  # $1 = dir with air.tif (source orientation)
  python -c "import tifffile,numpy as np,sys;a=tifffile.imread(sys.argv[1]);tifffile.imwrite(sys.argv[2],np.where(a>0,2,0).astype(np.uint8))" \
    $1/air.tif $1/air_label.tif && \
  python scripts/analysis/paint_material_limits.py --grey $SRC --label $1/air_label.tif --thickness-axis 1 \
    --out $1/lines_rgb.tif >> $L 2>&1; echo "paint $1 rc=$?" >> $L
}

python -m poregen.experiments.train_unet_air_ft --exp uaf01 \
  --init runs/unet_seg/us01-run-0001-20261118-090407/best.ckpt \
  --grey $S/grey.tif --air $S/air.tif --surfaces $S/surfaces.npz --thickness-axis 1 --steps 8000 >> $L 2>&1
echo "UAF01 train rc=$? $(date)" >> $L
U=$(ls -d runs/unet_seg/uaf01-run-* | tail -1)
python scripts/analysis/unet_air_ft_segment.py --ckpt $U/last.ckpt --tif $SRC --thickness-axis 1 \
  --out $O/unet_airft >> $L 2>&1; echo "uaf01 mediapulg rc=$?" >> $L
paint $O/unet_airft

(cd /home/jorgecabrejas/ext/sam2 && $E training/train.py -c configs/sam2.1_training/sam2.1_hiera_l_teijin_air.yaml \
  --use-cluster 0 --num-gpus 1) >> $L 2>&1
echo "SF02 train rc=$? $(date)" >> $L
for tile in 256 192 384; do
  for t in A B; do
    d=$O/sam2L_ft_${t}_tile$tile
    $E scripts/analysis/fm_sam2_air.py --model large --ckpt $R/checkpoints/checkpoint.pt --tif $SRC --thickness-axis 1 \
      --time-axis $t --tile $tile --out $d >> $L 2>&1; echo "sf02 mediapulg $t tile $tile rc=$?" >> $L
    paint $d
  done
done
echo "C40FT2 DONE $(date)" >> $L
