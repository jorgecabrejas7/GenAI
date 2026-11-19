#!/bin/bash
# Campaign 40, rest of queue 2 (c40_ft2.sh was stopped on 2026-11-19 14:06 because its SAM2.1 large training
# hung before step 0: GPU 0 %, one CPU core 100 %, 14 min). The training was restarted by hand (PID in $1,
# same config, with faulthandler stack dumps). This script waits for that PID to exit, then runs the
# fine-tuned large model on MediaPulg along A and B at tiles 256, 192 and 384, and paints the RGB line volumes.
# Ends with "C40FT2 DONE" in c40_ft2.log, even if a step fails.
cd /home/jorgecabrejas/Dev/GenAI
C=runs/campaigns/40-teijin-air
L=$C/c40_ft2.log
O=$C/mediapulg
SRC=data/incoming/teijin/01_GL_Teijin_MediaPulg_2_8b_1570x222x225.tif
E=/home/jorgecabrejas/miniforge3/envs/segfm/bin/python
R=runs/sam2_ft/sf02-run-0001-large-cuarto-all
echo "C40SF02REST waiting for training PID $1 $(date)" >> $L
while kill -0 "$1" 2>/dev/null; do sleep 30; done
echo "SF02 train ended $(date); checkpoint: $(ls -la $R/checkpoints/checkpoint.pt 2>&1)" >> $L

paint() {  # $1 = dir with air.tif (source orientation)
  python -c "import tifffile,numpy as np,sys;a=tifffile.imread(sys.argv[1]);tifffile.imwrite(sys.argv[2],np.where(a>0,2,0).astype(np.uint8))" \
    $1/air.tif $1/air_label.tif && \
  python scripts/analysis/paint_material_limits.py --grey $SRC --label $1/air_label.tif --thickness-axis 1 \
    --out $1/lines_rgb.tif >> $L 2>&1; echo "paint $1 rc=$?" >> $L
}

for tile in 256 192 384; do
  for t in A B; do
    d=$O/sam2L_ft_${t}_tile$tile
    $E scripts/analysis/fm_sam2_air.py --model large --ckpt $R/checkpoints/checkpoint.pt --tif $SRC --thickness-axis 1 \
      --time-axis $t --tile $tile --out $d >> $L 2>&1; echo "sf02 mediapulg $t tile $tile rc=$?" >> $L
    paint $d
  done
done
echo "C40FT2 DONE $(date)" >> $L
