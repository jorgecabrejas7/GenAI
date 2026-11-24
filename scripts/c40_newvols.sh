#!/bin/bash
# Campaign 40, new volumes (author request 2026-11-24): run the three best air methods on three new Teijin
# scans and paint the RGB line volumes (grey + red upper / green lower surface) for the author to open in Fiji.
# Volumes: MediaPulgada_2 (full 1938 slices), CuartoPulgada_2, CuartoPulgada_4. None of them was used in
# training (the fine-tunes used CuartoPulg_1 only). Methods, with the same settings as on the MediaPulg crop:
#   unet_airft  : U-Net air-only fine-tune (uaf01 last.ckpt), 96^3 windows
#   sam2L_ft_A  : SAM2.1 large fine-tune (sf02), video along A, tile 256, one key frame in the middle
#   sam2bp_ft_A : SAM2.1 base+ fine-tune (sf01-run-0002), same
# Order: U-Net on all three, then SAM2 large on all three, then base+, so the fast results come first.
# One GPU job at a time; CPU-only steps in a 24G scope. Ends with "C40NEW DONE" in the log, even if a step fails.
cd /home/jorgecabrejas/Dev/GenAI
C=runs/campaigns/40-teijin-air
L=$C/c40_newvols.log
N=$C/newvols
RAW=data/incoming/teijin/raw
E=/home/jorgecabrejas/miniforge3/envs/segfm/bin/python
CPU="systemd-run --user --scope -p MemoryMax=24G nice -n 19"
UNET=runs/unet_seg/uaf01-run-0001-20261119-104642/last.ckpt
SL=runs/sam2_ft/sf02-run-0001-large-cuarto-all/checkpoints/checkpoint.pt
SB=runs/sam2_ft/sf01-run-0002-cuarto-all/checkpoints/checkpoint.pt
VOLS="MediaPulgada_2 CuartoPulgada_2 CuartoPulgada_4"
echo "C40NEW start $(date)" >> $L
mkdir -p $N

# grey.tif per volume (raw -> tif, shape (Z, Y, X) from the name XxYxZ; the tif is linked as is)
$CPU python - >> $L 2>&1 <<'EOF'
import re, os, numpy as np, tifffile
from pathlib import Path
raw, out = Path("data/incoming/teijin/raw"), Path("runs/campaigns/40-teijin-air/newvols")
for d in sorted(raw.iterdir()):
    f = next(d.iterdir()); o = out / d.name; o.mkdir(parents=True, exist_ok=True); g = o / "grey.tif"
    if g.exists():
        continue
    if f.suffix == ".tif":
        os.link(f, g)
    else:
        x, y, z = map(int, re.search(r"_(\d+)x(\d+)x(\d+)\.raw$", f.name).groups())
        v = np.fromfile(f, np.uint8); assert v.size == x * y * z, (f, v.size)
        tifffile.imwrite(g, v.reshape(z, y, x), imagej=True)
    print(d.name, tifffile.TiffFile(g).series[0].shape, flush=True)
EOF
echo "grey rc=$?" >> $L

paint() {  # $1 = volume dir, $2 = method dir (air.tif in source orientation)
  $CPU python -c "import tifffile,numpy as np,sys;a=tifffile.imread(sys.argv[1]);tifffile.imwrite(sys.argv[2],np.where(a>0,2,0).astype(np.uint8))" \
    $2/air.tif $2/air_label.tif && \
  $CPU python scripts/analysis/paint_material_limits.py --grey $1/grey.tif --label $2/air_label.tif --thickness-axis 1 \
    --out $2/lines_rgb.tif >> $L 2>&1; echo "paint $2 rc=$?" >> $L
  rm -f $2/air_label.tif
}

for v in $VOLS; do
  python scripts/analysis/unet_air_ft_segment.py --ckpt $UNET --tif $N/$v/grey.tif --thickness-axis 1 \
    --out $N/$v/unet_airft >> $L 2>&1; echo "unet $v rc=$? $(date)" >> $L
  paint $N/$v $N/$v/unet_airft
done
for m in L B; do
  [ $m = L ] && { name=sam2L_ft_A; model=large; ck=$SL; } || { name=sam2bp_ft_A; model=base+; ck=$SB; }
  for v in $VOLS; do
    $E scripts/analysis/fm_sam2_air.py --model $model --ckpt $ck --tif $N/$v/grey.tif --thickness-axis 1 \
      --time-axis A --tile 256 --out $N/$v/$name >> $L 2>&1; echo "$name $v rc=$? $(date)" >> $L
    paint $N/$v $N/$v/$name
  done
done
echo "C40NEW DONE $(date)" >> $L
