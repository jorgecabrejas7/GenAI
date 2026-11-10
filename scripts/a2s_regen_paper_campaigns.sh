#!/usr/bin/env bash
# a2s regeneration of the paper's split_v4 campaigns (author 2026-11-06: "Regenerate and we have both results for the
# paper"). Each campaign gets a NEW root <old>-a2s; the old roots stay untouched. The one-chunk cases (every axis
# <= 192: no chunk join, identical under either sampler) are COPIED from the old root with the D87 air rule applied
# post hoc (every air voxel -> the volume's mean air grey, the same operation the generator applies since 7bbfc82;
# labels untouched; recorded in the manifest), so generate skips them and every case of the new root follows D87.
# Only the multi-chunk cases are generated, with a2s (the sampler default since 7bbfc82). Order: c18 first (the
# main table), then c24, c25, c27. Waits for the current chain (twins -> memorisation -> c37) to end.
#
#     bash scripts/a2s_regen_paper_campaigns.sh      (vault D89; launched 2026-11-06)
cd "$(dirname "$0")/.."
export POREGEN_SPLIT=split_v4
R=runs/campaigns
say() { printf '%s  C38 %s\n' "$(date -Is)" "$*" | tee -a runs/campaigns/chain.log; }
#: Parameters (vault D91, 2026-11-08): SUFFIX names the new roots; GEN_EXTRA is
#: appended to every generate call (e.g. "--solver heun --ddim-steps 25");
#: COPY_ONECHUNK=0 regenerates the one-chunk cases too (a solver change touches
#: them; a sampler change does not); WAIT_LINE is the chain.log line to wait for.
SUFFIX="${SUFFIX:--a2s}"; GEN_EXTRA="${GEN_EXTRA:-}"; COPY_ONECHUNK="${COPY_ONECHUNK:-1}"
WAIT_LINE="${WAIT_LINE:-C37 precision/recall rc=}"
TAG="${SUFFIX#-}"
#: ONLY: the campaigns to run, e.g. ONLY="c27" (recovery 2026-11-10).
ONLY="${ONLY:-c18 c24 c25 c27}"
want() { case " $ONLY " in *" $1 "*) return 0 ;; *) return 1 ;; esac; }
until grep -q "$WAIT_LINE" runs/campaigns/chain.log; do sleep 60; done
sleep 20
seed_root() {  # $1 old root, $2 new root: copy one-chunk cases (air rule applied) and link the real floor
    mkdir -p "$2"; [ -e "$2/real_floor" ] || ln -s "$(readlink -f "$1/real_floor")" "$2/real_floor"
    [ "$COPY_ONECHUNK" = 1 ] || { echo "one-chunk cases regenerated too (COPY_ONECHUNK=0)"; return 0; }
    python - "$1" "$2" <<'PY'
import json, os, shutil, sys, glob
import numpy as np, tifffile
old, new = sys.argv[1:3]; n = 0
for m in glob.glob(f"{old}/*/volumes/*/manifest.json"):
    a, case = m.split("/")[-4], m.split("/")[-2]
    if a == "real_floor": continue
    d = json.load(open(m))
    if max(d["volume_shape"]) <= 192:
        dst = f"{new}/{a}/volumes/{case}"
        if os.path.exists(dst): continue
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        shutil.copytree(os.path.dirname(m), dst)
        v = tifffile.imread(f"{dst}/volume.tif"); lab = tifffile.imread(f"{dst}/label.tif")
        air = lab == 2
        g = int(round(float(v[air].mean()))) if air.any() else None
        if g is not None:
            v[air] = g; tifffile.imwrite(f"{dst}/volume.tif", v, imagej=True)
        gs = (d.setdefault("notes", {}) or {}).setdefault("generation_stats", {})  # under notes: a top-level key is rejected by Manifest
        gs["air_grey_value"] = g
        gs["air_grey_applied_post_hoc"] = (
            "2026-11-06: one-chunk case copied from the unoverlapped-sampler root; D87 air rule applied to volume.tif")
        json.dump(d, open(f"{dst}/manifest.json", "w"), indent=2)
        n += 1
print(f"{n} one-chunk cases copied into {new} with the air rule applied")
PY
}
# c18 via the same script that built the original campaign
if want c18; then
OLD=$R/18-eval-v4-final-v4-run0005; NEW=${OLD}${SUFFIX}
say "c18 $TAG start"; seed_root "$OLD" "$NEW" | tee -a runs/campaigns/chain.log
CKPT_RUN=runs/ldm/ldm06-run-0005-20261022-081725-z4-c128-bs256-lr1e-04-dsv4 CKPT=latest SAMPLER=a2s EXTRA="$GEN_EXTRA" OUT=$NEW SCRATCH=runs/campaigns/chain_logs/v8/c18_$TAG bash scripts/regen_eval_v4.sh > runs/campaigns/chain_logs/c18_$TAG.log 2>&1
say "c18 $TAG rc=$? (sub-stages: grep rc= $NEW/regen.log)"
fi
run_campaign() {  # $1 tag $2 old root $3 model $4 ckpt $5.. assessments
    tag=$1; old=$2; model=$3; ckpt=$4; shift 4; new=${old}${SUFFIX}
    say "$tag $TAG start"; seed_root "$old" "$new" | tee -a runs/campaigns/chain.log
    for A in "$@"; do
        python -m poregen.eval_v4.cli generate "$A" --model "$model" --ckpt "$ckpt" --weights ema --out "$new" $GEN_EXTRA > runs/campaigns/chain_logs/${tag}_${TAG}_gen_$A.log 2>&1
        say "$tag generate $A rc=$?"
        python -m poregen.eval_v4.cli measure "$A" --root "$new" > runs/campaigns/chain_logs/${tag}_${TAG}_meas_$A.log 2>&1
        say "$tag measure $A rc=$?"
    done
    python -m poregen.eval_v4.cli report --root "$new" > runs/campaigns/chain_logs/${tag}_${TAG}_report.log 2>&1
    say "$tag report rc=$?"
}
want c24 && run_campaign c24 $R/24-ablation-v4-run0005 runs/ldm/ldm06-run-0005-20261022-081725-z4-c128-bs256-lr1e-04-dsv4 latest ablation
want c25 && run_campaign c25 $R/25-ldm-phi-only-v4-run0002 runs/ldm/ldm25-run-0002-20261025-062027-z4-c128-bs256-lr1e-04-dsv4 40000 sampler porosity_global microstructure cfg
want c27 && run_campaign c27 $R/27-budget-matched-40k-v4-run0005 runs/ldm/ldm06-run-0005-20261022-081725-z4-c128-bs256-lr1e-04-dsv4 40000 sampler porosity_global microstructure
say "REGENERATION COMPLETE ($TAG)"; [ "$TAG" = a2s ] && say "A2S REGENERATION COMPLETE"
