#!/usr/bin/env bash
# The five remaining r08 rungs on split_v4, standalone, for a second GB10.
#
# rf-2, rf-4, rf-16 (r08/base), rf-32, rf-64 — each exactly its split_v3 config
# (pin_memory true, as the whole VAE family from 2026-10-08) with chain v8's
# smoke and rf-8's four checks. It reads only the repo and data/split_v4; it
# resolves no VAE or LDM and does not regenerate docs/SPLIT_V3_GATES.md, which
# is in git. Results land exactly where they land on the first machine:
# runs/vae/r08-run-*-ds v4 and runs/campaigns/09-r08-latent-sweep/*-dsv4, so
# scripts/pull_remote_results.sh can bring them back unchanged.
#
#     setsid nohup bash scripts/chain_rungs_remote.sh &
#     touch runs/campaigns/STOP_CHAIN     # halt cleanly before the next stage
#
# RUNGS overrides the list, e.g. RUNGS="32:2:r08/reduction-factor-32".
set -u
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO" || exit 1
export POREGEN_SPLIT=split_v4
SPLIT=$POREGEN_SPLIT
LOG="$REPO/runs/campaigns/chain.log"
S="$REPO/runs/campaigns/chain_logs"
STOP="$REPO/runs/campaigns/STOP_CHAIN"
SMOKE_FRACTION=0.8
MIN_AVAIL_GB=8
RUNGS="${RUNGS:-2:32:r08/reduction-factor-2 4:16:r08/reduction-factor-4 16:4:r08/base 32:2:r08/reduction-factor-32 64:1:r08/reduction-factor-64}"
mkdir -p "$S"

say() { printf '%s  RUNGS %s\n' "$(date -Is)" "$*" | tee -a "$LOG"; }
mem_line() { printf 'host available %s GB; swap used %s MB' \
    "$(free -g | awk '/^Mem:/{print $7}')" "$(free -m | awk '/^Swap:/{print $3}')"; }
gate() {
    if [ -e "$STOP" ]; then
        say "STOP FILE PRESENT — halting cleanly before: $*"
        exit 0
    fi
}
run_watched() {
    local label="$1" tag="$2"; shift 2
    gate "$label"
    say "$label start"
    local t0=$SECONDS
    choom -n 1000 -- "$@" > "$S/$tag.log" 2>&1 &
    local pid=$!
    # A minute-by-minute pressure timeline beside every trainer, so a stall
    # leaves evidence (scripts/analysis/host_pressure_log.sh exits with it).
    bash scripts/analysis/host_pressure_log.sh "$pid" "$REPO/runs/vae" \
        > "$S/pressure_$tag.tsv" 2>&1 &
    ( sleep 300
      if kill -0 "$pid" 2>/dev/null; then
          say "$tag +5min: $(mem_line)"
          [ "$(free -g | awk '/^Mem:/{print $7}')" -lt "$MIN_AVAIL_GB" ] && {
              say "$tag STOPPING: host available under ${MIN_AVAIL_GB} GB"; kill -TERM "$pid"; }
      fi ) &
    local w=$!
    wait "$pid"; local rc=$?
    kill "$w" 2>/dev/null; wait "$w" 2>/dev/null
    say "$tag rc=$rc in $((SECONDS-t0))s — $(mem_line)"
    return "$rc"
}
check() {
    local label="$1" tag="$2"; shift 2
    gate "check $label"
    local t0=$SECONDS
    choom -n 1000 -- "$@" > "$S/check_$tag.log" 2>&1
    local rc=$?
    say "CHECK $label $([ "$rc" -eq 0 ] && echo PASS || echo FAIL) (rc=$rc) in $((SECONDS-t0))s"
}

say "=== the five r08 rungs on $SPLIT, $(hostname) at $(git rev-parse --short HEAD) === $(mem_line)"
for f in patch_index.parquet splits.json class_weights.json patches_meta.json \
         patches_xct.bin patches_label.bin volumes.zarr holes holes.json; do
    [ -e "data/$SPLIT/$f" ] || { say "STOP NOTICE: data/$SPLIT/$f is missing"; exit 1; }
done
say "data/$SPLIT present"

for spec in $RUNGS; do
    RF=${spec%%:*}; rest=${spec#*:}; Z=${rest%%:*}; EXP=${rest#*:}
    gate "rf-$RF smoke"
    POREGEN_CUDA_MEM_FRACTION=$SMOKE_FRACTION choom -n 1000 -- \
        python scripts/analysis/vae_smoke_step.py --experiment "$EXP" > "$S/smoke_rf$RF.log" 2>&1
    SRC=$?
    say "SMOKE $EXP rc=$SRC — $(grep -E 'peak allocated|OOM at batch|FITS|NOT A MEMORY' "$S/smoke_rf$RF.log" | head -2 | tr '\n' ' ')"
    if [ "$SRC" -eq 1 ]; then say "rf-$RF SKIPPED: does not fit at its own batch (a real OOM)."; continue
    elif [ "$SRC" -ne 0 ]; then say "rf-$RF SKIPPED: smoke failed without measuring memory (rc=$SRC)."; continue; fi
    run_watched "r08 rf-$RF (z=$Z, $EXP) on $SPLIT" "rf$RF" \
        python scripts/train_vae.py run "$EXP" || { say "rf-$RF training failed — checks not run"; continue; }
    RUNG=$(ls -dt runs/vae/r08-run-*-z$Z-*-ds${SPLIT#split_}/ 2>/dev/null | head -1)
    [ -z "$RUNG" ] && { say "rf-$RF: no run directory found"; continue; }
    say "rf-$RF = $(basename "${RUNG%/}")"
    check "rf-$RF rung report" "rf${RF}_report" \
        python scripts/analysis/r08_rung_report.py --run "${RUNG%/}"
    check "rf-$RF calibration probe (dense panels)" "rf${RF}_calib" \
        python scripts/analysis/r08_calibration_probe.py --run "${RUNG%/}"
    check "rf-$RF vae_val_l1 harness (L1, texture, sharpness)" "rf${RF}_l1" \
        python scripts/analysis/vae_val_l1.py --run "${RUNG%/}" --split "$SPLIT" --n-batches 20
    check "rf-$RF recon figure" "rf${RF}_fig" \
        python scripts/analysis/vae_recon_figure.py --run "v4 rf-$RF=${RUNG%/}" --split "$SPLIT" \
        --out "runs/campaigns/09-r08-latent-sweep/figures/recon_v4_rf$RF"
done
say "BLOCK COMPLETE — $(mem_line)"
