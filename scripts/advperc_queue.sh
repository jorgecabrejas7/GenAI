#!/usr/bin/env bash
# The adversarial + perceptual reduction-factor table (r08/advperc-rfN), one
# machine's queue: smoke, train inside a capped user scope, then the four
# rung checks on best.ckpt, then the next rung.
#
#     RUNGS="16 32 64" setsid nohup bash scripts/advperc_queue.sh &
#     FAMILY=advperc-ldm RUNGS="16 32 64" setsid nohup bash scripts/advperc_queue.sh &
#     touch runs/campaigns/STOP_ADVPERC     # halt cleanly before the next stage
#
# FAMILY (default advperc) picks the config family r08/<FAMILY>-rfN; every
# file name below carries <FAMILY>-rfN, so families never overwrite each other.
#
# Everything is named advperc-rfN (logs chain_logs/advperc-rfN.log,
# smoke_/check_/vmrate_ files, campaign-09 dirs r08_advperc-rfN-dsv4), so no
# rf-N log or campaign directory of the earlier rungs is overwritten — which
# chain_rungs_remote.sh would do, as it names a rung by its reduction factor.
#
# The trainer runs in a systemd user scope with MemoryHigh=60G and no swap:
# on the 2026-10-14 remote test the cap held the trainer's page cache and
# took a resumed rf-16 from about 5 s/step back to 2.6 s/step.
# A trainer that exits non-zero stops the queue: resume it by hand inside the
# same scope, run its checks, then start the queue again on the rest.
set -u
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")" && cd .. && pwd)"
cd "$REPO" || exit 1
export POREGEN_SPLIT=split_v4
SPLIT=$POREGEN_SPLIT
S="$REPO/runs/campaigns/chain_logs"
LOG="$S/advperc_queue_$(hostname -s).log"
STOP="$REPO/runs/campaigns/STOP_ADVPERC"
RUNGS="${RUNGS:?set RUNGS, e.g. RUNGS=\"16 32 64\"}"
FAMILY="${FAMILY:-advperc}"
mkdir -p "$S"

say() { printf '%s  ADVPERC %s\n' "$(date -Is)" "$*" | tee -a "$LOG"; }
gate() { [ -e "$STOP" ] && { say "STOP FILE PRESENT, halting before: $*"; exit 0; }; }
check() {
    local label="$1" tag="$2"; shift 2
    gate "check $label"
    local t0=$SECONDS
    choom -n 1000 -- "$@" > "$S/check_$tag.log" 2>&1
    local rc=$?
    say "CHECK $label $([ "$rc" -eq 0 ] && echo PASS || echo FAIL) (rc=$rc) in $((SECONDS - t0))s"
}

for N in $RUNGS; do
    EXP="r08/$FAMILY-rf$N"; T="$FAMILY-rf$N"
    gate "$T smoke"
    POREGEN_CUDA_MEM_FRACTION=0.8 choom -n 1000 -- \
        python scripts/analysis/vae_smoke_step.py --experiment "$EXP" > "$S/smoke_$T.log" 2>&1
    src=$?
    say "SMOKE $EXP rc=$src: $(grep -E 'FITS|peak allocated|OOM at batch|NOT A MEMORY' "$S/smoke_$T.log" | head -2 | tr '\n' ' ')"
    grep -q 'FITS' "$S/smoke_$T.log" || { say "$T does not FIT, queue stopped"; exit 1; }

    gate "$T training"
    t0=$SECONDS; tstart=$(date +%s)
    systemd-run --user --scope -q -p MemoryHigh=60G -p MemorySwapMax=0 \
        choom -n 1000 -- python scripts/train_vae.py run "$EXP" > "$S/$T.log" 2>&1 &
    pid=$!
    RUN=""
    for _ in $(seq 120); do
        for d in $(ls -dt runs/vae/r08-run-*-dsv4 2>/dev/null | head -5); do
            [ "$(stat -c %Y "$d")" -ge "$tstart" ] && grep -q "\"experiment_id\": \"$EXP\"" "$d/run_metadata.json" 2>/dev/null && { RUN=$d; break; }
        done
        [ -n "$RUN" ] && break
        kill -0 "$pid" 2>/dev/null || break
        sleep 5
    done
    cg=/sys/fs/cgroup$(cut -d: -f3 "/proc/$pid/cgroup" 2>/dev/null)
    say "$T training start pid=$pid run=${RUN:-NOT FOUND} scope=$(basename "$cg") memory.high=$(cat "$cg/memory.high" 2>/dev/null) swap.max=$(cat "$cg/memory.swap.max" 2>/dev/null)"
    [ -n "$RUN" ] && nice -n 19 bash scripts/analysis/vmrate_sampler.sh "$pid" "$RUN" > "$S/vmrate_$T.tsv" 2>&1 &
    wait "$pid"; rc=$?
    echo "rc=$rc" >> "$S/$T.log"
    say "$T rc=$rc in $((SECONDS - t0))s"
    [ "$rc" -eq 0 ] && [ -n "$RUN" ] || { say "$T training did not finish cleanly, queue stopped"; exit 1; }

    check "$T rung report" "${T}_report" python scripts/analysis/r08_rung_report.py --run "$RUN" --checkpoint best.ckpt
    check "$T calibration probe" "${T}_calib" python scripts/analysis/r08_calibration_probe.py --run "$RUN" --checkpoint best.ckpt
    check "$T vae_val_l1 harness" "${T}_l1" python scripts/analysis/vae_val_l1.py --run "$RUN" --ckpt best.ckpt \
        --split "$SPLIT" --n-batches 20 --out "$S/check_${T}_l1.json"
    check "$T recon figure" "${T}_fig" python scripts/analysis/vae_recon_figure.py --run "v4 $T=$RUN" \
        --split "$SPLIT" --out "runs/campaigns/09-r08-latent-sweep/figures/recon_v4_$T"
    say "$T DONE"
done
say "QUEUE COMPLETE ($FAMILY: $RUNGS)"
