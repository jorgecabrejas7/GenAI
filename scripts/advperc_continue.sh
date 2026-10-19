#!/usr/bin/env bash
# Continue finished advperc-ldm rungs to a fixed step, one after another on
# this machine (one CUDA job per machine). Author, 2026-10-19: "Keep on
# training up to step 30[000] for all of them": the discriminator phase was
# cut at 3-4k steps by the L1-driven early stop, so each rung resumes from
# latest.ckpt (AE, discriminator and both optimisers) with early stopping
# off. Order on each machine set by the author (rf-8 first on the remote).
#
#     RUNS="run-0020 run-0021 run-0022" TOTAL=30000 setsid nohup bash scripts/advperc_continue.sh &
#     touch runs/campaigns/STOP_CONTINUE      # halt before the next rung
set -u
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")" && cd .. && pwd)"
cd "$REPO" || exit 1
export POREGEN_SPLIT=split_v4
RUNS="${RUNS:?set RUNS, e.g. RUNS=\"run-0025 run-0026\"}"; TOTAL="${TOTAL:-30000}"
S="$REPO/runs/campaigns/chain_logs"; LOG="$S/advperc_continue_$(hostname -s).log"
STOP="$REPO/runs/campaigns/STOP_CONTINUE"
mkdir -p "$S"
say() { printf '%s  CONTINUE %s\n' "$(date -Is)" "$*" | tee -a "$LOG"; }
for R in $RUNS; do
    [ -e "$STOP" ] && { say "STOP FILE PRESENT, halting before $R"; exit 0; }
    D=$(ls -d runs/vae/r08-"$R"-* 2>/dev/null | head -1)
    [ -n "$D" ] || { say "$R NOT FOUND"; continue; }
    say "$R start: resume latest.ckpt to step $TOTAL, early stopping off ($D)"
    t0=$SECONDS
    choom -n 1000 -- python scripts/train_vae.py resume "$D" latest.ckpt \
        --total-steps "$TOTAL" --no-early-stopping > "$S/continue_$R.log" 2>&1
    say "$R rc=$? in $((SECONDS - t0))s"
done
say "QUEUE COMPLETE ($RUNS)"
