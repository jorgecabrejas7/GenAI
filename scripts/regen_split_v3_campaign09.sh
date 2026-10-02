#!/usr/bin/env bash
# Regenerate the split_v3 rf-8 files in campaign 09 that the split_v4 rf-8
# replaced on 2026-10-02: the rung report, the calibration probe and the latent
# std reference, all of r08-run-0004 on split_v3, into their original paths.
# The paper's six-rung table traces to these JSONs. --n 50000 is the row count
# the split_v3 store's channel_stats were measured on (the default 8000 is sized
# to run beside a trainer; this job runs with the card to itself).
#
# Waits for PID $1 (the GPU job ahead of it) to exit first: never two CUDA jobs.
#     setsid nohup bash scripts/regen_split_v3_campaign09.sh <pid> &
set -u
REPO=/home/jorgecabrejas/Dev/GenAI
S="/tmp/claude-1001/-home-jorgecabrejas-Dev-GenAI/ce0b3db0-2aa0-4a97-9bc4-7bb0db078739/scratchpad"
LOG="$S/regen_v3_c09.log"
RUN="$REPO/runs/vae/r08-run-0004-20260904-035134-archv2-conv_noattn_dualbranch_cls-z8-c32-bs128-lr2e-04-b0.050-fb0.1-klw0-schednone"
export POREGEN_SPLIT=split_v3
unset POREGEN_VAE_CHECKPOINT
say() { printf '%s  REGEN %s\n' "$(date -Is)" "$*" | tee -a "$LOG"; }
cd "$REPO" || exit 1

if [ -n "${1:-}" ]; then
    say "waiting for PID $1 to exit"
    while kill -0 "$1" 2>/dev/null; do sleep 30; done
fi
say "start: r08-run-0004 on split_v3"
fail=0
for job in \
    "rung_report|python scripts/analysis/r08_rung_report.py --run $RUN" \
    "probe|python scripts/analysis/r08_calibration_probe.py --run $RUN" \
    "latent_std|python scripts/analysis/latent_std_reference.py --store data/split_v3/latents_r08z8 --n 50000"; do
    tag=${job%%|*}; cmd=${job#*|}
    t0=$(date +%s)
    choom -n 1000 -- $cmd > "$S/regen_v3_$tag.log" 2>&1
    rc=$?
    say "$tag rc=$rc in $(( $(date +%s) - t0 ))s — see $S/regen_v3_$tag.log"
    [ $rc -eq 0 ] || fail=1
done
say "done (any failure: $fail)"
exit $fail
