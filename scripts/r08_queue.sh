#!/usr/bin/env bash
# Sequential runner for the r08 compression sweep.
#
# Why this exists: rf-8 finished at 09:51 and the next rung was not launched
# until 13:22 because a completion notification was missed. 3 h 31 m of GPU
# went nowhere. A human (or an agent) noticing a process exit is not a
# scheduling mechanism; this is.
#
# Rules:
#   - the next rung starts the moment the previous process exits, pass or fail
#   - a FAILED rung is logged and the chain continues; only a rung that cannot
#     be launched at all stops it
#   - CPU-able end-of-run reports run in parallel with the next rung
#   - every transition is appended to queue.log with an rc
#
# Usage:  bash scripts/r08_queue.sh
set -uo pipefail          # deliberately NOT -e: a failing rung must not kill the chain

REPO=/home/jorgecabrejas/Dev/GenAI
CAMP="$REPO/runs/campaigns/09-r08-latent-sweep"
SCRATCH=/tmp/claude-1001/-home-jorgecabrejas-Dev-GenAI/ce0b3db0-2aa0-4a97-9bc4-7bb0db078739/scratchpad
LOG="$CAMP/queue.log"
RUNGS=(reduction-factor-4 reduction-factor-2 reduction-factor-64)

mkdir -p "$CAMP" "$SCRATCH"
cd "$REPO" || exit 1

say() { printf '%s  %s\n' "$(date -Is)" "$*" >> "$LOG"; printf '%s  %s\n' "$(date -Is)" "$*"; }

newest_run() { ls -dt "$REPO"/runs/vae/r08-run-*/ 2>/dev/null | head -1; }

# CPU-only reports, in parallel with the next rung. tile-seam, sanity and the
# rung report need the GPU, which the next rung now owns — running them here
# would risk the OOM that already killed one job, so they are deferred to the
# gap after the chain and listed in queue.log as owed.
post_reports() {
    local run_dir="$1" tag="$2"
    [ -z "$run_dir" ] && { say "REPORTS $tag skipped: no run dir"; return; }
    [ -f "${run_dir}best.ckpt" ] || { say "REPORTS $tag skipped: no best.ckpt in $run_dir"; return; }
    say "REPORTS $tag start (calibration probe, CPU) $run_dir"
    ( CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=6 \
      python scripts/analysis/r08_calibration_probe.py \
        --run "$run_dir" --device cpu --per-bin 96 \
        --out-name "calibration_probe_r08_${tag}" \
        > "$SCRATCH/probe_${tag}.log" 2>&1
      say "REPORTS $tag done rc=$?" ) &
    say "OWED $tag: rung report, tile-seam, sanity (need GPU, deferred to the gap after the chain)"
}

say "QUEUE start; rungs: ${RUNGS[*]}"

# A rung may already be running (rf-32 was launched by hand). Wait it out
# rather than launching a second job onto the same GPU.
if pgrep -f "train_vae.py run r08/" >/dev/null 2>&1; then
    running=$(pgrep -af "train_vae.py run r08/" | head -1)
    say "WAIT for the rung already on the GPU: $running"
    while pgrep -f "train_vae.py run r08/" >/dev/null 2>&1; do sleep 30; done
    say "WAIT done"
    post_reports "$(newest_run)" "rf32"
fi

for exp in "${RUNGS[@]}"; do
    tag="${exp/reduction-factor-/rf}"
    say "START r08/$exp"
    python scripts/train_vae.py run "r08/$exp" > "$SCRATCH/r08_${tag}.log" 2>&1
    rc=$?
    say "DONE r08/$exp rc=$rc"
    [ "$rc" -ne 0 ] && say "FAIL r08/$exp rc=$rc — chain continues; see $SCRATCH/r08_${tag}.log"
    post_reports "$(newest_run)" "$tag"
done

say "START rung report --compare"
python scripts/analysis/r08_rung_report.py --compare > "$SCRATCH/r08_compare.log" 2>&1
say "DONE rung report --compare rc=$?"
say "QUEUE finished. Latent choice and store build wait for a decision."
