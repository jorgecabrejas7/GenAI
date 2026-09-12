#!/usr/bin/env bash
# Everything the GPU does after the facedrop fine-tune exits.
#
# The exit test is RUN PROGRESS, not process absence: a training run that is
# stopped for two minutes to change something is not a finished run, and a
# watcher that cannot tell the difference starts hours of generation on a
# half-trained checkpoint. See scripts/_ldm_reached_total.py.
#
# downstream_utility is deliberately NOT here. It trains on generated volumes
# and the regeneration set is the author's open choice; a day of GPU on a set
# that may be replaced is the waste this ordering exists to avoid.
set -uo pipefail
REPO=/home/jorgecabrejas/Dev/GenAI
FD="runs/ldm/ldm06-run-0002-20260912-085535-z8-c128-bs256-lr2e-05/"
CAMP="$REPO/runs/campaigns/09-r08-latent-sweep"
FT_CAMP="$REPO/runs/campaigns/11-decoder-ft"
FT_GO="$REPO/runs/campaigns/decoder_ft_go"
TRIAL="$REPO/runs/campaigns/17-chunk-band-trial"
S="/tmp/claude-1001/-home-jorgecabrejas-Dev-GenAI/ce0b3db0-2aa0-4a97-9bc4-7bb0db078739/scratchpad"
LOG="$CAMP/post_ldm06.log"
TAIL_RUNGS=(reduction-factor-2 reduction-factor-64)
mkdir -p "$S" "$FT_CAMP"; cd "$REPO" || exit 1
say() { printf '%s  FDPOST %s\n' "$(date -Is)" "$*" >> "$LOG"; printf '%s  FDPOST %s\n' "$(date -Is)" "$*"; }

run_dir_for() {
    local want="r08/$1" d got
    for d in $(ls -dt "$REPO"/runs/vae/r08-run-*/ 2>/dev/null); do
        [ -f "${d}best.ckpt" ] || continue
        got=$(python - "$d" <<'PY'
import sys, json, pathlib
try:
    print(json.loads((pathlib.Path(sys.argv[1]) / "run_metadata.json").read_text())
          .get("experiment_id", ""))
except Exception:
    print("")
PY
)
        [ "$got" = "$want" ] && { printf '%s' "$d"; return 0; }
    done
    return 1
}

say "armed; watching $FD"
while true; do
    if pgrep -f "scripts/train_ldm\.py run ldm06/facedrop" >/dev/null 2>&1; then sleep 60; continue; fi
    if python "$REPO/scripts/_ldm_reached_total.py" "$FD"; then break; fi
    say "no training process, but the run has not reached total_steps — treating this as a pause"
    sleep 60
done
say "facedrop reached total_steps"

# ── 1. the trial on the new checkpoint ──────────────────────────────────────
for arm in fd_baseline fd_a2s; do
    say "TRIAL $arm start"
    python scripts/analysis/chunk_band_trial.py --model "$FD" --ckpt latest \
        --out "$TRIAL" --arms "$arm" > "$S/fd_${arm}.log" 2>&1
    say "TRIAL $arm done rc=$?"
done
say "SCORE start"
python scripts/analysis/chunk_band_trial_report.py --trial "$TRIAL" \
    --baseline "$REPO/runs/campaigns/12-eval-v4" > "$S/fd_score.log" 2>&1
say "SCORE done rc=$? — REPORT TO SUPERVISOR"

# ── 2. the mechanism check: does the per-face contrast go away? ─────────────
say "RIM mixed-set start"
python scripts/analysis/window_rim_test.py --model "$FD" --ckpt latest \
    --out "$REPO/runs/campaigns/16-window-rim/mixed_facedrop" --split val \
    --seed 101 --tiles 3 10 10 --neighbour-mode canvas --s-nb 1.0 \
    > "$S/fd_rim.log" 2>&1
say "RIM mixed-set done rc=$? — REPORT TO SUPERVISOR"

# ── 3. STOP. ────────────────────────────────────────────────────────────────
# The decoder fine-tune and the r08 rungs USED to run here, behind
# runs/campaigns/decoder_ft_go. They moved to scripts/final_queue.sh, which puts
# them after the campaign-18 regeneration the gate table has to be read on.
# The GATE ITSELF is not deleted: final_queue.sh stage 5 waits on the same file.
#
# Keeping them here as well would mean whichever script saw the gate file first
# ran the fine-tune, and this one would always win by hours.
say "RIM done — STOPPING. decoder-ft and the r08 rungs belong to final_queue.sh now."
say "FACEDROP POST COMPLETE — trial arms and rim test only, by design."
