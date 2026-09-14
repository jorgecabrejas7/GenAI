#!/usr/bin/env bash
# The tail of the queue, reordered: the three new campaign-14 arms go BEFORE
# the r08 rungs, because paper-critical beats the outer rungs.
#
# `final_queue.sh` is already running and holds its own inode, so its stage 8
# cannot be reordered by editing it. This waits for its stage 7 to finish, stops
# it, and runs the rest itself:
#
#   A. campaign 14, arms real_8k / real_plus_synthetic_aug / synthetic_relabelled
#   B. r08 rf-2 and rf-64
#   C. the notebooks into campaigns 18 and 12
#
# Stopping `final_queue.sh` may catch its stage 8 a few seconds after it starts.
# That costs the first seconds of a VAE run and nothing else; the run is
# restarted from scratch in step B.
set -uo pipefail
REPO=/home/jorgecabrejas/Dev/GenAI
S="/tmp/claude-1001/-home-jorgecabrejas-Dev-GenAI/ce0b3db0-2aa0-4a97-9bc4-7bb0db078739/scratchpad"
QLOG="$REPO/runs/campaigns/final_queue.log"
LOG="$REPO/runs/campaigns/queue_tail.log"
C12="$REPO/runs/campaigns/12-eval-v4"
C14="$REPO/runs/campaigns/14-downstream-utility"
C18="$REPO/runs/campaigns/18-eval-v4-final"
NEW_ARMS=(real_8k real_plus_synthetic_aug synthetic_relabelled)
TAIL_RUNGS=(reduction-factor-2 reduction-factor-64)
cd "$REPO" || exit 1
say() { printf '%s  TAIL %s\n' "$(date -Is)" "$*" | tee -a "$LOG"; }

say "armed — waiting for final_queue.sh stage 7 to finish"
while ! grep -q "STAGE 7 done" "$QLOG" 2>/dev/null; do sleep 60; done
say "stage 7 finished"

# Stop the old queue by VERIFIED pid. A pattern kill here would match this
# script and the shell that launched it — that mistake has cost this project
# two aborted runs.
for pid in $(pgrep -f "bash .*final_queue\.sh" || true); do
    cmd=$(tr '\0' ' ' < "/proc/$pid/cmdline" 2>/dev/null || true)
    case "$cmd" in
        *final_queue.sh*) say "stopping final_queue.sh pid $pid"; kill -TERM "$pid" 2>/dev/null ;;
    esac
done
sleep 5
# and any VAE training it had just launched
for pid in $(pgrep -f "train_vae\.py run r08/reduction-factor" || true); do
    say "stopping a just-started rung, pid $pid"; kill -TERM "$pid" 2>/dev/null
done
sleep 5
say "old queue stopped"

# ── A. the three new campaign-14 arms ───────────────────────────────────────
# The relabelled arm needs label_onlypores.tif beside each case; written
# already by scripts/analysis/relabel_synthetic_onlypores.py.
say "A campaign 14 new arms start: ${NEW_ARMS[*]}"
nice -n 5 python scripts/analysis/downstream_utility.py \
    --campaign-root "$C18" --out "$C14" --arms "${NEW_ARMS[@]}" \
    > "$S/tail_downstream.log" 2>&1
say "A done rc=$? -> $C14"

# ── B. the two remaining VAE rungs ──────────────────────────────────────────
for exp in "${TAIL_RUNGS[@]}"; do
    say "B r08/$exp start"
    python scripts/train_vae.py run "r08/$exp" > "$S/tail_r08_${exp}.log" 2>&1
    say "B r08/$exp done rc=$?"
done

# ── C. the notebooks ────────────────────────────────────────────────────────
for root in "$C18" "$C12"; do
    say "C notebook into $(basename "$root") start"
    python scripts/analysis/build_eval_v4_notebook.py --root "$root" \
        > "$S/tail_nb_$(basename "$root").log" 2>&1
    say "C notebook into $(basename "$root") done rc=$?"
done
say "QUEUE TAIL COMPLETE"
