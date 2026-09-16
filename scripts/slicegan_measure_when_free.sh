#!/usr/bin/env bash
# Measure and report the SliceGAN baseline as soon as the downstream arms let
# go of the card.
#
# WHY THIS IS A SEPARATE SCRIPT. chain.sh already ran its stage 3, and both
# halves of it failed silently: `slicegan` was not a registered assessment, so
# `eval_v4 measure slicegan` exited on an argparse error and report then failed
# on the missing results.json. The chain logged rc=2 and rc=1 and carried on.
# The chain is STILL RUNNING, and bash reads a script by file offset, so editing
# it in place is undefined — this takes the owed work instead.
#
# It does NOT stop the chain. The measure reads campaign volumes and never the
# patch store, so it cannot deadlock a dataloader the way store I/O did to rf-2;
# and its only GPU work is the FID extractor, which is minutes against stage 5's
# 24-hour cap. Killing and restarting a training run to save that would cost far
# more than it saves.
#
# THE WAIT IS AGAINST THIS RUN AND NOT AGAINST THE LOG. chain.log is APPENDED
# to by every chain that has ever run, so `grep "STAGE 4 done"` matched a line
# from a previous run and this script fired at once, beside the arms it was
# written to wait for. The offset is taken before the wait, and only bytes after
# it are read. The process check is the second half of the same answer: a line
# without a finished process, or a finished process without a line, is not the
# end of stage 4.
set -uo pipefail
REPO=/home/jorgecabrejas/Dev/GenAI
S="/tmp/claude-1001/-home-jorgecabrejas-Dev-GenAI/ce0b3db0-2aa0-4a97-9bc4-7bb0db078739/scratchpad"
LOG="$REPO/runs/campaigns/chain.log"
C22="$REPO/runs/campaigns/22-slicegan-baseline"
cd "$REPO" || exit 1
say() { printf '%s  SGMEAS %s\n' "$(date -Is)" "$*" | tee -a "$LOG"; }

#: Bytes already in the log. Everything before this belongs to an earlier chain.
OFFSET=$(( $(stat -c %s "$LOG" 2>/dev/null || echo 0) + 1 ))

running() {  # downstream_utility alive, excluding this script and its children
    ps -eo pid,args | awk -v self="$$" '
        $1 != self && /downstream_utility\.py/ && !/awk/ && !/slicegan_measure_when_free/ {n++}
        END {exit !n}'
}

say "armed at log offset $OFFSET; waiting for STAGE 4 of THIS chain"
while true; do
    if tail -c "+$OFFSET" "$LOG" 2>/dev/null | grep -q "STAGE 4 done"; then
        running || break
        say "stage 4 logged done but downstream_utility is still up — still waiting"
    fi
    sleep 60
done
say "STAGE 4 finished; measuring the baseline"

python -m poregen.eval_v4.cli measure slicegan --root "$C22" \
    > "$S/sg_measure.log" 2>&1
say "measure rc=$?"
python -m poregen.eval_v4.cli report --root "$C22" --assessment slicegan \
    > "$S/sg_report.log" 2>&1
say "report rc=$? -> $C22/slicegan/findings.md"
