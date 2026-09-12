#!/usr/bin/env bash
# Stop facedrop_post.sh at the rim-test boundary, so it can never run the
# decoder fine-tune out of order.
#
# WHY THIS EXISTS AND AN EDIT WOULD NOT DO. facedrop_post.sh is ALREADY RUNNING.
# Editing the file gives the new text a new inode; the running bash keeps
# reading the old one, so the running instance would still reach its
# decoder-ft stage the moment runs/campaigns/decoder_ft_go appeared. The author's
# order puts that stage after the campaign-18 regeneration, hours later. The
# only way to bind the RUNNING process is to stop it.
#
# It is stopped at the rim-test boundary and not now, because fd_a2s and the rim
# test are stages the queue still wants and that script is the thing running
# them. After "RIM mixed-set done" everything it would do next has moved to
# scripts/final_queue.sh.
#
# The PID is resolved and then VERIFIED against /proc before anything is
# signalled. A pattern kill here would match this script and the shell that
# launched it — that mistake has cost this project two aborted runs.
set -uo pipefail
REPO=/home/jorgecabrejas/Dev/GenAI
FD_LOG="$REPO/runs/campaigns/09-r08-latent-sweep/post_ldm06.log"
LOG="$REPO/runs/campaigns/final_queue.log"
say() { printf '%s  STOPFD %s\n' "$(date -Is)" "$*" | tee -a "$LOG"; }

say "armed — will stop facedrop_post.sh once the rim test is done"
while ! grep -q "RIM mixed-set done" "$FD_LOG" 2>/dev/null; do sleep 30; done
say "rim test finished"

for pid in $(pgrep -f "bash scripts/facedrop_post\.sh" || true); do
    [ "$pid" = "$$" ] && continue
    cmd=$(tr '\0' ' ' < "/proc/$pid/cmdline" 2>/dev/null || true)
    case "$cmd" in
        *facedrop_post.sh*)
            case "$cmd" in
                *stop_fdpost_after_rim*) continue ;;   # never this watcher
            esac
            say "stopping pid $pid: $cmd"
            kill -TERM "$pid" 2>/dev/null
            for _ in $(seq 1 10); do
                kill -0 "$pid" 2>/dev/null || break
                sleep 1
            done
            kill -0 "$pid" 2>/dev/null && { say "pid $pid ignored TERM, sending KILL"; kill -KILL "$pid" 2>/dev/null; }
            ;;
    esac
done
sleep 2
if pgrep -f "bash scripts/facedrop_post\.sh" >/dev/null 2>&1; then
    say "WARNING facedrop_post.sh is STILL RUNNING — do not create decoder_ft_go"
else
    say "facedrop_post.sh stopped; the decoder-ft stage now belongs to final_queue.sh only"
fi
