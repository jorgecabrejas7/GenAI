#!/usr/bin/env bash
# One line a minute of host memory pressure and a trainer's state, so a stall
# leaves a timeline (the 2026-10-07 VAE stall left none: swap, page cache and
# the trainer's own swap at the moment it hung are unknown).
#   bash scripts/analysis/host_pressure_log.sh <trainer pid> <run dir> > log.tsv
set -u
PID=$1; RUN=$2
printf 'time\tswap_used_mb\tavail_mb\tcached_mb\tdirty_mb\tpswpin\tpswpout\ttrainer_swap_mb\ttrainer_rss_mb\tmain_state\tmain_utime\tgpu_util\tlog_mtime\n'
while kill -0 "$PID" 2>/dev/null; do
    read -r st ut < <(awk '{print $3, $14}' /proc/$PID/stat)
    printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' "$(date +%FT%T)" \
        "$(awk '/^SwapTotal/{t=$2}/^SwapFree/{f=$2}END{print int((t-f)/1024)}' /proc/meminfo)" \
        "$(awk '/^MemAvailable/{print int($2/1024)}' /proc/meminfo)" \
        "$(awk '/^Cached:/{print int($2/1024)}' /proc/meminfo)" \
        "$(awk '/^Dirty:/{print int($2/1024)}' /proc/meminfo)" \
        "$(awk '/^pswpin/{print $2}' /proc/vmstat)" "$(awk '/^pswpout/{print $2}' /proc/vmstat)" \
        "$(awk '/^VmSwap/{print int($2/1024)}' /proc/$PID/status)" \
        "$(awk '/^VmRSS/{print int($2/1024)}' /proc/$PID/status)" \
        "$st" "$ut" \
        "$(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits 2>/dev/null)" \
        "$(stat -c %Y "$RUN/log.jsonl" 2>/dev/null)"
    sleep 60
done
