#!/usr/bin/env bash
# One row a minute of host memory-management rates beside a trainer, so a
# slowdown can be read against compaction and reclaim work (the 2026-10-14
# test: GPU low-power phases followed kcompactd scan work, and a MemoryHigh
# cgroup cap on the trainer brought 5 s/step back to 2.6 s/step).
#   bash scripts/analysis/vmrate_sampler.sh <trainer pid> <run dir> >> out.tsv
# Columns: time, last logged step, GPU W and util, MemFree and Cached (kB),
# the 60 s deltas of nine /proc/vmstat counters (in /proc/vmstat order),
# kcompactd0 CPU % over the minute, the trainer's VmSwap (kB), and the
# trainer cgroup's memory.current (bytes) and memory.events "high" count.
# Exits when the trainer exits.
set -u
PID=$1; RUN=$2
K="pgscan_kswapd pgsteal_kswapd compact_migrate_scanned compact_isolated pgmigrate_success compact_stall workingset_refault_file pswpin pswpout"
KC=$(pgrep -x kcompactd0 | head -1)
CG=/sys/fs/cgroup$(cut -d: -f3 "/proc/$PID/cgroup")
HZ=$(getconf CLK_TCK)
snap() { awk -v k=" $K " 'index(k, " "$1" ") {printf "%s ", $2}' /proc/vmstat; }
kc() { awk '{print $14 + $15}' "/proc/$KC/stat"; }
printf 'time\tstep\tgpu_w\tgpu_util\tmemfree_kb\tcached_kb\t%skcompactd_pct\ttrainer_vmswap_kb\tcg_mem_current\tcg_events_high\n' \
    "$(awk -v k=" $K " 'index(k, " "$1" ") {printf "%s\t", $1}' /proc/vmstat)"
prev=($(snap)); pk=$(kc)
while kill -0 "$PID" 2>/dev/null; do
    sleep 60
    cur=($(snap)); ck=$(kc)
    d=""; for i in "${!cur[@]}"; do d+="$(( cur[i] - prev[i] ))\t"; done
    step=$(tail -n 1 "$RUN/log.jsonl" 2>/dev/null | grep -o '"step": [0-9]*' | grep -o '[0-9]*')
    g=$(nvidia-smi --query-gpu=power.draw,utilization.gpu --format=csv,noheader,nounits 2>/dev/null | tr -d ' ' | tr ',' '\t')
    m=$(awk '/^MemFree:/{f=$2} /^Cached:/{c=$2} END{printf "%s\t%s", f, c}' /proc/meminfo)
    printf '%s\t%s\t%s\t%s\t%b%s\t%s\t%s\t%s\n' "$(date +%FT%T)" "$step" "$g" "$m" "$d" \
        "$(( (ck - pk) * 100 / (60 * HZ) ))" \
        "$(awk '/^VmSwap:/{print $2}' "/proc/$PID/status" 2>/dev/null)" \
        "$(cat "$CG/memory.current" 2>/dev/null)" \
        "$(awk '$1 == "high" {print $2}' "$CG/memory.events" 2>/dev/null)"
    prev=("${cur[@]}"); pk=$ck
done
