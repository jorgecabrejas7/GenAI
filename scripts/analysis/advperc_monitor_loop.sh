#!/usr/bin/env bash
# Refresh the training monitor of the running r08/<FAMILY>-rfN rungs on both
# machines every INTERVAL seconds (default 1800). Author request 2026-10-15:
# "more metrics tables and visual plots for checking how the training goes".
#
#     FAMILY=advperc-ldm DISC_START=15000 setsid nohup bash scripts/analysis/advperc_monitor_loop.sh &
#     touch runs/campaigns/STOP_MONITOR      # stop it
#
# Each pass: mirror the remote runs' log.jsonl, metrics.jsonl and every
# step checkpoint (41 MB each) into runs/remote/, then run
# vae_training_monitor.py on CPU in a 16 GiB scope at nice 19 with 4 threads
# (a CPU job beside a trainer must not starve it). The SN-LSGAN runs 0019 and
# 0024 are drawn as dashed references. Output: runs/campaigns/33-advperc-ldm/monitor/.
set -u
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")" && cd ../.. && pwd)"
cd "$REPO" || exit 1
FAMILY="${FAMILY:-advperc-ldm}"; DISC_START="${DISC_START:-15000}"; INTERVAL="${INTERVAL:-1800}"
REMOTE=jorge@192.168.8.92; MIRROR="runs/remote/$REMOTE/runs/vae"
OUT=runs/campaigns/33-advperc-ldm/monitor; STOP=runs/campaigns/STOP_MONITOR
mkdir -p "$OUT" "$MIRROR"
runs_of() {  # run dirs under $1 whose experiment_id starts with r08/$FAMILY-
    grep -l "\"experiment_id\": \"r08/$FAMILY-rf" "$1"/r08-run-*/run_metadata.json 2>/dev/null | xargs -r -n1 dirname
}
while [ ! -e "$STOP" ]; do
    for d in $(ssh -o ConnectTimeout=10 "$REMOTE" "cd ~/Dev/GenAI && grep -l '\"experiment_id\": \"r08/$FAMILY-rf' runs/vae/r08-run-*/run_metadata.json 2>/dev/null | xargs -r -n1 dirname | xargs -r -n1 basename"); do
        mkdir -p "$MIRROR/$d"
        rsync -a --include='log.jsonl' --include='metrics.jsonl' --include='resolved_config.yaml' \
            --include='run_metadata.json' --include='*_step*.ckpt' --exclude='*' \
            "$REMOTE:Dev/GenAI/runs/vae/$d/" "$MIRROR/$d/"
    done
    args=()
    for r in $(runs_of runs/vae) $(runs_of "$MIRROR"); do
        exp=$(grep -o "\"experiment_id\": \"[^\"]*\"" "$r/run_metadata.json" | cut -d'"' -f4)
        id=$(basename "$r" | cut -d- -f2,3)
        args+=(--run "${exp#r08/} ($id)=$r")
    done
    if [ ${#args[@]} -gt 0 ]; then
        env CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=4 POREGEN_SPLIT=split_v4 \
            systemd-run --user --scope -q -p MemoryMax=16G nice -n 19 \
            python scripts/analysis/vae_training_monitor.py "${args[@]}" \
            --ref "SN-LSGAN rf-16 (run-0019)=$(ls -d runs/vae/r08-run-0019-*)" \
            --ref "SN-LSGAN rf-8 (run-0024)=$(ls -d "$MIRROR"/r08-run-0024-*)" \
            --disc-start "$DISC_START" --recon --out "$OUT" > "$OUT/last_pass.log" 2>&1
        echo "$(date -Is) pass rc=$? runs=${#args[@]}" >> "$OUT/passes.log"
    fi
    sleep "$INTERVAL"
done
