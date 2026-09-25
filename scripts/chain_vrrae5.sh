#!/usr/bin/env bash
# conv_k8's table and figure, then B to completion, then A0 full, rf-64, nbs.
#
# ORDER, as decided: conv_k8 -> B -> A0 -> rf-64 -> notebooks. B jumps ahead of
# A0 because it costs 1.2 h against A0's 10 h and it is the row that isolates
# the FC, while A0's own 400-step trial has already answered the question A0
# exists to ask (45.2 % of the error removed, at step 400).
#
# TWO DEFECTS OF v4 ARE FIXED HERE.
#
#   1  A CHAIN IS STOPPED BETWEEN STAGES, NEVER INSIDE ONE. v3 was signalled
#      while B was mid-step; the TERM reached the trainer through the process
#      group and B died 22 steps from the end, which skipped the U_f
#      finalization pass and cost it its table row and its figure entirely —
#      not 22 steps of training. So this chain reads a STOP file between
#      stages. Touch it and the chain stops at the next boundary with every
#      trainer intact. Nothing is ever signalled.
#
#   2  vae_recon_figure.py splits "label=path" on the FIRST "=", so a label
#      reading "r08 spatial, z=8" was parsed as label "r08 spatial, z" and
#      path "8)=/...". No label here contains "=".
#
# And V0 is vrrae-run-0001 (variant beta0, the vrrae_linear at beta 0, the
# 54.7 % row). v4 pointed at run-0005, which is the 400-step a0 trial.
set -uo pipefail
REPO=/home/jorgecabrejas/Dev/GenAI
S="/tmp/claude-1001/-home-jorgecabrejas-Dev-GenAI/ce0b3db0-2aa0-4a97-9bc4-7bb0db078739/scratchpad"
LOG="$REPO/runs/campaigns/chain.log"
STOP="$REPO/runs/campaigns/STOP_CHAIN"
C12="$REPO/runs/campaigns/12-eval-v4"; C18="$REPO/runs/campaigns/18-eval-v4-final"
C28="$REPO/runs/campaigns/28-vrrae-family"
FIG="$C28/figures"
MIN_AVAIL_GB=15
#: Per-process CUDA cap for the smoke probe. Without it "set -u" makes smoke()
#  return non-zero before it allocates anything, which reads as "does not fit" —
#  the same false negative that sent B to its fallback on 2026-09-24.
SMOKE_FRACTION=0.75
R08=$(ls -dt "$REPO"/runs/vae/r08-run-0004-*/ 2>/dev/null | head -1)
V0=$(ls -dt "$REPO"/runs/vae/vrrae-run-0001-*/ 2>/dev/null | head -1)
B_RUN=$(ls -dt "$REPO"/runs/vae/vrrae-run-0006-*/ 2>/dev/null | head -1)
CONV_RUN=$(ls -dt "$REPO"/runs/vae/vrrae-run-*vrrae_conv*/ 2>/dev/null | head -1)
cd "$REPO" || exit 1
say() { printf '%s  V5 %s\n' "$(date -Is)" "$*" | tee -a "$LOG"; }
mem_line() {
    local avail; avail=$(free -g | awk '/^Mem:/{print $7}')
    local cuda; cuda=$(python - <<'PY' 2>/dev/null || echo "n/a"
import torch
if torch.cuda.is_available():
    f,t = torch.cuda.mem_get_info(); print(f"{f/2**30:.1f}/{t/2**30:.1f} GiB free")
else: print("no cuda")
PY
)
    printf 'host available %s GB; cuda %s' "$avail" "$cuda"
}
#: The only way this chain stops early. Between stages, never inside one.
gate() {
    if [ -e "$STOP" ]; then
        say "STOP FILE PRESENT — halting cleanly before: $*"
        say "  every trainer is intact; remove $STOP and rerun to continue."
        exit 0
    fi
}
rate_line() {
    local run="$1" elapsed="$2"
    python - "$run" "$elapsed" <<'PY' 2>/dev/null || echo "rate n/a"
import json, os, sys
run, elapsed = sys.argv[1], float(sys.argv[2])
m = os.path.join(run, "metrics.jsonl")
steps = []
if os.path.exists(m):
    steps = [json.loads(l)["step"] for l in open(m) if l.strip()]
cfg = open(os.path.join(run, "resolved_config.yaml")).read()
total = next((int(l.split(":")[1]) for l in cfg.splitlines() if "total_steps:" in l), 0)
if not steps:
    print(f"no validation row yet; target {total}"); raise SystemExit
n = max(steps)
sps = elapsed / n
print(f"step {n}/{total}, {sps:.2f} s/step, ETA {(total-n)*sps/3600:.1f} h")
PY
}
run_watched() {
    local label="$1" tag="$2"; shift 2
    say "$label start"
    local t0=$SECONDS
    choom -n 1000 -- "$@" > "$S/$tag.log" 2>&1 &
    local pid=$!
    ( sleep 300
      if kill -0 "$pid" 2>/dev/null; then
          local avail; avail=$(free -g | awk '/^Mem:/{print $7}')
          local run; run=$(ls -dt "$REPO"/runs/vae/*/ 2>/dev/null | head -1)
          say "$tag +5min: $(mem_line) — $(rate_line "${run%/}" 300)"
          if [ "${avail:-99}" -lt "$MIN_AVAIL_GB" ]; then
              say "$tag STOPPING: host available ${avail} GB under ${MIN_AVAIL_GB} GB"
              kill -TERM "$pid" 2>/dev/null
          fi
      fi ) &
    local w=$!
    wait "$pid"; local rc=$?
    kill "$w" 2>/dev/null; wait "$w" 2>/dev/null
    local dt=$((SECONDS - t0))
    if [ "$rc" -ne 0 ]; then
        say "$tag rc=$rc FAILED after ${dt}s — $(mem_line) — see $S/$tag.log"
        say "STOP NOTICE: the block halts here on the first failure inside a run."
        exit "$rc"
    fi
    say "$tag rc=0 in ${dt}s — $(mem_line)"
    return 0
}
table() {
    choom -n 1000 -- python scripts/analysis/vrrae_family_table.py --out "$C28" \
        > "$S/tbl_$1.log" 2>&1
    say "campaign-28 table after $1 rc=$? — $(grep -oE '\*\*[0-9.]+%\*\*' "$C28/family_table.md" 2>/dev/null | tr '\n' ' ')"
}
#: No label may contain "=" — see the header.
figure() {
    local label="$1" run="$2" tag="$3"
    [ -n "$run" ] && [ -d "$run" ] || { say "FIGURE $tag SKIPPED: no run directory"; return 0; }
    choom -n 1000 -- python scripts/analysis/vae_recon_figure.py \
        --run "r08 spatial z8=${R08%/}" \
        --run "V0 flat KL off=${V0%/}" \
        --run "${label}=${run%/}" \
        --out "$FIG/recon_$tag" > "$S/fig_$tag.log" 2>&1
    say "FIGURE $tag rc=$? — $FIG/recon_$tag.png"
}
rung() { table "$1"; figure "$2" "$3" "$1"; }

mkdir -p "$FIG"
say "=== conv_k8 table, then B, then A0 === $(mem_line)"
say "r08=$(basename "${R08%/}")  V0=$(basename "${V0%/}")"

# ── 1. conv_k8, resumed from step 10000 ────────────────────────────────────
# The first attempt was OOM-killed at step ~11 700 when a CPU diagnostic asked
# for 275 GB beside it (40d99ff). choom -n 1000 makes the TRAINER the kernel's
# preferred victim, so the trainer dies and the offender survives — the reason
# nothing else may run in this pool while a card is up.
gate "conv_k8 resume"
CONV_RUN=$(ls -dt "$REPO"/runs/vae/vrrae-run-*vrrae_conv*/ 2>/dev/null | head -1)
CONV_CKPT=$(ls "${CONV_RUN%/}" | grep -E 'step[0-9]+\.ckpt$' | sort | tail -1)
if [ -n "$CONV_CKPT" ]; then
    run_watched "conv_k8 resume from $CONV_CKPT" vrrae_conv_k8_resume \
        python scripts/train_vae.py resume "$(basename "${CONV_RUN%/}")" "$CONV_CKPT"
else
    run_watched "conv_k8 from scratch" vrrae_conv_k8 \
        python scripts/train_vae.py run vrrae/conv_k8
    CONV_RUN=$(ls -dt "$REPO"/runs/vae/vrrae-run-*vrrae_conv*/ 2>/dev/null | head -1)
fi
rung conv_k8 "conv k8 per-cell" "${CONV_RUN%/}"

# ── 2. B to completion, so the U_f finalization pass runs ──────────────────
gate "B resume"
if [ -n "$B_RUN" ]; then
    run_watched "B resume from latest.ckpt (step 7000 of 7857)" vrrae_b_resume \
        python scripts/train_vae.py resume "$(basename "${B_RUN%/}")" latest.ckpt
    rung b "B flat FC1024 rank768" "${B_RUN%/}"
fi

# ── 3. A0 in full ──────────────────────────────────────────────────────────
gate "A0 full"
run_watched "A0 (vrrae/a0, beta 0)" vrrae_a0 python scripts/train_vae.py run vrrae/a0
A0RUN=$(ls -dt "$REPO"/runs/vae/vrrae-run-*vd1024*b0.000*/ 2>/dev/null | head -1)
rung a0 "A0 flat KL off full" "${A0RUN%/}"

# ── 4. rf-64 and the notebooks ─────────────────────────────────────────────
gate "rf-64"
RF64=$(ls -dt "$REPO"/runs/vae/r08-run-0012-*/ 2>/dev/null | head -1)
if [ -n "$RF64" ]; then
    run_watched "rf-64 resume from step 1000" rf64_resume \
        python scripts/train_vae.py resume "$(basename "${RF64%/}")" latest.ckpt
fi
for root in "$C18" "$C12"; do
    gate "notebook $(basename "$root")"
    run_watched "NOTEBOOK $(basename "$root")" "nb_$(basename "$root")" \
        python scripts/analysis/build_eval_v4_notebook.py --root "$root"
done
say "BLOCK COMPLETE — $(mem_line)"
