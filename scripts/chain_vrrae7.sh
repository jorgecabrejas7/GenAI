#!/usr/bin/env bash
# (3) tail -> (i) campaign 27 microstructure -> conv_k8 at beta 1e-3 -> free.
#
# WHY THIS EXISTS. Three times now a hand-off between stages has been left to a
# message rather than a script, and the last one cost seventeen idle hours: an
# intention was reported in the same words an action would have been, and there
# was no process to notice the difference. Every stage below is declared here,
# in order, with its own rc line; the chain runs to the end without anyone
# reading a message.
#
# STOPPING. Never signal it. Touch runs/campaigns/STOP_CHAIN and it halts at
# the next stage boundary with every job intact. A trainer signalled mid-step
# lost 22 steps AND its whole U_f finalisation on 2026-09-27.
set -uo pipefail
REPO=/home/jorgecabrejas/Dev/GenAI
S="/tmp/claude-1001/-home-jorgecabrejas-Dev-GenAI/ce0b3db0-2aa0-4a97-9bc4-7bb0db078739/scratchpad"
LOG="$REPO/runs/campaigns/chain.log"
STOP="$REPO/runs/campaigns/STOP_CHAIN"
C27="$REPO/runs/campaigns/27-budget-matched-40k"
C28="$REPO/runs/campaigns/28-vrrae-family"
FIG="$C28/figures"
MIN_AVAIL_GB=15
SMOKE_FRACTION=0.75
#: The (3) job this chain waits on, if it is still running. Empty = nothing.
WAIT_PID="${WAIT_PID:-}"
#: ldm06/base at the phi-only baseline's own budget — run-0001, NOT run-0002,
#: which is the 15k facedrop fine-tune and has no step 40000 at all.
LDM06_BASE=$(ls -dt "$REPO"/runs/ldm/ldm06-run-0001-*/ 2>/dev/null | head -1)
R08=$(ls -dt "$REPO"/runs/vae/r08-run-0004-*/ 2>/dev/null | head -1)
V0=$(ls -dt "$REPO"/runs/vae/vrrae-run-0001-*/ 2>/dev/null | head -1)
cd "$REPO" || exit 1
say() { printf '%s  V7 %s\n' "$(date -Is)" "$*" | tee -a "$LOG"; }
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
gate() {
    if [ -e "$STOP" ]; then
        say "STOP FILE PRESENT — halting cleanly before: $*"
        say "  every job is intact; remove $STOP and rerun to continue."
        exit 0
    fi
}
step() {
    local label="$1" tag="$2"; shift 2
    say "$label start"
    local t0=$SECONDS
    choom -n 1000 -- "$@" > "$S/v7_$tag.log" 2>&1
    local rc=$?
    say "$tag rc=$rc in $((SECONDS-t0))s — $(mem_line)"
    if [ "$rc" -ne 0 ]; then
        say "STOP NOTICE: $tag failed — see $S/v7_$tag.log"
        exit "$rc"
    fi
}
run_watched() {
    local label="$1" tag="$2"; shift 2
    say "$label start"
    local t0=$SECONDS
    choom -n 1000 -- "$@" > "$S/v7_$tag.log" 2>&1 &
    local pid=$!
    ( sleep 300
      if kill -0 "$pid" 2>/dev/null; then
          local avail; avail=$(free -g | awk '/^Mem:/{print $7}')
          say "$tag +5min: $(mem_line)"
          if [ "${avail:-99}" -lt "$MIN_AVAIL_GB" ]; then
              say "$tag STOPPING: host available ${avail} GB under ${MIN_AVAIL_GB} GB"
              kill -TERM "$pid" 2>/dev/null
          fi
      fi ) &
    local w=$!
    wait "$pid"; local rc=$?
    kill "$w" 2>/dev/null; wait "$w" 2>/dev/null
    say "$tag rc=$rc in $((SECONDS-t0))s — $(mem_line)"
    if [ "$rc" -ne 0 ]; then
        say "STOP NOTICE: $tag failed — see $S/v7_$tag.log"
        exit "$rc"
    fi
}
smoke() {
    local exp="$1" tag="$2"
    say "SMOKE $exp (cap $SMOKE_FRACTION)"
    POREGEN_CUDA_MEM_FRACTION=$SMOKE_FRACTION \
        choom -n 1000 -- python scripts/analysis/vae_smoke_step.py --experiment "$exp" \
        > "$S/v7_smoke_$tag.log" 2>&1
    local rc=$?
    say "SMOKE $exp rc=$rc — $(grep -E 'peak allocated|OOM at batch|FITS|NOT A MEMORY' "$S/v7_smoke_$tag.log" | head -2 | tr '\n' ' ')"
    return $rc
}

say "=== v7: campaign 27 microstructure, then conv_k8 at beta 1e-3 === $(mem_line)"

# ── 0. adopt the running decoder fine-tune ─────────────────────────────────
if [ -n "$WAIT_PID" ]; then
    say "waiting on the decoder fine-tune (pid $WAIT_PID)"
    while kill -0 "$WAIT_PID" 2>/dev/null; do sleep 30; done
    say "decoder fine-tune ended — $(mem_line)"
fi

# ── 1. campaign 27's microstructure, so the phi-only comparison is matched ──
gate "campaign 27 microstructure"
if [ -z "$LDM06_BASE" ]; then
    say "STOP NOTICE: no ldm06-run-0001 found; the budget-matched row needs it."
    exit 1
fi
say "ldm06 base = $(basename "${LDM06_BASE%/}")  step 40000"
run_watched "c27 microstructure generate (9 cases, DDIM-200, 192^3)" c27_gen \
    python -m poregen.eval_v4.cli generate microstructure \
    --model "${LDM06_BASE%/}" --ckpt 40000 --weights ema --out "$C27"
step "c27 measure" c27_measure \
    python -m poregen.eval_v4.cli measure microstructure --root "$C27"
step "c27 report" c27_report \
    python -m poregen.eval_v4.cli report --root "$C27"

# ── 2. the untried beta, the arm that can answer the posterior question ────
gate "conv_k8_b1e-3"
if smoke vrrae/conv_k8_b1e-3 conv_b1e3; then
    run_watched "conv_k8 beta 1e-3 (per-cell, k*=8, batch 128)" conv_b1e3 \
        python scripts/train_vae.py run vrrae/conv_k8_b1e-3
    step "campaign-28 table" c28_table \
        python scripts/analysis/vrrae_family_table.py --out "$C28"
    CONV=$(ls -dt "$REPO"/runs/vae/vrrae-run-*vrrae_conv*b0.001*/ 2>/dev/null | head -1)
    [ -z "$CONV" ] && CONV=$(ls -dt "$REPO"/runs/vae/vrrae-run-*vrrae_conv*/ 2>/dev/null | head -1)
    mkdir -p "$FIG"
    step "recon figure" c28_figure \
        python scripts/analysis/vae_recon_figure.py \
        --run "r08 spatial z8=${R08%/}" \
        --run "V0 flat KL off=${V0%/}" \
        --run "conv k8 beta 1e-3=${CONV%/}" \
        --out "$FIG/recon_conv_b1e3"
else
    say "conv_k8_b1e-3 SKIPPED: it does not fit at batch 128 under the cap."
    say "STOP NOTICE: the batch is r08's on purpose and no smaller one is invented here."
    exit 1
fi

say "BLOCK COMPLETE — the card is free — $(mem_line)"
