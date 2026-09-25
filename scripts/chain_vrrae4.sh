#!/usr/bin/env bash
# conv_k8, then A0 in full, then rf-64 and the notebooks.
#
# WHY A NEW FILE. chain_vrrae3.sh was RUNNING when the order changed, and bash
# reads a script by byte offset while it executes it — editing a live script
# corrupts the run. So v3 was stopped between stages instead, with B's training
# process left alive as an orphan, and the remaining stages are re-declared
# here in the order the author asked for:
#
#   B (already running, adopted below) -> conv_k8 -> A0 full -> rf-64 -> nbs
#
# conv_k8 goes BEFORE A0 because it is the arm that answers the question. Every
# flat rung — A, B, V0, vrrae03, vrrae04 — asks the decoder to rebuild 64^3
# voxels from one vector, and the per-cell model is the first that does not.
# Its k*=8 over 16^3 cells is 32 768 numbers per patch, r08's count exactly, so
# for the first time the comparison is like-for-like.
#
# THE VERDICT IS THE FIGURE, NOT THE L1. Every rung here therefore also draws
# vae_recon_figure.py next to r08 and V0 on the harness's held-out patches.
set -uo pipefail
REPO=/home/jorgecabrejas/Dev/GenAI
S="/tmp/claude-1001/-home-jorgecabrejas-Dev-GenAI/ce0b3db0-2aa0-4a97-9bc4-7bb0db078739/scratchpad"
LOG="$REPO/runs/campaigns/chain.log"
C12="$REPO/runs/campaigns/12-eval-v4"; C18="$REPO/runs/campaigns/18-eval-v4-final"
C28="$REPO/runs/campaigns/28-vrrae-family"
FIG="$C28/figures"
MIN_AVAIL_GB=15
SMOKE_FRACTION=0.75
#: B's orphaned trainer, adopted from v3. Empty means "nothing to wait for".
B_PID="${B_PID:-}"
B_RUN="${B_RUN:-}"
R08=$(ls -dt "$REPO"/runs/vae/r08-run-0004-*/ 2>/dev/null | head -1)
V0=$(ls -dt "$REPO"/runs/vae/vrrae-run-0005-*/ 2>/dev/null | head -1)
cd "$REPO" || exit 1
say() { printf '%s  V4 %s\n' "$(date -Is)" "$*" | tee -a "$LOG"; }
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
#: Steps done and seconds per step, read from the run's own metrics stream.
#  This is what turns "it is running" into "it finishes at 04:10", which is the
#  number the card is actually planned against.
rate_line() {
    local tag="$1" elapsed="$2"
    python - "$tag" "$elapsed" <<'PY' 2>/dev/null || echo "rate n/a"
import glob, json, os, sys, time
tag, elapsed = sys.argv[1], float(sys.argv[2])
runs = sorted(glob.glob("runs/vae/*/metrics.jsonl"), key=os.path.getmtime)
if not runs:
    print("rate n/a"); raise SystemExit
rows = [json.loads(l) for l in open(runs[-1]) if l.strip()]
steps = [r["step"] for r in rows if "step" in r]
if not steps:
    print("no step yet"); raise SystemExit
n = max(steps)
cfg = open(os.path.join(os.path.dirname(runs[-1]), "resolved_config.yaml")).read()
total = next((int(l.split(":")[1]) for l in cfg.splitlines() if "total_steps:" in l), 0)
sps = elapsed / n if n else 0.0
eta = (total - n) * sps / 3600.0
print(f"step {n}/{total}, {sps:.2f} s/step, ETA {eta:.1f} h")
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
          say "$tag +5min: $(mem_line) — $(rate_line "$tag" 300)"
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
smoke() {
    local exp="$1" tag="$2"
    say "SMOKE $exp (cap $SMOKE_FRACTION)"
    POREGEN_CUDA_MEM_FRACTION=$SMOKE_FRACTION \
        choom -n 1000 -- python scripts/analysis/vae_smoke_step.py --experiment "$exp" \
        > "$S/smoke_$tag.log" 2>&1
    local rc=$?
    say "SMOKE $exp rc=$rc — $(grep -E 'peak allocated|OOM at batch|FITS|NOT A MEMORY' "$S/smoke_$tag.log" | head -2 | tr '\n' ' ')"
    return $rc
}
table() {
    choom -n 1000 -- python scripts/analysis/vrrae_family_table.py --out "$C28" \
        > "$S/tbl_$1.log" 2>&1
    say "campaign-28 table after $1 rc=$? — $(grep -oE '\*\*[0-9.]+%\*\*' "$C28/family_table.md" 2>/dev/null | tr '\n' ' ')"
}
#: The figure, next to r08 and V0. CPU, so it is safe beside nothing else.
figure() {
    local label="$1" run="$2" tag="$3"
    [ -d "$run" ] || { say "FIGURE $tag SKIPPED: no run directory"; return 0; }
    choom -n 1000 -- python scripts/analysis/vae_recon_figure.py \
        --run "r08 (spatial, z=8)=${R08%/}" \
        --run "V0 (flat, KL off)=${V0%/}" \
        --run "${label}=${run%/}" \
        --out "$FIG/recon_$tag" > "$S/fig_$tag.log" 2>&1
    say "FIGURE $tag rc=$? — $FIG/recon_$tag.png"
}

mkdir -p "$FIG"
say "=== conv_k8, then A0 full === $(mem_line)"
say "r08=$(basename "${R08%/}")  V0=$(basename "${V0%/}")"

# ── 1. adopt B, which v3 started and which outlived it ─────────────────────
if [ -n "$B_PID" ]; then
    say "waiting on B (pid $B_PID), adopted from v3"
    while kill -0 "$B_PID" 2>/dev/null; do sleep 30; done
    say "B trainer exited — $(mem_line)"
fi
if [ -n "$B_RUN" ]; then
    if [ -f "$B_RUN/best.ckpt" ]; then
        table b
        figure "B (flat, FC 1024, rank 768)" "$B_RUN" b
    else
        say "B produced no best.ckpt. STOP NOTICE: conv_k8 is NOT launched on a"
        say "  broken predecessor; the card needs a human before it moves."
        exit 1
    fi
fi

# ── 2. conv_k8, the arm the block exists for ───────────────────────────────
if smoke vrrae/conv_k8 conv_k8; then
    run_watched "conv_k8 (per-cell, k*=8, batch 128)" vrrae_conv_k8 \
        python scripts/train_vae.py run vrrae/conv_k8
    table conv_k8
    CONV=$(ls -dt "$REPO"/runs/vae/vrrae-run-*vrrae_conv*/ 2>/dev/null | head -1)
    figure "conv k*=8 (per-cell)" "$CONV" conv_k8
else
    say "conv_k8 SKIPPED: it does not fit at batch 128 under the cap. No"
    say "  smaller batch is invented here — the batch is r08's on purpose."
fi

# ── 3. A0 in full; its 400-step trial passed at 0.0473 against a 0.12 gate ──
run_watched "A0 (vrrae/a0, beta 0)" vrrae_a0 python scripts/train_vae.py run vrrae/a0
table a0
A0RUN=$(ls -dt "$REPO"/runs/vae/vrrae-run-*b0.000*/ 2>/dev/null | head -1)
figure "A0 (flat, KL off)" "$A0RUN" a0

# ── 4. rf-64 and the notebooks ─────────────────────────────────────────────
RF64=$(ls -dt "$REPO"/runs/vae/r08-run-0012-*/ 2>/dev/null | head -1)
if [ -n "$RF64" ]; then
    run_watched "rf-64 resume from step 1000" rf64_resume \
        python scripts/train_vae.py resume "$(basename "${RF64%/}")" latest.ckpt
fi
for root in "$C18" "$C12"; do
    run_watched "NOTEBOOK $(basename "$root")" "nb_$(basename "$root")" \
        python scripts/analysis/build_eval_v4_notebook.py --root "$root"
done
say "BLOCK COMPLETE — $(mem_line)"
