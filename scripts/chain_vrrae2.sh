#!/usr/bin/env bash
# The VRRAE block, relaunched cautiously. NO MORE HOST OOMs is the requirement.
#
# The previous attempt took the host down twice. Both times the cause was a
# process allocating on a pool the host and the card share, with nothing to
# stop it: once a batch-1024 training run started on a FALSE smoke result, once
# the smoke test itself at batch 1536. Neither could fail on its own — the
# kernel had to pick a victim, and it picked wireplumber, pipewire and tmux.
#
# FOUR THINGS MAKE THAT IMPOSSIBLE HERE, and none of them is judgement:
#   * `select_device()` caps CUDA memory per process (02852d9), so an oversize
#     allocation raises torch.cuda.OutOfMemoryError instead of draining the
#     host. The smoke tests run under a STRICTER cap than the runs.
#   * every python runs under `choom -n 1000`, so if the kernel ever does have
#     to choose, it chooses the training process and never tmux or dbus.
#   * each config is smoked IMMEDIATELY BEFORE its own run, never all at once.
#   * five minutes into every run, available host memory is logged; under
#     15 GB the run is STOPPED rather than left to ride.
#
# The chain stops on the first failure inside a run. The only fallback is the
# one written file, vrrae/b_fallback — nothing is invented at runtime.
set -uo pipefail
REPO=/home/jorgecabrejas/Dev/GenAI
S="/tmp/claude-1001/-home-jorgecabrejas-Dev-GenAI/ce0b3db0-2aa0-4a97-9bc4-7bb0db078739/scratchpad"
LOG="$REPO/runs/campaigns/chain.log"
C12="$REPO/runs/campaigns/12-eval-v4"; C18="$REPO/runs/campaigns/18-eval-v4-final"
C28="$REPO/runs/campaigns/28-vrrae-family"
#: Below this many GB of AVAILABLE host memory, a run is stopped rather than
#: allowed to continue toward another kernel-level kill.
MIN_AVAIL_GB=15
SMOKE_FRACTION=0.75          # stricter than the runs, on purpose
cd "$REPO" || exit 1
say() { printf '%s  V2 %s\n' "$(date -Is)" "$*" | tee -a "$LOG"; }

mem_line() {   # what the machine actually has, host and card
    local avail; avail=$(free -g | awk '/^Mem:/{print $7}')
    local cuda; cuda=$(python - <<'PY' 2>/dev/null || echo "n/a"
import torch
if torch.cuda.is_available():
    free, total = torch.cuda.mem_get_info()
    print(f"{free/2**30:.1f}/{total/2**30:.1f} GiB free")
else:
    print("no cuda")
PY
)
    printf 'host available %s GB; cuda %s' "$avail" "$cuda"
}

# Run one job under choom, and WATCH IT. The watcher is the point: a run that
# is quietly eating the host is stopped at the five-minute mark instead of
# being discovered by the OOM killer twenty minutes later.
run_watched() {
    local label="$1" tag="$2"; shift 2
    say "$label start"
    local t0=$SECONDS
    choom -n 1000 -- "$@" > "$S/$tag.log" 2>&1 &
    local pid=$!
    ( sleep 300
      if kill -0 "$pid" 2>/dev/null; then
          local avail; avail=$(free -g | awk '/^Mem:/{print $7}')
          say "$tag +5min: $(mem_line)"
          if [ "${avail:-99}" -lt "$MIN_AVAIL_GB" ]; then
              say "$tag STOPPING: host available ${avail} GB is under ${MIN_AVAIL_GB} GB"
              kill -TERM "$pid" 2>/dev/null
          fi
      fi ) &
    local watcher=$!
    wait "$pid"; local rc=$?
    kill "$watcher" 2>/dev/null; wait "$watcher" 2>/dev/null
    local dt=$((SECONDS - t0))
    if [ "$rc" -ne 0 ]; then
        say "$tag rc=$rc FAILED after ${dt}s — $(mem_line) — see $S/$tag.log"
        say "STOPPING the block on the first failure inside a run, as instructed."
        exit "$rc"
    fi
    say "$tag rc=0 in ${dt}s — $(mem_line)"
    return 0
}

# A smoke test under the STRICTER cap. rc=1 now means a real OOM; rc=2 means a
# fault in the test or the config and is NOT evidence about memory.
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

say "=== VRRAE block, cautious relaunch === $(mem_line)"
say "guards: per-process CUDA cap (runs 0.8 / smoke $SMOKE_FRACTION), logvar clamp, choom -n 1000"

# ── 1. V0 — RESUME from step 6000, with the clamp ──────────────────────────
# Decision taken upstream: it stays "vrrae04 with the KL off". No beta, no free
# bits. If early stopping fires, THAT IS THE RESULT and not a failure.
V0_RUN=$(ls -dt "$REPO"/runs/vae/vrrae-run-0001-*/ 2>/dev/null | head -1)
if [ -z "$V0_RUN" ]; then
    say "V0 ABORT: no vrrae-run-0001 to resume"; exit 1
fi
run_watched "V0 resume from $(basename "${V0_RUN%/}") step 6000" vrrae_v0_resume \
    python scripts/train_vae.py resume "$(basename "${V0_RUN%/}")" latest.ckpt
table v0

# ── 2. A ───────────────────────────────────────────────────────────────────
if smoke vrrae/a a; then
    run_watched "A (vrrae/a, batch 1024)" vrrae_a python scripts/train_vae.py run vrrae/a
else
    say "A SKIPPED: smoke rc above; see $S/smoke_a.log"
fi
table a

# ── 3. B, then its ONE written fallback ────────────────────────────────────
if smoke vrrae/b b; then
    run_watched "B (vrrae/b, batch 1536)" vrrae_b python scripts/train_vae.py run vrrae/b
elif smoke vrrae/b_fallback b_fb; then
    say "B at 1536 does not fit under the cap; the written fallback does"
    run_watched "B fallback (batch 1024, rank 768)" vrrae_b_fb \
        python scripts/train_vae.py run vrrae/b_fallback
else
    say "B SKIPPED ENTIRELY: neither 1536 nor the written fallback fits."
    say "  No third configuration is invented here."
fi
table b

# ── 4. A0 ──────────────────────────────────────────────────────────────────
if smoke vrrae/a0 a0; then
    run_watched "A0 (vrrae/a0, beta 0)" vrrae_a0 python scripts/train_vae.py run vrrae/a0
else
    say "A0 SKIPPED: smoke rc above; see $S/smoke_a0.log"
fi
table a0

# ── 5. rf-64, then the notebooks ───────────────────────────────────────────
RF64_RUN=$(ls -dt "$REPO"/runs/vae/r08-run-0012-*/ 2>/dev/null | head -1)
if [ -n "$RF64_RUN" ]; then
    run_watched "rf-64 resume from step 1000" rf64_resume \
        python scripts/train_vae.py resume "$(basename "${RF64_RUN%/}")" latest.ckpt
else
    say "rf-64 SKIPPED: no r08-run-0012"
fi
for root in "$C18" "$C12"; do
    run_watched "NOTEBOOK $(basename "$root")" "nb_$(basename "$root")" \
        python scripts/analysis/build_eval_v4_notebook.py --root "$root"
done
say "VRRAE BLOCK COMPLETE — $(mem_line)"
