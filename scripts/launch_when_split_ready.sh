#!/usr/bin/env bash
# Launch the v4 chain THE MOMENT the dataset is ready, without anyone reading a
# message. Three times a hand-off was left to a message and once it cost
# seventeen idle hours; this is the hand-off as a process.
#
# IT DOES NOT TRUST AN rc=0 LINE. The build's tmux command reports the status
# of `tee`, not of python, so its rc=0 is written whether the build succeeded
# or crashed at volume 61. Success is judged by what each step LEFT BEHIND:
#   build    build_report.md + class_weights.json exist, and its log holds no
#            Traceback / SystemExit / Killed
#   extract  its rc line is honest (set -o pipefail, rc=${PIPESTATUS[0]}) and
#            must be 0; patches_meta.json and both memmaps exist; its --verify
#            pass did not report a failure
# Any failure: launch NOTHING, and write the tails to chain.log and to a file
# the supervising session reads.
set -uo pipefail
REPO=/home/jorgecabrejas/Dev/GenAI
SPLIT="${1:-split_v4}"
BUILD_LOG="${BUILD_LOG:?set BUILD_LOG to the build log that carries its rc line}"
EXTRACT_LOG="${EXTRACT_LOG:-$REPO/runs/logs/extract_split_v4.log}"
CHAIN="${CHAIN:-$REPO/scripts/chain_v8_split_v4.sh}"
LOG="$REPO/runs/campaigns/chain.log"
FAIL_NOTE="$REPO/runs/campaigns/LAUNCH_REFUSED_$SPLIT.txt"
D="${DATA_DIR:-$REPO/data/$SPLIT}"   # overridable only so the launcher can be tested without touching data/
cd "$REPO" || exit 1
say() { printf '%s  LAUNCH %s\n' "$(date -Is)" "$*" | tee -a "$LOG"; }
refuse() {
    say "REFUSED: $*"
    {
        echo "LAUNCH REFUSED for $SPLIT at $(date -Is): $*"
        echo; echo "=== build log tail ($BUILD_LOG) ==="; tail -n 25 "$BUILD_LOG" 2>&1
        echo; echo "=== extract log tail ($EXTRACT_LOG) ==="; tail -n 25 "$EXTRACT_LOG" 2>&1
    } > "$FAIL_NOTE"
    say "tails written to $FAIL_NOTE — nothing launched"
    exit 1
}

say "waiting on $SPLIT: build log $BUILD_LOG, extract log $EXTRACT_LOG"

# ── the build ──────────────────────────────────────────────────────────────
while ! grep -q '^rc=' "$BUILD_LOG" 2>/dev/null; do sleep 60; done
say "build wrote its rc line: $(grep '^rc=' "$BUILD_LOG" | tail -1) (not trusted on its own)"
if grep -qE 'Traceback|SystemExit|Killed' "$BUILD_LOG"; then
    refuse "the build log holds a Traceback / SystemExit / Killed"
fi
for f in build_report.md class_weights.json; do
    [ -e "$D/$f" ] || refuse "the build left no $D/$f — it did not finish"
done
say "build OK by its artefacts: build_report.md and class_weights.json present, no traceback"

# ── the extraction ─────────────────────────────────────────────────────────
while ! grep -qE '^rc=|build failed' "$EXTRACT_LOG" 2>/dev/null; do sleep 60; done
if grep -q 'build failed' "$EXTRACT_LOG"; then
    refuse "the extractor reports the build failed and did not run"
fi
RC=$(grep -E '^rc=' "$EXTRACT_LOG" | tail -1 | sed 's/^rc=//')
[ "$RC" = "0" ] || refuse "extraction rc=$RC"
# The extractor's own strings (scripts/extract_patches_memmap.py): it prints
# "XCT mismatch" / "Label mismatch" per bad patch, then "Verification FAILED"
# or "Verification OK: N patches checked, 0 errors."
if grep -qE 'Traceback|Killed|Verification FAILED|mismatch at parquet row' "$EXTRACT_LOG"; then
    refuse "the extraction log reports a failure or a verify mismatch"
fi
# AND THE POSITIVE MARKER. A --verify that never ran would leave no failure
# line either, so an absence of bad news is not good news.
grep -q 'Verification OK:' "$EXTRACT_LOG" \
    || refuse "the extraction log has no 'Verification OK' line — --verify did not run"
for f in patches_meta.json patches_xct.bin patches_label.bin patch_index.parquet \
         splits.json volumes.zarr; do
    [ -e "$D/$f" ] || refuse "extraction left no $D/$f"
done
say "extraction OK: rc=0 (honest), verify clean, memmaps and meta present"

# ── launch ─────────────────────────────────────────────────────────────────
rm -f "$FAIL_NOTE"
say "LAUNCHING $CHAIN with POREGEN_SPLIT=$SPLIT"
POREGEN_SPLIT="$SPLIT" exec bash "$CHAIN"
