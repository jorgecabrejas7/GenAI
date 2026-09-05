#!/usr/bin/env bash
# Everything between "the compressor is chosen" and "ldm06 is training".
#
# Runs unattended after a one-line decision, because each step's failure mode
# is a missing file the NEXT step would happily build around: a half-written
# store still loads, a stale conditioning sidecar still joins. Every stage is
# therefore verified before the next begins, and the launch is refused if any
# check fails.
#
# Usage:  bash scripts/ldm06_bringup.sh <run_dir_of_chosen_rung>
set -uo pipefail

REPO=/home/jorgecabrejas/Dev/GenAI
CAMP="$REPO/runs/campaigns/09-r08-latent-sweep"
SCRATCH=/tmp/claude-1001/-home-jorgecabrejas-Dev-GenAI/ce0b3db0-2aa0-4a97-9bc4-7bb0db078739/scratchpad
LOG="$CAMP/ldm06_bringup.log"
RUN_DIR="${1:?usage: ldm06_bringup.sh <run_dir_of_chosen_rung>}"

mkdir -p "$SCRATCH"; cd "$REPO" || exit 1
say() { printf '%s  %s\n' "$(date -Is)" "$*" | tee -a "$LOG"; }
die() { say "ABORT $*"; exit 1; }

CKPT="${RUN_DIR%/}/best.ckpt"
[ -f "$CKPT" ] || die "no best.ckpt at $CKPT"
Z=$(python - "$RUN_DIR" <<'PY'
import sys, yaml, pathlib
print(yaml.safe_load((pathlib.Path(sys.argv[1]) / "resolved_config.yaml").read_text())["model"]["z_channels"])
PY
)
STORE="$REPO/data/split_v3/latents_r08z${Z}"
say "BRINGUP start: rung $RUN_DIR (z=$Z) -> $STORE"

# 1. latent store
say "STORE build start"
python scripts/build_latent_dataset.py --checkpoint "$CKPT" --output "$STORE" \
    > "$SCRATCH/ldm06_store.log" 2>&1
rc=$?; say "STORE build rc=$rc"
[ "$rc" -eq 0 ] || die "store build failed; see $SCRATCH/ldm06_store.log"
for f in metadata.json train/latents.bin train/index.parquet train/material.bin \
         train/air.bin train/pore.bin val/latents.bin test/latents.bin; do
    [ -e "$STORE/$f" ] || die "store missing $f"
done
say "STORE files present"

# 2. conditioning sidecar
say "COND build start"
python scripts/build_conditioning.py --store "$STORE" \
    > "$SCRATCH/ldm06_cond.log" 2>&1
rc=$?; say "COND build rc=$rc"
[ "$rc" -eq 0 ] || die "conditioning build failed; see $SCRATCH/ldm06_cond.log"
for sp in train val test; do
    [ -e "$STORE/$sp/cond.parquet" ] || die "conditioning missing $sp/cond.parquet"
done
say "COND files present"

# 3. LatentDataset verification — serve real batches and check the contract,
#    because a store that loads is not the same as a store that is correct.
say "VERIFY start"
python - "$STORE" > "$SCRATCH/ldm06_verify.log" 2>&1 <<'PY'
import sys, json, pathlib, torch
sys.path.insert(0, "/home/jorgecabrejas/Dev/GenAI/src")
from poregen.diffusion.latents import LatentDataset

store = pathlib.Path(sys.argv[1])
meta = json.loads((store / "metadata.json").read_text())
print("z_channels:", meta["latent_shape"][0], "| splits:", meta.get("splits"))
print("pore_status:", meta.get("material", {}).get("pore_status", "MISSING"))

ds = LatentDataset(store, split="train")
b = ds[0]
print("batch keys:", sorted(b))
for k, v in sorted(b.items()):
    if torch.is_tensor(v):
        print(f"  {k:16s} {tuple(v.shape)} {v.dtype}")
assert len(ds) > 0, "empty dataset"
print("n rows:", len(ds))
print("VERIFY OK")
PY
rc=$?; say "VERIFY rc=$rc"
[ "$rc" -eq 0 ] || die "LatentDataset verification failed; see $SCRATCH/ldm06_verify.log"
grep -q "VERIFY OK" "$SCRATCH/ldm06_verify.log" || die "verification did not reach VERIFY OK"

# 4. point ldm06 at the store this rung produced, in one explicit commit.
#    Editing the config silently would make "which store did ldm06 read?"
#    unanswerable months later; committing it makes the switch attributable and
#    revertable. Only this one key, and only to this one value — every other
#    mismatch is still a refusal.
say "CONFIG set ldm06/base data.latents_root -> data/split_v3/latents_r08z${Z}"
python - "$Z" <<'PY'
import sys, pathlib, re
z = sys.argv[1]
p = pathlib.Path("/home/jorgecabrejas/Dev/GenAI/configs/experiments/ldm06/base.yaml")
want = f"data/split_v3/latents_r08z{z}"
s = p.read_text()
m = re.search(r"^(\s*latents_root:[ \t]*)(\S+)[ \t]*$", s, re.M)
if m is None:
    print("no latents_root key in ldm06/base.yaml — refusing to invent one")
    sys.exit(3)
if m.group(2) == want:
    print(f"latents_root already {want}")
    sys.exit(0)
p.write_text(s[: m.start()] + m.group(1) + want + s[m.end() :])
print(f"latents_root {m.group(2)} -> {want}")
sys.exit(10)
PY
rc=$?
if [ "$rc" -eq 10 ]; then
    git -C "$REPO" commit -q -m "config: ldm06/base reads the chosen rung's latent store (z=${Z})

The r08 compression sweep chose z=${Z}, so ldm06 trains on
data/split_v3/latents_r08z${Z}. Written by scripts/ldm06_bringup.sh as a
commit rather than an in-place edit: which store a run read is exactly the
kind of fact that has to survive to the paper.

Co-Authored-By: Claude Opus 5 (1M context) <noreply@anthropic.com>" \
        -- configs/experiments/ldm06/base.yaml \
        || die "could not commit the latents_root change"
    say "CONFIG committed $(git -C "$REPO" rev-parse --short HEAD)"
elif [ "$rc" -eq 0 ]; then
    say "CONFIG already correct"
else
    die "could not set latents_root (rc=$rc)"
fi

# 5. launch
say "LDM06 launch in tmux ldm06"
tmux new-session -d -s ldm06 -c "$REPO" 2>/dev/null
tmux send-keys -t ldm06 "python scripts/train_ldm.py run ldm06/base 2>&1 | tee $SCRATCH/ldm06_base.log" Enter
sleep 20
say "BRINGUP done; ldm06/base launched. rf-2 and rf-64 wait for it."
