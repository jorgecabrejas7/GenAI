#!/usr/bin/env bash
# Bring a remote GB10's split_v4 rung results here: runs/remote/<host>/ mirrors
# the remote's runs/ layout, so the results can be read without a Claude
# session there.
#
#   bash scripts/pull_remote_results.sh <host> [--full]
#
# Light (default, for periodic reads): per run metrics/log/config/metadata/
# summary and best/latest.ckpt; campaign-09 -dsv4 outputs and recon figures;
# chain.log and chain_logs/.
# --full (before the borrowed machine is wiped): the WHOLE run directories
# (every step checkpoint, tb/), the same campaign outputs and logs, then a
# path+size+sha256 manifest of both sides is compared and MIRROR OK or MIRROR
# MISMATCH printed. MIRROR OK writes runs/remote/<host>/MIRROR_OK, which
# scripts/remote_wipe.sh requires.
#
# RREPO (default Dev/GenAI) is the repo path on the remote, relative to the
# remote user's home or absolute.
set -euo pipefail
HOST="${1:?usage: pull_remote_results.sh <host> [--full]}"
MODE="${2:-light}"
RREPO="${RREPO:-Dev/GenAI}"
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DEST="$REPO/runs/remote/$HOST"
mkdir -p "$DEST/runs/vae" "$DEST/runs/campaigns/09-r08-latent-sweep"

if [ "$MODE" = "--full" ]; then
    VAE_FILTER=(--include='*-dsv4/***' --exclude='*')
else
    VAE_FILTER=(--include='*-dsv4/' --include='*-dsv4/metrics.jsonl' --include='*-dsv4/log.jsonl'
                --include='*-dsv4/resolved_config.yaml' --include='*-dsv4/run_metadata.json'
                --include='*-dsv4/summary.json' --include='*-dsv4/best.ckpt'
                --include='*-dsv4/latest.ckpt' --exclude='*')
fi
rsync -a --prune-empty-dirs "${VAE_FILTER[@]}" "$HOST:$RREPO/runs/vae/" "$DEST/runs/vae/"
rsync -a --include='*-dsv4/***' --include='figures/' --include='figures/recon_v4_*/***' --exclude='*' \
    "$HOST:$RREPO/runs/campaigns/09-r08-latent-sweep/" "$DEST/runs/campaigns/09-r08-latent-sweep/"
rsync -a "$HOST:$RREPO/runs/campaigns/chain.log" "$DEST/runs/campaigns/" || true
rsync -a "$HOST:$RREPO/runs/campaigns/chain_logs/" "$DEST/runs/campaigns/chain_logs/" || true
echo "pulled ($MODE) into $DEST at $(date -Is)"

[ "$MODE" = "--full" ] || { tail -3 "$DEST/runs/campaigns/chain.log" 2>/dev/null || true; exit 0; }

# ── the mirror check: every remote result file, here, byte-identical ────────
# The same file set on both sides: run dirs, campaign -dsv4 dirs, recon
# figures, chain.log, chain_logs/. One line per file: path, size, sha256.
LIST='find runs/vae -path "runs/vae/*-dsv4/*" -type f;
      find runs/campaigns/09-r08-latent-sweep -path "*-dsv4/*" -type f;
      find runs/campaigns/09-r08-latent-sweep/figures -path "*/recon_v4_*/*" -type f 2>/dev/null;
      ls runs/campaigns/chain.log 2>/dev/null; find runs/campaigns/chain_logs -type f 2>/dev/null'
MANIFEST='| sort | while read -r f; do printf "%s %s %s\n" "$f" "$(stat -c %s "$f")" "$(sha256sum "$f" | cut -c1-64)"; done'
STAMP=$(date -Is)
ssh "$HOST" "cd $RREPO && { $LIST; } $MANIFEST" > "$DEST/manifest.remote"
(cd "$DEST" && eval "{ $LIST; } $MANIFEST") > "$DEST/manifest.local"
# the remote's newest write under runs/, for remote_wipe.sh
ssh "$HOST" "cd $RREPO && find runs -type f -printf '%T@\n' | sort -n | tail -1" > "$DEST/remote_last_write"
n=$(wc -l < "$DEST/manifest.remote")
if [ "$n" -gt 0 ] && cmp -s "$DEST/manifest.remote" "$DEST/manifest.local"; then
    printf '%s %s files, %s bytes, remote last write %s\n' "$STAMP" "$n" \
        "$(awk '{s+=$2} END{print s}' "$DEST/manifest.remote")" "$(cat "$DEST/remote_last_write")" \
        > "$DEST/MIRROR_OK"
    echo "MIRROR OK — $(cat "$DEST/MIRROR_OK")"
else
    rm -f "$DEST/MIRROR_OK"
    echo "MIRROR MISMATCH — $n remote files; differences:"
    diff "$DEST/manifest.remote" "$DEST/manifest.local" | head -40 || true
    exit 1
fi
