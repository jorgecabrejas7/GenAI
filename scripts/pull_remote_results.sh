#!/usr/bin/env bash
# Bring the second GB10's rung results here, for reading without a Claude
# session there: runs/remote/<host>/ mirrors the remote's runs/ layout.
#     bash scripts/pull_remote_results.sh <host> [remote repo path]
set -euo pipefail
HOST="${1:?usage: pull_remote_results.sh <host> [remote repo path]}"
RREPO="${2:-/home/jorgecabrejas/Dev/GenAI}"
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DEST="$REPO/runs/remote/$HOST"
mkdir -p "$DEST/runs/vae" "$DEST/runs/campaigns/09-r08-latent-sweep"
rsync -a --prune-empty-dirs \
    --include='*-dsv4/' \
    --include='*-dsv4/metrics.jsonl' --include='*-dsv4/log.jsonl' \
    --include='*-dsv4/resolved_config.yaml' --include='*-dsv4/run_metadata.json' \
    --include='*-dsv4/summary.json' --include='*-dsv4/best.ckpt' --include='*-dsv4/latest.ckpt' \
    --exclude='*' \
    "$HOST:$RREPO/runs/vae/" "$DEST/runs/vae/"
rsync -a --include='*-dsv4/***' --include='figures/' --include='figures/recon_v4_*/***' --exclude='*' \
    "$HOST:$RREPO/runs/campaigns/09-r08-latent-sweep/" "$DEST/runs/campaigns/09-r08-latent-sweep/"
rsync -a "$HOST:$RREPO/runs/campaigns/chain.log" "$DEST/runs/campaigns/" || true
rsync -a "$HOST:$RREPO/runs/campaigns/chain_logs/" "$DEST/runs/campaigns/chain_logs/" || true
echo "pulled into $DEST at $(date -Is)"
tail -3 "$DEST/runs/campaigns/chain.log" 2>/dev/null || true
