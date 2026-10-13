#!/usr/bin/env bash
# Deprecated for the full corpus: overlays are not required.
# Use CSV-only mask batch instead.
set -euo pipefail
cd "$(dirname "$0")/.."
echo "[batch-mask] Overlay stage disabled for full corpus."
echo "[batch-mask] Starting CSV-only runner: scripts/run_batch_mask_csv.sh"
exec bash scripts/run_batch_mask_csv.sh
