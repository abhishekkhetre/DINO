#!/usr/bin/env bash
# CSV-only mask batch (no overlay). Prefer run_batch_mask_video.sh for full refresh.
set -euo pipefail
cd "$(dirname "$0")/.."
echo "[batch-mask] For fresh mask + VIDEO rendering use: bash scripts/run_batch_mask_video.sh"
echo "[batch-mask] Starting CSV-only stages via temporary config override is not separate;"
echo "[batch-mask] editing stages in YAML: use batch_sam3_full_fine_mask.yaml without overlay."
echo "[batch-mask] Redirecting to full mask+video runner…"
exec bash scripts/run_batch_mask_video.sh
