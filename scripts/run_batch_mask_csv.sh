#!/usr/bin/env bash
# Full-corpus mask batch → CSV only (detect + assign + sequences). No overlay.
# Run on the lab workstation with ONE GPU job.
set -euo pipefail

cd "$(dirname "$0")/.."
mkdir -p outputs/batch_sam3_full_fine_mask

echo "[batch-mask] pwd=$(pwd)"
echo "[batch-mask] activating checks…"
command -v nvidia-smi >/dev/null && nvidia-smi --query-gpu=memory.used,memory.free --format=csv,noheader || true

export PYTHONPATH=src:${PYTHONPATH:-}
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

LOG=outputs/batch_sam3_full_fine_mask/batch_run.log
echo "[batch-mask] logging → $LOG"
echo "[batch-mask] config=configs/batch_sam3_full_fine_mask.yaml"
echo "[batch-mask] output_root=outputs/batch_sam3_full_fine_mask"

nohup python -m gaze_objects.cli batch --config configs/batch_sam3_full_fine_mask.yaml \
  >"$LOG" 2>&1 &
PID=$!
echo "$PID" > outputs/batch_sam3_full_fine_mask/batch_run.pid
echo "[batch-mask] started pid=$PID"
echo "[batch-mask] monitor:  tail -f $LOG"
echo "[batch-mask] when done: python -m gaze_objects.cli aggregate-sequences --output-root outputs/batch_sam3_full_fine_mask"
