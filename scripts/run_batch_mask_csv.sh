#!/usr/bin/env bash
# Fresh full-corpus MASK batch → CSV only (detect + assign + sequences).
# No overlay video.
set -euo pipefail

cd "$(dirname "$0")/.."
ROOT_OUT=outputs/batch_sam3_full_fine_mask
mkdir -p "$ROOT_OUT"

echo "[batch-mask-csv] pwd=$(pwd)"
echo "[batch-mask-csv] output_root=$ROOT_OUT"
echo "[batch-mask-csv] stages=detect,assign,sequences (NO overlay)"
command -v nvidia-smi >/dev/null && nvidia-smi --query-gpu=memory.used,memory.free --format=csv,noheader || true

export PYTHONPATH=src:${PYTHONPATH:-}
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

LOG=$ROOT_OUT/batch_run.log
echo "[batch-mask-csv] logging → $LOG"

nohup python -m gaze_objects.cli batch --config configs/batch_sam3_full_fine_mask.yaml \
  >"$LOG" 2>&1 &
PID=$!
echo "$PID" > "$ROOT_OUT/batch_run.pid"
echo "[batch-mask-csv] started pid=$PID"
echo "[batch-mask-csv] monitor:  tail -f $LOG"
echo "[batch-mask-csv] when done:"
echo "  python -m gaze_objects.cli aggregate-sequences --output-root $ROOT_OUT"
