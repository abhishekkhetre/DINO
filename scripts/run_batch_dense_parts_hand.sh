#!/usr/bin/env bash
# Dense full-corpus MASK batch → CSV sheets only (no overlay videos).
# stride=2, AG parts + hand/table/floor, geometry on screwdriver/wrench only.
# SAM3 unloads after each recording; mask npz deleted after assign (disk).
set -euo pipefail

cd "$(dirname "$0")/.."
ROOT_OUT=outputs/batch_sam3_dense_parts_hand
mkdir -p "$ROOT_OUT"

echo "[batch-dense-parts] pwd=$(pwd)"
echo "[batch-dense-parts] output_root=$ROOT_OUT"
echo "[batch-dense-parts] stages=detect,assign,sequences (NO overlay)"
echo "[batch-dense-parts] stride=2 | discard_masks_after_assign=true"
command -v nvidia-smi >/dev/null && nvidia-smi --query-gpu=memory.used,memory.free,memory.total --format=csv,noheader || true
df -h . | tail -n 1 || true

export PYTHONPATH=src:${PYTHONPATH:-}
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# One GPU job only — refuse to start if another compute process holds the GPU.
if command -v nvidia-smi >/dev/null; then
  OTHER=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | grep -v '^$' | wc -l || true)
  if [[ "${OTHER// /}" -gt 0 ]]; then
    echo "ERROR: another GPU process is running. Stop it first (one job only)." >&2
    nvidia-smi
    exit 4
  fi
fi

LOG=$ROOT_OUT/batch_run.log
echo "[batch-dense-parts] logging → $LOG"

nohup python -m gaze_objects.cli batch --config configs/batch_sam3_dense_parts_hand.yaml \
  >"$LOG" 2>&1 &
PID=$!
echo "$PID" > "$ROOT_OUT/batch_run.pid"
echo "[batch-dense-parts] started pid=$PID"
echo "[batch-dense-parts] monitor:  tail -f $LOG"
echo "[batch-dense-parts] when done:"
echo "  python -m gaze_objects.cli aggregate-sequences --output-root $ROOT_OUT"
