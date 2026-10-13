#!/usr/bin/env bash
# Fresh full-corpus MASK batch: detect → assign → sequences → overlay video.
# Fine labels include gaze tip/handle/body. CSV + per-recording mp4.
set -euo pipefail

cd "$(dirname "$0")/.."
ROOT_OUT=outputs/batch_sam3_full_fine_mask
mkdir -p "$ROOT_OUT"

echo "[batch-mask-video] pwd=$(pwd)"
echo "[batch-mask-video] output_root=$ROOT_OUT"
echo "[batch-mask-video] stages=detect,assign,sequences,overlay"
command -v nvidia-smi >/dev/null && nvidia-smi --query-gpu=memory.used,memory.free --format=csv,noheader || true

# ffmpeg strongly recommended so overlays become H.264 playable
if ! command -v ffmpeg >/dev/null 2>&1; then
  echo "[batch-mask-video] WARNING: ffmpeg not on PATH — overlays may stay mp4v."
  echo "  Install once:  conda install -y -c conda-forge ffmpeg"
fi

export PYTHONPATH=src:${PYTHONPATH:-}
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

LOG=$ROOT_OUT/batch_run.log
echo "[batch-mask-video] logging → $LOG"

nohup python -m gaze_objects.cli batch --config configs/batch_sam3_full_fine_mask.yaml \
  >"$LOG" 2>&1 &
PID=$!
echo "$PID" > "$ROOT_OUT/batch_run.pid"
echo "[batch-mask-video] started pid=$PID"
echo "[batch-mask-video] monitor:  tail -f $LOG"
echo "[batch-mask-video] when ALL recordings done:"
echo "  python -m gaze_objects.cli aggregate-sequences --output-root $ROOT_OUT"
