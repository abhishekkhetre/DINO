#!/usr/bin/env bash
# Dense full-corpus MASK batch → CSV sheets only (no overlay videos).
# stride=2, AG parts + hand/table/floor, geometry on elongated tools only.
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

# Block only real CUDA compute jobs (python/torch), not desktop (Xorg/GNOME/Firefox).
# gnome-remote-desktop often shows as C+G and must NOT abort the batch.
if command -v nvidia-smi >/dev/null; then
  BLOCKERS=""
  while IFS=, read -r pid name; do
    pid=$(echo "$pid" | tr -d ' ')
    name=$(echo "$name" | tr -d ' ')
    [[ -z "$pid" || "$pid" == "pid" ]] && continue
    base=$(basename "$name" 2>/dev/null || echo "$name")
    case "$base" in
      python|python3|pt_main_thread|*torch*|jupyter*)
        BLOCKERS="${BLOCKERS}${pid} ${name}\n"
        ;;
    esac
  done < <(nvidia-smi --query-compute-apps=pid,process_name --format=csv,noheader 2>/dev/null || true)

  if [[ -n "$BLOCKERS" ]]; then
    echo "ERROR: another Python/CUDA job is on the GPU. Stop it first (one job only):" >&2
    echo -e "$BLOCKERS" >&2
    nvidia-smi
    exit 4
  fi
  echo "[batch-dense-parts] GPU free of other python/CUDA jobs (desktop display OK)"
fi

# Soft disk warning — masks are discarded after assign, but CSVs still need headroom.
AVAIL_G=$(df -BG . | awk 'NR==2 {gsub(/G/,""); print $4}')
if [[ -n "${AVAIL_G:-}" ]] && [[ "$AVAIL_G" -lt 25 ]]; then
  echo "WARNING: only ~${AVAIL_G}G free on disk. Batch may fail mid-run; free space if possible." >&2
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
