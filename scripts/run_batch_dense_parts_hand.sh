#!/usr/bin/env bash
# Dense full-corpus MASK batch → CSV sheets only (no overlay videos).
# Prefers tmux so a frozen desktop / closed terminal does NOT kill the job.
# Monitor over SSH:  bash scripts/check_batch_dense_parts_hand.sh
#                   tmux attach -t dense-parts-batch
set -euo pipefail

cd "$(dirname "$0")/.."
ROOT_OUT=outputs/batch_sam3_dense_parts_hand
mkdir -p "$ROOT_OUT"
SESSION=dense-parts-batch

echo "[batch-dense-parts] pwd=$(pwd)"
echo "[batch-dense-parts] output_root=$ROOT_OUT"
echo "[batch-dense-parts] stages=detect,assign,sequences (NO overlay)"
echo "[batch-dense-parts] stride=2 | discard_masks_after_assign=true"
command -v nvidia-smi >/dev/null && nvidia-smi --query-gpu=memory.used,memory.free,memory.total --format=csv,noheader || true
df -h . | tail -n 1 || true

export PYTHONPATH=src:${PYTHONPATH:-}
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Block only real CUDA compute jobs (python/torch), not desktop (Xorg/GNOME/Firefox).
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

AVAIL_G=$(df -BG . | awk 'NR==2 {gsub(/G/,""); print $4}')
if [[ -n "${AVAIL_G:-}" ]] && [[ "$AVAIL_G" -lt 25 ]]; then
  echo "WARNING: only ~${AVAIL_G}G free on disk. Free space if possible." >&2
fi

LOG=$ROOT_OUT/batch_run.log

# Already running?
if [[ -f "$ROOT_OUT/batch_run.pid" ]]; then
  old=$(cat "$ROOT_OUT/batch_run.pid" || true)
  if [[ -n "${old:-}" ]] && ps -p "$old" >/dev/null 2>&1; then
    echo "ERROR: batch already running as pid=$old — do not start a second job." >&2
    echo "  check:  bash scripts/check_batch_dense_parts_hand.sh" >&2
    exit 5
  fi
fi

run_batch() {
  # Append so resume keeps history across restarts
  python -m gaze_objects.cli batch --config configs/batch_sam3_dense_parts_hand.yaml \
    >>"$LOG" 2>&1
}

if command -v tmux >/dev/null 2>&1; then
  if tmux has-session -t "$SESSION" 2>/dev/null; then
    echo "ERROR: tmux session '$SESSION' already exists. Attach: tmux attach -t $SESSION" >&2
    exit 5
  fi
  # Fresh log marker
  {
    echo ""
    echo "======== batch start $(date -Is) ========"
  } >>"$LOG"
  tmux new-session -d -s "$SESSION" -c "$PWD" \
    "export PYTHONPATH=src:\${PYTHONPATH:-}; export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True; python -m gaze_objects.cli batch --config configs/batch_sam3_dense_parts_hand.yaml >>$LOG 2>&1; echo EXIT=\$? >>$LOG; exec bash"
  # Resolve python pid inside the session (best-effort)
  sleep 2
  PID=$(pgrep -f "gaze_objects.cli batch --config configs/batch_sam3_dense_parts_hand" | head -n 1 || true)
  echo "${PID:-tmux}" >"$ROOT_OUT/batch_run.pid"
  echo "[batch-dense-parts] started in tmux session: $SESSION  (pid≈${PID:-unknown})"
  echo "[batch-dense-parts] DO NOT reboot if the screen freezes — SSH in instead."
  echo "[batch-dense-parts] check:   bash scripts/check_batch_dense_parts_hand.sh"
  echo "[batch-dense-parts] log:     tail -f $LOG"
  echo "[batch-dense-parts] attach:  tmux attach -t $SESSION"
else
  {
    echo ""
    echo "======== batch start $(date -Is) ========"
  } >>"$LOG"
  nohup python -m gaze_objects.cli batch --config configs/batch_sam3_dense_parts_hand.yaml \
    >>"$LOG" 2>&1 &
  PID=$!
  echo "$PID" >"$ROOT_OUT/batch_run.pid"
  echo "[batch-dense-parts] tmux not found — started with nohup pid=$PID"
  echo "[batch-dense-parts] Prefer: sudo apt install tmux   (next time)"
  echo "[batch-dense-parts] check: bash scripts/check_batch_dense_parts_hand.sh"
  echo "[batch-dense-parts] log:   tail -f $LOG"
fi

echo "[batch-dense-parts] when done:"
echo "  python -m gaze_objects.cli aggregate-sequences --output-root $ROOT_OUT"
