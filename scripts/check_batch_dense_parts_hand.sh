#!/usr/bin/env bash
# Progress check for the dense parts batch — safe over SSH when the GUI freezes.
set -euo pipefail
cd "$(dirname "$0")/.."
OUT=outputs/batch_sam3_dense_parts_hand

echo "=== disk ==="
df -h . | tail -n 1

echo "=== gpu ==="
nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.free --format=csv,noheader 2>/dev/null || true
nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader 2>/dev/null || true

echo "=== process ==="
pid=$(cat "$OUT/batch_run.pid" 2>/dev/null || true)
if [[ -n "${pid:-}" ]] && ps -p "$pid" >/dev/null 2>&1; then
  ps -p "$pid" -o pid,etime,pcpu,pmem,cmd
  echo "STILL RUNNING"
else
  echo "No live batch pid (safe to resume if incomplete)"
fi

echo "=== tmux (if used) ==="
tmux has-session -t dense-parts-batch 2>/dev/null && echo "tmux session dense-parts-batch: alive" || echo "no tmux session dense-parts-batch"

echo "=== qc ==="
if [[ -f "$OUT/batch_qc_summary.csv" ]]; then
  python3 - <<'PY'
import csv
from collections import Counter
from pathlib import Path
rows=list(csv.DictReader(Path("outputs/batch_sam3_dense_parts_hand/batch_qc_summary.csv").open()))
print(dict(Counter(r["status"] for r in rows)), "n=", len(rows))
for r in rows[-3:]:
    print(r["recording_id"], r["status"], (r.get("error") or "")[:60])
PY
else
  echo "No QC yet — first recording not finished"
fi

echo "=== last detect lines ==="
grep '\[detect\]\|\[batch\]' "$OUT/batch_run.log" 2>/dev/null | tail -n 15 || echo "(no log)"
