#!/usr/bin/env bash
# Heartbeat / stall detector for the dense batch.
# Install (cron every 10 min):
#   crontab -e
#   */10 * * * * /home/ifab-agiprobotw-0001/KHETRE/DINO_KHETRE/DINO/scripts/watchdog_batch_dense_parts_hand.sh
#
# Check from phone/SSH anytime:
#   cat outputs/batch_sam3_dense_parts_hand/watchdog_status.txt
#   tail outputs/batch_sam3_dense_parts_hand/watchdog.log
set -euo pipefail

ROOT="${BATCH_ROOT:-$HOME/KHETRE/DINO_KHETRE/DINO}"
cd "$ROOT"
OUT=outputs/batch_sam3_dense_parts_hand
mkdir -p "$OUT"
STATUS="$OUT/watchdog_status.txt"
LOG="$OUT/watchdog.log"
BATCH_LOG="$OUT/batch_run.log"
PID_FILE="$OUT/batch_run.pid"
STALL_MINUTES="${STALL_MINUTES:-45}"

ts=$(date -Is)
pid=$(cat "$PID_FILE" 2>/dev/null || true)
alive=0
etime="-"
if [[ -n "${pid:-}" ]] && ps -p "$pid" >/dev/null 2>&1; then
  alive=1
  etime=$(ps -p "$pid" -o etime= | tr -d ' ')
fi

log_age="missing"
log_stall=0
if [[ -f "$BATCH_LOG" ]]; then
  # seconds since last log write
  now=$(date +%s)
  mtime=$(stat -c %Y "$BATCH_LOG")
  age=$((now - mtime))
  log_age="${age}s"
  if [[ "$alive" -eq 1 && "$age" -gt $((STALL_MINUTES * 60)) ]]; then
    log_stall=1
  fi
fi

last_detect=$(grep '\[detect\]\|\[batch\]' "$BATCH_LOG" 2>/dev/null | tail -n 1 || echo "(no batch lines)")
gpu=$(nvidia-smi --query-gpu=utilization.gpu,memory.used,temperature.gpu --format=csv,noheader 2>/dev/null | head -n 1 || echo "n/a")
disk=$(df -h . | awk 'NR==2 {print $4" free ("$5" used)"}')

qc="no QC yet"
if [[ -f "$OUT/batch_qc_summary.csv" ]]; then
  qc=$(python3 - <<'PY'
import csv
from collections import Counter
from pathlib import Path
rows=list(csv.DictReader(Path("outputs/batch_sam3_dense_parts_hand/batch_qc_summary.csv").open()))
c=Counter(r["status"] for r in rows)
print(f"n={len(rows)} {dict(c)}")
PY
)
fi

state="OK"
if [[ "$alive" -eq 0 ]]; then
  state="DEAD_OR_FINISHED"
elif [[ "$log_stall" -eq 1 ]]; then
  state="STALL_LOG_SILENT"
fi

{
  echo "time=$ts"
  echo "state=$state"
  echo "pid=${pid:-none} alive=$alive etime=$etime"
  echo "batch_log_age=$log_age stall_threshold=${STALL_MINUTES}m"
  echo "gpu=$gpu"
  echo "disk=$disk"
  echo "qc=$qc"
  echo "last=$last_detect"
} >"$STATUS"

echo "$ts state=$state alive=$alive log_age=$log_age qc=$qc | $last_detect" >>"$LOG"

# Optional email if mail is configured (uncomment and set ADDRESS):
# if [[ "$state" != "OK" ]]; then
#   mail -s "dense-parts batch $state" you@example.com <"$STATUS" || true
# fi

exit 0
