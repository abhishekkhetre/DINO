#!/usr/bin/env bash
# AFTER stopping any old stride-2 job — start feasible stride-4 + lean-label batch.
#   bash scripts/start_stable_dense_batch_fast.sh
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
OUT=outputs/batch_sam3_dense_parts_hand_fast
CFG=configs/batch_sam3_dense_parts_hand_fast.yaml
UNIT_NAME=dense-parts-batch-fast.service
mkdir -p "$OUT"

echo "======== start_stable_dense_batch_fast $(date -Is) ========"

# Stop old heavy batch service / leftover python if present
systemctl --user stop dense-parts-batch.service 2>/dev/null || true
systemctl --user disable dense-parts-batch.service 2>/dev/null || true
OLD_PID_FILE=outputs/batch_sam3_dense_parts_hand/batch_run.pid
if [[ -f "$OLD_PID_FILE" ]]; then
  opid=$(cat "$OLD_PID_FILE" || true)
  if [[ -n "${opid:-}" ]] && ps -p "$opid" >/dev/null 2>&1; then
    echo "[stop] killing old batch pid=$opid"
    kill "$opid" 2>/dev/null || true
    sleep 2
  fi
fi
# Also stop this fast unit if already running (restart clean)
systemctl --user stop "$UNIT_NAME" 2>/dev/null || true

if command -v nvidia-smi >/dev/null; then
  if nvidia-smi --query-compute-apps=pid,process_name --format=csv,noheader 2>/dev/null | grep -qiE 'python|torch'; then
    echo "ERROR: GPU still has a python job. Kill it, then re-run this script." >&2
    nvidia-smi
    exit 4
  fi
fi

bash "$ROOT/scripts/prepare_workstation_for_batch.sh" || true
pkill -x firefox 2>/dev/null || true

# Locate conda
CONDA_SH=""
for c in "$HOME/conda/etc/profile.d/conda.sh" "$HOME/miniconda3/etc/profile.d/conda.sh" "$HOME/anaconda3/etc/profile.d/conda.sh"; do
  [[ -f "$c" ]] && CONDA_SH="$c" && break
done
if [[ -z "$CONDA_SH" ]]; then
  echo "ERROR: conda.sh not found" >&2
  exit 2
fi

WRAP="$ROOT/scripts/_systemd_dense_parts_batch_fast.sh"
cat >"$WRAP" <<EOF
#!/usr/bin/env bash
set -euo pipefail
cd "$ROOT"
bash "$ROOT/scripts/prepare_workstation_for_batch.sh" >> "$OUT/batch_run.log" 2>&1 || true
# shellcheck disable=SC1090
source "$CONDA_SH"
conda activate sam3
export PYTHONPATH=src:\${PYTHONPATH:-}
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
mkdir -p "$OUT"
echo "======== systemd FAST batch start \$(date -Is) ========" >> "$OUT/batch_run.log"
exec python -m gaze_objects.cli batch --config $CFG >> "$OUT/batch_run.log" 2>&1
EOF
chmod +x "$WRAP"

UNIT_DIR="${XDG_CONFIG_HOME:-$HOME/.config}/systemd/user"
mkdir -p "$UNIT_DIR"
cat >"$UNIT_DIR/$UNIT_NAME" <<EOF
[Unit]
Description=Dense SAM3 parts-hand FAST corpus batch (stride 4, lean labels)
After=default.target

[Service]
Type=simple
WorkingDirectory=$ROOT
ExecStart=$WRAP
Restart=on-failure
RestartSec=60
Nice=5
KillMode=mixed
TimeoutStopSec=30

[Install]
WantedBy=default.target
EOF

loginctl enable-linger "$USER" 2>/dev/null || true
systemctl --user daemon-reload
systemctl --user enable "$UNIT_NAME"
systemctl --user restart "$UNIT_NAME"
sleep 3
PID=$(pgrep -f "gaze_objects.cli batch --config $CFG" | head -n 1 || true)
[[ -n "${PID:-}" ]] && echo "$PID" >"$OUT/batch_run.pid"
echo "[ok] fast batch service=$UNIT_NAME pid=${PID:-unknown}"
echo "[ok] output_root=$OUT  stride=4  lean prompts (~16 concepts)"

# Point watchdog at fast output via env file for cron? Keep a dedicated watchdog line.
WATCH="$ROOT/scripts/watchdog_batch_dense_parts_hand_fast.sh"
cat >"$WATCH" <<'EOF'
#!/usr/bin/env bash
export BATCH_ROOT="${BATCH_ROOT:-$HOME/KHETRE/DINO_KHETRE/DINO}"
# Reuse logic by temporarily swapping paths via OUT override inside a thin copy:
set -euo pipefail
ROOT="${BATCH_ROOT:-$HOME/KHETRE/DINO_KHETRE/DINO}"
cd "$ROOT"
OUT=outputs/batch_sam3_dense_parts_hand_fast
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
# also treat systemd unit as alive
if [[ "$alive" -eq 0 ]] && systemctl --user is-active --quiet dense-parts-batch-fast.service 2>/dev/null; then
  alive=1
  etime="systemd"
  pid=$(systemctl --user show -p MainPID --value dense-parts-batch-fast.service 2>/dev/null || echo systemd)
  echo "$pid" >"$PID_FILE" 2>/dev/null || true
fi
log_age="missing"
log_stall=0
if [[ -f "$BATCH_LOG" ]]; then
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
rows=list(csv.DictReader(Path("outputs/batch_sam3_dense_parts_hand_fast/batch_qc_summary.csv").open()))
print(f"n={len(rows)} {dict(Counter(r['status'] for r in rows))}")
PY
)
fi
state="OK"
[[ "$alive" -eq 0 ]] && state="DEAD_OR_FINISHED"
[[ "$log_stall" -eq 1 ]] && state="STALL_LOG_SILENT"
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
EOF
chmod +x "$WATCH"

CRON_LINE="*/10 * * * * $WATCH"
if crontab -l 2>/dev/null | grep -qF "watchdog_batch_dense_parts_hand_fast.sh"; then
  echo "[ok] fast watchdog cron already installed"
else
  (crontab -l 2>/dev/null | grep -v "watchdog_batch_dense_parts_hand.sh" || true; echo "$CRON_LINE") | crontab -
  echo "[ok] installed fast watchdog cron (replaced old watchdog line if present)"
fi
bash "$WATCH" || true

echo
systemctl --user status "$UNIT_NAME" --no-pager -l | head -n 15 || true
echo
echo "Monitor:"
echo "  cat $OUT/watchdog_status.txt"
echo "  tail -f $OUT/batch_run.log"
echo "When done:"
echo "  python -m gaze_objects.cli aggregate-sequences --output-root $OUT"
