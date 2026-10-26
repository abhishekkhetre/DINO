#!/usr/bin/env bash
# AFTER REBOOT — one script for a more freeze-resistant, auto-resuming batch.
# No IT/SSH required. Run on the workstation (prefer Ctrl+Alt+F3 text login).
#
#   cd ~/KHETRE/DINO_KHETRE/DINO
#   git pull origin cursor/batch-dense-parts-hand-e54c
#   bash scripts/start_stable_dense_batch.sh
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
OUT=outputs/batch_sam3_dense_parts_hand
mkdir -p "$OUT"

echo "======== start_stable_dense_batch $(date -Is) ========"

# --- refuse if something already owns the GPU ---
if command -v nvidia-smi >/dev/null; then
  if nvidia-smi --query-compute-apps=pid,process_name --format=csv,noheader 2>/dev/null | grep -qiE 'python|torch'; then
    echo "ERROR: a Python/CUDA job is already on the GPU. Stop it first." >&2
    nvidia-smi
    exit 4
  fi
fi

# --- lighten desktop load on this GPU ---
bash "$ROOT/scripts/prepare_workstation_for_batch.sh" || true
pkill -x firefox 2>/dev/null || true
pkill -x chrome 2>/dev/null || true

# --- install / refresh systemd user service ---
bash "$ROOT/scripts/install_batch_systemd_user.sh"

# linger: service can run without an interactive desktop session
loginctl enable-linger "$USER" 2>/dev/null || echo "[!] loginctl enable-linger failed (continue anyway)"

systemctl --user daemon-reload
systemctl --user enable dense-parts-batch.service
systemctl --user restart dense-parts-batch.service

# record pid for watchdog/check scripts
sleep 3
PID=$(pgrep -f "gaze_objects.cli batch --config configs/batch_sam3_dense_parts_hand" | head -n 1 || true)
if [[ -n "${PID:-}" ]]; then
  echo "$PID" >"$OUT/batch_run.pid"
  echo "[ok] batch service running pid=$PID"
else
  echo "[!] service started but python pid not found yet — check status below" >&2
fi

# --- watchdog cron (idempotent) ---
CRON_LINE="*/10 * * * * $ROOT/scripts/watchdog_batch_dense_parts_hand.sh"
if crontab -l 2>/dev/null | grep -qF "watchdog_batch_dense_parts_hand.sh"; then
  echo "[ok] watchdog cron already installed"
else
  (crontab -l 2>/dev/null || true; echo "$CRON_LINE") | crontab -
  echo "[ok] installed watchdog cron"
fi
bash "$ROOT/scripts/watchdog_batch_dense_parts_hand.sh" || true

echo
echo "=== status ==="
systemctl --user status dense-parts-batch.service --no-pager -l | head -n 20 || true
echo
echo "=== IMPORTANT — keep freezes rare ==="
echo "1) Prefer working from text console: Ctrl+Alt+F3 (leave GUI unused)."
echo "2) Do NOT open Firefox / remote-desktop / video on this PC during the run."
echo "3) If GUI freezes: Ctrl+Alt+F3 → killall -SIGQUIT gnome-shell  (do NOT reboot)."
echo "4) Daily check:  cat $OUT/watchdog_status.txt"
echo "5) Live log:     tail -f $OUT/batch_run.log"
echo "6) Service:      systemctl --user status dense-parts-batch.service"
echo
echo "If motherboard has Intel/AMD video: plug monitor there (iGPU) for best stability."
