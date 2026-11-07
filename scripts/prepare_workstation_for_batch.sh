#!/usr/bin/env bash
# Reduce GPU freezes during long SAM batches — LOCAL ONLY (no IT/SSH needed).
# Run once before starting/resuming a long batch, on the workstation keyboard.
#
#   bash scripts/prepare_workstation_for_batch.sh
set -euo pipefail

echo "=== prepare workstation for long CUDA batch ==="

# 1) Stop remote-desktop (often C+G on the NVIDIA GPU — major freeze source)
if systemctl --user status gnome-remote-desktop.service >/dev/null 2>&1; then
  systemctl --user stop gnome-remote-desktop.service || true
  systemctl --user disable gnome-remote-desktop.service 2>/dev/null || true
  echo "[ok] stopped gnome-remote-desktop (user)"
else
  echo "[i] gnome-remote-desktop not active"
fi

# 2) Lighten compositor
if command -v gsettings >/dev/null; then
  gsettings set org.gnome.desktop.interface enable-animations false 2>/dev/null || true
  gsettings set org.gnome.desktop.interface toolkit-accessibility false 2>/dev/null || true
  echo "[ok] disabled GNOME animations"
fi

# 3) Ask to close GPU browsers / players
echo "[!] Close Firefox / Chrome / video players on this machine (they use the GPU)."
if pgrep -x firefox >/dev/null 2>&1; then
  echo "    Detected firefox — recommend:  pkill firefox"
fi

# 4) Detect possible iGPU (best permanent cure: plug monitor into motherboard)
echo
echo "=== display GPUs (DRM) ==="
ls -1 /dev/dri/by-path 2>/dev/null || ls -1 /dev/dri 2>/dev/null || true
if lspci 2>/dev/null | grep -qiE 'VGA|3D|Display'; then
  lspci | grep -iE 'VGA|3D|Display' || true
fi
echo
echo "PERMANENT (best): If you see Intel/AMD + NVIDIA above, plug the monitor"
echo "  into the MOTHERBOARD video port (not the NVIDIA card), enable iGPU in BIOS,"
echo "  then reboot once. NVIDIA becomes compute-only → freezes become rare."
echo

# 5) Persistence mode (may need sudo — ignore if denied)
if command -v nvidia-smi >/dev/null; then
  nvidia-smi -pm 1 2>/dev/null && echo "[ok] NVIDIA persistence mode on" \
    || echo "[i] persistence mode needs:  sudo nvidia-smi -pm 1"
  nvidia-smi --query-gpu=name,memory.used,memory.total,temperature.gpu --format=csv,noheader
fi

# 6) Disk headroom
df -h . | tail -n 1
AVAIL_G=$(df -BG . | awk 'NR==2 {gsub(/G/,""); print $4}')
if [[ -n "${AVAIL_G:-}" ]] && [[ "$AVAIL_G" -lt 80 ]]; then
  echo "[!] Low disk (~${AVAIL_G}G free). Free space to avoid hangs (aim >100G)."
fi

echo
echo "=== if the screen freezes LATER (no reboot) ==="
echo "  Ctrl+Alt+F3 → login → then:"
echo "    killall -SIGQUIT gnome-shell"
echo "  or only check batch:"
echo "    bash scripts/check_batch_dense_parts_hand.sh"
echo "  Ctrl+Alt+F2 (or F1) to return to GUI."
echo
echo "Done. Start/leave the batch running. Avoid using this desktop for browsing."
