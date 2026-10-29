#!/usr/bin/env bash
# Install a user systemd service that keeps the dense batch alive across
# crashes/reboots (skip_if_done resumes finished recordings).
#
# One-time on the workstation:
#   cd ~/KHETRE/DINO_KHETRE/DINO
#   git pull origin cursor/batch-dense-parts-hand-e54c
#   bash scripts/install_batch_systemd_user.sh
#
# Then:
#   systemctl --user start dense-parts-batch.service
#   systemctl --user status dense-parts-batch.service
#   journalctl --user -u dense-parts-batch.service -f
#
# Enable start after login/boot (with lingering):
#   loginctl enable-linger "$USER"
#   systemctl --user enable dense-parts-batch.service
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
UNIT_DIR="${XDG_CONFIG_HOME:-$HOME/.config}/systemd/user"
UNIT="$UNIT_DIR/dense-parts-batch.service"
OUT="$ROOT/outputs/batch_sam3_dense_parts_hand"
CONDA_SH="${CONDA_SH:-$HOME/conda/etc/profile.d/conda.sh}"
# Fallback common locations
for c in "$HOME/conda/etc/profile.d/conda.sh" "$HOME/miniconda3/etc/profile.d/conda.sh" "$HOME/anaconda3/etc/profile.d/conda.sh"; do
  [[ -f "$c" ]] && CONDA_SH="$c" && break
done

mkdir -p "$UNIT_DIR" "$OUT"

WRAP="$ROOT/scripts/_systemd_dense_parts_batch.sh"
cat >"$WRAP" <<EOF
#!/usr/bin/env bash
set -euo pipefail
cd "$ROOT"
# Lighten desktop GPU load each (re)start — ignore failures
bash "$ROOT/scripts/prepare_workstation_for_batch.sh" >> "$OUT/batch_run.log" 2>&1 || true
# shellcheck disable=SC1090
source "$CONDA_SH"
conda activate sam3
export PYTHONPATH=src:\${PYTHONPATH:-}
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
mkdir -p "$OUT"
echo "======== systemd batch start \$(date -Is) ========" >> "$OUT/batch_run.log"
exec python -m gaze_objects.cli batch --config configs/batch_sam3_dense_parts_hand.yaml >> "$OUT/batch_run.log" 2>&1
EOF
chmod +x "$WRAP"

cat >"$UNIT" <<EOF
[Unit]
Description=Dense SAM3 parts-hand corpus batch (CSV sheets)
After=default.target

[Service]
Type=simple
WorkingDirectory=$ROOT
ExecStart=$WRAP
Restart=on-failure
RestartSec=60
Nice=5
# Do not kill on logout if lingering is enabled
KillMode=mixed
TimeoutStopSec=30

[Install]
WantedBy=default.target
EOF

systemctl --user daemon-reload
echo "Installed: $UNIT"
echo "Wrapper:   $WRAP"
echo
echo "NEXT (read carefully if a batch is already running with nohup/pid 3975):"
echo "  1) Only start the service when NO other batch python is using the GPU."
echo "  2) systemctl --user start dense-parts-batch.service"
echo "  3) loginctl enable-linger \"\$USER\"   # so it survives SSH logout / reboot login"
echo "  4) systemctl --user enable dense-parts-batch.service"
echo
echo "Status:  systemctl --user status dense-parts-batch.service"
echo "Logs:    journalctl --user -u dense-parts-batch.service -f"
echo "         tail -f $OUT/batch_run.log"
