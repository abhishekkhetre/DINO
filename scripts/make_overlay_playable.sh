#!/usr/bin/env bash
# Convert an OpenCV mp4v overlay to playable H.264 (Windows Media Player / browsers).
# Usage:
#   bash scripts/make_overlay_playable.sh outputs/am07_sam3_granular_overlay/sam_gaze_overlay.mp4

set -euo pipefail
SRC="${1:-outputs/am07_sam3_granular_overlay/sam_gaze_overlay.mp4}"
DST="${2:-${SRC%.mp4}_playable.mp4}"

if [[ ! -f "$SRC" ]]; then
  echo "ERROR: missing $SRC" >&2
  exit 2
fi

FFMPEG="$(command -v ffmpeg || true)"
if [[ -z "$FFMPEG" ]]; then
  for c in \
    "$HOME/miniconda3/envs/sam3/bin/ffmpeg" \
    "$HOME/miniconda3/bin/ffmpeg" \
    "$HOME/anaconda3/envs/sam3/bin/ffmpeg" \
    /usr/bin/ffmpeg
  do
    [[ -x "$c" ]] && FFMPEG="$c" && break
  done
fi

if [[ -z "${FFMPEG:-}" ]]; then
  echo "ffmpeg not found. Install into the sam3 env (no sudo needed):"
  echo "  conda activate sam3"
  echo "  conda install -y -c conda-forge ffmpeg"
  exit 3
fi

echo "[playable] $FFMPEG -i $SRC → $DST"
"$FFMPEG" -y -i "$SRC" -c:v libx264 -pix_fmt yuv420p -preset veryfast -crf 20 \
  -movflags +faststart -an "$DST"
echo "[playable] wrote $DST"
ls -lh "$DST"
