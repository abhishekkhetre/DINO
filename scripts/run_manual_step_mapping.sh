#!/usr/bin/env bash
# Map corpus attention sequences onto the instruction-manual steps (CPU only).
# Run on the bwUniCluster login node (a few minutes) or inside any job.
#
#   bash scripts/run_manual_step_mapping.sh                       # newest corpus sheet
#   SEQ=/path/corpus_attention_sequences.csv bash scripts/run_manual_step_mapping.sh
#   PARTICIPANTS=$WS/sheets/participants_expertise.csv ANNOTATIONS=... bash ...
set -euo pipefail

WS="${WS:-/pfs/work9/workspace/scratch/ka_wn6139-gaze_sam3}"
cd "$(dirname "$0")/.."
export PYTHONPATH=src:${PYTHONPATH:-}

if [[ -z "${SEQ:-}" ]]; then
  SEQ=$(find "$WS/outputs" "$WS/sheets" -name 'corpus_attention_sequences.csv' -printf '%T@ %p\n' 2>/dev/null \
        | sort -nr | head -1 | cut -d' ' -f2-)
fi
if [[ -z "${SEQ:-}" || ! -f "$SEQ" ]]; then
  echo "No corpus_attention_sequences.csv found. Set SEQ=/path/to/file." >&2
  exit 1
fi

STAMP="$(date +%Y%m%d_%H%M)"
OUT="${OUT:-$WS/outputs/manual_step_mapping_$STAMP}"
ARGS=(--sequences "$SEQ" --out "$OUT" --em-iterations "${EM_ITERS:-5}")
if [[ -z "${PARTICIPANTS:-}" && -f "$WS/sheets/participants_expertise.csv" ]]; then
  PARTICIPANTS="$WS/sheets/participants_expertise.csv"
fi
[[ -n "${PARTICIPANTS:-}" ]] && ARGS+=(--participants "$PARTICIPANTS")
[[ -n "${ANNOTATIONS:-}" ]] && ARGS+=(--annotations "$ANNOTATIONS")
[[ -n "${PANELS_ROOT:-}" ]] && ARGS+=(--panels-root "$PANELS_ROOT")

echo "[manual-steps] sequences: $SEQ"
echo "[manual-steps] output:    $OUT"
python -m gaze_objects.cli map-manual-steps "${ARGS[@]}"

tar -C "$(dirname "$OUT")" -czf "$HOME/manual_step_mapping_$STAMP.tar.gz" "$(basename "$OUT")"
echo
echo "Sheets: $OUT"
echo "Laptop download:"
echo "  scp ka_wn6139@uc3.scc.kit.edu:~/manual_step_mapping_$STAMP.tar.gz ~/Downloads/"
