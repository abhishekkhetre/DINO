#!/usr/bin/env bash
# Pack latest corpus / QC / report sheets on bwUniCluster for laptop download.
# Run on HPC login or compute node (no GPU needed).
set -euo pipefail

WS="${WS:-/pfs/work9/workspace/scratch/ka_wn6139-gaze_sam3}"
STAMP="$(date +%Y%m%d_%H%M)"
OUT_TAR="${OUT_TAR:-$HOME/gaze_latest_sheets_${STAMP}.tar.gz}"

if [[ ! -d "$WS" ]]; then
  echo "Workspace not found: $WS" >&2
  echo "Set WS=... to your Lustre path (ws_find gaze_sam3)." >&2
  exit 1
fi

mkdir -p "$WS/sheets/_index"
INDEX="$WS/sheets/_index/found_sheets_${STAMP}.txt"
: > "$INDEX"

echo "[pack] scanning $WS/outputs and $WS/sheets ..."
while IFS= read -r -d '' f; do
  rel="${f#"$WS/"}"
  dest_dir="$WS/sheets/_mirror/$(dirname "$rel")"
  mkdir -p "$dest_dir"
  cp -f "$f" "$dest_dir/"
  echo "$rel" >> "$INDEX"
done < <(find "$WS/outputs" "$WS/sheets" -type f \( \
  -name 'batch_qc_summary.csv' \
  -o -name 'batch_summary.json' \
  -o -name 'corpus_*.csv' \
  -o -name 'corpus_*.json' \
  -o -iname '*gaze*report*.html' \
  -o -iname '*Final_Gaze*' \
\) -print0 2>/dev/null)

n=$(wc -l < "$INDEX" | tr -d ' ')
echo "[pack] collected $n sheet files into $WS/sheets/_mirror"
if [[ "$n" -eq 0 ]]; then
  echo "[pack] WARNING: no corpus/QC sheets found yet under outputs/." >&2
  echo "[pack] After batch+aggregate, re-run this script." >&2
fi

# Also include any flat copies already in sheets/
tar -C "$WS" -czf "$OUT_TAR" \
  sheets/_mirror \
  sheets/_index \
  $(find sheets -maxdepth 1 -type f -printf 'sheets/%f\n' 2>/dev/null || true)

ls -lh "$OUT_TAR"
echo
echo "Download on laptop:"
echo "  scp ka_wn6139@uc3.scc.kit.edu:$OUT_TAR ~/Downloads/"
