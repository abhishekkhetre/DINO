# Gaze results — download pack (7 Nov 2026)

## What is in this folder (available now)

| File | What it is |
|------|------------|
| `SUPERVISOR_FEEDBACK_RESULTS_INTERPRETATION.md` | Full answers to supervisor feedback on Final Gaze Visual Report |
| `tier_a_expertise_findings.csv` | Summarised expertise metrics (manual / screwdriver / AG / motifs) |
| `tier_a_process_structure.csv` | Summarised process / temporal / coverage findings |
| `claims_guardrails.csv` | Explicit can-claim / cannot-claim boundaries |
| `corpus_sheet_inventory.csv` | Map of every corpus sheet and where the full files live |
| `INTERNSHIP_PRESENTATION_BRIEF.md` | Full internship narrative / slide brief |
| `WORKSTATION_TO_HPC_TRANSFER.md` | Migration playbook (paths on bwUniCluster) |

These summarised sheets are derived from the **16 Oct 2026 Final Gaze Visual Report** snapshot (~370 recordings; 17 experts / 20 non-experts; ~19.7% fixation label coverage).

## Full raw corpus sheets (on HPC — not in this cloud VM)

Workspace:
```text
WS=/pfs/work9/workspace/scratch/ka_wn6139-gaze_sam3
```

Expected locations:
```text
$WS/sheets/                  # survival-kit copies of corpus_*.csv / batch_qc_*
$WS/outputs/<batch_name>/    # full batch roots with same sheets
```

### On HPC — zip the latest sheets

```bash
WS=/pfs/work9/workspace/scratch/ka_wn6139-gaze_sam3
mkdir -p "$WS/sheets"
# refresh survival kit from every batch root that has aggregates
find "$WS/outputs" -maxdepth 2 -type f \( \
  -name 'batch_qc_summary.csv' -o -name 'batch_summary.json' \
  -o -name 'corpus_*.csv' -o -name 'corpus_*.json' \
  -o -name '*Gaze*Report*.html' -o -name '*gaze*report*.html' \
\) -print0 | while IFS= read -r -d '' f; do
  bn=$(basename "$(dirname "$f")")
  mkdir -p "$WS/sheets/$bn"
  cp -n "$f" "$WS/sheets/$bn/" 2>/dev/null || cp "$f" "$WS/sheets/$bn/"
done
cd "$WS"
tar -czvf "$HOME/gaze_latest_sheets_$(date +%Y%m%d).tar.gz" sheets
ls -lh "$HOME/gaze_latest_sheets_"*.tar.gz
```

### On your laptop — download

```bash
scp ka_wn6139@uc3.scc.kit.edu:~/gaze_latest_sheets_*.tar.gz ~/Downloads/
# or pull the live folder:
scp -r ka_wn6139@uc3.scc.kit.edu:/pfs/work9/workspace/scratch/ka_wn6139-gaze_sam3/sheets ~/Downloads/gaze_sheets_from_hpc
```

If `sheets/` is empty on HPC, the aggregates may only sit under a specific output root — list them first:

```bash
WS=/pfs/work9/workspace/scratch/ka_wn6139-gaze_sam3
find "$WS/outputs" -maxdepth 2 -type f \( -name 'corpus_*.csv' -o -name 'batch_qc_summary.csv' \) | head -50
```

## Note on denser rebatch

The stride-4 / hand+AG-parts rebatch on HPC was **not finished** when data were migrated. Until that Slurm job completes and you re-run `aggregate-sequences`, the **latest finished corpus sheets** remain the prior mask/fine batch aggregates under `$WS/outputs/` and `$WS/sheets/`.
