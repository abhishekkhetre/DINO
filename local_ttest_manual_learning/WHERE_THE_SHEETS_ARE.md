# Where the last successful batch sheets are

## Short answer

Yes — after the workstation → HPC migration, the finished corpus sheets should be on **bwUniCluster**, not on the workstation, and **not** in this Cursor cloud agent or in GitHub.

| Place | Has full `corpus_*.csv`? |
|---|---|
| Lab workstation | No longer the home of the data (moved away) |
| HPC `$WS/sheets/` and `$WS/outputs/<batch>/` | **Yes — expected location** |
| GitHub `abhishekkhetre/DINO` | **No** — only summarised tables |
| This Cursor VM (`/workspace/...`) | **No** — same summaries only; cannot SSH to HPC from here |

`WS=/pfs/work9/workspace/scratch/ka_wn6139-gaze_sam3`

Most likely last **finished** corpus batch names mentioned in project docs:

- `batch_sam3_full_fine` (full-video sequences; ~290 ok recordings in the presentation brief)
- denser hand/parts rebatch was **not** finished at migration time → do **not** wait for the pending Slurm job for these t-tests

The Oct 2026 Final Gaze Visual Report used tables under a path like `analysis/final_20261016` (compressed tokens / aggregates). Those sheets, if copied in the transfer, should also be under `$WS/` somewhere.

## On HPC — find the newest finished sheets

```bash
ssh ka_wn6139@uc3.scc.kit.edu
WS=/pfs/work9/workspace/scratch/ka_wn6139-gaze_sam3

# newest corpus attention sheets
find "$WS" -type f -name 'corpus_attention_sequences.csv' -printf '%T+ %p\n' 2>/dev/null | sort -r | head -20

# also look for the report snapshot
find "$WS" -type f \( -name '*Gaze*Report*.html' -o -name 'corpus_results.csv' \) -printf '%T+ %p\n' 2>/dev/null | sort -r | head -20

ls -la "$WS/sheets" 2>/dev/null
ls -la "$WS/outputs" 2>/dev/null | head
```

Copy the chosen sheet (plus expertise list if you have it) to your laptop, then into this agent folder:

`/workspace/local_ttest_manual_learning/01_inputs_needed/`

Minimum files for t-tests:

1. `corpus_attention_sequences.csv`  
2. `participants_expertise.csv` (`participant_id,expertise`)  

Optional: per-recording `task_duration_s` for the timestamp decay.
