# Status

Last updated: 2026-10-06.

## Fine SAM3 batch — DONE on workstation

`outputs/batch_sam3_full_fine/` finished **290/290 ok** (detect → assign → sequences).

| Metric | Value |
| --- | --- |
| Recordings | 290 |
| Errors | 0 |
| Frames processed | ~192k |
| Detections | ~1.27M |
| Fixations | ~126k |
| Fixations labelled | ~26k |
| Sequences | ~18k |

`conditional_accuracy` is empty on purpose — this run did **not** evaluate against Tobii AOIs
(fine labels ≠ AOI names).

## Same deliverable as previous corpus (next step)

On the workstation, build the corpus tables (like the earlier AOI sequence export):

```bash
cd ~/KHETRE/DINO_KHETRE/DINO
conda activate sam3
git pull origin cursor/sam3-fine-labels-full-e54c
export PYTHONPATH=src:$PYTHONPATH

python -m gaze_objects.cli aggregate-sequences \
  --output-root outputs/batch_sam3_full_fine
```

Writes under `outputs/batch_sam3_full_fine/`:

| File | What |
| --- | --- |
| `corpus_sequence_strings.csv` | one gaze-order string per ID (e.g. `angle grinder > screwdriver > …`) |
| `corpus_transitions.csv` | from→to transition counts |
| `corpus_attention_sequences.csv` | all sequence rows |
| `corpus_results.csv` | QC metrics + sequence_string (one row per recording) |
| `corpus_sequences_summary.json` | counts + top transitions |

IDs only:

```bash
python -m gaze_objects.cli discover-pairs \
  --data-dir /home/ifab-agiprobotw-0001/KHETRE/Tobii_Data \
  --ids-only --out outputs/tobii_inventory
```
