# Status after IFAB uploads (21 Nov 2026)

## Present in this folder
| File | Use |
|---|---|
| `participants_expertise.csv` | Built from `Expertise.xlsx`: Trained Yes→Expert (17), No→Non-expert (20). IDs uppercased. |
| `Expertise.xlsx` | Original |
| `corpus_sequence_strings.csv` | One sequence string per recording (object order) |
| `corpus_results.csv` | Per-recording QC + sequence string |
| `Corpus_Analysis_Summary.xlsx` | Corpus rollup only (not per-participant shares) |

## Still missing for the planned t-tests
`corpus_attention_sequences.csv` (the Nov 2 HPC sheet inside `gaze_sheets_20261102.tar.gz`)

Why: the learning-curve metric in the report is **share of labelled fixation time** on the manual (duration-weighted).  
Sequence strings only give **token counts** (how many times “instruction manual” appears in the collapsed sequence), which is a **different** metric. We will not invent duration shares from token counts.

## Next step
Upload either:
- `gaze_sheets_20261102.tar.gz`, or
- unpacked `corpus_attention_sequences.csv`

into this folder, then t-tests can run.
