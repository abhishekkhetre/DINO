# Status

Last updated: 2026-09-30.

## Completed

### Full corpus dense SAM3 batch
Config: `configs/batch_sam3_dense_all.yaml`  
QC: `outputs/batch_sam3_dense/batch_qc_summary.csv`

| metric | value |
|--------|------:|
| n_recordings | 291 |
| n_ok | **290** |
| n_error | **1** (`NN08WE29_01_fehlerhaft` — no Eye Tracker rows; quarantine) |
| cond_acc mean / median | ~0.94 / 1.0 |
| cond_acc < 0.80 | 21 |

Empty-fixation Stage E crash fixed; 3 former errors now `ok` with `conditional_accuracy=NaN` (no labelled assigned samples in 30–60s clip).

## Current goal

Triage the **21 recordings with cond_acc &lt; 0.80** (confusion matrices); decide
prompt/assign fixes vs accept as hard scenes. Then dwell/sequence analysis on the
usable corpus.

## Not yet

- Safe Tools AOI detector coverage
- Process-step / dwell-transition analysis
- Exclude `*_fehlerhaft` from discover/batch by default
