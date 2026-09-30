# Status

Last updated: 2026-09-30.

## Completed

### Pipeline
- Detectors: `mock`, `idea_dino`, `grounding_dino`, `sam3`
- Batch over `~/KHETRE/Tobii_Data` with tool-free assembly prompts
- Assign: `prefer_nested_parent` for Tools→Angle Grinder only

### Full corpus dense SAM3 batch (workstation)
Config: `configs/batch_sam3_dense_all.yaml`  
QC: `outputs/batch_sam3_dense/batch_qc_summary.csv`

| metric | value |
|--------|------:|
| n_recordings | 291 |
| n_ok | 287 |
| n_error | 4 |
| cond_acc mean / median | 0.942 / 1.0 |
| cond_acc min–max | 0.029–1.0 |
| cond_acc < 0.90 | 37 |
| cond_acc < 0.80 | 21 |

- **Median accuracy 1.0** — most recordings look strong.
- Tail: ~21 recordings <0.80 (cluster: `KI05KO01_*`, `LE07UF1_*`, `KJ03JM25_*`, `ER*` outliers).
- 4 hard errors: 3× `KeyError: attended_status` (empty fixations in clip) + 1× `NN08WE29_01_fehlerhaft` (bad TSV).

## Current goal

Patch empty-fixation / bad-TSV crash; re-run only the 4 failed IDs. Then triage
low-acc cluster (confusion matrices) without re-running the full 291.

## Not yet

- Safe Tools AOI detector coverage
- Process-step / dwell-transition analysis
- Deep dive on <0.80 outliers
