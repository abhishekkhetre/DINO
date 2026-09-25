# Status

Last updated: 2026-09-25.

## Completed

### Stage A–D AM07 pilots
See prior rows: SAM3 v2 + same-AOI assign ≈ **97%** conditional AOI agreement on the sparse 60-frame clip.

### Stage E (sparse v2)
- 39 fixations, 20 labelled, 12 sequences (Manual / Angle Grinder / Tools / Boxes).

### Pipeline
- Detectors: `mock`, `idea_dino`, `grounding_dino`, `sam3`
- Stage E: `sequences` CLI with optional quality gates

## Current goal

**5-video dense SAM3 smoke batch** (`configs/batch_sam3_dense_5.yaml`):
`data_dir` → `~/KHETRE/Tobii_Data` (recursive discover) → up to 5 TSV+video pairs →
detect → assign → evaluate → sequences → `batch_qc_summary.csv`.

## Not yet

- Full ~400 batch
- Process-step alignment / dwell-transition analysis
