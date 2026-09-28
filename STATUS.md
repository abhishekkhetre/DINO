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
- Batch: `discover-pairs` + `batch` over `~/KHETRE/Tobii_Data`

### 5-video dense SAM3 smoke batch (workstation)
Config: `configs/batch_sam3_dense_5.yaml` → `outputs/batch_sam3_dense_5/batch_qc_summary.csv`

| recording | frames | dets | cond. AOI acc | cond. n | fixations | labelled | sequences |
|-----------|-------:|-----:|--------------:|--------:|----------:|---------:|----------:|
| AM07AM07_01 | 150 | 2972 | 0.997 | 319 | 39 | 22 | 12 |
| AA05ED02_01 | 150 | 1934 | 0.972 | 327 | 52 | 15 | 6 |
| AA05ED02_02 | 150 | 1391 | 1.000 | 341 | 60 | 19 | 4 |
| AA05ED02_03 | 150 | 1481 | **0.443 → 0.482** | 318→342 | 45 | 13 | 5 |
| AA05ED02_04 | 150 | 1829 | 1.000 | 411 | 68 | 16 | 7 |

- **5/5 ok**, 0 errors
- `AA05ED02_03` failure = **Angle Grinder → screwdriver (Tools)** (165 rows); nested-parent
  re-assign alone only +4pp (no grinder box around many screwdrivers).
- Fix in flight: drop screwdriver/wrench prompts (`prompts_assembly_aoi.txt`) + stronger
  nested/near parent assign → **requires re-detect**.

## Current goal

Re-detect + re-assign the 5-video batch with assembly AOI prompts; confirm
`AA05ED02_03` conditional accuracy recovers toward the other recordings.

## Not yet

- Full ~400 batch
- Process-step alignment / dwell-transition analysis
