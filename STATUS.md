# Status

Last updated: 2026-09-28.

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

Baseline (pilot prompts + screwdriver):

| recording | cond. acc | dets |
|-----------|----------:|-----:|
| AM07AM07_01 | 0.997 | 2972 |
| AA05ED02_01 | 0.972 | 1934 |
| AA05ED02_02 | 1.000 | 1391 |
| AA05ED02_03 | **0.443** | 1481 |
| AA05ED02_04 | 1.000 | 1829 |

After `tool`/`tools` assembly prompts (re-detect):

| recording | cond. acc | dets |
|-----------|----------:|-----:|
| AM07AM07_01 | 1.000 | 7343 |
| AA05ED02_01 | 0.893 | 5010 |
| AA05ED02_02 | 0.959 | 4245 |
| AA05ED02_03 | **0.469** | 2970 |
| AA05ED02_04 | 0.987 | 4906 |

- 03 still broken; generic `tool`/`tools` inflated detections and did not fix AG theft.
- Next prompts: **grinder / manual / storage box only** (no tool concepts).

## Current goal

Re-detect with tool-free `prompts_assembly_aoi.txt`; confirm 03 recovers without
regressing the other four.

## Not yet

- Full ~400 batch
- Tobii "Tools" AOI detector coverage (deferred — needs safer prompts)
- Process-step alignment / dwell-transition analysis
