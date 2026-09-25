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

**Dense SAM3 + gated Stage E** on AM07 (`configs/am07_stage_c_sam3_dense.yaml`):
- stride 2, max_frames 150
- `min_assigned_samples: 2`, `min_label_fraction: 0.5`
- output: `outputs/am07_stage_c_sam3_dense/`

Then second recording pilot before any full batch.

## Not yet

- Second-recording validation
- Batch / ~400 videos
- Process-step alignment / dwell-transition analysis
