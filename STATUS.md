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
Config: `configs/batch_sam3_dense_5.yaml` → tool-free `prompts_assembly_aoi.txt`

| recording | cond. acc | cond. n | dets | labelled fix. | sequences |
|-----------|----------:|--------:|-----:|--------------:|----------:|
| AM07AM07_01 | 0.984 | 307 | 2376 | 21 | 12 |
| AA05ED02_01 | 0.988 | 322 | 1434 | 14 | 5 |
| AA05ED02_02 | 1.000 | 341 | 841 | 19 | 4 |
| AA05ED02_03 | **0.906** | 212 | 923 | 9 | 3 |
| AA05ED02_04 | 1.000 | 407 | 1245 | 15 | 6 |

- **5/5 ok**. Dropping tool prompts fixed AG→Tools theft on 03 (was 0.44–0.47).
- Tradeoff on 03: smaller assigned+labelled n (409→212) and fewer labelled fixations (14→9).
- Tobii **Tools** AOI has no detector concept in this prompt set (deferred).

## Current goal

Remaining 03 errors are **AG→Manual** (20/212). Prefer Manual nested under AG
on re-assign; optionally raise `instruction manual` detect threshold. Then scale.

## Not yet

- Full ~400 batch
- Safe Tools AOI detector coverage
- Process-step alignment / dwell-transition analysis
