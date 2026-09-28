# Status

Last updated: 2026-09-28.

## Completed

### Stage A–D AM07 pilots
SAM3 + same-AOI assign ≈ **97%** conditional AOI on sparse pilot; dense AM07 ~98–100%.

### Stage E
Fixation → attended-object sequences with quality gates.

### Pipeline
- Detectors: `mock`, `idea_dino`, `grounding_dino`, `sam3`
- Batch: `discover-pairs` + `batch` over `~/KHETRE/Tobii_Data`
- Assembly prompts: `metaSAM3/prompts_assembly_aoi.txt` (grinder / manual / box; no tools)
- Assign: `prefer_nested_parent` for **Tools→Angle Grinder only** (not Manual)

### 5-video dense SAM3 smoke — stable QC
Config: `configs/batch_sam3_dense_5.yaml`  
Output: `outputs/batch_sam3_dense_5/batch_qc_summary.csv`

| recording | cond. acc | cond. n | dets | labelled fix. | sequences |
|-----------|----------:|--------:|-----:|--------------:|----------:|
| AM07AM07_01 | 0.984 | 307 | 2376 | 21 | 12 |
| AA05ED02_01 | 0.988 | 322 | 1434 | 14 | 5 |
| AA05ED02_02 | 1.000 | 341 | 841 | 19 | 4 |
| AA05ED02_03 | 0.906 | 212 | 923 | 9 | 3 |
| AA05ED02_04 | 1.000 | 407 | 1245 | 15 | 6 |

- **5/5 ok**. Four recordings ≥0.98; 03 at 0.91 with residual **AG→Manual** (20).
- Do not nest Manual under AG (collapsed AM07). Tools prompts steal AG — deferred.

## Current goal

**Full corpus batch** via `configs/batch_sam3_dense_all.yaml` (`select.mode: all`,
incremental QC, `skip_if_done`). Reuse the 5 finished smoke dirs under
`outputs/batch_sam3_dense/`.

## Not yet

- Safe Tools AOI detector coverage
- Process-step alignment / dwell-transition analysis
- Optional: raise manual threshold + re-detect AA05ED02_03 only
