# Status

Last updated: 2026-09-30.

## Completed

### Full corpus dense SAM3 batch
290/291 ok (1 faulty TSV quarantined). Median conditional accuracy **1.0**.

### Low-acc tail (21 recordings) — confusion patterns
1. **AG → Manual** (dominant): Manual boxes steal Angle Grinder gaze  
   (`LE07UF1_*`, `ER10WE06_06`, `KJ03JM25_06`, `KI05KO01_02`, …).
2. **Tools → Manual/Boxes/AG**: Tobii Tools AOI with **no Tools detector**  
   (tool prompts removed on purpose). Expected until Tools coverage returns.
3. **Boxes ↔ AG**: smaller secondary confusion.

### Tools-excluded QC (offline from evaluation_joined)
mean **0.958** / median **1.0** · **14** recordings &lt; 0.80 (was 21).

## Current goal
Re-detect 11 AG→Manual heavies via `configs/batch_sam3_dense_ag_manual_fix.yaml`
(`instruction manual: 0.70`). Leave Boxes↔AG cases (`KI05KO01_03`, `ER05RG05_08`) aside.

## Not yet
- Safe Tools AOI detector coverage
- Process-step / dwell-transition analysis
