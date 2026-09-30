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

## Current goal
- Score QC with `evaluation.ignore_reference_labels: [Tools]`.
- Raise `instruction manual` detect threshold to 0.70; re-detect only AG→Manual
  heavy outliers (not full 291).
- Skip `*fehlerhaft*` in discover.

## Not yet
- Safe Tools AOI detector coverage
- Process-step / dwell-transition analysis
