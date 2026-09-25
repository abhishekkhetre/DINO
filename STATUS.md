# Status

Last updated: 2026-09-24.

## Completed

### Stage A — TSV audit
Reproducible on AM07; SHA-256 and Section 7 counts matched.

### Stage B — sync + overlay
- Pilot overlay generated and **visually verified against Tobii**.
- Sync config marked `verified`.

### Workstation GPU stack (Ubuntu)
- RTX 3080 Ti; conda `dino_gpu` (Grounding) and conda `sam3` (Meta SAM 3).
- GroundingDINO + IDEA-Research DINO previously validated.

### Stage C/D — AM07 pilots (same 30–60 s, 60 frames)

| Backend | Detections | Assigned / ambiguous | Conditional AOI accuracy |
| --- | ---: | --- | ---: |
| COCO IDEA-DINO | 135 | — | **0%** |
| Grounding DINO (raw) | 597 | 112 / 82 | **~55%** (60/109) |
| SAM 3 (thr 0.30, +tool) | 1938 | 16 / 219 | unusable (almost all ambiguous) |
| **SAM 3 tuned (thr 0.55, no tool)** | **512** | **150 / 9** | **~91.5%** (130/142) |
| **SAM 3 v2 (grinder synonyms)** | **1198** | **95 / 129** | **~95.5%** (85/89); Angle Grinder recall 0.8 |
| **SAM 3 v2 + same-AOI assign** | 1198 | **145 / 79** | **~97.1%** (132/136); Angle Grinder recall **0.94** |

SAM 3 tuned: Boxes/Tools perfect; Manual strong; **Angle Grinder → all called Manual** (12/12).  
SAM 3 v2: Angle Grinder recovered; synonym overlaps raised ambiguous until same-AOI merge.

### Pipeline package
- Detectors: `mock`, `idea_dino`, `grounding_dino`, **`sam3`**.
- Stage E: `gaze_objects.cli sequences` (fixation majority → attention runs).

## Current goal

Run Stage E on AM07 SAM3 v2 outputs (no re-detect):
`python -m gaze_objects.cli sequences --config configs/am07_stage_c_sam3.yaml`

## Not yet

- Batch / full ~400 recordings
- Second-recording SAM3 validation
- Gaze-in-mask / video tracker upgrades
