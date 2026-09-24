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

SAM 3 tuned: Boxes/Tools perfect; Manual strong; **Angle Grinder → all called Manual** (12/12).

### Pipeline package
- Detectors: `mock`, `idea_dino`, `grounding_dino`, **`sam3`**.

## Current goal

1. Persist tuned SAM 3 settings + per-concept thresholds for grinder.
2. Re-run AM07 SAM 3 detect/assign/evaluate; check Angle Grinder recall.
3. Then Stage E (attention sequences) or a second recording.

## Not yet

- Stage E attention sequences
- Batch / full ~400 recordings
