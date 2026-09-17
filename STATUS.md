# Status

Last updated: 2026-09-16.

## Completed

### Stage A — TSV audit
Reproducible on AM07; SHA-256 and Section 7 counts matched.

### Stage B — sync + overlay
- Pilot overlay generated and **visually verified against Tobii**.
- Sync config marked `verified`.

### Workstation GPU stack (Ubuntu)
- RTX 3080 Ti, driver 595.91, conda `dino_gpu`, PyTorch cu118.
- IDEA-Research DINO ops built (`MSDA OK`).
- AM07 Stage C with `idea_dino` completed (60 frames, 135 detections).
- Stage D evaluate completed: **0% AOI name agreement** because COCO labels (`refrigerator`, `person`, `hair drier`) do not match study AOIs.

### Pipeline package
- `gaze_objects`: audit, sync, overlay, detect, assign, evaluate.
- Detectors: `mock`, `idea_dino`; Grounding DINO adapter added locally (pending push/pull if not on remote yet).

## Current goal

Try **Grounding DINO** (text prompts for study objects) on the same AM07 30–60 s clip, then re-run assign + evaluate. Goal: labels closer to `Manual` / `Tools` / `Boxes` / `Angle Grinder`.

## Not yet

- Stage E attention sequences
- Batch / full ~400 recordings (wait until category detection is credible)

## Notes

- Assigned baseline remains IDEA-Research DINO; Grounding DINO is a comparison backend.
- Prompts and phrase→AOI maps are provisional hypotheses.
- Do not scale to all videos while category agreement stays near zero.
