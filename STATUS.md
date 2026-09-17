# Status

Last updated: 2026-09-17.

## Completed

### Stage A — TSV audit
Reproducible on AM07; SHA-256 and Section 7 counts matched.

### Stage B — sync + overlay
- Pilot overlay generated and **visually verified against Tobii**.
- Sync config marked `verified`.

### Workstation GPU stack (Ubuntu)
- RTX 3080 Ti, driver 595.91, conda `dino_gpu`, PyTorch cu118.
- IDEA-Research DINO ops built (`MSDA OK`).
- GroundingDINO installed under `~/gaze_project/GroundingDINO` with `transformers==4.37.2`.

### Stage C/D — AM07 pilot (COCO IDEA-DINO)
- 60 frames, 135 detections.
- Stage D: **0%** conditional AOI agreement (COCO names ≠ study AOIs).

### Stage C/D — AM07 pilot (Grounding DINO)
- Same 30–60 s clip, 60 frames, **597** detections.
- Conditional AOI agreement: **55%** (60/109) before phrase-map fix.
- Strong classes when assigned: Angle Grinder / Boxes / Tools.
- Main error: raw phrase `instruction` not mapped to `Manual` (48 cases).
- Fix in `label_map.py` (`instruction` → `Manual`); assign re-applies map so **re-detect not required**.

### Pipeline package
- `gaze_objects`: audit, sync, overlay, detect, assign, evaluate.
- Detectors: `mock`, `idea_dino`, `grounding_dino`.

## Current goal

Re-run **assign + evaluate** on existing Grounding detections after pulling the `instruction`→`Manual` map fix. Expect conditional agreement near ceiling on this clip (Manual FP should collapse). Then Stage E or a second recording pilot.

## Not yet

- Stage E attention sequences
- Batch / full ~400 recordings (wait until category detection is credible on more than one clip)

## Notes

- Assigned baseline remains IDEA-Research DINO; Grounding DINO is the stronger AM07 pilot for AOI-name agreement.
- Prompts and phrase→AOI maps are provisional hypotheses.
- Do not scale to all videos until agreement holds beyond this single pilot.
