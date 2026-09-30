# Status

Last updated: 2026-09-30.

## Completed
- Trimmed 30–60s corpus batch (~290 ok, median AOI acc ~1.0 Tools-excluded).
- Full-video path: `clip.mode=full` + no `max_frames` cap + sequence aggregation.

## Current goal
**Full-recording batch** for gaze-order sequences (`Angle Grinder > Manual > Boxes > …`).

Config: `configs/batch_sam3_full_sequences.yaml`  
Output: `outputs/batch_sam3_full/`  
Aggregate: `corpus_sequence_strings.csv`, `corpus_transitions.csv`

## Notes
- Full video is much slower than the 30s pilot (stride=5, uncapped frames).
- Primary deliverable is Stage E attention sequences / corpus sequence strings.
- Tools AOI still excluded from scoring; assembly prompts unchanged.
