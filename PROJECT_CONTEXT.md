# Project context

Derived from the 9 September 2026 handoff brief. Update when evidence or supervisor instructions change.

## Confirmed

- Internship goal: workflow for real-world gaze recordings; immediate focus is attended-object identification in the **scene video** (not Pro Lab writeback / reference-image geometry as a prerequisite).
- Hardware described for detector work: NVIDIA GeForce RTX 3080 on a workstation (must be verified on the machine actually used for inference).
- Assigned detector: IDEA-Research DINO (DETR with improved denoising anchor boxes). Separate from Grounding DINO and Meta DINOv2.
- Tobii provides gaze coordinates; we do not estimate gaze from eye images in this prototype.
- Pilot recording identity fields from export: participant `AM07AM07`, recording `Recording 50`, project `CRC 1574`, media `scenevideo.mp4`, scene 1920×1080.
- Expected TSV fingerprint (handoff): size 45,916,285 bytes; SHA-256 `10cea3c917e2b7df1c39f6101845390977c5c46e3d0b195a62af248f0571350a`.

## Implementation proposals (not supervisor mandates)

- Offline Python package `gaze_objects` with staged CLI.
- Millisecond timestamp hypothesis and `video_time_s = recording_time_s - video_start_recording_s` offset model.
- Box-containment gaze assignment after detection (Stage C).
- Reuse Tobii I-VT fixations for early temporal aggregation (Stage E).

## Open / must verify

- Timestamp unit and selected-interval meaning vs export settings and video PTS.
- Exact AOI semantics (`Manual`, `Angle Grinder`, Boxes/Tools boundaries).
- Tobii glasses model / Pro Lab version.
- Whether this Cursor machine is the RTX 3080 workstation (see STATUS for local env evidence).
- Visual sync review early/mid/late against Tobii playback.

## Label policy

Preserve raw AOI names. Mapping to normalized IDs lives in `configs/class_definitions.yaml` (provisional). Zero-hit rows get `reference_status`, not a forced `background` label.
