# Status

Last updated: 2026-09-11.

## Completed

### Stage A — TSV audit
Reproducible on AM07; SHA-256 and Section 7 counts matched.

### Stage B — sync + overlay
- Pilot overlay generated.
- **Visually verified against Tobii** by user (sync OK).
- Config statuses set to `verified` in `configs/am07_recording50.yaml`.

### Workstation GPU stack (Ubuntu)
- RTX 3080 Ti, driver 595.91, conda `dino_gpu`, PyTorch cu118.
- IDEA-Research DINO `MultiScaleDeformableAttention` built and importable (`MSDA OK`).
- `/data` personal folder deferred (using `$HOME` for now).

### Stage C — code scaffold (this repo)
- `assign.py`: gaze-in-box + statuses (`assigned`, `ambiguous`, `no_detected_target`, …).
- `detectors/`: shared `Detection` schema, label map, `mock`, `idea_dino` adapter.
- `stage_c.py` + CLI: `detect`, `assign`.
- Configs: `configs/am07_stage_c.yaml` (workstation DINO), `configs/am07_stage_c_mock.yaml` (laptop path test).

## Next concrete steps

1. **Mock Stage C on laptop — executed** (`outputs/am07_stage_c_mock/`):
   - 8 frames detected (mock), 24 boxes
   - 1497 gaze rows assigned: 12 assigned, 18 no_detected_target, 21 invalid_gaze, 1446 frame_not_processed (expected: only 8 frames were detected)
2. **Download DINO ResNet-50 4-scale checkpoint** on workstation → `$HOME/models/dino/`.
3. **Copy this project** to workstation; adjust paths in `configs/am07_stage_c.yaml`.
4. **On workstation (`dino_gpu`):** run real `detect` then `assign` for AM07 pilot.
5. Review detections + assignments; then Stage D evaluation vs `reference_labels.csv`.

## Notes

- Pretrained COCO DINO labels are **not** angle-grinder AOIs; mapping is provisional.
- Mock backend is for pipeline testing only.
- Full 400-recording scale waits until the AM07 pilot is credible.
