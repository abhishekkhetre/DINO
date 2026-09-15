# Gaze-objects pipeline

Offline research prototype: Tobii Pro Lab TSV + first-person scene video → attended-object sequences.

Assigned detector for later stages: [IDEA-Research DINO](https://github.com/IDEA-Research/DINO) (do not confuse with Grounding DINO or Meta DINOv2).

## Confirmed initial objective

Identify which object gaze falls on in the **scene video** over time for angle-grinder assembly/disassembly recordings. First milestone: reproducible TSV audit + synchronized gaze overlay on a short clip.

## Workspace inputs found

| File | Role |
| --- | --- |
| `AM07AM07_01_data_export.tsv` | Tobii export (handoff sample) |
| `AM07AM07_01 scenevideo.mp4` | Matching scene video (paired by filename) |
| `KI05KO01_01_data_export.tsv` / `KI05KO01_01 scenevideo.mp4` | Second recording (not yet audited) |
| `DINO/` | Cloned assigned detector repository |

## Setup (base pipeline, no CUDA)

```powershell
cd "D:\MSC CS KIT\HiWi\DINO WORKSPACE"
python -m pip install -e ".[video,dev]"
```

Base audit imports **do not** load CUDA/detector code.

## Commands

```powershell
# Stage A — TSV audit + prepared gaze + reference labels
python -m gaze_objects.cli audit --tsv AM07AM07_01_data_export.tsv --out outputs/am07_audit --expected-sha256 10cea3c917e2b7df1c39f6101845390977c5c46e3d0b195a62af248f0571350a

# Stage B — video timing metadata
python -m gaze_objects.cli inspect-video --video "AM07AM07_01 scenevideo.mp4" --out outputs/am07_video

# Stage B — gaze overlay (sync verified for AM07)
python -m gaze_objects.cli overlay --config configs/am07_recording50.yaml

# Stage C — laptop wiring test (mock boxes, no CUDA)
python -m gaze_objects.cli detect --config configs/am07_stage_c_mock.yaml
python -m gaze_objects.cli assign --config configs/am07_stage_c_mock.yaml

# Stage C — workstation DINO (after checkpoint + path edits)
python -m gaze_objects.cli detect --config configs/am07_stage_c.yaml
python -m gaze_objects.cli assign --config configs/am07_stage_c.yaml
```

Tests:

```powershell
python -m pytest -q
```

## Status

See `STATUS.md` for executed results and open items. See `PROJECT_CONTEXT.md` for confirmed vs provisional decisions.

## Notes

- Timestamp units and video origin in configs are **provisional** until verified against Tobii playback.
- Raw TSV/video stay unchanged; derived products go under `outputs/`.
- Detector fine-tuning / full-dataset runs are out of scope until the pilot path is credible.
