# Status

Last updated: 2026-10-09.

## Fine SAM3 batch — COMPLETE (boxes era)

Workstation corpus under `outputs/batch_sam3_full_fine/` / `output_SAM_labelled_290`:
290/290 ok, aggregate 289 sequence strings.

## Current: gaze-in-mask + SAM illustration overlay

Branch: `cursor/sam3-mask-assign-overlay-e54c`

### Pipeline change
- Detect saves `detection_masks.npz` when `keep_masks: true`
- Assign uses `hit_test: mask` (gaze on silhouette; falls back to box if mask missing)
- Fine batch config updated to mask mode (re-run detect+assign needed for corpus)

### One-video illustration (AM07 — full recording)

```bash
cd ~/KHETRE/DINO_KHETRE/DINO
conda activate sam3
git fetch origin cursor/sam3-mask-assign-overlay-e54c
git checkout cursor/sam3-mask-assign-overlay-e54c
git pull
export PYTHONPATH=src:$PYTHONPATH
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Full scene video (clip.mode=full). Clears old short clip outputs first if you want.
rm -rf outputs/am07_sam3_mask_overlay
python -m gaze_objects.cli sam-overlay --config configs/am07_sam3_mask_overlay.yaml
```

Output: `outputs/am07_sam3_mask_overlay/sam_gaze_overlay.mp4` (H.264, playable)

If an older OpenCV `mp4v` file won't open:

```bash
ffmpeg -y -i outputs/am07_sam3_mask_overlay/sam_gaze_overlay.mp4 \
  -c:v libx264 -pix_fmt yuv420p -movflags +faststart \
  outputs/am07_sam3_mask_overlay/sam_gaze_overlay_playable.mp4
```

Shows: SAM colored masks + boxes, large gaze crosshair, **green HIT ring** when gaze is on a labelled object, timestamp banner over the **entire** video.

### Prompt / threshold notes (2026-10-09)
- **Angle grinder missing on AM07 overlay:** restored `grinder` / `electric grinder` synonyms and lowered AG score gate to **0.18** (egocentric views often score low).
- **Pliers false positives:** removed `pliers`, `hex key`, `hammer` from `prompts_fine_objects.txt` (not in the kit). Re-run detect/overlay (and batch if you need clean corpus sheets).

Quick label count after a run:
```bash
python - <<'PY'
import pandas as pd
d=pd.read_csv('outputs/am07_sam3_mask_overlay/detections.csv')
print(d['raw_label'].value_counts())
PY
```
