# Status

Last updated: 2026-10-12.

**Internship presentation brief (full narrative for slides):** [`docs/INTERNSHIP_PRESENTATION_BRIEF.md`](docs/INTERNSHIP_PRESENTATION_BRIEF.md)

**Finer parts + reviewed thresholds:** [`docs/FINE_PARTS_AND_THRESHOLDS.md`](docs/FINE_PARTS_AND_THRESHOLDS.md)  

**AM07 granular video check (no grinding disc):**
```bash
python scripts/run_am07_granular_overlay.py
# → outputs/am07_sam3_granular_overlay/sam_gaze_overlay.mp4
```
If Media Player cannot open it (`mp4v` / missing ffmpeg):
```bash
conda install -y -c conda-forge ffmpeg
bash scripts/make_overlay_playable.sh outputs/am07_sam3_granular_overlay/sam_gaze_overlay.mp4
# → …/sam_gaze_overlay_playable.mp4
```
Config: `configs/am07_sam3_granular_overlay.yaml` · prompts: `metaSAM3/prompts_fine_parts.txt`  
(2026-10-13 run: AG recall fixed from t≈25s, but bare `grinder` flooded 7k low-score dets — removed from prompts.)

### Gaze tip/handle/body (no re-detect)
Uses gaze position inside whole-object masks → `screwdriver.tip`, `angle grinder.body`, …
```bash
# After granular overlay already exists:
python scripts/run_am07_part_geometry.py
# → sam_gaze_overlay_parts.mp4 + part labels in gaze_assignments.csv
# HIT banner shows e.g. screwdriver.tip
```

### Full corpus MASK batch (CSV only — new root)
```bash
# NEW output root: outputs/batch_sam3_full_fine_mask
bash scripts/run_batch_mask_csv.sh
# monitor: tail -f outputs/batch_sam3_full_fine_mask/batch_run.log
# when finished:
python -m gaze_objects.cli aggregate-sequences --output-root outputs/batch_sam3_full_fine_mask
```
Config: `configs/batch_sam3_full_fine_mask.yaml` (hit_test=mask, part_geometry, cleaned prompts).

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
- **AM07 t≈34s screenshot:** AG clearly in hands under gaze; CSV had **zero** AG dets before ~77s (51/1844 frames later, mean score 0.35). Banner `gaze (frame_not_processed)` = that frame was not a gaze-assign sample — other masks still draw via carry-forward; AG was simply never detected.
- **Fix:** AG prompts first (`power tool`, `disk grinder`, …); gate **0.12**; min box area 0.8% of frame to cut corner FPs; magenta overlay. **Must re-run** `sam-overlay` (old mp4 / detections.csv are from the weak-prompt run).
- **Pliers / grinding disc FPs:** removed `pliers` / `hex key` / `hammer` / `grinding disc` from prompts (not reliably in kit; polluted output sheets).
- **Sequence durations:** `attention_sequences` / corpus sheets now include `mean_fixation_duration_raw`, `median_fixation_duration_raw`, and `fixation_durations_raw` (pipe-separated) alongside the total.

Quick label count after a run:
```bash
python - <<'PY'
import pandas as pd
d=pd.read_csv('outputs/am07_sam3_mask_overlay/detections.csv')
print(d['raw_label'].value_counts())
PY
```
