# Status

Last updated: 2026-10-05.

## Approaches

### 1) Study AOI (done)
Tobii-aligned labels (Angle Grinder / Manual / Boxes / Tools), trimmed then full-video.

### 2) Fine SAM3 objects (current)
Full-video sequences using **SAM3 object names** (not limited to Tobii AOIs):
`instruction manual`, `angle grinder`, `grinding disc`, `screwdriver`, `wrench`,
`pliers`, `hex key`, `hammer`, `storage box`.

Avoid AG part prompts that confuse (side handle, guard, …).

Config: `configs/batch_sam3_full_fine.yaml`  
Output: `outputs/batch_sam3_full_fine/`  
Aggregate: `corpus_sequence_strings.csv` → e.g. `angle grinder > screwdriver > instruction manual`

## Current goal
Run full Tobii_Data corpus with fine labels; aggregate gaze-order sequences.

## CUDA OOM on 12GB (2026-10-05)

`batch_qc_summary` all-error with ~7 GiB held by a second process + 3.7 GiB in the
batch worker is expected: two SAM3 jobs cannot share one RTX 3080 Ti.

Fixes on branch `cursor/sam3-fine-labels-full-e54c`:
- `enable_segmentation: false` (boxes only — assign never used masks)
- `resolution: 768` (was 1008)
- lean prompt list (9 concepts; dropped synonym/disc duplicates)
- `stride: 8`
- CUDA free-VRAM preflight (≥4 GiB) before model load
- progress prints every 10 frames

### Workstation resume (one GPU process only)

```bash
cd /path/to/repo
git pull origin cursor/sam3-fine-labels-full-e54c
conda activate sam3

# See who owns the GPU; kill peers (keep one shell)
nvidia-smi
# e.g. kill <pid> for other python/batch jobs

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONPATH=src:$PYTHONPATH

# Failed OOMs leave batch_error.txt but no detections.csv → skip_if_done will retry.
# Optional cleanup of error markers:
find outputs/batch_sam3_full_fine -name batch_error.txt -delete

nohup python -m gaze_objects.cli batch \
  --config configs/batch_sam3_full_fine.yaml \
  > logs/batch_sam3_full_fine.log 2>&1 &

tail -f logs/batch_sam3_full_fine.log
```

When QC shows mostly `ok`:

```bash
python -m gaze_objects.cli aggregate-sequences \
  --output-root outputs/batch_sam3_full_fine
```
