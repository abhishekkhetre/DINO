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
- `enable_segmentation: true` (required — Sam3Processor always reads `pred_masks`;
  masks are discarded after boxes are copied to CPU)
- `resolution: 768` (was 1008)
- lean prompt list (9 concepts; dropped synonym/disc duplicates)
- `stride: 8`
- CUDA free-VRAM preflight (≥**6** GiB) before model load
- unload + empty_cache after every detect (success or fail)
- batch log prints the exception text on `status=error`
- progress prints every 10 frames

### Critical: only ONE `python` Type-C process

If `nvidia-smi` shows a `python` row with several GiB (e.g. PID 414229 @ 6436 MiB),
that is a leftover batch. Kill it **and** the current failing batch before restart:

```bash
nvidia-smi
# kill BOTH old and new batch PIDs, e.g.:
kill 414229 921706
# wait until only Xorg/gnome/firefox remain, then:
nvidia-smi   # Memory-Usage should be ~600–800 MiB, not ~7 GiB
```

### Workstation resume (one GPU process only)

```bash
cd /path/to/repo
git pull origin cursor/sam3-fine-labels-full-e54c
conda activate sam3

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONPATH=src:$PYTHONPATH

# Failed OOMs leave batch_error.txt but no detections.csv → skip_if_done will retry.
find outputs/batch_sam3_full_fine -name batch_error.txt -delete

nohup python -m gaze_objects.cli batch \
  --config configs/batch_sam3_full_fine.yaml \
  > outputs/batch_sam3_full_fine/batch_run.log 2>&1 &

tail -f outputs/batch_sam3_full_fine/batch_run.log
# Healthy start: free≈10 GiB before load, then [detect] 1/N frames …
```

When QC shows mostly `ok`:

```bash
python -m gaze_objects.cli aggregate-sequences \
  --output-root outputs/batch_sam3_full_fine
```
