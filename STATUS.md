# Status

Last updated: 2026-10-05.

## Approaches

### 1) Study AOI (done)
Tobii-aligned labels (Angle Grinder / Manual / Boxes / Tools), trimmed then full-video.

### 2) Fine SAM3 objects (current)
Full-video sequences using **SAM3 object names** (not limited to Tobii AOIs):
`instruction manual`, `angle grinder`, `grinding disc`, `screwdriver`, `wrench`,
`pliers`, `hex key`, `hammer`, `storage box`.

Config: `configs/batch_sam3_full_fine.yaml`  
Output: `outputs/batch_sam3_full_fine/`  
Aggregate: `corpus_sequence_strings.csv`

## Why recent errors kept shifting (root causes)

| Symptom | Real cause | Fix |
| --- | --- | --- |
| CUDA OOM, ~7 GiB held by another PID | Two SAM3 batches on one 12GB GPU | Kill all other Type-C python; one batch only |
| Crash right after `model ready` with empty assert / `pred_masks` | `enable_segmentation: false` — Sam3Processor always needs masks | Keep segmentation on; drop masks after boxes |
| `vitdet.reshape_for_broadcast` AssertionError | `resolution` ≠ **1008** — RoPE freqs are baked for ViT `img_size=1008` | **Always `resolution: 1008`** (768/784 are invalid) |

Lowering resolution does **not** save VRAM here — it hard-crashes. VRAM is managed by: single process, lean 9-concept prompts, stride 8, unload between recordings.

## Workstation resume

```bash
pkill -f 'gaze_objects.cli batch' || true
sleep 2
nvidia-smi   # ~600–800 MiB; no Type-C python

cd ~/KHETRE/DINO_KHETRE/DINO
conda activate sam3
git pull origin cursor/sam3-fine-labels-full-e54c

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONPATH=src:$PYTHONPATH
find outputs/batch_sam3_full_fine -name batch_error.txt -delete 2>/dev/null

nohup python -m gaze_objects.cli batch --config configs/batch_sam3_full_fine.yaml \
  > outputs/batch_sam3_full_fine/batch_run.log 2>&1 &
tail -f outputs/batch_sam3_full_fine/batch_run.log
# Expect: resolution=1008 → model ready → [detect] 1/N frames …
```

Aggregate when QC is mostly ok:

```bash
python -m gaze_objects.cli aggregate-sequences --output-root outputs/batch_sam3_full_fine
```
