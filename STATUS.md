# Status

Last updated: 2026-10-02.

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
