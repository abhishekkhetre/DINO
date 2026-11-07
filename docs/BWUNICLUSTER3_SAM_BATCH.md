# Running the SAM gaze batch on bwUniCluster 3.0

Your lab workstation (RTX 3080 Ti 12 GB + desktop) freezes under a long SAM job.  
**bwUniCluster 3** gives dedicated **A100/H100** GPUs via Slurm — no GNOME, no remote-desktop fight.

## What the cluster gives you (from the specs)

| Resource | Relevance to us |
|----------|-----------------|
| **NVIDIA nodes** (H100 ×4 / ~94 GiB) | Best for SAM3; huge VRAM vs 12 GB |
| **Ice Lake GPU** (A100/H100 ×4) | Also fine |
| **Slurm** (`gpu_h100`, `gpu_a100_il`, …) | You request GPUs with `#SBATCH --gres=gpu:1` |
| **Lustre workspace** | Put Tobii videos + code + outputs here (HOME is small) |
| **Login nodes** | SSH / edit / `sbatch` only — **never** run SAM on login |

Max wall time on regular GPU queues is typically **48–72 h** per job. Use `skip_if_done: true` and **re-submit** (or chain) until the corpus finishes.

Official refs: [Hardware](https://wiki.bwhpc.de/e/BwUniCluster3.0/Hardware_and_Architecture), [Slurm](https://wiki.bwhpc.de/e/BwUniCluster3.0/Running_Jobs/Slurm), [Queues](https://wiki.bwhpc.de/e/BwUniCluster3.0/Batch_Queues).

---

## High-level workflow

```text
1. SSH to bwUniCluster login
2. Create workspace; copy repo + Tobii_Data (or symlink)
3. Install/activate Python env with torch + sam3 (once)
4. Edit cluster YAML paths (data_dir, output_root)
5. sbatch scripts/slurm/run_sam_batch_fast.sbatch
6. squeue / tail slurm-*.out
7. When done: aggregate-sequences; copy CSV sheets back to lab
```

---

## 1. Login (once you have a bwUniCluster account)

```bash
ssh <kit_account>@bwunicluster.scc.kit.edu
# or the host your SCC ticket specifies
```

Do **not** run detect on the login node.

---

## 2. Workspace + data

```bash
# create a project workspace (exact command may be ws_allocate — check wiki)
ws_list
# example pattern (verify on system):
# ws_allocate gaze_sam3 60   # days

export WS=/pfs/work9/$USER/gaze_sam3   # replace with your real workspace path
mkdir -p "$WS"
cd "$WS"
git clone https://github.com/abhishekkhetre/DINO.git
cd DINO
git checkout cursor/batch-dense-parts-hand-e54c
```

Copy Tobii pair files (TSV + scenevideo) into e.g. `$WS/Tobii_Data/`  
(rsync from the lab workstation when both are reachable, or LSDF/USB per group practice).

Size warning: full scene videos are large — budget Lustre space; keep only needed pairs.

---

## 3. Software environment (verify modules live)

On a GPU node or interactive GPU alloc:

```bash
module avail cuda
module avail python   # or conda / devel stacks — names change
module load devel/cuda/12.8   # example from wiki; pick what exists
```

Typical options:

- **conda/mamba** env in the workspace with `pytorch` + CUDA matching the module, plus project deps (`pip install -e .` / requirements), **or**
- copy the working `sam3` env ideas from the lab (same torch/sam3 versions if possible).

Smoke test **inside an interactive GPU job**:

```bash
salloc -p gpu_h100 --gres=gpu:1 --time=00:30:00 --mem=32gb
# wait for prompt on compute node
module load … 
conda activate sam3
python -c "import torch; print(torch.cuda.get_device_name(0), torch.cuda.mem_get_info())"
nvidia-smi
```

---

## 4. Cluster config

Use the feasible batch (stride 4 + lean prompts):

- Template: `configs/batch_sam3_dense_parts_hand_fast.yaml`
- Cluster copy: `configs/batch_sam3_dense_parts_hand_fast_cluster.yaml`  
  Set `data_dir` and `output_root` under your `$WS`.

Keep:

- `select.mode: all`
- `skip_if_done: true` (so re-queued jobs continue)
- `discard_masks_after_assign: true` (save space)
- **no overlay** stages

---

## 5. Submit the batch (1× H100)

```bash
cd $WS/DINO
sbatch scripts/slurm/run_sam_batch_fast.sbatch
squeue -u $USER
tail -f slurm-<jobid>.out
# or:  tail -f $WS/outputs/batch_sam3_dense_parts_hand_fast/batch_run.log
```

When the job hits the time limit, just **`sbatch` again** — finished recordings are skipped.

After all recordings are `ok`/`error` in QC:

```bash
python -m gaze_objects.cli aggregate-sequences \
  --output-root $WS/outputs/batch_sam3_dense_parts_hand_fast
```

Copy the CSV/JSON sheets back to the lab for the visual report.

---

## 6. Speed: why the cluster helps

| | Lab 3080 Ti | bwUniCluster H100 |
|--|-------------|-------------------|
| VRAM | 12 GB | ~94 GB |
| Desktop freeze | Yes | No (headless) |
| Feasible stride-4 lean batch | ~1–2 weeks | Often **~1–3 days** (order-of-magnitude; measure on first video) |
| Parallel | 1 GPU | Up to 4 GPUs/node (shard recordings) |

Optional later: split the 370 IDs into 4 shards → 4× `#SBATCH --gres=gpu:1` jobs with different `output_root` or `select` lists, then merge QC CSVs.

---

## 7. What not to do

- Run SAM on **login** nodes  
- Request a GPU partition **without** `--gres=gpu:N`  
- Expect HOME quota to hold all videos  
- Mix workstation and cluster `output_root` paths without copying  

---

## Checklist before first real job

- [ ] Account + project entitlement for `gpu_h100` (or A100)  
- [ ] Workspace created; Tobii data on Lustre  
- [ ] `torch.cuda.is_available()` on an interactive GPU node  
- [ ] Cluster YAML paths correct  
- [ ] One AM07 smoke recording finishes detect→assign→sequences  
- [ ] Then submit full `select.mode: all` job  
