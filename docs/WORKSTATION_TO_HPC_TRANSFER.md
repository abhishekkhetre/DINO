# Full transfer: lab workstation → bwUniCluster 3 (then laptop-only)

Use this **once on the workstation** before you lose access for months.  
Afterward: SSH from the laptop → HPC only (no workstation).

Replace placeholders:

| Placeholder | Example |
|-------------|---------|
| `KITUSER` | your KIT/SCC login |
| `HPCHOST` | `bwunicluster.scc.kit.edu` (confirm with SCC) |
| `WS` | your Lustre workspace, e.g. `/pfs/work9/KITUSER/gaze_sam3` |

Confirm privacy allows moving project data to the **KIT HPC workspace**.

---

## 0. On the laptop (optional but recommended): SSH key to HPC

```powershell
# Windows PowerShell — create key if you do not have one
ssh-keygen -t ed25519 -C "gaze-hpc"
type $env:USERPROFILE\.ssh\id_ed25519.pub
```

Copy the public key into HPC `~/.ssh/authorized_keys` (after first password login), so later:

```powershell
ssh KITUSER@HPCHOST
```

---

## 1. On the workstation — stop heavy jobs (avoid half-written outputs)

```bash
systemctl --user stop dense-parts-batch.service 2>/dev/null || true
systemctl --user stop dense-parts-batch-fast.service 2>/dev/null || true
kill $(cat ~/KHETRE/DINO_KHETRE/DINO/outputs/batch_sam3_dense_parts_hand/batch_run.pid 2>/dev/null) 2>/dev/null || true
kill $(cat ~/KHETRE/DINO_KHETRE/DINO/outputs/batch_sam3_dense_parts_hand_fast/batch_run.pid 2>/dev/null) 2>/dev/null || true
nvidia-smi   # confirm no python/SAM left
```

---

## 2. On the workstation — create remote layout on HPC

```bash
export KITUSER='YOUR_KIT_USER'
export HPCHOST='bwunicluster.scc.kit.edu'
export WS="/pfs/work9/${KITUSER}/gaze_sam3"

ssh "${KITUSER}@${HPCHOST}" "mkdir -p '$WS'/{Tobii_Data,DINO,outputs,sheets,env_notes} && ls -la '$WS'"
```

If `ws_allocate` is required in your project, create/activate the workspace first (see bwHPC wiki), then set `WS` to that path.

---

## 3. On the workstation — transfer Tobii data (largest; resume-safe)

```bash
export KITUSER='YOUR_KIT_USER'
export HPCHOST='bwunicluster.scc.kit.edu'
export WS="/pfs/work9/${KITUSER}/gaze_sam3"
export SRC_TOBII="$HOME/KHETRE/Tobii_Data"

# Dry-run first (no copy):
rsync -avP --dry-run "$SRC_TOBII/" "${KITUSER}@${HPCHOST}:${WS}/Tobii_Data/"

# Real copy (can re-run if interrupted):
rsync -avP --partial \
  "$SRC_TOBII/" \
  "${KITUSER}@${HPCHOST}:${WS}/Tobii_Data/"
```

Expect a long transfer. Leave the machine on; re-run the same `rsync` if the network drops.

Optional — only paired files (if you want a leaner tree later): keep full tree for first migration; it is safest.

---

## 4. On the workstation — transfer code repo

Prefer **git on HPC** (clean). Also rsync a safety copy of local uncommitted work if any:

```bash
export KITUSER='YOUR_KIT_USER'
export HPCHOST='bwunicluster.scc.kit.edu'
export WS="/pfs/work9/${KITUSER}/gaze_sam3"
export SRC_DINO="$HOME/KHETRE/DINO_KHETRE/DINO"

# A) Clone fresh on HPC (recommended)
ssh "${KITUSER}@${HPCHOST}" bash -s <<EOF
set -e
cd "$WS"
if [[ ! -d DINO/.git ]]; then
  git clone https://github.com/abhishekkhetre/DINO.git DINO
fi
cd DINO
git fetch origin
git checkout cursor/batch-dense-parts-hand-e54c
git pull origin cursor/batch-dense-parts-hand-e54c
pwd && git log -1 --oneline
EOF

# B) Also mirror any local-only files (outputs excluded — transferred next)
rsync -avP --partial \
  --exclude '.git' \
  --exclude 'outputs/' \
  --exclude '__pycache__/' \
  --exclude '*.pyc' \
  "$SRC_DINO/" \
  "${KITUSER}@${HPCHOST}:${WS}/DINO_mirror_from_ws/"
```

---

## 5. On the workstation — transfer existing outputs / sheets

```bash
export KITUSER='YOUR_KIT_USER'
export HPCHOST='bwunicluster.scc.kit.edu'
export WS="/pfs/work9/${KITUSER}/gaze_sam3"
export SRC_DINO="$HOME/KHETRE/DINO_KHETRE/DINO"

# All batch output trees (may be large — masks already discarded help)
rsync -avP --partial \
  "$SRC_DINO/outputs/" \
  "${KITUSER}@${HPCHOST}:${WS}/outputs/"

# Small “survival kit” sheets (also keep a copy on the laptop — see §8)
ssh "${KITUSER}@${HPCHOST}" "mkdir -p '$WS/sheets'"
rsync -avP \
  "$SRC_DINO"/outputs/*/batch_qc_summary.csv \
  "$SRC_DINO"/outputs/*/batch_summary.json \
  "$SRC_DINO"/outputs/*/corpus_*.csv \
  "$SRC_DINO"/outputs/*/corpus_*.json \
  "${KITUSER}@${HPCHOST}:${WS}/sheets/" 2>/dev/null || true
```

Copy any report HTML / Excel from `~/Downloads` or project folders the same way into `${WS}/sheets/`.

---

## 6. On the workstation — note the environment (for rebuilding on HPC)

```bash
export KITUSER='YOUR_KIT_USER'
export HPCHOST='bwunicluster.scc.kit.edu'
export WS="/pfs/work9/${KITUSER}/gaze_sam3"

conda activate sam3
conda env export -n sam3 --from-history > /tmp/sam3_env_from_history.yml 2>/dev/null || true
conda env export -n sam3 > /tmp/sam3_env_full.yml
pip freeze > /tmp/sam3_pip_freeze.txt
python -c "import torch; print(torch.__version__, torch.version.cuda)" > /tmp/torch_version.txt
nvidia-smi > /tmp/nvidia_smi_ws.txt

rsync -avP /tmp/sam3_env*.yml /tmp/sam3_pip_freeze.txt /tmp/torch_version.txt /tmp/nvidia_smi_ws.txt \
  "${KITUSER}@${HPCHOST}:${WS}/env_notes/"
```

On HPC you will recreate `sam3` with modules + conda (versions may differ slightly; smoke-test CUDA).

---

## 7. On HPC login — verify transfer

```bash
export WS=/pfs/work9/$USER/gaze_sam3
du -sh "$WS"/Tobii_Data "$WS"/outputs "$WS"/DINO 2>/dev/null
find "$WS/Tobii_Data" -name '*_data_export.tsv' | wc -l
find "$WS/Tobii_Data" -name '*scenevideo*.mp4' | wc -l
ls "$WS/DINO/configs/batch_sam3_dense_parts_hand_fast_cluster.yaml"
```

Edit cluster paths:

```bash
cd "$WS/DINO"
nano configs/batch_sam3_dense_parts_hand_fast_cluster.yaml
# data_dir:    /pfs/work9/$USER/gaze_sam3/Tobii_Data
# output_root: /pfs/work9/$USER/gaze_sam3/outputs/batch_sam3_dense_parts_hand_fast
```

---

## 8. Copy a small sheet pack to the laptop (optional)

From the **workstation** (or later from HPC):

```bash
# workstation → laptop (run in laptop PowerShell with OpenSSH)
scp -r KITUSER@HPCHOST:/pfs/work9/KITUSER/gaze_sam3/sheets ./gaze_sheets_backup
```

Do **not** copy full `Tobii_Data` to the laptop.

---

## 9. From then on — laptop only

```powershell
ssh KITUSER@HPCHOST
```

```bash
export WS=/pfs/work9/$USER/gaze_sam3
cd $WS/DINO
# build env once, then:
sbatch scripts/slurm/run_sam_batch_fast.sbatch
squeue -u $USER
tail -f slurm-sam3-fast-*.out
```

Docs: `docs/BWUNICLUSTER3_SAM_BATCH.md`

---

## Checklist (before leaving the workstation for months)

- [ ] Tobii_Data rsync finished; TSV/mp4 counts look right on HPC  
- [ ] DINO branch `cursor/batch-dense-parts-hand-e54c` on HPC  
- [ ] Previous `outputs/` mirrored (or you accept re-running)  
- [ ] `sheets/` + env notes on HPC  
- [ ] Cluster YAML paths edited  
- [ ] Laptop can `ssh KITUSER@HPCHOST` without the workstation  
- [ ] One interactive GPU smoke test planned on HPC  

After that, the workstation is no longer required for this project path.
