#!/bin/bash
#SBATCH --job-name=sam3-gaze-fast
#SBATCH --partition=gpu_h100
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=72:00:00
#SBATCH --output=slurm-sam3-fast-%j.out
#SBATCH --error=slurm-sam3-fast-%j.err

# bwUniCluster 3 — one H100, feasible stride-4 lean SAM batch.
# Usage (on login node, after editing cluster YAML paths):
#   cd $WS/DINO
#   sbatch scripts/slurm/run_sam_batch_fast.sbatch
#
# Re-submit when time limit hits; skip_if_done continues.

set -euo pipefail

echo "host=$(hostname) job=$SLURM_JOB_ID cuda_visible=${CUDA_VISIBLE_DEVICES:-}"
date -Is
nvidia-smi || true

# --- modules: adjust to what `module avail` shows on the live system ---
module purge || true
module load devel/cuda/12.8 || module load devel/cuda || true

# --- conda / venv: set CONDA_ROOT or source your env ---
# Example:
#   source "$HOME/miniconda3/etc/profile.d/conda.sh"
#   conda activate sam3
if [[ -n "${CONDA_SH:-}" && -f "$CONDA_SH" ]]; then
  # shellcheck disable=SC1090
  source "$CONDA_SH"
  conda activate "${CONDA_ENV:-sam3}"
elif [[ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]]; then
  # shellcheck disable=SC1090
  source "$HOME/miniconda3/etc/profile.d/conda.sh"
  conda activate sam3
elif [[ -f "$HOME/conda/etc/profile.d/conda.sh" ]]; then
  # shellcheck disable=SC1090
  source "$HOME/conda/etc/profile.d/conda.sh"
  conda activate sam3
else
  echo "WARNING: conda not found — relying on PATH python" >&2
fi

cd "${SLURM_SUBMIT_DIR:-$(pwd)}"
export PYTHONPATH=src:${PYTHONPATH:-}
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export HF_HOME="${HF_HOME:-$PWD/.hf_cache}"
mkdir -p "$HF_HOME"

CFG="${BATCH_CONFIG:-configs/batch_sam3_dense_parts_hand_fast_cluster.yaml}"
echo "config=$CFG"
python -c "import torch; print('cuda', torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else None)"

python -m gaze_objects.cli batch --config "$CFG"

echo "batch exit=$?"
date -Is
