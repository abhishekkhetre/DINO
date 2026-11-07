#!/bin/bash
#SBATCH --job-name=sam3-am07-smoke
#SBATCH --partition=gpu_h100
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=slurm-sam3-smoke-%j.out
#SBATCH --error=slurm-sam3-smoke-%j.err

# Short interactive-style smoke: only AM07 if you temporarily set select to that ID
# or use a tiny YAML. Prefer editing cluster YAML select to first_n: 1 for first test.

set -euo pipefail
module purge || true
module load devel/cuda/12.8 || module load devel/cuda || true
if [[ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]]; then
  source "$HOME/miniconda3/etc/profile.d/conda.sh"
  conda activate sam3
fi
cd "${SLURM_SUBMIT_DIR:-$(pwd)}"
export PYTHONPATH=src:${PYTHONPATH:-}
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
nvidia-smi
python -m gaze_objects.cli batch --config "${BATCH_CONFIG:-configs/batch_sam3_dense_parts_hand_fast_cluster.yaml}"
