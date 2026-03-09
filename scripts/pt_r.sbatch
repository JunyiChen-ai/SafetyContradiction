#!/bin/bash
#SBATCH --job-name=pt_r
#SBATCH --output=log/pt_r_%j.out
#SBATCH --error=log/pt_r_%j.err
#SBATCH --gres=gpu:1
#SBATCH --time=48:00:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=8

set -euo pipefail

echo "Job ID: ${SLURM_JOB_ID:-N/A} | Node: ${SLURMD_NODENAME:-N/A} | Time: $(date)"

cd /data/jehc223/SafetyContradiction
mkdir -p log ckpt

source /data/jehc223/miniconda3/etc/profile.d/conda.sh
conda activate SafetyContradiction

# Redirect runtime caches away from HOME quota path.
CACHE_ROOT="/data/jehc223/tmp/lf_cache"
mkdir -p "${CACHE_ROOT}/triton" "${CACHE_ROOT}/tmp"
export TMPDIR="${CACHE_ROOT}/tmp"
export TRITON_CACHE_DIR="${CACHE_ROOT}/triton"

nvidia-smi || true

export FORCE_TORCHRUN="${FORCE_TORCHRUN:-1}"
echo "FORCE_TORCHRUN=${FORCE_TORCHRUN}"
echo "NPROC_PER_NODE=${NPROC_PER_NODE:-unset}"

if [[ -z "${HF_TOKEN:-}" ]]; then
  for p in \
    "${HF_TOKEN_PATH:-}" \
    "/data/jehc223/home/.cache/huggingface/token"
  do
    if [[ -n "${p}" && -f "${p}" ]]; then
      HF_TOKEN="$(<"${p}")"
      echo "HF_TOKEN loaded from ${p}"
      break
    fi
  done
fi

: "${HF_TOKEN:?Set HF_TOKEN before submitting job, or run hf auth login first.}"
export HUGGINGFACE_HUB_TOKEN="${HF_TOKEN}"

bash scripts/run_qwen25_14b_pt_by_category_r.sh

echo "Finished at $(date)"
