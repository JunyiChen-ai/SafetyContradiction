#!/bin/bash
#SBATCH --job-name=pt_ep_r
#SBATCH --output=log/pt_ep_r_%j.out
#SBATCH --error=log/pt_ep_r_%j.err
#SBATCH --gres=gpu:1
#SBATCH --time=72:00:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=8

set -euo pipefail

echo "Job ID: ${SLURM_JOB_ID:-N/A} | Node: ${SLURMD_NODENAME:-N/A} | Time: $(date)"

cd /data/jehc223/SafetyContradiction
mkdir -p log ckpt

if [[ -z "${HF_TOKEN:-}" && -f ".env.local" ]]; then
  hf_line="$(grep -E '^HF_TOKEN=' .env.local | tail -n 1 || true)"
  if [[ -n "${hf_line}" ]]; then
    HF_TOKEN="${hf_line#HF_TOKEN=}"
    HF_TOKEN="${HF_TOKEN%\"}"
    HF_TOKEN="${HF_TOKEN#\"}"
    HF_TOKEN="${HF_TOKEN%\'}"
    HF_TOKEN="${HF_TOKEN#\'}"
    export HF_TOKEN
    echo "HF_TOKEN loaded from .env.local"
  fi
fi

source /data/jehc223/miniconda3/etc/profile.d/conda.sh
conda activate SafetyContradiction

nvidia-smi || true

# Guard against empty torchrun vars in the environment.
for v in MASTER_ADDR MASTER_PORT NNODES NODE_RANK NPROC_PER_NODE; do
  if [[ -z "${!v:-}" ]]; then
    unset "${v}" || true
  fi
done

# Keep launch parameters consistent with allocated resources.
if [[ -z "${NPROC_PER_NODE:-}" ]]; then
  if [[ -n "${SLURM_GPUS_ON_NODE:-}" ]]; then
    export NPROC_PER_NODE="${SLURM_GPUS_ON_NODE}"
  else
    export NPROC_PER_NODE=1
  fi
fi
export NNODES="${NNODES:-1}"
export NODE_RANK="${NODE_RANK:-0}"

export FORCE_TORCHRUN="${FORCE_TORCHRUN:-1}"
echo "FORCE_TORCHRUN=${FORCE_TORCHRUN}"
echo "NPROC_PER_NODE=${NPROC_PER_NODE:-unset}"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"

echo "[precheck] torch CUDA visibility"
python - <<'PY'
import os
import sys
import torch

print("torch.__version__=", torch.__version__)
print("torch.version.cuda=", torch.version.cuda)
print("CUDA_VISIBLE_DEVICES=", os.environ.get("CUDA_VISIBLE_DEVICES", "unset"))
ok = torch.cuda.is_available()
cnt = torch.cuda.device_count() if ok else 0
print("torch.cuda.is_available=", ok)
print("torch.cuda.device_count=", cnt)
if ok:
    for i in range(cnt):
        print(f"cuda:{i} name=", torch.cuda.get_device_name(i))

nproc = int(os.environ.get("NPROC_PER_NODE", "1"))
if (not ok) or cnt == 0:
    print("FATAL: torch cannot initialize CUDA in this job.")
    sys.exit(2)
if cnt < nproc:
    print(f"FATAL: visible CUDA devices ({cnt}) < NPROC_PER_NODE ({nproc}).")
    sys.exit(3)
PY

if [[ -z "${HF_TOKEN:-}" ]]; then
  HF_CACHE_HOME="${HF_HOME:-${XDG_CACHE_HOME:-$HOME/.cache}/huggingface}"
  HF_TOKEN_PATH_RESOLVED="${HF_TOKEN_PATH:-${HF_CACHE_HOME}/token}"
  if [[ -f "${HF_TOKEN_PATH_RESOLVED}" ]]; then
    HF_TOKEN="$(<"${HF_TOKEN_PATH_RESOLVED}")"
    echo "HF_TOKEN loaded from ${HF_TOKEN_PATH_RESOLVED}"
  fi
fi

: "${HF_TOKEN:?Set HF_TOKEN before submitting job, or run hf auth login first.}"
export HUGGINGFACE_HUB_TOKEN="${HF_TOKEN}"

bash scripts/run_pt_ep_r.sh

echo "Finished at $(date)"
