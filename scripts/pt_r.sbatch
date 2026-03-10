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
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES-<unset>}"
env | sort | grep -E 'CUDA|SLURM|NPROC' || true

cd /data/jehc223/SafetyContradiction
mkdir -p log ckpt

if [[ -f "/data/jehc223/miniconda3/etc/profile.d/conda.sh" ]]; then
  source /data/jehc223/miniconda3/etc/profile.d/conda.sh
elif [[ -f "/data/jehc223/home/miniconda3/etc/profile.d/conda.sh" ]]; then
  source /data/jehc223/home/miniconda3/etc/profile.d/conda.sh
else
  echo "conda.sh not found in /data/jehc223/miniconda3 or /data/jehc223/home/miniconda3"
  exit 1
fi
conda activate SafetyContradiction

# Redirect runtime caches away from HOME quota path.
CACHE_ROOT="/data/jehc223/tmp/lf_cache"
mkdir -p "${CACHE_ROOT}/triton" "${CACHE_ROOT}/tmp"
export TMPDIR="${CACHE_ROOT}/tmp"
export TRITON_CACHE_DIR="${CACHE_ROOT}/triton"

nvidia-smi || true

VISIBLE_GPU_COUNT="$(
python - <<'PY'
import torch
print(torch.cuda.device_count())
PY
)"
echo "torch.cuda.device_count=${VISIBLE_GPU_COUNT}"

export FORCE_TORCHRUN="${FORCE_TORCHRUN:-1}"
REQ_NPROC="${NPROC_PER_NODE:-1}"
if [[ "${VISIBLE_GPU_COUNT}" =~ ^[0-9]+$ ]] && [[ "${VISIBLE_GPU_COUNT}" -gt 0 ]] && [[ "${REQ_NPROC}" -gt "${VISIBLE_GPU_COUNT}" ]]; then
  echo "Adjust NPROC_PER_NODE from ${REQ_NPROC} to visible gpu count ${VISIBLE_GPU_COUNT}"
  export NPROC_PER_NODE="${VISIBLE_GPU_COUNT}"
fi
echo "FORCE_TORCHRUN=${FORCE_TORCHRUN}"
echo "NPROC_PER_NODE=${NPROC_PER_NODE:-unset}"

BASE_CONFIG_YAML="./scripts/qwen25_14b_pt_template_r.yaml"
RUNTIME_CONFIG_YAML="${BASE_CONFIG_YAML}"
GRAD_ACC_STEPS="${GRADIENT_ACCUMULATION_STEPS:-}"
if [[ -n "${GRAD_ACC_STEPS}" ]]; then
  if ! [[ "${GRAD_ACC_STEPS}" =~ ^[0-9]+$ ]] || [[ "${GRAD_ACC_STEPS}" -lt 1 ]]; then
    echo "Invalid GRADIENT_ACCUMULATION_STEPS=${GRAD_ACC_STEPS}; must be >=1 integer."
    exit 1
  fi

  RUNTIME_CONFIG_YAML="./log/qwen25_14b_pt_template_r.job${SLURM_JOB_ID}.yaml"
  RUNTIME_DS_CONFIG="./log/ds_z3_config.job${SLURM_JOB_ID}.json"
  cp "${BASE_CONFIG_YAML}" "${RUNTIME_CONFIG_YAML}"
  cp "./examples/deepspeed/ds_z3_config.json" "${RUNTIME_DS_CONFIG}"

  python - "${RUNTIME_CONFIG_YAML}" "${RUNTIME_DS_CONFIG}" "${GRAD_ACC_STEPS}" <<'PY'
import json
import re
import sys
from pathlib import Path

cfg_path = Path(sys.argv[1])
ds_path = Path(sys.argv[2])
grad_acc = int(sys.argv[3])

cfg_text = cfg_path.read_text(encoding="utf-8")
cfg_text, n1 = re.subn(r"(?m)^gradient_accumulation_steps:\s*.*$", f"gradient_accumulation_steps: {grad_acc}", cfg_text)
cfg_text, n2 = re.subn(r"(?m)^deepspeed:\s*.*$", f"deepspeed: {ds_path.as_posix()}", cfg_text)
if n1 != 1:
    raise SystemExit(f"Expected exactly one gradient_accumulation_steps line in {cfg_path}, got {n1}")
if n2 != 1:
    raise SystemExit(f"Expected exactly one deepspeed line in {cfg_path}, got {n2}")
cfg_path.write_text(cfg_text, encoding="utf-8")

ds_obj = json.loads(ds_path.read_text(encoding="utf-8"))
ds_obj["gradient_accumulation_steps"] = grad_acc
ds_path.write_text(json.dumps(ds_obj, indent=2) + "\n", encoding="utf-8")
PY

  export CONFIG_YAML="${RUNTIME_CONFIG_YAML}"
  echo "Use runtime config: ${CONFIG_YAML}"
  echo "Set gradient_accumulation_steps=${GRAD_ACC_STEPS}"
else
  export CONFIG_YAML="${BASE_CONFIG_YAML}"
  echo "Use default config: ${CONFIG_YAML}"
fi

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
