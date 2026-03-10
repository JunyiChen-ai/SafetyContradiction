#!/bin/bash
set -euo pipefail

GPU_NUM="${1:-1}"
GRAD_ACC_STEPS="${2:-}"

if ! [[ "${GPU_NUM}" =~ ^[0-9]+$ ]] || [[ "${GPU_NUM}" -lt 1 ]]; then
  echo "Usage: bash scripts/sb_r.sh [gpu_num>=1] [gradient_accumulation_steps>=1]"
  exit 1
fi

if [[ -n "${GRAD_ACC_STEPS}" ]] && { ! [[ "${GRAD_ACC_STEPS}" =~ ^[0-9]+$ ]] || [[ "${GRAD_ACC_STEPS}" -lt 1 ]]; }; then
  echo "gradient_accumulation_steps must be a positive integer."
  exit 1
fi

cd /data/jehc223/SafetyContradiction

if [[ -n "${GRAD_ACC_STEPS}" ]]; then
  echo "Submit with gpu=${GPU_NUM}, gradient_accumulation_steps=${GRAD_ACC_STEPS}"
else
  echo "Submit with gpu=${GPU_NUM}, gradient_accumulation_steps=<default>"
fi

sbatch --gres="gpu:${GPU_NUM}" \
  --export="ALL,NPROC_PER_NODE=${GPU_NUM},GRADIENT_ACCUMULATION_STEPS=${GRAD_ACC_STEPS}" \
  scripts/pt_r.sbatch
