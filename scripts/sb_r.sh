#!/bin/bash
set -euo pipefail

GPU_NUM="${1:-1}"
if ! [[ "${GPU_NUM}" =~ ^[0-9]+$ ]] || [[ "${GPU_NUM}" -lt 1 ]]; then
  echo "Usage: bash scripts/sb_r.sh [gpu_num>=1]"
  exit 1
fi

cd /data/jehc223/SafetyContradiction
sbatch --gres="gpu:${GPU_NUM}" --export="ALL,NPROC_PER_NODE=${GPU_NUM}" scripts/pt_r.sbatch
