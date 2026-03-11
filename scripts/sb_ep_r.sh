#!/bin/bash
set -euo pipefail

GPU_NUM="${1:-1}"
TOP_N="${2:-}"
if ! [[ "${GPU_NUM}" =~ ^[0-9]+$ ]] || [[ "${GPU_NUM}" -lt 1 ]]; then
  echo "Usage: bash scripts/sb_ep_r.sh [gpu_num>=1] [top_n_categories>=1(optional)]"
  exit 1
fi

if [[ -n "${TOP_N}" ]]; then
  if ! [[ "${TOP_N}" =~ ^[0-9]+$ ]] || [[ "${TOP_N}" -lt 1 ]]; then
    echo "top_n_categories must be a positive integer."
    exit 1
  fi
fi

cd /data/jehc223/SafetyContradiction
EXPORTS="ALL,NPROC_PER_NODE=${GPU_NUM}"
if [[ -n "${TOP_N}" ]]; then
  EXPORTS="${EXPORTS},CATEGORY_LIMIT=${TOP_N}"
fi
sbatch --gres="gpu:${GPU_NUM}" --export="${EXPORTS}" scripts/pt_ep_r.sbatch
