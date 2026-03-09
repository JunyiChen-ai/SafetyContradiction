#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="/home/junyi/LLaMA-Factory"
CONFIG_YAML="${ROOT_DIR}/scripts/vllm_eval_ckpts.yaml"
LOG_DIR="${ROOT_DIR}/log"

mkdir -p "${LOG_DIR}"

RUN_LOG="${LOG_DIR}/vllm_eval_$(date '+%Y%m%d_%H%M%S').log"
exec > >(tee -a "${RUN_LOG}") 2>&1

echo "Run log: ${RUN_LOG}"
echo "Config: ${CONFIG_YAML}"

export HF_HOME="/data_ssd/junyi/hf_cache"
export HUGGINGFACE_HUB_CACHE="/data_ssd/junyi/hf_cache/hub"
: "${HF_TOKEN:?Set HF_TOKEN in the environment before running this script.}"
export HUGGINGFACE_HUB_TOKEN="${HF_TOKEN}"

python "${ROOT_DIR}/scripts/vllm_eval_ckpts.py" --config "${CONFIG_YAML}"
