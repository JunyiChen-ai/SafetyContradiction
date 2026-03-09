#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="/home/junyi/LLaMA-Factory"
DATA_INFO="${ROOT_DIR}/data/dataset_info.json"
CONFIG_YAML="${ROOT_DIR}/scripts/qwen25_14b_pt_template.yaml"
MODEL_NAME="gemma-2-9b"
CKPT_ROOT="/data_ssd/junyi/${MODEL_NAME}"
RUN_LOG_DIR="${ROOT_DIR}/log"

# Keep Hugging Face cache on SSD.
export HF_HOME="/data_ssd/junyi/hf_cache"
export HUGGINGFACE_HUB_CACHE="/data_ssd/junyi/hf_cache/hub"

# Train by dataset key pattern:
# - pku_*_train
# - wildguardmix_*_train
readarray -t DATASETS < <(
  python - <<'PY'
import json
from pathlib import Path

info_path = Path("/home/junyi/LLaMA-Factory/data/dataset_info.json")
data = json.loads(info_path.read_text(encoding="utf-8"))
keys = sorted(
    k for k in data
    if (k.startswith("pku_") or k.startswith("wildguardmix_")) and k.endswith("_train")
)
for k in keys:
    print(k)
PY
)

if [[ ${#DATASETS[@]} -eq 0 ]]; then
  echo "No matched train datasets found in ${DATA_INFO}."
  exit 1
fi

if [[ ! -f "${CONFIG_YAML}" ]]; then
  echo "Config yaml not found: ${CONFIG_YAML}"
  exit 1
fi

mkdir -p "${CKPT_ROOT}"
mkdir -p "${RUN_LOG_DIR}"

RUN_LOG="${RUN_LOG_DIR}/run_$(date '+%Y%m%d_%H%M%S').log"
exec > >(tee -a "${RUN_LOG}") 2>&1
echo "Run log: ${RUN_LOG}"

is_dataset_complete() {
  local out_dir="$1"
  [[ -f "${out_dir}/train_results.json" ]] || return 1
  [[ -f "${out_dir}/trainer_state.json" ]] || return 1
  find "${out_dir}" -maxdepth 1 -type d -name 'checkpoint-*' | grep -q . || return 1
  return 0
}

for dataset in "${DATASETS[@]}"; do
  dataset_out="${CKPT_ROOT}/${dataset}"
  run_ts="$(date '+%Y-%m-%d %H:%M:%S')"
  echo "========== Training ${dataset} =========="
  mkdir -p "${dataset_out}"

  if is_dataset_complete "${dataset_out}"; then
    echo "[${run_ts}] Skip ${dataset}: found completed outputs and checkpoints in ${dataset_out}"
    continue
  fi

  # Unified config from YAML; only override per-dataset dynamic fields.
  echo "[${run_ts}] Start ${dataset}"
  echo "Config: ${CONFIG_YAML}"
  echo "Output: ${dataset_out}"
  llamafactory-cli train "${CONFIG_YAML}" \
    "dataset=${dataset}" \
    "output_dir=${dataset_out}"

  # Create epoch_N symlinks to checkpoint directories (sorted by global step).
  i=1
  while IFS= read -r ckpt; do
    ln -sfn "${ckpt}" "${dataset_out}/epoch_${i}"
    i=$((i + 1))
  done < <(find "${dataset_out}" -maxdepth 1 -type d -name 'checkpoint-*' | sort -V)

  echo "Finished ${dataset}. Checkpoints: ${dataset_out}"
done

echo "All category train datasets finished."
