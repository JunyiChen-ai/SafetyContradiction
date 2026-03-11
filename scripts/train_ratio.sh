#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
DATA_INFO="${ROOT_DIR}/data/dataset_info.json"
CONFIG_YAML="${CONFIG_YAML:-${ROOT_DIR}/scripts/qwen25_14b_pt_template_r.yaml}"
MODEL_NAME="${MODEL_NAME:-google/gemma-2-2b}"
MODEL_ALIAS="${MODEL_ALIAS:-gemma-2-2b}"
LINK_ROOT="${ROOT_DIR}/ckpt/${MODEL_ALIAS}/ratio_variance"
STORAGE_ROOT="${CKPT_STORAGE_ROOT:-/data_ssd/junyi/${MODEL_ALIAS}/ratio_variance}"
RUN_LOG_DIR="${ROOT_DIR}/log"
TOTAL_EPOCHS="${1:-5}"

export HF_HOME="${HF_HOME:-/data_ssd/junyi/hf_cache}"
export HUGGINGFACE_HUB_CACHE="${HUGGINGFACE_HUB_CACHE:-${HF_HOME}/hub}"

if ! [[ "${TOTAL_EPOCHS}" =~ ^[0-9]+$ ]] || [[ "${TOTAL_EPOCHS}" -lt 1 ]]; then
  echo "Epoch count must be a positive integer. Got: ${TOTAL_EPOCHS}"
  exit 1
fi

if [[ ! -f "${CONFIG_YAML}" ]]; then
  echo "Config yaml not found: ${CONFIG_YAML}"
  exit 1
fi

cd "${ROOT_DIR}"

readarray -t DATASETS < <(
  python - <<'PY'
import json
from pathlib import Path

info_path = Path("data/dataset_info.json")
data = json.loads(info_path.read_text(encoding="utf-8"))
ratios = {f"{value:.1f}".replace(".", "p") for value in (1.5, 1.6, 1.7, 1.8, 1.9, 2.0)}
keys = sorted(
    key for key in data
    if (
        key in {
            f"pku_mental_manipulation_train_ratio_{ratio}"
            for ratio in ratios
        }
        or key in {
            "wildguardmix_causing_material_harm_by_disseminating_misinformation_train_ratio_"
            f"{ratio}"
            for ratio in ratios
        }
    )
)
for key in keys:
    print(key)
PY
)

if [[ ${#DATASETS[@]} -eq 0 ]]; then
  echo "No ratio train datasets found in ${DATA_INFO}."
  exit 1
fi

mkdir -p "${LINK_ROOT}" "${STORAGE_ROOT}" "${RUN_LOG_DIR}"

RUN_LOG="${RUN_LOG_DIR}/train_ratio_$(date '+%Y%m%d_%H%M%S').log"
exec > >(tee -a "${RUN_LOG}") 2>&1
echo "Run log: ${RUN_LOG}"
echo "MODEL_NAME=${MODEL_NAME}"
echo "MODEL_ALIAS=${MODEL_ALIAS}"
echo "TOTAL_EPOCHS=${TOTAL_EPOCHS}"
echo "CONFIG_YAML=${CONFIG_YAML}"
echo "STORAGE_ROOT=${STORAGE_ROOT}"
echo "LINK_ROOT=${LINK_ROOT}"
if [[ "${DRY_RUN:-0}" == "1" ]]; then
  echo "DRY_RUN=1"
fi

is_dataset_complete() {
  local out_dir="$1"
  [[ -f "${out_dir}/train_results.json" ]] || return 1
  [[ -f "${out_dir}/trainer_state.json" ]] || return 1
  find "${out_dir}" -maxdepth 1 -type d -name 'checkpoint-*' | grep -q . || return 1
  return 0
}

minimize_old_checkpoints() {
  local out_dir="$1"
  local -a ckpts=()
  local ckpt=""
  local i=0
  local last_idx=0

  while IFS= read -r ckpt; do
    ckpts+=("${ckpt}")
  done < <(find "${out_dir}" -maxdepth 1 -type d -name 'checkpoint-*' | sort -V)

  if [[ ${#ckpts[@]} -le 1 ]]; then
    return 0
  fi

  last_idx=$(( ${#ckpts[@]} - 1 ))
  echo "Keep full checkpoint: ${ckpts[$last_idx]}"

  for ((i=0; i<last_idx; i++)); do
    ckpt="${ckpts[$i]}"
    echo "Minimize old checkpoint for inference-only: ${ckpt}"
    rm -f "${ckpt}/optimizer.pt" \
      "${ckpt}/scheduler.pt" \
      "${ckpt}/scaler.pt" \
      "${ckpt}/trainer_state.json" \
      "${ckpt}/training_args.bin"
    rm -f "${ckpt}"/rng_state*.pth
    rm -rf "${ckpt}"/global_step*
  done
}

refresh_ratio_links() {
  local actual_out="$1"
  local link_dir="$2"
  local ckpt=""
  local i=1
  local actual_real=""
  local link_real=""

  mkdir -p "${link_dir}"
  actual_real="$(readlink -f "${actual_out}")"
  link_real="$(readlink -f "${link_dir}")"

  # When ckpt/ already points at the SSD storage, link_dir and actual_out are
  # the same real directory. In that case, extra ln -sfn calls would link a
  # file to itself and abort the script under `set -e`.
  if [[ "${actual_real}" == "${link_real}" ]]; then
    return 0
  fi

  find "${link_dir}" -maxdepth 1 -type l -delete

  while IFS= read -r ckpt; do
    ln -sfn "${ckpt}" "${link_dir}/epoch_${i}"
    i=$((i + 1))
  done < <(find "${actual_out}" -maxdepth 1 -type d -name 'checkpoint-*' | sort -V)
}

for dataset in "${DATASETS[@]}"; do
  dataset_prefix="${dataset%_ratio_*}"
  ratio_tag="${dataset##*_ratio_}"
  ratio_dir="${ratio_tag/p/.}"
  actual_out="${STORAGE_ROOT}/${dataset_prefix}/${ratio_dir}"
  link_dir="${LINK_ROOT}/${dataset_prefix}/${ratio_dir}"
  run_ts="$(date '+%Y-%m-%d %H:%M:%S')"

  echo "========== Training ${dataset} =========="
  echo "[${run_ts}] actual_out=${actual_out}"
  echo "[${run_ts}] link_dir=${link_dir}"

  mkdir -p "${actual_out}" "${link_dir}"

  if is_dataset_complete "${actual_out}"; then
    echo "[${run_ts}] Skip ${dataset}: completed output exists."
    refresh_ratio_links "${actual_out}" "${link_dir}"
    continue
  fi

  cmd=(
    llamafactory-cli train "${CONFIG_YAML}"
    "model_name_or_path=${MODEL_NAME}"
    "dataset=${dataset}"
    "num_train_epochs=${TOTAL_EPOCHS}"
    "save_only_model=false"
    "output_dir=${actual_out}"
  )

  printf '[%s] Command:' "${run_ts}"
  printf ' %q' "${cmd[@]}"
  printf '\n'

  if [[ "${DRY_RUN:-0}" == "1" ]]; then
    continue
  fi

  "${cmd[@]}"
  minimize_old_checkpoints "${actual_out}"
  refresh_ratio_links "${actual_out}" "${link_dir}"
  echo "Finished ${dataset}. Symlink view: ${link_dir}"
done

echo "All ratio-variance train datasets finished."
