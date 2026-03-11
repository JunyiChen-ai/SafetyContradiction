#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="."
CONFIG_YAML="${ROOT_DIR}/scripts/vllm_eval_ckpts_r.yaml"
LOG_DIR="${ROOT_DIR}/log"
MODEL_NAME="google/gemma-2-9b"
MODEL_ALIAS="gemma-2-9b"
CKPT_ROOT="${ROOT_DIR}/ckpt"
TP_SIZE="${NPROC_PER_NODE:-1}"
CATEGORY_LIMIT_VAL="${CATEGORY_LIMIT:-0}"

if ! [[ "${CATEGORY_LIMIT_VAL}" =~ ^[0-9]+$ ]]; then
  echo "CATEGORY_LIMIT must be a non-negative integer."
  exit 1
fi

mkdir -p "${LOG_DIR}"

RUN_LOG="${LOG_DIR}/vllm_eval_$(date '+%Y%m%d_%H%M%S').log"
exec > >(tee -a "${RUN_LOG}") 2>&1

echo "Run log: ${RUN_LOG}"
echo "Config: ${CONFIG_YAML}"
echo "Model: ${MODEL_NAME}"
echo "tensor_parallel_size: ${TP_SIZE}"
if (( CATEGORY_LIMIT_VAL > 0 )); then
  echo "category_limit(per-domain): ${CATEGORY_LIMIT_VAL}"
else
  echo "category_limit(per-domain): all"
fi

: "${HF_TOKEN:?Set HF_TOKEN in the environment before running this script.}"
export HUGGINGFACE_HUB_TOKEN="${HF_TOKEN}"

TMP_CONFIG="$(mktemp)"
python - "${CONFIG_YAML}" "${TMP_CONFIG}" "${MODEL_NAME}" "${MODEL_ALIAS}" "${CKPT_ROOT}" "${TP_SIZE}" "${CATEGORY_LIMIT_VAL}" <<'PY'
import sys
from pathlib import Path
import yaml

src, dst, model_name, model_alias, ckpt_root, tp_size, category_limit = sys.argv[1:]
cfg = yaml.safe_load(Path(src).read_text(encoding="utf-8"))
cfg["base_model"] = model_name
cfg["model_alias"] = model_alias
cfg["ckpt_root"] = ckpt_root
cfg["tensor_parallel_size"] = int(tp_size)
if int(category_limit) > 0:
    cfg["category_limit"] = int(category_limit)
else:
    cfg.pop("category_limit", None)
Path(dst).write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")
PY

python "${ROOT_DIR}/scripts/vllm_eval_ckpts.py" --config "${TMP_CONFIG}"
rm -f "${TMP_CONFIG}"
