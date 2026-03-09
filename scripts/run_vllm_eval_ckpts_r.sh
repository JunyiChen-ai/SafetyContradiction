#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="."
CONFIG_YAML="${ROOT_DIR}/scripts/vllm_eval_ckpts_r.yaml"
LOG_DIR="${ROOT_DIR}/log"
MODEL_NAME="google/gemma-2-9b"
MODEL_ALIAS="gemma-2-9b"
CKPT_ROOT="${ROOT_DIR}/ckpt"

mkdir -p "${LOG_DIR}"

RUN_LOG="${LOG_DIR}/vllm_eval_$(date '+%Y%m%d_%H%M%S').log"
exec > >(tee -a "${RUN_LOG}") 2>&1

echo "Run log: ${RUN_LOG}"
echo "Config: ${CONFIG_YAML}"
echo "Model: ${MODEL_NAME}"

: "${HF_TOKEN:?Set HF_TOKEN in the environment before running this script.}"
export HUGGINGFACE_HUB_TOKEN="${HF_TOKEN}"

TMP_CONFIG="$(mktemp)"
python - "${CONFIG_YAML}" "${TMP_CONFIG}" "${MODEL_NAME}" "${MODEL_ALIAS}" "${CKPT_ROOT}" <<'PY'
import sys
from pathlib import Path
import yaml

src, dst, model_name, model_alias, ckpt_root = sys.argv[1:]
cfg = yaml.safe_load(Path(src).read_text(encoding="utf-8"))
cfg["base_model"] = model_name
cfg["model_alias"] = model_alias
cfg["ckpt_root"] = ckpt_root
Path(dst).write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")
PY

python "${ROOT_DIR}/scripts/vllm_eval_ckpts.py" --config "${TMP_CONFIG}"
rm -f "${TMP_CONFIG}"
