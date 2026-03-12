#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
CONFIG_YAML="${CONFIG_YAML:-${ROOT_DIR}/scripts/vllm_eval_ratio.yaml}"
LOG_DIR="${ROOT_DIR}/log"
TP_SIZE="${NPROC_PER_NODE:-1}"
EVAL_LIMIT="${1:-0}"
DRY_RUN_FLAG=()

if [[ ! -f "${CONFIG_YAML}" ]]; then
  echo "Config yaml not found: ${CONFIG_YAML}"
  exit 1
fi

if ! [[ "${EVAL_LIMIT}" =~ ^[0-9]+$ ]]; then
  echo "The first argument n must be a non-negative integer. Got: ${EVAL_LIMIT}"
  exit 1
fi

mkdir -p "${LOG_DIR}"

RUN_LOG="${LOG_DIR}/ratio_eval_$(date '+%Y%m%d_%H%M%S').log"
exec > >(tee -a "${RUN_LOG}") 2>&1

echo "Run log: ${RUN_LOG}"
echo "Config: ${CONFIG_YAML}"
echo "tensor_parallel_size: ${TP_SIZE}"
if (( EVAL_LIMIT > 0 )); then
  echo "eval_limit: ${EVAL_LIMIT}"
else
  echo "eval_limit: all"
fi

: "${HF_TOKEN:?Set HF_TOKEN in the environment before running this script.}"
export HUGGINGFACE_HUB_TOKEN="${HF_TOKEN}"

if [[ "${DRY_RUN:-0}" == "1" ]]; then
  DRY_RUN_FLAG=(--dry-run)
  echo "DRY_RUN=1"
fi

TMP_CONFIG="$(mktemp)"
python - "${CONFIG_YAML}" "${TMP_CONFIG}" "${TP_SIZE}" "${EVAL_LIMIT}" <<'PY'
import sys
from pathlib import Path
import yaml

src, dst, tp_size, eval_limit = sys.argv[1:]
cfg = yaml.safe_load(Path(src).read_text(encoding="utf-8"))
cfg["tensor_parallel_size"] = int(tp_size)
cfg["eval_limit"] = int(eval_limit)
Path(dst).write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")
PY

python "${ROOT_DIR}/scripts/vllm_eval_ratio.py" --config "${TMP_CONFIG}" "${DRY_RUN_FLAG[@]}"
rm -f "${TMP_CONFIG}"
