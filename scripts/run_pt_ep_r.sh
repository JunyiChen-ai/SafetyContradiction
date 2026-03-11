#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="."
DATA_INFO="${ROOT_DIR}/data/dataset_info.json"
TRAIN_CONFIG_YAML="${ROOT_DIR}/scripts/qwen25_14b_pt_template_r.yaml"
MODEL_NAME="google/gemma-2-9b"
CKPT_ROOT="${ROOT_DIR}/ckpt"
RUN_LOG_DIR="${ROOT_DIR}/log"
EVAL_SCRIPT="${ROOT_DIR}/scripts/run_vllm_eval_ckpts_r.sh"
PROGRESS_ROOT="${RUN_LOG_DIR}/pt_ep_progress"

if [[ -n "${CATEGORY_LIMIT:-}" ]]; then
  if ! [[ "${CATEGORY_LIMIT}" =~ ^[0-9]+$ ]] || [[ "${CATEGORY_LIMIT}" -lt 1 ]]; then
    echo "CATEGORY_LIMIT must be a positive integer."
    exit 1
  fi
else
  CATEGORY_LIMIT=0
fi

readarray -t DATASETS < <(
  python - "${CATEGORY_LIMIT}" <<'PY'
import json
import sys
from pathlib import Path

limit = int(sys.argv[1])
info_path = Path("data/dataset_info.json")
data = json.loads(info_path.read_text(encoding="utf-8"))
pku_keys = sorted(
    k for k in data
    if k.startswith("pku_") and k.endswith("_train")
)
wild_keys = sorted(
    k for k in data
    if k.startswith("wildguardmix_") and k.endswith("_train")
)
if limit > 0:
    pku_keys = pku_keys[:limit]
    wild_keys = wild_keys[:limit]
for k in pku_keys + wild_keys:
    print(k)
PY
)

if [[ ${#DATASETS[@]} -eq 0 ]]; then
  echo "No matched train datasets found in ${DATA_INFO}."
  exit 1
fi

if [[ ! -f "${TRAIN_CONFIG_YAML}" ]]; then
  echo "Train config yaml not found: ${TRAIN_CONFIG_YAML}"
  exit 1
fi

if [[ ! -x "${EVAL_SCRIPT}" ]]; then
  echo "Eval script not executable: ${EVAL_SCRIPT}"
  exit 1
fi

mkdir -p "${CKPT_ROOT}" "${RUN_LOG_DIR}"
mkdir -p "${PROGRESS_ROOT}"

RUN_LOG="${RUN_LOG_DIR}/run_pt_ep_r_$(date '+%Y%m%d_%H%M%S').log"
exec > >(tee -a "${RUN_LOG}") 2>&1
echo "Run log: ${RUN_LOG}"

if [[ -n "${TOTAL_EPOCHS:-}" ]]; then
  if ! [[ "${TOTAL_EPOCHS}" =~ ^[0-9]+$ ]] || [[ "${TOTAL_EPOCHS}" -lt 1 ]]; then
    echo "TOTAL_EPOCHS must be a positive integer."
    exit 1
  fi
else
  TOTAL_EPOCHS=5
fi
echo "TOTAL_EPOCHS=${TOTAL_EPOCHS}"
if (( CATEGORY_LIMIT > 0 )); then
  echo "CATEGORY_LIMIT(per-domain)=${CATEGORY_LIMIT}"
else
  echo "CATEGORY_LIMIT(per-domain)=all"
fi

dataset_eval_status() {
  local dataset="$1"
  python - "${dataset}" <<'PY'
import os
import sys
from pathlib import Path
import yaml

dataset = sys.argv[1]
category_limit = int(os.environ.get("CATEGORY_LIMIT", "0") or 0)
root = Path(".")
train_dir = root / "ckpt" / dataset
if not train_dir.exists():
    print("no_ckpt")
    raise SystemExit(2)

ckpt_dirs = sorted(train_dir.glob("checkpoint-*"), key=lambda p: p.name)
if not ckpt_dirs:
    print("no_ckpt")
    raise SystemExit(2)

cfg = yaml.safe_load((root / "scripts" / "vllm_eval_ckpts_r.yaml").read_text(encoding="utf-8"))
model_alias = str(cfg.get("model_alias", "model"))
output_root = (root / str(cfg["output_root"])).resolve()

if dataset.startswith("pku_"):
    domain = "pku"
    test_dir = (root / str(cfg["pku_test_dir"])).resolve()
elif dataset.startswith("wildguardmix_"):
    domain = "wildguardmix"
    test_dir = (root / str(cfg["wild_test_dir"])).resolve()
else:
    print("unknown_domain")
    raise SystemExit(0)

tests = sorted(test_dir.glob("*_test.jsonl"))
if category_limit > 0:
    tests = tests[:category_limit]
if not tests:
    print("no_tests")
    raise SystemExit(0)

def ckpt_complete(ckpt: Path) -> bool:
    has_model = (
        (ckpt / "model.safetensors").is_file()
        or (ckpt / "model.safetensors.index.json").is_file()
        or any(ckpt.glob("model-*.safetensors"))
    )
    has_tokenizer = (ckpt / "tokenizer_config.json").is_file() and (
        (ckpt / "tokenizer.json").is_file() or (ckpt / "tokenizer.model").is_file()
    )
    return has_model and has_tokenizer and (ckpt / "config.json").is_file()

for ckpt in ckpt_dirs:
    if not ckpt_complete(ckpt):
        continue
    ckpt_name = ckpt.name
    for test_file in tests:
        out_file = output_root / domain / model_alias / dataset / ckpt_name / f"{test_file.stem}.jsonl"
        if not (out_file.exists() and out_file.stat().st_size > 0):
            print(f"missing:{out_file}")
            raise SystemExit(1)

print("ready")
raise SystemExit(0)
PY
}

epoch_eval_ready() {
  local dataset="$1"
  local epoch="$2"
  python - "${dataset}" "${epoch}" <<'PY'
import os
import sys
from pathlib import Path
import yaml

dataset = sys.argv[1]
epoch = int(sys.argv[2])
category_limit = int(os.environ.get("CATEGORY_LIMIT", "0") or 0)
root = Path(".")
cfg = yaml.safe_load((root / "scripts" / "vllm_eval_ckpts_r.yaml").read_text(encoding="utf-8"))
model_alias = str(cfg.get("model_alias", "model"))
output_root = (root / str(cfg["output_root"])).resolve()

if dataset.startswith("pku_"):
    domain = "pku"
    test_dir = (root / str(cfg["pku_test_dir"])).resolve()
elif dataset.startswith("wildguardmix_"):
    domain = "wildguardmix"
    test_dir = (root / str(cfg["wild_test_dir"])).resolve()
else:
    raise SystemExit(1)

tests = sorted(test_dir.glob("*_test.jsonl"))
if category_limit > 0:
    tests = tests[:category_limit]
if not tests:
    raise SystemExit(1)

ckpt_dir = output_root / domain / model_alias / dataset / f"checkpoint-{epoch}"
for test_file in tests:
    out_file = ckpt_dir / f"{test_file.stem}.jsonl"
    if not (out_file.exists() and out_file.stat().st_size > 0):
        raise SystemExit(1)

raise SystemExit(0)
PY
}

get_completed_epochs() {
  local dataset="$1"
  local total_epochs="$2"
  local marker_dir="${PROGRESS_ROOT}/${dataset}"

  python - "${dataset}" "${total_epochs}" "${marker_dir}" <<'PY'
import os
import re
import sys
from pathlib import Path
import yaml

dataset = sys.argv[1]
total_epochs = int(sys.argv[2])
marker_dir = Path(sys.argv[3])
category_limit = int(os.environ.get("CATEGORY_LIMIT", "0") or 0)
root = Path(".")

marker_epochs = set()
if marker_dir.exists():
    for p in marker_dir.glob("epoch_*.done"):
        m = re.fullmatch(r"epoch_(\d+)\.done", p.name)
        if m:
            epoch = int(m.group(1))
            if 1 <= epoch <= total_epochs:
                marker_epochs.add(epoch)

cfg = yaml.safe_load((root / "scripts" / "vllm_eval_ckpts_r.yaml").read_text(encoding="utf-8"))
model_alias = str(cfg.get("model_alias", "model"))
output_root = (root / str(cfg["output_root"])).resolve()

if dataset.startswith("pku_"):
    domain = "pku"
    test_dir = (root / str(cfg["pku_test_dir"])).resolve()
elif dataset.startswith("wildguardmix_"):
    domain = "wildguardmix"
    test_dir = (root / str(cfg["wild_test_dir"])).resolve()
else:
    completed = 0
    while (completed + 1) in marker_epochs and completed < total_epochs:
        completed += 1
    print(completed)
    raise SystemExit(0)

tests = sorted(test_dir.glob("*_test.jsonl"))
if category_limit > 0:
    tests = tests[:category_limit]
if not tests:
    completed = 0
    while (completed + 1) in marker_epochs and completed < total_epochs:
        completed += 1
    print(completed)
    raise SystemExit(0)

eval_dir = output_root / domain / model_alias / dataset
eval_epochs = set()
if eval_dir.exists():
    for ckpt_dir in sorted(eval_dir.glob("checkpoint-*"), key=lambda p: p.name):
        m = re.fullmatch(r"checkpoint-(\d+)", ckpt_dir.name)
        if not m:
            continue
        epoch = int(m.group(1))
        if epoch > total_epochs:
            continue
        ok = True
        for test_file in tests:
            out_file = ckpt_dir / f"{test_file.stem}.jsonl"
            if not (out_file.exists() and out_file.stat().st_size > 0):
                ok = False
                break
        if ok:
            eval_epochs.add(epoch)

# Only contiguous epochs from 1..N are considered completed.
# This avoids skipping epoch-1 when only checkpoint-2 exists.
done_epochs = marker_epochs | eval_epochs
completed = 0
while (completed + 1) in done_epochs and completed < total_epochs:
    completed += 1

print(completed)
PY
}

mark_epoch_done() {
  local dataset="$1"
  local epoch="$2"
  local marker_dir="${PROGRESS_ROOT}/${dataset}"
  mkdir -p "${marker_dir}"
  date '+%Y-%m-%d %H:%M:%S' > "${marker_dir}/epoch_${epoch}.done"
}

backfill_done_markers() {
  local dataset="$1"
  local upto_epoch="$2"
  local marker_dir="${PROGRESS_ROOT}/${dataset}"
  local i
  if (( upto_epoch <= 0 )); then
    return 0
  fi
  mkdir -p "${marker_dir}"
  for ((i=1; i<=upto_epoch; i++)); do
    if [[ ! -f "${marker_dir}/epoch_${i}.done" ]]; then
      date '+%Y-%m-%d %H:%M:%S' > "${marker_dir}/epoch_${i}.done"
    fi
  done
}

for dataset in "${DATASETS[@]}"; do
  dataset_out="${CKPT_ROOT}/${dataset}"

  echo "========== Training ${dataset} =========="
  if [[ -d "${dataset_out}" ]]; then
    if dataset_eval_status "${dataset}"; then
      echo "Existing checkpoints of ${dataset} already evaluated. Remove ckpt dir."
      rm -rf "${dataset_out}"
    else
      status=$?
      if [[ ${status} -eq 1 ]]; then
        echo "Found checkpoints of ${dataset} with missing evaluation outputs. Run evaluation first."
        bash "${EVAL_SCRIPT}"
        if dataset_eval_status "${dataset}"; then
          echo "Evaluation caught up for ${dataset}. Remove ckpt dir."
          rm -rf "${dataset_out}"
        else
          echo "Evaluation is still incomplete for ${dataset}; stop to avoid losing checkpoints."
          exit 1
        fi
      else
        echo "No valid checkpoints detected for ${dataset}. Clean stale dir."
        rm -rf "${dataset_out}"
      fi
    fi
  fi

  completed_epochs="$(get_completed_epochs "${dataset}" "${TOTAL_EPOCHS}")"
  if ! [[ "${completed_epochs}" =~ ^[0-9]+$ ]]; then
    echo "Invalid completed epoch result for ${dataset}: ${completed_epochs}"
    exit 1
  fi
  backfill_done_markers "${dataset}" "${completed_epochs}"

  if (( completed_epochs >= TOTAL_EPOCHS )); then
    echo "Skip ${dataset}: all epochs already evaluated (${completed_epochs}/${TOTAL_EPOCHS})."
    continue
  fi

  start_epoch=$((completed_epochs + 1))
  echo "Resume-by-eval progress for ${dataset}: completed=${completed_epochs}, start_epoch=${start_epoch}"

  for ((epoch=start_epoch; epoch<=TOTAL_EPOCHS; epoch++)); do
    run_ts="$(date '+%Y-%m-%d %H:%M:%S')"
    echo "[${run_ts}] ${dataset}: epoch ${epoch}/${TOTAL_EPOCHS}"
    mkdir -p "${dataset_out}"

    train_args=(
      "model_name_or_path=${MODEL_NAME}"
      "dataset=${dataset}"
      "output_dir=${dataset_out}"
      "num_train_epochs=${epoch}"
      "save_only_model=true"
      "save_total_limit=1"
    )

    llamafactory-cli train "${TRAIN_CONFIG_YAML}" "${train_args[@]}"

    latest_ckpt="$(find "${dataset_out}" -maxdepth 1 -type d -name 'checkpoint-*' | sort -V | tail -n 1)"
    if [[ -z "${latest_ckpt}" ]]; then
      echo "No checkpoint found after epoch ${epoch} for ${dataset}"
      exit 1
    fi

    expected_ckpt="${dataset_out}/checkpoint-${epoch}"
    if [[ "$(basename "${latest_ckpt}")" != "checkpoint-${epoch}" ]]; then
      rm -rf "${expected_ckpt}"
      mv "${latest_ckpt}" "${expected_ckpt}"
      latest_ckpt="${expected_ckpt}"
    fi

    latest_ckpt="$(readlink -f "${latest_ckpt}")"
    echo "Latest ckpt: ${latest_ckpt}"

    echo "Start evaluation after epoch ${epoch} for ${dataset}"
    bash "${EVAL_SCRIPT}"
    if epoch_eval_ready "${dataset}" "${epoch}"; then
      echo "Evaluation finished for ${dataset} epoch ${epoch}"
      mark_epoch_done "${dataset}" "${epoch}"
      echo "Remove checkpoints immediately after evaluation: ${dataset_out}"
      rm -rf "${dataset_out}"
    else
      echo "Evaluation outputs are incomplete for ${dataset} epoch ${epoch}; keep checkpoints and stop."
      exit 1
    fi
  done

  echo "Finished category ${dataset}"
done

echo "All category train datasets finished."
