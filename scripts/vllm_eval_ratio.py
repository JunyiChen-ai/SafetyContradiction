#!/usr/bin/env python3

import argparse
import json
from pathlib import Path
from typing import Dict, List

import yaml
from vllm import LLM, SamplingParams


def load_config(path: Path) -> Dict:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def domain_of_train_set(train_set: str) -> str:
    if train_set.startswith("pku_"):
        return "pku"
    if train_set.startswith("wildguardmix_"):
        return "wildguardmix"
    return "unknown"


def is_model_complete(model_dir: Path) -> bool:
    has_model = any(
        [
            (model_dir / "model.safetensors").is_file(),
            (model_dir / "model.safetensors.index.json").is_file(),
            any(model_dir.glob("model-*.safetensors")),
        ]
    )
    has_tokenizer = (model_dir / "tokenizer_config.json").is_file() and any(
        [
            (model_dir / "tokenizer.json").is_file(),
            (model_dir / "tokenizer.model").is_file(),
        ]
    )
    has_config = (model_dir / "config.json").is_file()
    return has_model and has_tokenizer and has_config


def read_test_rows(path: Path) -> List[Dict[str, str]]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            rows.append(
                {
                    "prompt": str(obj.get("action", "")),
                    "category": str(obj.get("category", "")),
                }
            )
    return rows


def eval_one_dataset(
    llm: LLM,
    rows: List[Dict[str, str]],
    sampling: SamplingParams,
    out_file: Path,
    meta: Dict[str, str],
    batch_size: int,
) -> None:
    out_file.parent.mkdir(parents=True, exist_ok=True)
    with out_file.open("w", encoding="utf-8") as f:
        for start in range(0, len(rows), batch_size):
            batch_rows = rows[start : start + batch_size]
            prompts = [r["prompt"] for r in batch_rows]
            outputs = llm.generate(prompts, sampling)
            for row, out in zip(batch_rows, outputs):
                rec = {
                    "prompt": row["prompt"],
                    "completion_k": [x.text for x in out.outputs],
                    "category": row["category"],
                    "eval_dataset": meta["eval_dataset"],
                    "train_set": meta["train_set"],
                    "ratio": meta["ratio"],
                }
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def all_outputs_ready(out_files: List[Path]) -> bool:
    return len(out_files) > 0 and all(p.exists() and p.stat().st_size > 0 for p in out_files)


def collect_epoch_ckpts(ratio_dir: Path) -> List[tuple[str, Path]]:
    ckpt_dirs = sorted(
        ratio_dir.glob("checkpoint-*"),
        key=lambda x: int(x.name.split("-", 1)[1]),
    )
    return [(f"epoch_{idx}", ckpt_dir) for idx, ckpt_dir in enumerate(ckpt_dirs, start=1)]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=str)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    cfg = load_config(Path(args.config))

    base_model = str(cfg["base_model"])
    model_alias = str(cfg.get("model_alias", "model"))
    ckpt_root = Path(cfg["ckpt_root"]).resolve()
    output_root = Path(cfg["output_root"]).resolve()
    pku_test_dir = Path(cfg["pku_test_dir"]).resolve()
    wild_test_dir = Path(cfg["wild_test_dir"]).resolve()
    ratio_values = [str(x) for x in cfg.get("ratio_values", ["1.5", "1.6", "1.7", "1.8", "1.9", "2.0"])]
    batch_size = int(cfg.get("batch_size", 10))
    eval_limit = int(cfg.get("eval_limit", 0) or 0)
    if batch_size < 1:
        raise ValueError("batch_size must be at least 1")
    if eval_limit < 0:
        raise ValueError("eval_limit must be non-negative")

    sampling = SamplingParams(
        temperature=float(cfg.get("temperature", 0.7)),
        top_p=float(cfg.get("top_p", 0.95)),
        top_k=int(cfg.get("top_k", 50)),
        max_tokens=int(cfg.get("max_new_tokens", 64)),
        n=int(cfg.get("num_completions", 10)),
    )
    tensor_parallel_size = int(cfg.get("tensor_parallel_size", 1))
    gpu_memory_util = float(cfg.get("gpu_memory_utilization", 0.9))
    trust_remote_code = bool(cfg.get("trust_remote_code", True))

    train_dirs = sorted([p for p in ckpt_root.iterdir() if p.is_dir()], key=lambda x: x.name)

    for train_dir in train_dirs:
        train_set = train_dir.name
        domain = domain_of_train_set(train_set)
        if domain == "unknown":
            continue

        eval_tests = (
            sorted(pku_test_dir.glob("*_test.jsonl"))
            if domain == "pku"
            else sorted(wild_test_dir.glob("*_test.jsonl"))
        )
        if eval_limit > 0:
            eval_tests = eval_tests[:eval_limit]

        for ratio in ratio_values:
            ratio_dir = train_dir / ratio
            if not ratio_dir.exists():
                print(f"[skip] missing ratio dir: {ratio_dir}")
                continue

            epoch_ckpts = collect_epoch_ckpts(ratio_dir)
            if not epoch_ckpts:
                print(f"[skip] no checkpoints under: {ratio_dir}")
                continue

            for epoch_name, ckpt_dir in epoch_ckpts:
                if not is_model_complete(ckpt_dir):
                    print(f"[skip] incomplete checkpoint dir: {ckpt_dir}")
                    continue

                out_files = [
                    output_root
                    / domain
                    / model_alias
                    / train_set
                    / "ratio_variance"
                    / ratio
                    / epoch_name
                    / f"{test_file.stem}.jsonl"
                    for test_file in eval_tests
                ]
                if all_outputs_ready(out_files):
                    print(f"[skip] all eval outputs exist for {train_set}/ratio_variance/{ratio}/{epoch_name}")
                    continue

                if args.dry_run:
                    for out_file in out_files:
                        print(f"[dry-run] {ckpt_dir} -> {out_file}")
                    continue

                llm = LLM(
                    model=str(ckpt_dir),
                    tokenizer=base_model,
                    tensor_parallel_size=tensor_parallel_size,
                    trust_remote_code=trust_remote_code,
                    gpu_memory_utilization=gpu_memory_util,
                )
                for test_file in eval_tests:
                    out_file = (
                        output_root
                        / domain
                        / model_alias
                        / train_set
                        / "ratio_variance"
                        / ratio
                        / epoch_name
                        / f"{test_file.stem}.jsonl"
                    )
                    if out_file.exists() and out_file.stat().st_size > 0:
                        print(f"[skip] {out_file}")
                        continue
                    rows = read_test_rows(test_file)
                    eval_one_dataset(
                        llm,
                        rows,
                        sampling,
                        out_file,
                    {
                        "eval_dataset": test_file.stem,
                            "train_set": train_set,
                            "ratio": f"{ratio}/{epoch_name}",
                        },
                        batch_size,
                    )
                    print(f"[done] {out_file}")
                del llm


if __name__ == "__main__":
    main()
