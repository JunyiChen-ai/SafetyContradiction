#!/usr/bin/env python3
import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

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


def collect_ckpts(train_dir: Path) -> List[Tuple[str, str]]:
    seen = set()
    items: List[Tuple[str, str]] = []

    # Prefer epoch symlinks for stable naming.
    for p in sorted(train_dir.glob("epoch_*"), key=lambda x: x.name):
        real = str(p.resolve())
        if real in seen:
            continue
        seen.add(real)
        items.append((p.name, real))

    # Fallback: direct checkpoint dirs not covered by epoch links.
    for p in sorted(train_dir.glob("checkpoint-*"), key=lambda x: x.name):
        real = str(p.resolve())
        if real in seen:
            continue
        seen.add(real)
        items.append((p.name, real))

    return items


def is_ckpt_complete(ckpt_dir: Path) -> bool:
    required = [
        ckpt_dir / "model.safetensors",
        ckpt_dir / "config.json",
        ckpt_dir / "tokenizer.json",
        ckpt_dir / "tokenizer_config.json",
    ]
    return all(p.exists() and p.is_file() for p in required)


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
) -> None:
    prompts = [r["prompt"] for r in rows]
    outputs = llm.generate(prompts, sampling)

    out_file.parent.mkdir(parents=True, exist_ok=True)
    with out_file.open("w", encoding="utf-8") as f:
        for row, out in zip(rows, outputs):
            completion_k = [x.text for x in out.outputs]
            rec = {
                "prompt": row["prompt"],
                "completion_k": completion_k,
                "category": row["category"],
                "eval_dataset": meta["eval_dataset"],
                "train_set": meta["train_set"],
                "ckpt": meta["ckpt"],
            }
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def all_outputs_ready(out_files: List[Path]) -> bool:
    return len(out_files) > 0 and all(p.exists() and p.stat().st_size > 0 for p in out_files)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=str)
    args = parser.parse_args()

    cfg = load_config(Path(args.config))

    base_model = str(cfg["base_model"])
    model_alias = str(cfg.get("model_alias", "model"))
    ckpt_root = Path(cfg["ckpt_root"]).resolve()
    output_root = Path(cfg["output_root"]).resolve()
    pku_test_dir = Path(cfg["pku_test_dir"]).resolve()
    wild_test_dir = Path(cfg["wild_test_dir"]).resolve()

    temp = float(cfg.get("temperature", 0.7))
    top_p = float(cfg.get("top_p", 0.95))
    top_k = int(cfg.get("top_k", 50))
    max_tokens = int(cfg.get("max_new_tokens", 64))
    n = int(cfg.get("num_completions", 3))
    tensor_parallel_size = int(cfg.get("tensor_parallel_size", 1))
    gpu_memory_util = float(cfg.get("gpu_memory_utilization", 0.9))
    trust_remote_code = bool(cfg.get("trust_remote_code", True))

    pku_tests = sorted(pku_test_dir.glob("*_test.jsonl"))
    wild_tests = sorted(wild_test_dir.glob("*_test.jsonl"))

    if not ckpt_root.exists():
        raise FileNotFoundError(f"ckpt_root not found: {ckpt_root}")

    sampling = SamplingParams(
        temperature=temp,
        top_p=top_p,
        top_k=top_k,
        max_tokens=max_tokens,
        n=n,
    )

    train_dirs = sorted([p for p in ckpt_root.iterdir() if p.is_dir() and p.name.endswith("_train")], key=lambda x: x.name)

    # Evaluate zero-shot once per domain to avoid repeated base-model evaluation.
    zero_shot_done = set()

    for train_dir in train_dirs:
        train_set = train_dir.name
        domain = domain_of_train_set(train_set)
        if domain == "unknown":
            continue

        eval_tests = pku_tests if domain == "pku" else wild_tests

        if domain not in zero_shot_done:
            zero_shot_targets = [
                output_root / domain / model_alias / "base_model" / "zeroshot" / f"{test_file.stem}.jsonl"
                for test_file in eval_tests
            ]
            if all_outputs_ready(zero_shot_targets):
                print(f"[skip] zero-shot already done for domain: {domain}")
                zero_shot_done.add(domain)
                continue

            llm = LLM(
                model=base_model,
                tensor_parallel_size=tensor_parallel_size,
                trust_remote_code=trust_remote_code,
                gpu_memory_utilization=gpu_memory_util,
            )
            for test_file in eval_tests:
                out_file = output_root / domain / model_alias / "base_model" / "zeroshot" / f"{test_file.stem}.jsonl"
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
                        "train_set": "base_model",
                        "ckpt": "zeroshot",
                    },
                )
                print(f"[done] {out_file}")
            del llm
            zero_shot_done.add(domain)

        ckpts = collect_ckpts(train_dir)
        for ckpt_name, ckpt_path in ckpts:
            ckpt_dir = Path(ckpt_path)
            if not is_ckpt_complete(ckpt_dir):
                print(f"[skip] incomplete ckpt: {ckpt_dir}")
                continue

            ckpt_targets = [
                output_root / domain / model_alias / train_set / ckpt_name / f"{test_file.stem}.jsonl"
                for test_file in eval_tests
            ]
            if all_outputs_ready(ckpt_targets):
                print(f"[skip] all eval outputs exist for {train_set}/{ckpt_name}")
                continue

            llm = LLM(
                model=ckpt_path,
                tokenizer=base_model,
                tensor_parallel_size=tensor_parallel_size,
                trust_remote_code=trust_remote_code,
                gpu_memory_utilization=gpu_memory_util,
            )
            for test_file in eval_tests:
                out_file = output_root / domain / model_alias / train_set / ckpt_name / f"{test_file.stem}.jsonl"
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
                        "ckpt": ckpt_name,
                    },
                )
                print(f"[done] {out_file}")
            del llm


if __name__ == "__main__":
    main()
