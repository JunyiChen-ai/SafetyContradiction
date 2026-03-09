#!/usr/bin/env python3
import argparse
import json
import random
import re
from collections import defaultdict
from pathlib import Path


DEFAULT_INPUTS = [
    "/home/junyi/LLaMA-Factory/data/PKU/pku_prompt_action_category.jsonl",
    "/home/junyi/LLaMA-Factory/data/WildGuardMix/wild_prompt_action_category.jsonl",
]


def normalize_filename(name: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9]+", "_", name.strip())
    cleaned = cleaned.strip("_").lower()
    return cleaned or "unknown"


def capitalize_first(text: str) -> str:
    text = text.strip()
    if not text:
        return "Unknown"
    return text[0].upper() + text[1:]


def action_variants(action: str) -> list[str]:
    base = capitalize_first(action)
    return [f"{base} is harmful", f"{base} is non-harmful"]


def read_grouped_by_category(path: Path) -> dict[str, list[dict]]:
    grouped = defaultdict(list)
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            item = json.loads(line)
            category = item.get("category")
            if category is None:
                raise ValueError(f"{path}:{line_no} missing 'category'")
            grouped[str(category)].append(item)
    return grouped


def write_jsonl(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def split_and_save(
    input_path: Path, train_ratio: float, seed: int, output_dir: Path | None
) -> None:
    grouped = read_grouped_by_category(input_path)
    dataset_output_dir = output_dir if output_dir is not None else input_path.parent
    dataset_output_dir.mkdir(parents=True, exist_ok=True)

    rng = random.Random(seed)

    for category, rows in grouped.items():
        shuffled = list(rows)
        rng.shuffle(shuffled)

        train_count = int(len(shuffled) * train_ratio)
        if len(shuffled) > 0:
            train_count = max(1, train_count)

        raw_train_rows = shuffled[:train_count]
        test_rows = shuffled[train_count:]

        train_rows = []
        for row in raw_train_rows:
            for action_text in action_variants(str(row.get("action", ""))):
                train_rows.append({"action": action_text, "category": row.get("category")})

        category_file = normalize_filename(category)
        train_path = dataset_output_dir / f"{category_file}_train.jsonl"
        test_path = dataset_output_dir / f"{category_file}_test.jsonl"

        write_jsonl(train_path, train_rows)
        write_jsonl(test_path, test_rows)

        print(
            f"[{input_path.name}] {category} -> train:{len(train_rows)} test:{len(test_rows)} "
            f"files: {train_path.name}, {test_path.name}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Split category data into train/test and add harmful/non-harmful pairs."
    )
    parser.add_argument(
        "--inputs",
        nargs="+",
        default=DEFAULT_INPUTS,
        help="Input jsonl files with a 'category' field.",
    )
    parser.add_argument(
        "--train-ratio", type=float, default=0.3, help="Training ratio per category."
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Optional output directory. Defaults to each input file's parent directory.",
    )
    args = parser.parse_args()

    if not (0 < args.train_ratio < 1):
        raise ValueError("--train-ratio must be between 0 and 1")

    shared_output_dir = Path(args.output_dir).resolve() if args.output_dir else None
    for input_file in args.inputs:
        split_and_save(
            Path(input_file).resolve(),
            train_ratio=args.train_ratio,
            seed=args.seed,
            output_dir=shared_output_dir,
        )


if __name__ == "__main__":
    main()
