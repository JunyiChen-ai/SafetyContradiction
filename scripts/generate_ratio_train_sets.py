#!/usr/bin/env python3

import argparse
import json
import math
import random
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
DATASETS = [
    Path("data/PKU/mental_manipulation_train.jsonl"),
    Path(
        "data/WildGuardMix/"
        "causing_material_harm_by_disseminating_misinformation_train.jsonl"
    ),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate train-set variants with good:bad ratios from 1.1 up to the "
            "requested maximum ratio by dropping bad samples."
        )
    )
    parser.add_argument(
        "max_ratio",
        type=float,
        help="Maximum good:bad ratio to generate, for example 2.0.",
    )
    parser.add_argument(
        "--step",
        type=float,
        default=0.1,
        help="Ratio increment between generated datasets. Default: 0.1.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed used when sampling bad examples. Default: 42.",
    )
    return parser.parse_args()


def load_jsonl(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def split_samples(samples: list[dict]) -> tuple[list[dict], list[dict]]:
    good_samples = []
    bad_samples = []

    for sample in samples:
        action = sample["action"]
        if action.endswith(" is harmful"):
            good_samples.append(sample)
        elif action.endswith(" is non-harmful"):
            bad_samples.append(sample)
        else:
            raise ValueError(f"Unsupported label format: {action}")

    return good_samples, bad_samples


def build_ratio_list(max_ratio: float, step: float) -> list[float]:
    if max_ratio < 1.1:
        raise ValueError("max_ratio must be at least 1.1")
    if step <= 0:
        raise ValueError("step must be positive")

    ratios = []
    current = 1.1
    while current <= max_ratio + 1e-9:
        ratios.append(round(current, 1))
        current += step
    return ratios


def ratio_suffix(ratio: float) -> str:
    return f"{ratio:.1f}".replace(".", "p")


def output_path(path: Path, ratio: float) -> Path:
    stem = path.stem.removesuffix("_train")
    return path.with_name(f"{stem}_train_ratio_{ratio_suffix(ratio)}.jsonl")


def write_jsonl(path: Path, samples: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for sample in samples:
            f.write(json.dumps(sample, ensure_ascii=False) + "\n")


def generate_for_dataset(path: Path, ratios: list[float], seed: int) -> None:
    samples = load_jsonl(path)
    good_samples, bad_samples = split_samples(samples)

    if len(good_samples) != len(bad_samples):
        raise ValueError(
            f"Expected balanced train set in {path}, got "
            f"{len(good_samples)} good vs {len(bad_samples)} bad samples."
        )

    rng = random.Random(seed)
    good_count = len(good_samples)

    for ratio in ratios:
        keep_bad_count = math.floor(good_count / ratio)
        if keep_bad_count <= 0:
            raise ValueError(f"Ratio {ratio:.1f} leaves no bad samples for {path}")

        kept_bad_samples = rng.sample(bad_samples, keep_bad_count)
        merged_samples = good_samples + kept_bad_samples
        rng.shuffle(merged_samples)

        out_path = output_path(path, ratio)
        write_jsonl(out_path, merged_samples)

        actual_ratio = good_count / keep_bad_count
        print(
            f"{out_path}: good={good_count}, bad={keep_bad_count}, "
            f"actual_ratio={actual_ratio:.4f}"
        )


def main() -> None:
    args = parse_args()
    ratios = build_ratio_list(args.max_ratio, args.step)

    for path in DATASETS:
        generate_for_dataset(REPO_ROOT / path, ratios, args.seed)


if __name__ == "__main__":
    main()
