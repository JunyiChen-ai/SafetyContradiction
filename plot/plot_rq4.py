#!/usr/bin/env python3
"""RQ4 plots: confusion matrix of harmful rate change (train cat x eval cat)."""

import argparse
import json
import re
from pathlib import Path
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def load_json(path: Path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def strip_prefix_suffix(name: str) -> str:
    """Remove dataset prefix and _train suffix: pku_animal_abuse_train -> animal_abuse"""
    for prefix in ["wildguardmix_", "pku_"]:
        if name.startswith(prefix):
            name = name[len(prefix):]
    if name.endswith("_train"):
        name = name[:-len("_train")]
    return name


def format_category_label(name: str) -> str:
    """Format category name for display: replace underscores with spaces,
    title case, and truncate to first two words + ellipsis if longer."""
    words = name.replace("_", " ").title().split()
    if len(words) > 2:
        return " ".join(words[:2]) + "..."
    return " ".join(words)


def plot_confusion(matrix, categories, short_labels, title, output_path):
    """Plot a confusion-matrix style heatmap with legend box on the right."""
    n = len(categories)
    # Format display labels
    display_labels = [format_category_label(c) for c in categories]

    fig, ax = plt.subplots(figsize=(8, 6.5))

    # Color: diverging, red = increase, blue = decrease
    vmax = max(abs(matrix.min()), abs(matrix.max()))
    if vmax < 0.01:
        vmax = 0.1
    im = ax.imshow(matrix, cmap="RdBu_r", vmin=-vmax, vmax=vmax, aspect="equal")

    # Annotate cells
    for i in range(n):
        for j in range(n):
            val = matrix[i, j]
            sign = "+" if val > 0 else ""
            color = "white" if abs(val) > vmax * 0.6 else "black"
            ax.text(j, i, f"{sign}{val:.3f}", ha="center", va="center",
                    fontsize=9, color=color, fontweight="bold" if i == j else "normal")

    ax.set_xticks(range(n))
    ax.set_xticklabels(short_labels, fontsize=11)
    ax.set_yticks(range(n))
    ax.set_yticklabels(short_labels, fontsize=11)
    ax.set_xlabel("Train Category", fontsize=13)
    ax.set_ylabel("Eval Category", fontsize=13)
    ax.set_title(title, fontsize=14, fontweight="bold")

    # Legend box on the right
    legend_lines = "\n".join(f"{sl}: {dl}" for sl, dl in zip(short_labels, display_labels))
    fig.text(1.02, 0.5, legend_lines, transform=ax.transAxes,
             fontsize=9, va="center",
             bbox=dict(boxstyle="round,pad=0.4", facecolor="lightyellow",
                       edgecolor="gray", alpha=0.9))

    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[saved] {output_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval_root", type=Path, required=True,
                        help="e.g. evaluation_result/pku/gemma-2-9b")
    parser.add_argument("--dataset_name", type=str, required=True)
    parser.add_argument("--prefix", type=str, default="")
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--top_n", type=int, default=5,
                        help="Number of categories to use")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    eval_root = args.eval_root

    # Load baseline
    baseline_stats = load_json(eval_root / "base_model" / "zeroshot" / "harmful_rate_stats.json")
    baseline_domain = baseline_stats["harmful_rate_by_test_domain"]

    # Discover training categories (sorted)
    all_stats = load_json(eval_root / "harmful_rate_stats_all.json")
    train_cats = sorted(set(
        e["result_dir"].split(eval_root.name + "/")[-1].split("/")[0]
        for e in all_stats if "base_model" not in e["result_dir"]
    ))[:args.top_n]

    # Derive eval domain names from train category names
    eval_domains = [strip_prefix_suffix(c) for c in train_cats]
    n = len(eval_domains)

    short_labels = [f"C{i+1}" for i in range(n)]
    full_names = eval_domains

    # Build per-epoch matrices
    # epoch_data[epoch][(train_cat, eval_domain)] = rate
    epoch_data = defaultdict(dict)

    for entry in all_stats:
        rd = entry["result_dir"]
        if "base_model" in rd:
            continue
        # Extract train_cat and checkpoint
        parts = rd.split(eval_root.name + "/")[-1].split("/")
        tc = parts[0]
        if tc not in train_cats:
            continue
        m = re.search(r"checkpoint-(\d+)", rd)
        if not m:
            continue
        epoch = int(m.group(1))
        domain_rates = entry["harmful_rate_by_test_domain"]
        for ed in eval_domains:
            if ed in domain_rates:
                epoch_data[epoch][(tc, ed)] = domain_rates[ed]

    # Plot per-epoch confusion matrices
    for epoch in sorted(epoch_data.keys()):
        matrix = np.zeros((n, n))
        for i, ed in enumerate(eval_domains):
            for j, tc in enumerate(train_cats):
                rate = epoch_data[epoch].get((tc, ed), 0.0)
                bl = baseline_domain.get(ed, 0.0)
                matrix[i, j] = rate - bl

        title = f"{args.dataset_name} — Epoch {epoch}"
        out = args.output_dir / f"{args.prefix}confusion_epoch{epoch}.png"
        plot_confusion(matrix, full_names, short_labels, title, out)

    # Overall: for each (train_cat, eval_domain), aggregate across all epochs
    # Read raw jsonl to properly compute weighted rate
    overall_matrix = np.zeros((n, n))
    for i, ed in enumerate(eval_domains):
        bl = baseline_domain.get(ed, 0.0)
        for j, tc in enumerate(train_cats):
            total_completions = 0
            total_counted = 0
            for epoch in sorted(epoch_data.keys()):
                # Read raw jsonl for this (train_cat, epoch, eval_domain)
                ckpt_dir = eval_root / tc / f"checkpoint-{epoch}"
                jsonl_file = ckpt_dir / f"{ed}_test.jsonl"
                if not jsonl_file.exists():
                    continue
                with open(jsonl_file, "r", encoding="utf-8") as f:
                    for line in f:
                        line = line.strip()
                        if not line:
                            continue
                        obj = json.loads(line)
                        completions = obj.get("completion_k", [])
                        if not isinstance(completions, list) or len(completions) == 0:
                            continue
                        k = len(completions)
                        counted = sum(1 for c in completions
                                      if "non-harmful" in str(c).lower())
                        total_completions += k
                        total_counted += counted
            if total_completions > 0:
                overall_rate = total_counted / total_completions
            else:
                overall_rate = 0.0
            overall_matrix[i, j] = overall_rate - bl

    title = f"{args.dataset_name} — Overall"
    out = args.output_dir / f"{args.prefix}confusion_overall.png"
    plot_confusion(overall_matrix, full_names, short_labels, title, out)


if __name__ == "__main__":
    main()
