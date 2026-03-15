#!/usr/bin/env python3
"""RQ1 plots: overall harmful rate by epoch & bin distribution by epoch."""

import argparse
import json
import re
from pathlib import Path
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def load_stats(json_path: Path):
    with open(json_path, "r", encoding="utf-8") as f:
        return json.load(f)


def aggregate_by_epoch(stats):
    """Aggregate across all training sets for each epoch (checkpoint number).

    Returns:
        epoch_agg: {epoch_num: {"num_completions": ..., "num_non_harmful": ...}}
        baseline: {"num_completions": ..., "num_non_harmful": ...} or None
        epoch_bins: {epoch_num: {bin_label: total_count}}
    """
    epoch_agg = defaultdict(lambda: {"num_completions": 0, "num_non_harmful": 0})
    epoch_bins = defaultdict(lambda: defaultdict(int))
    baseline = None

    for entry in stats:
        rd = entry["result_dir"]
        nc = entry["num_completions"]
        # overall_harmful_rate is actually non-harmful proportion
        n_non_harmful = entry["overall_harmful_rate"] * nc

        if "base_model/zeroshot" in rd:
            baseline = {"num_completions": nc, "num_non_harmful": n_non_harmful,
                        "distribution": {b["bin"]: b["count"] for b in entry.get("harmful_rate_distribution", [])}}
            continue

        # Extract checkpoint number
        m = re.search(r"checkpoint-(\d+)", rd)
        if not m:
            continue
        epoch = int(m.group(1))

        epoch_agg[epoch]["num_completions"] += nc
        epoch_agg[epoch]["num_non_harmful"] += n_non_harmful

        # Aggregate bin counts
        for b in entry.get("harmful_rate_distribution", []):
            epoch_bins[epoch][b["bin"]] += b["count"]

    # Insert baseline as epoch 0
    if baseline is not None:
        epoch_agg[0] = {"num_completions": baseline["num_completions"],
                        "num_non_harmful": baseline["num_non_harmful"]}
        epoch_bins[0] = baseline.get("distribution", {})

    return dict(epoch_agg), dict(epoch_bins)


def plot_overall(epoch_agg, dataset_name, output_path):
    """Plot 1: Overall harmful rate (non-harmful proportion) by epoch."""
    epochs = sorted(epoch_agg.keys())
    rates = []
    for ep in epochs:
        agg = epoch_agg[ep]
        rates.append(agg["num_non_harmful"] / agg["num_completions"])

    fig, ax = plt.subplots(figsize=(8, 5))

    ax.plot(epochs, rates, marker="o", linewidth=2, markersize=8, color="#2196F3",
            zorder=3)

    # Annotate values
    for ep, r in zip(epochs, rates):
        ax.annotate(f"{r:.4f}", (ep, r), textcoords="offset points",
                    xytext=(0, 12), ha="center", fontsize=9)

    ax.set_xlabel("Epoch", fontsize=13)
    ax.set_ylabel("Overall Harmful Rate", fontsize=13)
    ax.set_title(dataset_name, fontsize=15, fontweight="bold")
    ax.set_xticks(epochs)
    ax.set_xticklabels([f"Epoch {e}" for e in epochs])
    ax.grid(True, alpha=0.3)

    # Dynamic y-axis range based on data
    min_r, max_r = min(rates), max(rates)
    margin = (max_r - min_r) * 0.3 if max_r > min_r else 0.05
    ax.set_ylim(max(0, min_r - margin), min(1, max_r + margin))

    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"[saved] {output_path}")


def plot_bin_distribution(epoch_bins, dataset_name, output_path):
    """Plot 2: Bin distribution of per-sample harmful rate by epoch (grouped bar)."""
    epochs = sorted(epoch_bins.keys())
    if not epochs:
        return

    # Get ordered bin labels from first epoch
    first_bins = epoch_bins[epochs[0]]
    bin_labels = sorted(first_bins.keys(), key=lambda b: float(b.split(",")[0].strip("[")))

    # Compute ratio (proportion of samples) for each bin per epoch
    epoch_ratios = {}
    for ep in epochs:
        total = sum(epoch_bins[ep].values())
        epoch_ratios[ep] = [epoch_bins[ep].get(bl, 0) / total if total > 0 else 0
                            for bl in bin_labels]

    # --- Grouped bar chart ---
    n_bins = len(bin_labels)
    n_epochs = len(epochs)
    x = np.arange(n_bins)
    width = 0.8 / n_epochs

    cmap = matplotlib.colormaps.get_cmap("viridis").resampled(n_epochs + 1)

    fig, ax = plt.subplots(figsize=(12, 6))
    for i, ep in enumerate(epochs):
        offset = (i - n_epochs / 2 + 0.5) * width
        bars = ax.bar(x + offset, epoch_ratios[ep], width, label=f"Epoch {ep}",
                      color=cmap(i), edgecolor="white", linewidth=0.5)

    # Shorten bin labels for display
    short_labels = [bl.replace(",", ", ") for bl in bin_labels]
    ax.set_xticks(x)
    ax.set_xticklabels(short_labels, fontsize=9, rotation=30, ha="right")
    ax.set_xlabel("Per-sample Harmful Rate Bin", fontsize=13)
    ax.set_ylabel("Proportion of Samples", fontsize=13)
    ax.set_title(f"{dataset_name} — Distribution by Epoch", fontsize=15, fontweight="bold")
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3, axis="y")

    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"[saved] {output_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True,
                        help="Path to harmful_rate_stats_all.json")
    parser.add_argument("--dataset_name", type=str, required=True,
                        help="Dataset name for title (e.g. WildGuardMix, PKU)")
    parser.add_argument("--prefix", type=str, default="",
                        help="Filename prefix for output PNGs (e.g. 'wildguardmix_')")
    parser.add_argument("--output_dir", type=Path, required=True,
                        help="Directory to save output plots")
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    stats = load_stats(args.input)
    epoch_agg, epoch_bins = aggregate_by_epoch(stats)

    pfx = args.prefix
    plot_overall(epoch_agg, args.dataset_name,
                 args.output_dir / f"{pfx}overall_harmful_rate.png")

    plot_bin_distribution(epoch_bins, args.dataset_name,
                          args.output_dir / f"{pfx}bin_distribution.png")


if __name__ == "__main__":
    main()
