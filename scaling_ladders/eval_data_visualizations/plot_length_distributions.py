"""
Script to load benchmark datasets and plot the distribution of text lengths (in characters).
Creates a main figure with subplots for each benchmark, showing histograms with median lines.
"""

import json
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from benchmark_paths import BENCHMARK_PATHS


def load_text_lengths(filepath: str) -> list[int]:
    """Load a JSONL file and return the character lengths of each 'text' field."""
    lengths = []
    with open(filepath, "r") as f:
        for line in f:
            if line.strip():
                data = json.loads(line)
                text = data.get("text", "")
                lengths.append(len(text))
    return lengths


def plot_length_distributions(benchmark_paths: dict[str, str], output_path: str = None):
    """
    Plot histograms of text lengths for each benchmark as subplots.

    Args:
        benchmark_paths: Dictionary mapping benchmark names to filepaths
        output_path: Optional path to save the figure
    """
    n_benchmarks = len(benchmark_paths)
    if n_benchmarks == 0:
        print("No benchmarks to plot.")
        return

    n_cols = min(4, n_benchmarks)
    n_rows = math.ceil(n_benchmarks / n_cols)

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(6 * n_cols, 4 * n_rows))

    if n_benchmarks == 1:
        axes = np.array([axes])
    axes = axes.flatten()

    for idx, (name, filepath) in enumerate(benchmark_paths.items()):
        ax = axes[idx]

        if not Path(filepath).exists():
            print(f"reading {name}.... file not found!")
            ax.text(
                0.5,
                0.5,
                f"File not found:\n{filepath}",
                ha="center",
                va="center",
                transform=ax.transAxes,
                fontsize=10,
            )
            ax.set_title(name)
            continue

        print(f"reading {name}....", end=" ", flush=True)
        lengths = load_text_lengths(filepath)
        print("done.")

        if not lengths:
            print(f"{name} summary statistics: no data found")
            ax.text(
                0.5,
                0.5,
                "No data found",
                ha="center",
                va="center",
                transform=ax.transAxes,
            )
            ax.set_title(name)
            continue

        median_length = np.median(lengths)
        min_length = np.min(lengths)
        max_length = np.max(lengths)
        q1 = np.percentile(lengths, 25)
        q3 = np.percentile(lengths, 75)

        print(
            f"{name} summary statistics of length in characters: "
            f"min={min_length}, q1={q1:.0f}, median={median_length:.0f}, q3={q3:.0f}, max={max_length}"
        )

        ax.hist(lengths, bins=50, edgecolor="black", alpha=0.7)
        ax.axvline(
            median_length,
            color="red",
            linestyle="--",
            linewidth=2,
            label="Median",
        )

        ax.set_xlabel("Text Length (characters)")
        ax.set_ylabel("Count")
        ax.set_title(
            f"{name}\n"
            f"n={len(lengths):,}, min={min_length}\n"
            f"Q1={q1:.0f}, median={median_length:.0f}, Q3={q3:.0f}, max={max_length}"
        )
        ax.legend()

    for idx in range(n_benchmarks, len(axes)):
        axes[idx].set_visible(False)

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        print(f"Saved figure to {output_path}")

    plt.show()


if __name__ == "__main__":
    plot_length_distributions(
        BENCHMARK_PATHS,
        output_path="plots/length_distributions.png",
    )
