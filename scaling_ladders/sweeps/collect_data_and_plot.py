#!/usr/bin/env python3
"""
collect_data_and_plot.py

Script to collect evaluation data from directory structure and plot results.
"""

import argparse
import json
import re
from pathlib import Path
from typing import Dict, Any, Optional, Set, Tuple
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
import yaml

from benchmark_metrics import BENCHMARK_METRICS


def read_config(config_path: Path) -> Dict[str, Any]:
    """
    Read the config.yaml file and extract relevant values.

    Args:
        config_path: Path to the config.yaml file

    Returns:
        Dictionary with batch_size, steps, tp_size, and seq_len
    """
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    return {
        'batch_size': config.get('batch_size'),
        'steps': config.get('steps'),
        'tp_size': config.get('distributed', {}).get('tp_size'),
        'seq_len': config.get('model', {}).get('max_seq_len'),
    }


def get_num_nodes(top_level_name: str) -> Optional[int]:
    """
    Determine the number of nodes based on the top-level directory name.

    Args:
        top_level_name: Name of the top-level directory

    Returns:
        Number of nodes, or None if pattern not recognized
    """

    if "e18_3p92" in top_level_name:
        return 1
    elif "e19_1p86" in top_level_name:
        return 2
    elif "e19_6p92" in top_level_name:
        return 4

    for model_size in ["_30M", "_70M", "_100M", "_200M", "_300M"]:
        if model_size in top_level_name:
            return 1
    if "_500M" in top_level_name:
        return 2
    elif "_1B" in top_level_name:
        return 4
    elif "_3B" in top_level_name:
        return 12

    return None


def read_data_from_directory(base_dir: str) -> Tuple[Dict[str, Dict[str, Dict[str, float]]], Dict[str, Dict[str, Any]]]:
    """
    Read data from the evaluation directory structure.

    Args:
        base_dir: Base directory path containing the subdirectories

    Returns:
        Tuple of:
        - Nested dictionary structure:
          {
              top_level_dir: {
                  eval_subdir: {
                      json_filename: nll_char_target_value
                  }
              }
          }
        - Config dictionary:
          {
              top_level_dir: {
                  'batch_size': int,
                  'steps': int,
                  'tp_size': int,
                  'seq_len': int,
                  'num_nodes': int
              }
          }
    """
    data = {}
    configs = {}

    base_path = Path(base_dir)

    if not base_path.exists():
        raise FileNotFoundError(f"Base directory not found: {base_dir}")

    # Loop over the top-level subdirectories (9 subdirectories)
    for top_level_dir in sorted(base_path.iterdir()):
        if not top_level_dir.is_dir():
            continue

        if "eval" in top_level_dir.name:
            continue

        top_level_name = top_level_dir.name
        print(f"Processing model: {top_level_name}")
        data[top_level_name] = {}

        # Read config.yaml
        config_path = top_level_dir / "config.yaml"
        if config_path.exists():
            try:
                configs[top_level_name] = read_config(config_path)
            except Exception as e:
                print(f"Error reading config for {top_level_name}: {e}")
                configs[top_level_name] = None
        else:
            print(f"Warning: config.yaml not found in {top_level_name}")
            configs[top_level_name] = None

        # Set num_nodes based on directory name
        if configs[top_level_name] is not None:
            num_nodes = get_num_nodes(top_level_name)
            if num_nodes is not None:
                configs[top_level_name]['num_nodes'] = num_nodes
            else:
                print(f"Warning: Could not determine num_nodes for {top_level_name}")
                configs[top_level_name]['num_nodes'] = None

        # Path to evals directory
        evals_path = top_level_dir / "evals"

        if not evals_path.exists():
            continue

        # Loop over eval subdirectories (eval_0000002000, eval_0000004000, etc.)
        for eval_subdir in sorted(evals_path.iterdir()):
            if not eval_subdir.is_dir():
                continue
            if not eval_subdir.name.startswith("eval_"):
                continue

            eval_name = eval_subdir.name
            data[top_level_name][eval_name] = {}

            # Path to results directory
            results_path = eval_subdir / "results"

            if not results_path.exists():
                continue

            # Loop over JSON files (arc_easy.json, arc_challenge.json, etc.)
            for json_file in sorted(results_path.glob("*.json")):
                json_filename = json_file.name

                # Handle ppl_v2.json separately
                if json_filename == "ppl_v2.json":
                    try:
                        with open(json_file, 'r') as f:
                            json_data = json.load(f)
                            ppl_metrics = extract_ppl_metrics(json_data)
                            data[top_level_name][eval_name].update(ppl_metrics)
                    except json.JSONDecodeError:
                        pass
                    except IOError:
                        pass
                    continue

                # Skip ppl.json files
                if json_filename == "ppl.json":
                    continue

                try:
                    with open(json_file, 'r') as f:
                        json_data = json.load(f)
                        value = extract_value_from_json(json_data, json_filename)
                        if value is not None:
                            data[top_level_name][eval_name][json_filename] = value
                except json.JSONDecodeError:
                    pass
                except IOError:
                    pass

    # Read additional benchmarks from the eval directory
    eval_dir = base_path / "eval"
    if eval_dir.exists() and eval_dir.is_dir():
        data = read_eval_directory(eval_dir, data)

    return data, configs


def read_eval_directory(
    eval_dir: Path,
    data: Dict[str, Dict[str, Dict[str, float]]]
) -> Dict[str, Dict[str, Dict[str, float]]]:
    """
    Read additional benchmark data from the eval directory structure.

    The eval directory has structure:
        eval/{model_name}/checkpoint_{number}/results/*.json

    This merges results into the existing data structure, matching
    checkpoint numbers to eval names (e.g., checkpoint_0000000266 -> eval_0000000266).

    Args:
        eval_dir: Path to the eval directory
        data: Existing data dictionary to merge into

    Returns:
        Updated data dictionary with additional benchmark results
    """
    # Loop over model subdirectories (e.g., _30M, _70M, etc.)
    for model_dir in sorted(eval_dir.iterdir()):
        if not model_dir.is_dir():
            continue

        model_name = model_dir.name

        # Skip if this model wasn't found in the top-level directories
        if model_name not in data:
            continue

        print(f"Processing eval directory for model: {model_name}")

        # Loop over checkpoint subdirectories (checkpoint_0000000266, etc.)
        for checkpoint_dir in sorted(model_dir.iterdir()):
            if not checkpoint_dir.is_dir():
                continue
            if not checkpoint_dir.name.startswith("checkpoint_"):
                continue

            # Convert checkpoint name to eval name
            # checkpoint_0000000266 -> eval_0000000266
            checkpoint_num = checkpoint_dir.name.replace("checkpoint_", "")
            eval_name = f"eval_{checkpoint_num}"

            # Initialize eval entry if it doesn't exist
            if eval_name not in data[model_name]:
                data[model_name][eval_name] = {}

            # Path to results directory
            results_path = checkpoint_dir / "results"

            if not results_path.exists():
                continue

            # Loop over JSON files
            for json_file in sorted(results_path.glob("*.json")):
                json_filename = json_file.name

                # Handle ppl_v2.json separately
                if json_filename == "ppl_v2.json":
                    try:
                        with open(json_file, 'r') as f:
                            json_data = json.load(f)
                            ppl_metrics = extract_ppl_metrics(json_data)
                            # Only add ppl metrics that don't already exist
                            for key, value in ppl_metrics.items():
                                if key not in data[model_name][eval_name]:
                                    data[model_name][eval_name][key] = value
                    except json.JSONDecodeError:
                        pass
                    except IOError:
                        pass
                    continue

                # Skip ppl.json files
                if json_filename == "ppl.json":
                    continue

                # Skip if we already have this benchmark from the main evals
                if json_filename in data[model_name][eval_name]:
                    continue

                try:
                    with open(json_file, 'r') as f:
                        json_data = json.load(f)
                        value = extract_value_from_json(json_data, json_filename)
                        if value is not None:
                            data[model_name][eval_name][json_filename] = value
                except json.JSONDecodeError:
                    pass
                except IOError:
                    pass

    return data


def extract_value_from_json(json_data: Dict[str, Any], json_filename: str) -> Optional[float]:
    """
    Extract the metric value from the JSON data based on benchmark name.

    Uses BENCHMARK_METRICS mapping to determine which key to extract.
    Falls back to bits_per_byte_target for unknown benchmarks.

    Args:
        json_data: Parsed JSON data
        json_filename: Name of the JSON file

    Returns:
        The metric value, or None if not found
    """
    benchmark_name = json_filename.replace(".json", "")
    metric_key = BENCHMARK_METRICS.get(benchmark_name, "bits_per_byte_target")

    try:
        return json_data["results"][metric_key]
    except KeyError:
        return None


def extract_ppl_metrics(json_data: Dict[str, Any]) -> Dict[str, float]:
    """
    Extract bits_per_byte metrics from ppl_v2.json data.

    Looks for keys containing "bits_per_byte" and extracts the benchmark name
    from the key. For example:
        "--checkpoint--transformer2--data--internal_ppl--notes.val__notes/bits_per_byte": 8.58
    extracts benchmark name "notes" (between "__" and "/bits_per_byte").

    Args:
        json_data: Parsed ppl_v2.json data

    Returns:
        Dictionary mapping "ppl_{benchmark_name}.json" to the bits_per_byte value
    """
    ppl_metrics = {}

    # The metrics are nested under "results"
    results = json_data.get("results", {})

    for key, value in results.items():
        if "/bits_per_byte" in key and isinstance(value, (int, float)):
            # Extract benchmark name: find text between "__" and "/bits_per_byte"
            # Example: "--checkpoint--transformer2--data--internal_ppl--notes.val__notes/bits_per_byte"
            # We want "notes"
            match = re.search(r'__([^/]+)/bits_per_byte', key)
            if match:
                benchmark_name = match.group(1)
                # Use a naming convention to distinguish ppl metrics
                json_key = f"ppl_{benchmark_name}.json"
                ppl_metrics[json_key] = value

    return ppl_metrics


def extract_checkpoint_number(eval_name: str) -> int:
    """
    Extract the checkpoint number from the eval subdirectory name.

    Args:
        eval_name: String like "eval_0000002000"

    Returns:
        Integer checkpoint number (e.g., 2000)
    """
    match = re.search(r'eval_(\d+)', eval_name)
    if match:
        return int(match.group(1))
    raise ValueError(f"Could not extract checkpoint number from: {eval_name}")


def calculate_tokens(checkpoint_num: int, config: Dict[str, Any]) -> Optional[int]:
    """
    Calculate the number of unique tokens trained up to a checkpoint.

    Args:
        checkpoint_num: The checkpoint/step number
        config: Dictionary with batch_size, tp_size, seq_len, and num_nodes

    Returns:
        Number of tokens: batch_size * checkpoint_num * seq_len * (8 / tp_size) * num_nodes
        Returns None if any required config value is missing
    """
    batch_size = config.get('batch_size')
    tp_size = config.get('tp_size')
    seq_len = config.get('seq_len')
    num_nodes = config.get('num_nodes')

    if any(v is None for v in [batch_size, tp_size, seq_len, num_nodes]):
        return None

    return int(batch_size * checkpoint_num * seq_len * (8 / tp_size) * num_nodes)


def get_all_json_filenames(data: Dict[str, Dict[str, Dict[str, float]]]) -> Set[str]:
    """
    Get all unique JSON filenames from the data.

    Args:
        data: The nested dictionary of collected data

    Returns:
        Set of unique JSON filenames
    """
    json_filenames = set()
    for top_level, eval_data in data.items():
        for eval_name, json_data in eval_data.items():
            json_filenames.update(json_data.keys())
    return json_filenames


def compute_average_benchmark(
    data: Dict[str, Dict[str, Dict[str, float]]],
    json_filenames: Set[str]
) -> Dict[str, Dict[str, float]]:
    """
    Compute the average nll_char_target across all benchmarks for each checkpoint.

    Args:
        data: The nested dictionary of collected data
        json_filenames: Set of all JSON filenames

    Returns:
        Dictionary structure:
        {
            top_level_dir: {
                eval_subdir: average_value
            }
        }
    """
    average_data = {}

    for top_level, eval_data in data.items():
        average_data[top_level] = {}

        for eval_name, json_data in eval_data.items():
            values = [v for k, v in json_data.items() if k in json_filenames]
            if values:
                average_data[top_level][eval_name] = sum(values) / len(values)

    return average_data


def plot_data(
    data: Dict[str, Dict[str, Dict[str, float]]],
    configs: Dict[str, Dict[str, Any]],
    output_dir: str = ".",
    last_point_only: bool = False
) -> None:
    """
    Create a grid of subplots, one per unique JSON filename plus an average.

    The grid size is calculated dynamically based on the number of benchmarks.

    Args:
        data: The nested dictionary of collected data
        configs: Dictionary of config values per top-level directory
        output_dir: Directory to save the plot
        last_point_only: If True, only plot the last point for each model
    """
    # Get all unique JSON filenames
    # Sort so that ppl_ benchmarks appear at the end
    all_filenames = get_all_json_filenames(data)
    non_ppl = sorted([f for f in all_filenames if not f.startswith('ppl_')])
    ppl = sorted([f for f in all_filenames if f.startswith('ppl_')])
    json_filenames = non_ppl + ppl

    # Calculate grid size: num_benchmarks + 1 for average
    num_plots = len(json_filenames) + 1
    num_cols = 4
    num_rows = (num_plots + num_cols - 1) // num_cols  # Ceiling division

    # Compute average benchmark
    average_data = compute_average_benchmark(data, set(json_filenames))

    # Get all top-level directories for consistent coloring
    top_level_dirs = sorted(data.keys())
    num_dirs = len(top_level_dirs)

    # Create a color map
    colors = cm.tab10(np.linspace(0, 1, max(num_dirs, 10)))
    color_map = {dir_name: colors[i] for i, dir_name in enumerate(top_level_dirs)}

    # Create output directory if it doesn't exist
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Create dynamic grid of subplots
    fig, axes = plt.subplots(num_rows, num_cols, figsize=(24, 5 * num_rows))
    axes = axes.flatten()

    # Plot each JSON file in a subplot
    for idx, json_filename in enumerate(json_filenames):
        ax = axes[idx]

        # Plot data for each top-level directory
        for top_level_dir in top_level_dirs:
            eval_data = data.get(top_level_dir, {})
            config = configs.get(top_level_dir)

            if config is None:
                continue

            # Collect x (tokens) and y (nll_char_target values)
            x_values = []
            y_values = []

            for eval_name, json_data in eval_data.items():
                if json_filename in json_data:
                    checkpoint_num = extract_checkpoint_number(eval_name)
                    tokens = calculate_tokens(checkpoint_num, config)
                    if tokens is not None:
                        x_values.append(tokens)
                        y_values.append(json_data[json_filename])

            if x_values:
                # Sort by tokens
                sorted_pairs = sorted(zip(x_values, y_values))
                x_values, y_values = zip(*sorted_pairs)

                if last_point_only:
                    # Only plot the last point
                    ax.scatter(
                        [x_values[-1]],
                        [y_values[-1]],
                        color=color_map[top_level_dir],
                        label=top_level_dir,
                        s=50
                    )
                else:
                    ax.plot(
                        x_values,
                        y_values,
                        marker='o',
                        linestyle='-',
                        color=color_map[top_level_dir],
                        label=top_level_dir,
                        markersize=3,
                        linewidth=1
                    )

        # Customize subplot
        task_name = json_filename.replace('.json', '')
        ax.set_xlabel('Tokens Trained', fontsize=9)
        ax.set_ylabel('nll_char_target', fontsize=9)
        ax.set_title(task_name, fontsize=10)
        ax.set_xscale('log')
        ax.grid(True, alpha=0.3)
        ax.tick_params(axis='both', labelsize=8)

    # Plot average benchmark in the next subplot after all benchmarks
    avg_idx = len(json_filenames)
    ax = axes[avg_idx]
    for top_level_dir in top_level_dirs:
        avg_eval_data = average_data.get(top_level_dir, {})
        config = configs.get(top_level_dir)

        if config is None:
            continue

        x_values = []
        y_values = []

        for eval_name, avg_value in avg_eval_data.items():
            checkpoint_num = extract_checkpoint_number(eval_name)
            tokens = calculate_tokens(checkpoint_num, config)
            if tokens is not None:
                x_values.append(tokens)
                y_values.append(avg_value)

        if x_values:
            # Sort by tokens
            sorted_pairs = sorted(zip(x_values, y_values))
            x_values, y_values = zip(*sorted_pairs)

            if last_point_only:
                # Only plot the last point
                ax.scatter(
                    [x_values[-1]],
                    [y_values[-1]],
                    color=color_map[top_level_dir],
                    label=top_level_dir,
                    s=50
                )
            else:
                ax.plot(
                    x_values,
                    y_values,
                    marker='o',
                    linestyle='-',
                    color=color_map[top_level_dir],
                    label=top_level_dir,
                    markersize=3,
                    linewidth=1
                )

    ax.set_xlabel('Tokens Trained', fontsize=9)
    ax.set_ylabel('nll_char_target', fontsize=9)
    ax.set_title('Average (all benchmarks)', fontsize=10, fontweight='bold')
    ax.set_xscale('log')
    ax.grid(True, alpha=0.3)
    ax.tick_params(axis='both', labelsize=8)

    # Hide any unused subplots
    for idx in range(avg_idx + 1, len(axes)):
        axes[idx].set_visible(False)

    # Create a single legend for the entire figure
    # Use custom handles to ensure all models are included with correct colors
    from matplotlib.lines import Line2D
    legend_handles = [
        Line2D([0], [0], color=color_map[dir_name], marker='o', linestyle='-',
               markersize=5, linewidth=2, label=dir_name)
        for dir_name in top_level_dirs
    ]
    fig.legend(
        handles=legend_handles,
        loc='center right',
        bbox_to_anchor=(0.99, 0.5),
        fontsize=9,
        title='Models'
    )

    # Adjust layout
    plt.tight_layout()
    plt.subplots_adjust(right=0.85)  # Make room for legend

    # Save plot
    suffix = "_last_point" if last_point_only else ""
    plot_filename = output_path / f"all_benchmarks{suffix}_bpb.png"
    plt.savefig(plot_filename, dpi=150, bbox_inches='tight')
    print(f"Saved plot: {plot_filename}")

    plt.close(fig)


def parse_args() -> argparse.Namespace:
    """
    Parse command-line arguments.

    Returns:
        Parsed arguments
    """
    parser = argparse.ArgumentParser(
        description="Collect evaluation data and plot results."
    )
    parser.add_argument(
        "--last-point-only",
        action="store_true",
        help="Only plot the last point (final checkpoint) for each model instead of the full curve."
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./plots/2026-02-11",
        help="Directory to save plots (default: ./plots)"
    )
    return parser.parse_args()


def main() -> None:
    """Main entry point for the script."""
    args = parse_args()

    base_dir = "/checkpoint/transformer2/jessedodge/amaia_dumps/sweep_20260209_v6-1_baseline/v4" #"/checkpoint/transformer2/jessedodge/amaia_dumps/sweep_ladder_new_eval/"
    output_dir = args.output_dir

    # Read data from directory
    data, configs = read_data_from_directory(base_dir)

    # Plot the data
    plot_data(data, configs, output_dir, last_point_only=args.last_point_only)

    print(f"\nPlot saved to: {output_dir}")


if __name__ == "__main__":
    main()
