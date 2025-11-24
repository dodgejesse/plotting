# conda activate /checkpoint/transformer2/envs/amaia_fair-sc_092625/

import os
import re
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
from collections import defaultdict

base_directory = "/checkpoint/transformer2/jessedodge/amaia_dumps/scaling_tokens/"
parent_experiment_name = "varying_model_size"
experiment_name = "train_with_tp_size=1"


sort_alphabetically = False

def read_config(config_file_path):
    """Read config.yaml and extract batch_size, steps, and max_seq_len."""
    batch_size = None
    steps = None
    max_seq_len = None

    with open(config_file_path, 'r') as f:
        for line in f:
            if line.startswith("batch_size: "):
                match = re.search(r'batch_size:\s*(\d+)', line)
                if match:
                    batch_size = int(match.group(1))
            if line.startswith("steps: "):
                match = re.search(r'steps:\s*(\d+)', line)
                if match:
                    steps = int(match.group(1))
            if line.startswith("seq_len: "):
                match = re.search(r'seq_len:\s*(\d+)', line)
                if match:
                    max_seq_len = int(match.group(1))

    return batch_size, steps, max_seq_len

def get_num_gpus(filepath):
    if "295Mparams" in filepath:
        return 8 # one node
    elif "603Mparams" in filepath:
        return 16 # two nodes
    elif "1Bparams" in filepath:
        return 4*8

def extract_loss_data(log_file_path, batch_size, max_seq_len):
    """Extract step numbers and loss values from train.log."""
    step_values = []
    loss_values = []

    with open(log_file_path, 'r') as f:
        for line in f:
            if "loss" in line.lower() and "step" in line.lower():
                # Extract step number
                step_match = re.search(r'step[:\s=]+(\d+)', line, re.IGNORECASE)
                # Extract loss value
                loss_match = re.search(r'loss[:\s=]+([0-9]+\.?[0-9]*(?:[eE][+-]?[0-9]+)?)', line, re.IGNORECASE)

                if step_match and loss_match:
                    step = int(step_match.group(1))
                    loss = float(loss_match.group(1))

                    # Multiply step by batch_size * max_seq_len * num_gpus
                    adjusted_step = step * batch_size * max_seq_len * get_num_gpus(log_file_path)

                    step_values.append(adjusted_step)
                    loss_values.append(loss)

    return step_values, loss_values

def extract_group_key(subdir_name):
    """Extract the grouping key from subdirectory name (between first '_' and first '-')."""
    # Find the part after the first '_' and before the first '-'
    match = re.search(r'_([^-]+)', subdir_name)
    if match:
        return match.group(1)
    return "unknown"

def process_subdirectory(subdir_path, subdir_name):
    """Process a single subdirectory and return its training data."""
    log_file_path = os.path.join(subdir_path, "train.log")
    config_file_path = os.path.join(subdir_path, "config.yaml")

    # Check if both files exist
    if not (os.path.exists(log_file_path) and os.path.exists(config_file_path)):
        return None

    # Read config
    batch_size, steps, max_seq_len = read_config(config_file_path)

    # Validate config values
    if batch_size is None or max_seq_len is None:
        return None

    # Extract loss data
    step_values, loss_values = extract_loss_data(log_file_path, batch_size, max_seq_len)

    # Return results if we found values
    if step_values and loss_values:
        return {
            'steps': step_values,
            'losses': loss_values,
            'batch_size': batch_size,
            'max_seq_len': max_seq_len
        }

    return None

def collect_training_data(base_directory):
    """Collect training data from all subdirectories."""
    results = {}

    for subdir in os.listdir(base_directory):
        subdir_path = os.path.join(base_directory, subdir)

        # Check if it's a directory
        if os.path.isdir(subdir_path):
            data = process_subdirectory(subdir_path, subdir)
            if data is not None:
                results[subdir] = data

    return results

def group_results_by_key(results):
    """Group results by the extracted key from subdirectory names."""
    grouped = defaultdict(dict)

    for subdir_name, data in results.items():
        group_key = extract_group_key(subdir_name)
        grouped[group_key][subdir_name] = data

    return grouped

def sort_groups(group_keys):
    sorted = ["300Mtokens", "1Btokens", "3Btokens", "10Btokens", "30Btokens", "100Btokens"]

    for item in sorted:
        if item not in group_keys:
            raise KeyError

    return sorted

def plot_training_curves(results, output_file='training_curves.pdf'):
    """Create and save plot of all training curves with grouped subplots."""
    # Group results by key
    grouped_results = group_results_by_key(results)

    #import pdb;
    #pdb.set_trace()

    # Sort groups alphabetically
    if sort_alphabetically:
        sorted_groups = sorted(grouped_results.keys())
    else:
        sorted_groups = sort_groups(grouped_results.keys())
    num_groups = len(sorted_groups)

    # Create subplots (arrange in a grid)
    cols = min(2, num_groups)  # Max 2 columns
    rows = (num_groups + cols - 1) // cols  # Ceiling division

    fig, axes = plt.subplots(rows, cols, figsize=(14, 6 * rows))

    # Handle case where there's only one subplot
    if num_groups == 1:
        axes = np.array([axes])
    axes = axes.flatten() if num_groups > 1 else axes

    # Plot each group in its own subplot
    for idx, group_key in enumerate(sorted_groups):
        ax = axes[idx]
        group_data = grouped_results[group_key]

        # Generate distinct colors for curves in this group
        sorted_subdirs = sorted(group_data.keys())
        num_curves = len(sorted_subdirs)
        colors = cm.tab20(np.linspace(0, 1, num_curves)) if num_curves <= 20 else cm.viridis(np.linspace(0, 1, num_curves))

        # Plot each curve in this group
        for curve_idx, subdir in enumerate(sorted_subdirs):
            data = group_data[subdir]
            step_values = data['steps']
            loss_values = data['losses']

            if "bszCrit" in subdir:
                cur_label = f"{subdir}={data['batch_size']}"
            else:
                cur_label = subdir

            # Plot the training curve with unique color
            ax.plot(step_values, loss_values, label=cur_label, color=colors[curve_idx], alpha=0.7)

        ax.set_xlabel('Tokens (Step × Batch Size × Max Seq Len)')
        ax.set_ylabel('Loss')
        ax.set_title(f'Training Curves - {group_key}')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    # Hide any unused subplots
    for idx in range(num_groups, len(axes)):
        axes[idx].axis('off')

    plt.tight_layout()

    # Save to PDF
    output_dir = "./plots/"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    plt.savefig(output_dir + output_file, format='pdf', bbox_inches='tight')
    print(f"Plot saved to {output_file}")

def main():
    directory_path = f"{base_directory}/{parent_experiment_name}/{experiment_name}/"

    # Collect training data from all subdirectories
    results = collect_training_data(directory_path)

    # Plot and save the results
    if results:
        plot_training_curves(results, f"{parent_experiment_name}_{experiment_name}_training_curves.pdf")
    else:
        print("No training data found.")

if __name__ == "__main__":
    main()
