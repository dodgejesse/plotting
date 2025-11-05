import os
import re
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np

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

                    # Multiply step by batch_size * max_seq_len
                    adjusted_step = step * batch_size * max_seq_len

                    step_values.append(adjusted_step)
                    loss_values.append(loss)

    return step_values, loss_values

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

def plot_training_curves(results, output_file='training_curves.pdf'):

    #import pdb; pdb.set_trace()


    """Create and save plot of all training curves."""
    plt.figure(figsize=(12, 8))

    # Generate distinct colors for each curve
    sorted_subdirs = sorted(results.keys())
    num_curves = len(sorted_subdirs)
    colors = cm.tab20(np.linspace(0, 1, num_curves)) if num_curves <= 20 else cm.viridis(np.linspace(0, 1, num_curves))

    # Plot each training curve
    for idx, subdir in enumerate(sorted_subdirs):
        data = results[subdir]
        step_values = data['steps']
        loss_values = data['losses']

        # Plot the training curve with unique color
        plt.plot(step_values, loss_values, label=subdir, color=colors[idx], alpha=0.7)

    plt.xlabel('Tokens (Step × Batch Size × Max Seq Len)')
    plt.ylabel('Loss')
    plt.title('Training Curves')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    # Save to PDF
    plt.savefig(output_file, format='pdf', bbox_inches='tight')
    print(f"Plot saved to {output_file}")

def main():
    # Hardcoded directory path
    base_directory = "/checkpoint/transformer2/jessedodge/amaia_dumps/scaling_tokens/batch_size_experiments/train/"

    # Collect training data from all subdirectories
    results = collect_training_data(base_directory)

    # Plot and save the results
    if results:
        plot_training_curves(results)
    else:
        print("No training data found.")

if __name__ == "__main__":
    main()
