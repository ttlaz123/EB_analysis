import matplotlib
matplotlib.use('Agg') 
import os
import re
import argparse
from getdist import plots
from getdist.mcsamples import loadMCSamples
import matplotlib.pyplot as plt
BIN_TYPES = ['2-8', '9-15', '2-15']
BIN_COLORS = {
    '2-8': 'blue',
    '9-15': 'orange',
    '2-15': 'green'
}


def load_all_chains_in_folder(folder):
    chain_files = sorted([
        f for f in os.listdir(folder)
        if re.match(r"sim\d+\.1\.txt", f)
    ])

    if not chain_files:
        raise FileNotFoundError(f"No matching chains in {folder}")

    samples_list = []
    param_names = None
    for cf in chain_files:
        root = os.path.join(folder, cf.split('.')[0])  # remove extension
        print('Loading: ' + str(root))
        samples = loadMCSamples(root)
        if param_names is None:
            param_names = [p.name for p in samples.getParamNames().names
                           if all(x not in p.name for x in ['chi2','weight','minuslogprior'])]
        samples_list.append(samples)
    return samples_list, param_names


def group_folders_by_prefix(base_dir):
    pattern = re.compile(r"(.+)_bin(2-8|9-15|2-15)_det_polrot_(.+)")
    groups = {}
    for folder in os.listdir(base_dir):
        match = pattern.match(folder)
        if match:
            prefix, bin_type, suffix = match.groups()
            key = f"{prefix}_det_polrot_{suffix}"
            full_path = os.path.join(base_dir, folder)
            groups.setdefault(key, {})[bin_type] = full_path
    return groups


def group_folders_by_prefix(base_dir):
    """
    Groups MCMC result folders by shared prefix and suffix, organizing them
    by bin type (`2-8`, `9-15`, `2-15`).

    Parameters
    ----------
    base_dir : str
        Path to the base directory containing MCMC result subfolders.

    Returns
    -------
    dict
        A dictionary mapping a group label (prefix + suffix) to a dict of
        bin_type -> folder path.
    """
    pattern = re.compile(r"(.+)_bin(2-8|9-15|2-15)_det_polrot_(.+)")
    groups = {}
    for folder in os.listdir(base_dir):
        match = pattern.match(folder)
        if match:
            prefix, bin_type, suffix = match.groups()
            key = f"{prefix}_det_polrot_{suffix}"
            full_path = os.path.join(base_dir, folder)
            groups.setdefault(key, {})[bin_type] = full_path
    return groups

def group_folders_by_prefix(base_dir):
    pattern = re.compile(r"(.+)_bin(2-8|9-15|2-15)_det_polrot_(.+)")
    groups = {}
    for folder in os.listdir(base_dir):
        match = pattern.match(folder)
        if match:
            prefix, bin_type, suffix = match.groups()
            key = f"{prefix}_det_polrot_{suffix}"
            full_path = os.path.join(base_dir, folder)
            groups.setdefault(key, {})[bin_type] = full_path
    return groups

def plot_triangle_for_group(group_label, bin_folders, output_root):
    if not all(bt in bin_folders for bt in BIN_TYPES):
        print(f"[Skipping] Incomplete bin types for group: {group_label}")
        return

    example_bin = BIN_TYPES[0]
    sim_files = sorted([
        f for f in os.listdir(bin_folders[example_bin])
        if re.match(r"sim\d+\.1\.txt", f)
    ])
    n_sims = len(sim_files)
    if n_sims == 0:
        print(f"[Warning] No sim files found in {bin_folders[example_bin]}")
        return

    print(f"[Info] Plotting {n_sims} sims for group {group_label}")

    plotter = plots.get_subplot_plotter()
    group_output_dir = os.path.join(output_root, group_label)
    os.makedirs(group_output_dir, exist_ok=True)

    first_root = os.path.join(bin_folders[example_bin], sim_files[0].split('.')[0])
    first_samples = loadMCSamples(first_root)
    param_names = [p.name for p in first_samples.getParamNames().params
                   if all(x not in p.name for x in ['chi2','weight','minuslogprior'])]

    for sim_idx in range(n_sims):
        samples_for_plot = []
        legend_labels = []
        sim_file_stub = f"sim{sim_idx:03d}"
        try:
            for bt in BIN_TYPES:
                folder = bin_folders[bt]
                root = os.path.join(folder, sim_file_stub)
                samples = loadMCSamples(root)
                samples_for_plot.append(samples)
                legend_labels.append(f"bin {bt}")
        except Exception as e:
            print(f"[Warning] Missing sim {sim_file_stub} in some bins: {e}")
            continue

        plotter.triangle_plot(samples_for_plot, param_names, filled=True,
                              legend_labels=legend_labels, legend_loc='upper right')

        outpath = os.path.join(group_output_dir, f"{sim_file_stub}.png")
        plotter.export(outpath)
        print(f"[Saved] {outpath}")

        plt.close('all')

def main():
    parser = argparse.ArgumentParser(description="Plot triangle plots for det_polrot MCMC folders across bins.")
    parser.add_argument("base_dir", help="Path to directory containing MCMC result folders.")
    parser.add_argument("--output_dir", default=None, help="Root directory to save all plots.")
    args = parser.parse_args()
    if(args.output_dir is None):
        args.output_dir = os.path.join(args.base_dir, 'bindiffs') 
    groups = group_folders_by_prefix(args.base_dir)
    for group_label, bin_folders in groups.items():
        plot_triangle_for_group(group_label, bin_folders, args.output_dir)

if __name__ == "__main__":
    main()
