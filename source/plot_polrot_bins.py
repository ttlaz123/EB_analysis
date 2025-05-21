import matplotlib
matplotlib.use('Agg') 
import os
import re
import argparse
from getdist import plots
from getdist.mcsamples import loadMCSamples

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
        samples = loadMCSamples(root)
        if param_names is None:
            param_names = [p.name for p in samples.getParamNames().params
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

    # Load all sims per bin type
    bin_samples = {}
    param_names = None
    for bt in BIN_TYPES:
        samples_list, param_names = load_all_chains_in_folder(bin_folders[bt])
        bin_samples[bt] = samples_list

    n_sims = len(next(iter(bin_samples.values())))
    print(f"[Info] Plotting {n_sims} sims for group {group_label}")

    plotter = plots.get_subplot_plotter()

    # Create output directory for this group
    group_output_dir = os.path.join(output_root, group_label)
    os.makedirs(group_output_dir, exist_ok=True)

    for sim_idx in range(n_sims):
        try:
            # Get the samples of sim_idx from all bins
            samples_for_plot = [bin_samples[bt][sim_idx] for bt in BIN_TYPES]
        except IndexError:
            print(f"[Warning] sim_idx {sim_idx} missing in some bins, skipping")
            continue

        legend_labels = [f"bin {bt}" for bt in BIN_TYPES]

        plotter.triangle_plot(samples_for_plot, param_names, filled=True,
                              legend_labels=legend_labels, legend_loc='upper right')

        outpath = os.path.join(group_output_dir, f"sim{sim_idx:03d}.png")
        plotter.export(outpath)
        print(f"[Saved] {outpath}")

def main():
    parser = argparse.ArgumentParser(description="Plot triangle plots for det_polrot MCMC folders across bins.")
    parser.add_argument("base_dir", help="Path to directory containing MCMC result folders.")
    parser.add_argument("--output_dir", default=None, help="Root directory to save all plots.")
    args = parser.parse_args()
    if(args.output_dir is None):
        args.output_dir = args.base_dir
    groups = group_folders_by_prefix(args.base_dir)
    for group_label, bin_folders in groups.items():
        plot_triangle_for_group(group_label, bin_folders, args.output_dir)

if __name__ == "__main__":
    main()
