import matplotlib
matplotlib.use('Agg') 
import os
import re
import argparse
import numpy as np
from getdist import plots, MCSamples
from getdist.mcsamples import loadMCSamples

BIN_TYPES = ['2-8', '9-15', '2-15']
BIN_COLORS = {
    '2-8': 'blue',
    '9-15': 'orange',
    '2-15': 'green'
}


def parse_chain_file(filepath):
    """
    Parses a GetDist MCMC chain text file.

    Parameters
    ----------
    filepath : str
        Path to the `.txt` file containing the chain samples.

    Returns
    -------
    header : list of str
        List of column names from the header row.
    data : ndarray
        Numpy array of shape (n_samples, n_columns) with the MCMC sample values.
    """
    with open(filepath) as f:
        header = f.readline().strip().split()
    data = np.loadtxt(filepath, skiprows=1)
    return header, data

def load_chains(folder):
    """
    Loads MCMC chains from a folder and returns a GetDist MCSamples object
    containing only polarization rotation parameters.

    Parameters
    ----------
    folder : str
        Path to the folder containing `.1.txt` MCMC chain files.

    Returns
    -------
    MCSamples
        GetDist MCSamples object containing alpha_* parameters and LaTeX labels.

    Raises
    ------
    FileNotFoundError
        If no matching chain files are found in the specified folder.
    """
    chain_files = sorted([
        f for f in os.listdir(folder)
        if re.match(r"sim\d+\.1\.txt", f)
    ])

    if not chain_files:
        raise FileNotFoundError(f"No matching chains in {folder}")
    root = chain_files[0].split('.')[0]
    print(chain_files[0])
    print(root)
    samples = loadMCSamples(root)
    
    param_names = [name.name for name in samples.getParamNames().names
                   if ('chi2' not in name.name and
                       'weight' not in name.name and
                       'minuslogprior' not in name.name)]
    print(param_names)
    
    # Clean latex labels
    latex_labels = [name.replace("_", "\\_") for name in param_names]
    return MCSamples(samples=samples, names=param_names, labels=latex_labels)

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

def plot_triangle_for_group(group_label, bin_folders, output_dir):
    """
    Generates and saves a GetDist triangle plot comparing posterior distributions
    across bin types for a given detector polarization rotation group.

    Parameters
    ----------
    group_label : str
        Label for the group of folders being plotted.
    bin_folders : dict
        Mapping from bin types (`2-8`, `9-15`, `2-15`) to their respective folder paths.
    output_dir : str
        Directory where the output plot will be saved.

    Returns
    -------
    None
    """
    if not all(bt in bin_folders for bt in BIN_TYPES):
        print(f"[Skipping] Incomplete bin types for group: {group_label}")
        return

    samples_list = []
    legend_labels = []

    for bin_type in BIN_TYPES:
        try:
            folder = bin_folders[bin_type]
            samples = load_chains(folder)
            samples_list.append(samples)
            legend_labels.append(f"bin {bin_type}")
        except Exception as e:
            print(f"[Error] Failed to load {bin_type} in {folder}: {e}")
            return

    plotter = plots.get_subplot_plotter()
    plotter.triangle_plot(samples_list, filled=True, legend_labels=legend_labels, legend_loc='upper right')

    os.makedirs(output_dir, exist_ok=True)
    outpath = os.path.join(output_dir, f"triangle_{group_label}.png")
    plotter.export(outpath)
    print(f"[Saved] {outpath}")

def main():
    parser = argparse.ArgumentParser(description="Plot triangle plots for det_polrot MCMC folders across bins.")
    parser.add_argument("base_dir", help="Path to directory containing MCMC result folders.")
    parser.add_argument("--output_dir", default="triangle_plots", help="Directory to save output plots.")

    args = parser.parse_args()
    groups = group_folders_by_prefix(args.base_dir)
    for group_label, bin_folders in groups.items():
        plot_triangle_for_group(group_label, bin_folders, args.output_dir)

if __name__ == "__main__":
    main()
