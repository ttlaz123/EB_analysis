import os
import glob
import argparse
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

def plot_beta_histogram_stack(sim_folder, real_file, param="alpha_cmb", bins=50, outpath="beta_histograms.png"):
    """
    Plot 1D histograms of `alpha_cmb` (renamed to `beta_cmb`) from MCMC chains:
      - All sim chains in `sim_folder`: gray, semi-transparent
      - Real constraint from `real_file`: red and bold

    Args:
        sim_folder (str): Folder containing simulation chain files.
        real_file (str): File path to real data chain.
        param (str): Name of parameter to plot (default: 'alpha_cmb').
        bins (int): Number of histogram bins.
        outpath (str): Path to save the plot.
    """
    fig, ax = plt.subplots(figsize=(8, 5))

    # --- Plot simulations ---
    sim_files = sorted(glob.glob(os.path.join(sim_folder, "*.txt")))
    sim_vals_all = []

    for f in sim_files:
        try:
            with open(f, 'r') as file:
                header = file.readline().strip().replace('#', '').split()
            df = pd.read_csv(f, delim_whitespace=True, comment='#', names=header)
            if param in df.columns:
                sim_vals = df[param].values
                sim_vals_all.append(sim_vals)
                ax.hist(sim_vals, bins=bins, color='gray', alpha=0.25, density=True)
        except Exception as e:
            print(f"Error loading {f}: {e}")

    n_sims = len(sim_vals_all)
    print(f"Loaded {n_sims} simulation chains.")

    # --- Plot real data ---
    with open(real_file, 'r') as f:
        header = f.readline().strip().replace('#', '').split()
    df_real = pd.read_csv(real_file, delim_whitespace=True, comment='#', names=header)

    if param not in df_real.columns:
        print(f"ERROR: '{param}' not found in real data file.")
        return

    real_vals = df_real[param].values
    ax.hist(real_vals, bins=bins, color='red', alpha=0.9, density=True, label="Real constraint")

    # --- Final touches ---
    ax.set_xlabel(r"$\beta_{\rm CMB}$", fontsize=13)
    ax.set_ylabel("Density", fontsize=13)
    ax.set_title(f"Histogram of β_CMB across {n_sims} simulations and real data", fontsize=14)
    ax.legend()
    ax.grid(True, linestyle="--", alpha=0.5)

    plt.tight_layout()
    plt.savefig(outpath)
    print(f"Saved plot to {outpath}")
    plt.close()

def main():
    parser = argparse.ArgumentParser(description="Plot histograms of β_CMB (alpha_cmb) from sim chains + real data.")
    parser.add_argument("--sim_folder", required=True, help="Path to folder with simulation chain .txt files.")
    parser.add_argument("--real_file", required=True, help="Path to the real chain file.")
    parser.add_argument("--outpath", default="beta_histograms.png", help="Output path for the saved plot.")
    parser.add_argument("--bins", type=int, default=50, help="Number of bins for the histograms.")

    args = parser.parse_args()

    plot_beta_histogram_stack(
        sim_folder=args.sim_folder,
        real_file=args.real_file,
        bins=args.bins,
        outpath=args.outpath
    )

if __name__ == "__main__":
    main()
