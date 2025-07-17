import os
import glob
import argparse
import matplotlib 
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from scipy.stats import gaussian_kde

def plot_beta_histogram_stack(sim_folder, real_file, param="alpha_CMB", bins=50, outpath="beta_histograms.png"):
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

    sim_files = sorted(glob.glob(os.path.join(sim_folder, "*.txt")))
    sim_vals_all = []
    sim_peaks = []
    count = 0

    with open(real_file, 'r') as f:
        header = f.readline().strip().replace('#', '').split()
    df_real = pd.read_csv(real_file, delim_whitespace=True, comment='#', names=header)
    real_vals = df_real[param].values
    if param not in df_real.columns:
        print(f"ERROR: '{param}' not found in real data file.")
        return

    # Load and KDE-smooth simulation chains
    for f in sim_files:
        count += 1
        if count % 10 == 0:
            print('Loading: ' + f)
        try:
            with open(f, 'r') as file:
                header = file.readline().strip().replace('#', '').split()
            df = pd.read_csv(f, delim_whitespace=True, comment='#', names=header)
            if param in df.columns:
                sim_vals = df[param].values
                sim_vals_all.append(sim_vals)

                kde = gaussian_kde(sim_vals)
                x_vals = np.linspace(min(sim_vals), max(sim_vals), 200)
                ax.plot(x_vals, kde(x_vals), color='gray', alpha=0.25, linewidth=1.0)

                # record peak of this sim chain
                peak = np.mean(sim_vals)
                sim_peaks.append(peak)
        except Exception as e:
            print(f"Error loading {f}: {e}")

    n_sims = len(sim_vals_all)
    print(f"Loaded {n_sims} simulation chains.")

    # KDE for real data
    kde_real = gaussian_kde(real_vals)
    x_real = np.linspace(min(real_vals), max(real_vals), 200)
    ax.plot(x_real, kde_real(x_real), color='red', linewidth=2.0,
            label=f"Real: peak = {np.mean(real_vals):.4g}, std = {np.std(real_vals):.4g}")

    # Vertical line at beta = 0
    ax.axvline(0.0, color='black', linestyle='--', linewidth=1.2, label=r"$\beta = 0$")

    # Add sim peaks mean and spread label (invisible plot just for legend)
    if sim_peaks:
        sim_mean = np.mean(sim_peaks)
        sim_std = np.std(sim_peaks)
        ax.plot([], [], color='gray', alpha=0.25, linewidth=1.0,
                label=f"Sims: mean peak = {sim_mean:.4g}, spread = {sim_std:.4g}")

    # Final touches
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
