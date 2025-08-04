import os
import glob
import argparse
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from scipy.stats import gaussian_kde

def load_sim_and_real_chains(sim_folder, real_file, param="alpha_CMB", cache_file="cached_data.npz"):
    """
    Load and optionally cache simulation and real MCMC chains.
    
    Returns:
        sim_vals_all (list of np.ndarray): List of simulation samples.
        sim_peaks (list of float): Peak values (means) of each simulation.
        real_vals (np.ndarray): Values from the real data chain.
    """
    if os.path.exists(cache_file):
        print(f"Loading cached data from {cache_file}")
        cached = np.load(cache_file, allow_pickle=True)
        return cached["sim_vals_all"], cached["sim_peaks"].tolist(), cached["real_vals"]

    sim_files = sorted(glob.glob(os.path.join(sim_folder, "*.txt")))
    sim_vals_all = []
    sim_peaks = []

    with open(real_file, 'r') as f:
        header = f.readline().strip().replace('#', '').split()
    df_real = pd.read_csv(real_file, delim_whitespace=True, comment='#', names=header)
    if param not in df_real.columns:
        raise ValueError(f"ERROR: '{param}' not found in real data file.")
    real_vals = df_real[param].values

    for i, fpath in enumerate(sim_files):
        if i % 10 == 0:
            print(f"Loading: {fpath}")
        try:
            with open(fpath, 'r') as f:
                header = f.readline().strip().replace('#', '').split()
            df = pd.read_csv(fpath, delim_whitespace=True, comment='#', names=header)
            if param in df.columns:
                vals = df[param].values
                sim_vals_all.append(vals)
                sim_peaks.append(np.mean(vals))
        except Exception as e:
            print(f"Error loading {fpath}: {e}")

    # Save for reuse
    np.savez_compressed(cache_file,
                        sim_vals_all=np.array(sim_vals_all, dtype=object),
                        sim_peaks=np.array(sim_peaks),
                        real_vals=real_vals)
    print(f"Saved cached data to {cache_file}")
    return sim_vals_all, sim_peaks, real_vals




def plot_beta_histogram_stack(sim_folder, real_file, param="alpha_CMB", bins=50, outpath="beta_histograms.png", cache_file='cache_file.npz'):
    """
    Plot 1D histograms of `alpha_cmb` (renamed to `beta_cmb`) from MCMC chains:
      - All sim chains in `sim_folder`: gray, semi-transparent
      - Real constraint from `real_file`: red and bold
    """
    sim_vals_all, sim_peaks, real_vals = load_sim_and_real_chains(sim_folder, real_file, param, cache_file=cache_file)

    plt.rcParams.update({
        "text.usetex": True,
        "font.size": 24,
        "font.family": "serif",
        "font.serif": ["Computer Modern"],
        "axes.labelsize": 24,
        "xtick.labelsize": 18,
        "ytick.labelsize": 18,
        "legend.fontsize": 18,
    })

    fig, ax = plt.subplots(figsize=(8, 5))

    # Plot simulation KDEs
    count = 0
    for sim_vals in sim_vals_all:
        count +=1
        if(count %10==0):
            print("Plotting: " + str(count))
        kde = gaussian_kde(sim_vals)
        x_vals = np.linspace(min(sim_vals), max(sim_vals), 200)
        ax.plot(x_vals, kde(x_vals), color='gray', alpha=0.25, linewidth=1.0)

    # Plot real data KDE
    kde_real = gaussian_kde(real_vals)
    x_real = np.linspace(min(real_vals), max(real_vals), 200)
    ax.plot(x_real, kde_real(x_real), color='red', linewidth=2.0,
            label=f"Real: peak = {np.mean(real_vals):.4g}, std = {np.std(real_vals):.4g}")

    # Vertical line at beta = 0
    ax.axvline(0.0, color='black', linestyle='--', linewidth=1.2, label=r"$\beta = 0$")

    # Add invisible line for sim stats
    if sim_peaks:
        sim_mean = np.mean(sim_peaks)
        sim_std = np.std(sim_peaks)
        ax.plot([], [], color='gray', alpha=0.25, linewidth=1.0,
                label=f"Sims: mean peak = {sim_mean:.4g}, spread = {sim_std:.4g}")

    # Final touches
    ax.set_xlabel(r"$\beta_{\rm CMB}$")
    ax.set_ylabel("Density")
    #ax.set_title(f"Histogram of β_CMB across {len(sim_vals_all)} simulations and real data")
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
    parser.add_argument("--cache_file", required=True, help="Path to cache file.")
    parser.add_argument("--outpath", default="beta_histograms.png", help="Output path for the saved plot.")
    parser.add_argument("--bins", type=int, default=50, help="Number of bins for the histograms.")

    args = parser.parse_args()

    plot_beta_histogram_stack(
        sim_folder=args.sim_folder,
        real_file=args.real_file,
        bins=args.bins,
        outpath=args.outpath,
        cache_file=args.cache_file
    )

if __name__ == "__main__":
    main()
