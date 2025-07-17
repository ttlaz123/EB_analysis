import os
import glob
import argparse
import matplotlib 
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from scipy.stats import gaussian_kde

def plot_beta_histogram_stack(real_vals, sim_vals, outpath=None, param_label='β'):
    """
    Plot smoothed histograms (KDE) of real and simulation β values.
    
    Parameters:
    - real_vals: array of β values from real data
    - sim_vals: 2D array where each row contains β values from a simulation
    - outpath: if provided, path to save the figure
    - param_label: label to use for x-axis
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    # --- Common bin range for all ---
    all_vals = np.concatenate([real_vals] + [s for s in sim_vals])
    x_grid = np.linspace(all_vals.min() - 0.1, all_vals.max() + 0.1, 1000)

    # --- Real data KDE ---
    kde_real = gaussian_kde(real_vals)
    real_density = kde_real(x_grid)
    peak_real = real_vals[np.argmax(kde_real(real_vals))]
    spread_real = np.std(real_vals)

    ax.plot(x_grid, real_density, label=f"Real: peak={peak_real:.3g}, σ={spread_real:.3g}",
            color='black', lw=2)

    # --- Simulations KDE ---
    sim_peaks = []
    for sim in sim_vals:
        kde_sim = gaussian_kde(sim)
        density_sim = kde_sim(x_grid)
        peak_sim = sim[np.argmax(kde_sim(sim))]
        sim_peaks.append(peak_sim)
        ax.plot(x_grid, density_sim, alpha=0.2, color='C0')

    # --- Mean and std of peaks ---
    mean_peak = np.mean(sim_peaks)
    std_peak = np.std(sim_peaks)
    ax.plot([], [], label=f"Sims: ⟨peak⟩={mean_peak:.3g}, σ(peak)={std_peak:.3g}", color='C0')

    # --- Add vertical line at beta=0 ---
    ax.axvline(0, color='gray', linestyle='--', lw=1, label='β = 0')

    # --- Final plot settings ---
    ax.set_xlabel(param_label)
    ax.set_ylabel("Density")
    ax.set_xlim(x_grid.min(), x_grid.max())
    ax.set_title(f"Histogram of {param_label}")
    ax.legend()

    if outpath:
        plt.savefig(outpath, dpi=150)
    plt.show()

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
