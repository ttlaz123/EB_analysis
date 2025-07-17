import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
import argparse
def plot_beta_histogram_stack(sim_folders, real_folder, filename="chain.npy", beta_key="beta_cmb", bins=30):
    """
    Plot smoothed histogram of beta values for multiple simulation folders
    and one real chain, with statistical summary in labels.
    """
    

    # Define cache file
    cache_file = "beta_hist_data.pkl"

    # Try loading from pickle if it exists
    if os.path.exists(cache_file):
        with open(cache_file, "rb") as f:
            sim_vals, real_vals = pickle.load(f)
    else:
        # If pickle doesn't exist, compute and save
        sim_vals = []
        for folder in sim_folders:
            path = os.path.join(folder, filename)
            if not os.path.exists(path): continue
            data = np.load(path, allow_pickle=True).item()
            if beta_key in data:
                beta = data[beta_key]
                if np.isscalar(beta): continue
                sim_vals.extend(beta.flatten())

        real_path = os.path.join(real_folder, filename)
        real_data = np.load(real_path, allow_pickle=True).item()
        real_vals = real_data[beta_key].flatten()

        # Save to pickle
        with open(cache_file, "wb") as f:
            pickle.dump((sim_vals, real_vals), f)

    # Convert to numpy
    sim_vals = np.array(sim_vals)
    real_vals = np.array(real_vals)

    # KDE smoothing
    kde_sim = gaussian_kde(sim_vals)
    kde_real = gaussian_kde(real_vals)
    x = np.linspace(min(sim_vals.min(), real_vals.min()) - 0.01,
                    max(sim_vals.max(), real_vals.max()) + 0.01, 1000)

    # Compute stats
    sim_peak = x[np.argmax(kde_sim(x))]
    sim_std = np.std([x[np.argmax(gaussian_kde(sim_vals[i::5])(x))] for i in range(5)])  # rough peak spread
    real_peak = x[np.argmax(kde_real(x))]
    real_std = np.std(real_vals)

    # Plot
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(x, kde_sim(x), color='gray', alpha=0.5, label=f"Simulations\nmean peak={sim_peak:.3g}±{sim_std:.2g}")
    ax.plot(x, kde_real(x), color='red', linewidth=2,
            label=f"Real constraint: β={real_peak:.3g}±{real_std:.2g}")
    ax.axvline(0, color='k', linestyle='--', label="β = 0")

    ax.set_xlabel(r"$\beta_{\mathrm{cmb}}$")
    ax.set_ylabel("Density")
    ax.legend()
    ax.grid(True)
    plt.tight_layout()
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
