import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
from getdist.mcsamples import loadMCSamples

def detect_alpha_params(sim_path):
    """
    Detects all parameters starting with 'alpha_' from a GetDist chain.
    """
    samples = loadMCSamples(sim_path, no_cache=True)
    return [p.name for p in samples.getParamNames().names if p.name.startswith('alpha_')]

def load_mean_std(path, param):
    """
    Loads the mean and std of a specific parameter from a GetDist MCMC chain.
    """
    samples = loadMCSamples(path, no_cache=True)
    return samples.mean(param), samples.std(param)

def compute_z_score(mu1, std1, mu2, std2):
    """
    Computes Z-score of mean difference between two distributions.
    """
    return abs(mu1 - mu2) / np.sqrt(std1**2 + std2**2)

def collect_z_scores(bin2_8_root, bin9_15_root, param, num_sims):
    """
    Collects Z-scores for one parameter across multiple simulations.
    """
    z_scores = []

    for i in range(num_sims):
        try:
            sim_path_1 = os.path.join(bin2_8_root, f"sim{i:03d}")
            sim_path_2 = os.path.join(bin9_15_root, f"sim{i:03d}")

            mu1, std1 = load_mean_std(sim_path_1, param)
            mu2, std2 = load_mean_std(sim_path_2, param)

            z = compute_z_score(mu1, std1, mu2, std2)
            z_scores.append(z)

        except Exception as e:
            print(f"[Warning] Skipping sim {i:03d} for {param}: {e}")
            continue

    return np.array(z_scores)

def plot_z_histogram(z_scores, param, outdir):
    """
    Plots histogram of Z-scores for a given parameter.
    """
    plt.figure(figsize=(6, 4))
    plt.hist(z_scores, bins=30, color='skyblue', edgecolor='black')
    plt.xlabel(f"Z-score for {param}")
    plt.ylabel("Count")
    plt.title(f"Z-score Distribution: {param}")
    plt.tight_layout()

    os.makedirs(outdir, exist_ok=True)
    outpath = os.path.join(outdir, f"zscore_hist_{param}.png")
    plt.savefig(outpath, dpi=200)
    plt.close()
    print(f"[Saved] {outpath}")

def main():
    parser = argparse.ArgumentParser(description="Compare alpha_ posteriors between bin 2-8 and 9-15 via Z-score histograms.")
    parser.add_argument("bin2_8_dir", help="Directory for bin 2-8 simulations (sim000, sim001, ...)")
    parser.add_argument("bin9_15_dir", help="Directory for bin 9-15 simulations (sim000, sim001, ...)")
    parser.add_argument("--num_sims", type=int, default=500, help="Number of simulations to check")
    parser.add_argument("--outdir", default="zscore_histograms", help="Directory to save histogram plots")

    args = parser.parse_args()

    # Detect parameters starting with alpha_ from first sim
    sim0_path = os.path.join(args.bin2_8_dir, "sim000")
    alpha_params = detect_alpha_params(sim0_path)

    for param in alpha_params:
        z_scores = collect_z_scores(args.bin2_8_dir, args.bin9_15_dir, param, args.num_sims)
        if len(z_scores) > 0:
            plot_z_histogram(z_scores, param, args.outdir)
        else:
            print(f"[Skipped] No Z-scores computed for {param}")

if __name__ == "__main__":
    main()
