import os
import argparse
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from getdist.mcsamples import loadMCSamples
import logging
import pandas as pd
from scipy.stats import norm
logging.getLogger().setLevel(logging.ERROR)

def get_group_label_from_paths(dir1, dir2):
    label1 = os.path.basename(os.path.normpath(dir1))
    label2 = os.path.basename(os.path.normpath(dir2))
    return f"{label1}_{label2}"

def detect_alpha_params(sim_folder):
    """
    Load one sim, return all parameter names starting with 'alpha_'.
    """
    root = os.path.join(sim_folder, "sim001")
    samples = loadMCSamples(root)
    all_names = [p.name for p in samples.getParamNames().names]
    alpha_params = [p for p in all_names if p.startswith("alpha_")]
    return alpha_params


def compute_z_score(mu1, std1, mu2, std2):
    denom = np.sqrt(std1**2 + std2**2)
    if denom == 0:
        return np.nan
    return (mu1 - mu2) / denom

def collect_all_zscores(bin2_8_root, bin9_15_root, params, num_sims):
    """
    Load both sims once per simulation, extract all params, compute z-scores.
    Returns dict param -> list of z-scores.
    """
    zscores = {p: [] for p in params}
    summary_csv_1 = os.path.join(bin2_8_root, os.path.basename(os.path.normpath(bin2_8_root) + "_summary.csv"))
    summary_csv_2 = os.path.join(bin9_15_root, os.path.basename(os.path.normpath(bin9_15_root) + "_summary.csv"))

    use_summary_1 = os.path.exists(summary_csv_1)
    use_summary_2 = os.path.exists(summary_csv_2)
    print('Summary files:' + summary_csv_1)
    print('Summary files:' + summary_csv_2)
    if use_summary_1 and use_summary_2:
        print(f"Loading summary CSVs:\n  {summary_csv_1}\n  {summary_csv_2}")
        df1 = pd.read_csv(summary_csv_1)
        df2 = pd.read_csv(summary_csv_2)

        # Assumes summary CSV has columns named like 'param_mean' and 'param_std'
        for i in range(min(len(df1), len(df2))):
            for p in params:
                mean_col = f"{p}_mean"
                std_col = f"{p}_std"
                try:
                    mu1 = df1.at[i, mean_col]
                    std1 = df1.at[i, std_col]
                    mu2 = df2.at[i, mean_col]
                    std2 = df2.at[i, std_col]

                    z = compute_z_score(mu1, std1, mu2, std2)
                    if not np.isnan(z):
                        zscores[p].append(z)
                except KeyError as e:
                    print(f"[Warning] Missing expected column {e} in summary CSV.")
                    continue
    else:
        # Fallback to loading individual sim folders
        for i in range(num_sims):
            sim_folder_1 = os.path.join(bin2_8_root, f"sim{i:03d}")
            sim_folder_2 = os.path.join(bin9_15_root, f"sim{i:03d}")
            if i % 10 == 0:
                print("Loading: " + sim_folder_1)
                print("Loading: " + sim_folder_2)
            try:
                samples1 = loadMCSamples(sim_folder_1)
                samples2 = loadMCSamples(sim_folder_2)

                for p in params:
                    mu1 = samples1.mean(p)
                    std1 = samples1.std(p)
                    mu2 = samples2.mean(p)
                    std2 = samples2.std(p)
                    z = compute_z_score(mu1, std1, mu2, std2)
                    if not np.isnan(z):
                        zscores[p].append(z)

            except Exception as e:
                print(f"[Warning] Skipping sim{i:03d} due to error: {e}")
                continue

    # Convert lists to numpy arrays
    for p in params:
        zscores[p] = np.array(zscores[p])

    return zscores

def plot_z_histogram(zscores, param, outdir, group_label):
    plt.figure(figsize=(6,4))
    bins = np.linspace(-4, 4, 30)
    bin_width = bins[1] - bins[0]
    counts, bins, patches =plt.hist(zscores, bins=bins, color='skyblue', edgecolor='black')
    (mu, sigma) = norm.fit(zscores)
    
    x = np.linspace(-4, 4, 100)
    gaussian_pdf = norm.pdf(x, mu, sigma)
    scaled_pdf = gaussian_pdf * bin_width * sum(counts)
    plt.plot(x, scaled_pdf, 'r-', lw=2, label=f'Gaussian fit\nMean={mu:.2f}\nStd={sigma:.2f}')
    
    
    plt.xlabel(f"Z-score for {param}")
    plt.ylabel("Count")
    plt.title(f"Z-score Distribution: {param}\nGroup: {group_label}")
    plt.legend()
    plt.tight_layout()
    os.makedirs(outdir, exist_ok=True)
    safe_label = group_label.replace(" ", "_").replace("/", "_")
    outpath = os.path.join(outdir, f"zscore_hist_{param}_{safe_label}.png")
    plt.savefig(outpath, dpi=200)
    plt.close()
    print(f"[Saved] {outpath}")

def main():
    parser = argparse.ArgumentParser(description="Compute and plot z-score histograms for alpha_ parameters.")
    parser.add_argument("bin2_8_dir", help="Directory for bin 2-8 simulations")
    parser.add_argument("bin9_15_dir", help="Directory for bin 9-15 simulations")
    parser.add_argument("--num_sims", type=int, default=500, help="Number of simulations")
    parser.add_argument("--outdir", default="zscore_histograms", help="Output directory for histograms")
    # Remove group_label argument
    args = parser.parse_args()

    group_label = get_group_label_from_paths(args.bin2_8_dir, args.bin9_15_dir)
    print(f"Auto-detected group label: {group_label}")

    alpha_params = detect_alpha_params(args.bin2_8_dir)
    print(f"Detected alpha parameters: {alpha_params}")

    zscores = collect_all_zscores(args.bin2_8_dir, args.bin9_15_dir, alpha_params, args.num_sims)

    for param, zs in zscores.items():
        if len(zs) > 0:
            plot_z_histogram(zs, param, args.outdir, group_label)
        else:
            print(f"[Skipped] No valid z-scores for {param}")

if __name__ == "__main__":
    main()