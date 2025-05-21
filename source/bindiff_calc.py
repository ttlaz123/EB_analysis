import os
import argparse
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from getdist.mcsamples import loadMCSamples
import logging
logging.getLogger().setLevel(logging.ERROR)

def get_common_group_label(dir1, dir2):
    # Find longest common prefix of the two absolute paths
    abs1 = os.path.abspath(dir1)
    abs2 = os.path.abspath(dir2)
    common_prefix = os.path.commonprefix([abs1, abs2])

    # commonprefix can cut inside folder names, so fix by going up to last os.sep
    if not common_prefix.endswith(os.sep):
        common_prefix = common_prefix.rsplit(os.sep, 1)[0]

    # Use the last folder name of this prefix as group label
    group_label = os.path.basename(common_prefix)
    if not group_label:
        # fallback if empty, e.g. root folder or something odd
        group_label = "default_group"
    return group_label

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
    return abs(mu1 - mu2) / denom

def collect_all_zscores(bin2_8_root, bin9_15_root, params, num_sims):
    """
    Load both sims once per simulation, extract all params, compute z-scores.
    Returns dict param -> list of z-scores.
    """
    zscores = {p: [] for p in params}

    for i in range(num_sims):
        sim_folder_1 = os.path.join(bin2_8_root, f"sim{i:03d}")
        sim_folder_2 = os.path.join(bin9_15_root, f"sim{i:03d}")
        try:
            samples1 = loadMCSamples(sim_folder_1)
            samples2 = loadMCSamples(sim_folder_2)

            means1 = samples1.getMeans(params)
            stds1 = samples1.std(params)
            means2 = samples2.getMeans(params)
            stds2 = samples2.std(params)

            for p in params:
                mu1, std1 = means1[p], stds1[p]
                mu2, std2 = means2[p], stds2[p]
                z = compute_z_score(mu1, std1, mu2, std2)
                if not np.isnan(z):
                    zscores[p].append(z)

        except Exception as e:
            print(f"[Warning] Skipping sim{i:03d} due to error: {e}")
            continue

    for p in params:
        zscores[p] = np.array(zscores[p])
    return zscores

def plot_z_histogram(zscores, param, outdir, group_label):
    plt.figure(figsize=(6,4))
    plt.hist(zscores, bins=30, color='skyblue', edgecolor='black')
    plt.xlabel(f"Z-score for {param}")
    plt.ylabel("Count")
    plt.title(f"Z-score Distribution: {param}\nGroup: {group_label}")
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

    group_label = get_common_group_label(args.bin2_8_dir, args.bin9_15_dir)
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