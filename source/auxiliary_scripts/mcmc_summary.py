import os
import glob
import pandas as pd
from getdist import loadMCSamples

def summarize_chain_file(chain_file):
    base_name = os.path.splitext(chain_file)[0]
    samples = loadMCSamples(file_root=base_name)

    stats = samples.getMargeStats()
    param_means = stats.means
    param_stddevs = stats.std_devs
    param_names = stats.parNames.names

    summary = {
        "chain_file": os.path.basename(chain_file),
    }
    for name, mean, std in zip(param_names, param_means, param_stddevs):
        summary[f"{name}_mean"] = mean
        summary[f"{name}_std"] = std

    return summary

def summarize_all_chains_in_dir(parent_dir):
    for subdir, dirs, files in os.walk(parent_dir):
        chain_files = sorted(glob.glob(os.path.join(subdir, '*.txt')))
        if not chain_files:
            continue

        print(f"Processing chains in: {subdir}")
        try:
            summaries = []
            for chain_file in chain_files:
                summary = summarize_chain_file(chain_file)
                summaries.append(summary)

            summary_df = pd.DataFrame(summaries)
            summary_csv_path = os.path.join(subdir, os.path.basename(subdir) + "_summary.csv")
            summary_df.to_csv(summary_csv_path, index=False)
            print(f"Saved summary to: {summary_csv_path}")
        except Exception as e:
            print(f"Failed to process {subdir}: {e}")

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Summarize MCMC chains using GetDist.")
    parser.add_argument("parent_dir", help="Top-level directory containing subdirs of MCMC chains.")
    args = parser.parse_args()

    summarize_all_chains_in_dir(args.parent_dir)
