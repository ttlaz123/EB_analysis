import os
import argparse
import pandas as pd
from getdist import loadMCSamples
def summarize_chain(root):
    """Return summary (mean, std) for all parameters in one chain."""
    samples = loadMCSamples(file_root=root)

    if samples is None or samples.paramNames is None:
        raise ValueError(f"Could not load MCMC samples from root: {root}")

    names = samples.getParamNames().names
    means = samples.mean(names)
    stds = samples.std(names)

    summary = {"chain_root": os.path.basename(root)}
    for name, mean, std in zip(names, means, stds):
        summary[f"{name.name}_mean"] = mean
        summary[f"{name.name}_std"] = std

    return summary

def process_directory(base_dir):
    """Walk through all subdirectories of base_dir and process each one."""
    for subdir, _, _ in os.walk(base_dir):
        out_path = os.path.join(subdir, os.path.basename(subdir) + "_summary.csv")
        if os.path.exists(out_path):
            print(f"Skipping {subdir}, summary CSV already exists: {out_path}")
            continue
        process_single_directory(subdir)

def process_single_directory(subdir):
    """Process a single directory and create a summary CSV from all chains inside it."""
    out_path = os.path.join(subdir, os.path.basename(subdir) + "_summary.csv")


    chain_files = sorted(f for f in os.listdir(subdir) if f.endswith('.txt'))
    if not chain_files:
        print(f"No .txt chain files found in: {subdir}")
        return

    print(f"Processing chains in: {subdir}")
    
    summaries = []
    for count, f in enumerate(chain_files, 1):
        full_path = os.path.join(subdir, f)
        root = os.path.splitext(full_path)[0].rsplit('.', 1)[0]  # removes .1/.2/.txt
        try:
            if count % 10 == 0:
                print(f"  → {root}")
            summary = summarize_chain(root)
            summaries.append(summary)
        except Exception as e:
            print(f"  !! Failed to process {root}: {e}")

    if summaries:
        summary_df = pd.DataFrame(summaries)
        summary_df.to_csv(out_path, index=False)
        print(f"Saved summary: {out_path}")
    else:
        print(f"No valid chains found in: {subdir}")
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Summarize all MCMC chains using getdist.")
    parser.add_argument("base_dir", type=str, help="Directory of chain subdirectories.")
    args = parser.parse_args()

    process_directory(args.base_dir)
