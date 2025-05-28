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
    """Process all subdirectories and create summary CSVs with all chains."""
    for subdir, _, files in os.walk(base_dir):
        out_path = os.path.join(subdir, os.path.basename(subdir) + "_summary.csv")
        if os.path.exists(out_path):
            print(f"Skipping {subdir}, summary CSV already exists: {out_path}")
            continue
        chain_files = sorted(f for f in files if f.endswith('.txt'))
        if not chain_files:
            continue

        print(f"Processing chains in: {subdir}")
        
        summaries = []
        count = 0
        for f in chain_files:
            full_path = os.path.join(subdir, f)
            root = os.path.splitext(full_path)[0].rsplit('.', 1)[0]  # removes .1/.2/.txt
            
            count += 1
            try:
                if(count % 10 == 0):
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
