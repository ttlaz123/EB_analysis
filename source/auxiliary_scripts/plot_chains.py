import os
import re
import argparse
from typing import List, Dict
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from getdist import loadMCSamples, plots
from matplotlib.lines import Line2D
model_config = {
    "BK18lf_nob_bin2-15_all":       ("with_fg", 1, "BK18 EE+EB with foregrounds",         "#57b9ff", 1),
    "BK18lf_all_bin2-15_all":       ("with_fg", 2, "BK18 EE+EB+BB with foregrounds",       "#1ea1ff", 1),
    "BK18lf_eb_bin2-15_gdust":      ("with_fg", 0, "BK18 EB with foregrounds","#84ccff", 1),
    "BK18lf_eb_bin2-15_fixed_dust": ("no_fg",   0, "BK18 EB no foregrounds",               "#ff7f0e", 3),
    "eskilt_only":                  ("eskilt",  0, "Eskilt 2023",                           "#2ca02c", 3),
    "eskilt_BK18lf":                ("combined",0, "Eskilt 2023 + BK18 EB no foregrounds", "#d62728", 5),
    "BK18lf_alens_bin2-15_all":     ("with_fg", 3, "BK18 EE+EB+scaled BB with foregrounds", "#1375bc", 1)
}


def find_chain_dirs(base_dir: str) -> List[str]:
    return [
        d for d in os.listdir(base_dir)
        if os.path.isdir(os.path.join(base_dir, d))
    ]


def get_samples(chain_path: str):
    try:
        return loadMCSamples(chain_path)
    except Exception as e:
        print(f"Warning: Failed to load {chain_path}: {e}")
        return None


def filter_params(samples, exclude_prefixes=("alpha", "beta",'chi2')) -> List[str]:
    print(samples.paramNames.list())

    return [
        p for p in samples.paramNames.list()
        if not any(p.startswith(prefix) for prefix in exclude_prefixes)
    ]

def extract_fede_tag(name: str) -> str:
    match = re.search(r'fede0\.?\d+', name)
    return match.group(0) if match else "no_fede_tag"


def group_samples_by_fede(chain_dirs: List[str], base_dir: str) -> Dict[str, List]:
    fede_groups = {}

    for chain_dir in chain_dirs:
        chain_file = os.path.join(base_dir, chain_dir, "real")
        samples = get_samples(chain_file)
        if not samples:
            continue

        fede_key = extract_fede_tag(chain_dir)
        if fede_key not in fede_groups:
            fede_groups[fede_key] = []
        fede_groups[fede_key].append((samples, chain_dir))
    for key in fede_groups:
        print(key)
        for x in fede_groups[key]:
            print(x[1])
    return fede_groups


def plot_grouped_posteriors(fede_groups: Dict[str, List], output_dir: str):
    os.makedirs(output_dir, exist_ok=True)

    group_order = {"with_fg": 0, "no_fg": 1, "eskilt": 2, "combined": 3}

    for fede_key, entries in fede_groups.items():
        print('Plotting:', fede_key)
        if fede_key == 'no_fede_tag':
            print('Skipping:', fede_key)
            continue

        plot_data = []
        param_name = "gMpl"

        for samples, dir_name in entries:
            match = None
            for key in model_config:
                if key in dir_name:
                    match = key
                    break
            if not match:
                continue

            group, subgroup_priority, label, color, lw = model_config[match]
            mean = samples.mean(param_name)
            std = samples.std(param_name)
            full_label = f"{label}: {mean:.2f} ± {std:.2f}"
            plot_data.append(((group_order[group], subgroup_priority), samples, full_label, color, lw))

        # Sort by (group group_priority, subgroup_priority)
        plot_data.sort(key=lambda x: x[0])

        g = plots.getSubplotPlotter(width_inch=10)
        g.settings.num_plot_contours = 1
        g.settings.alpha_filled_add = 0.4

        legend_labels = []
        for (_, samples, label, color, lw) in plot_data:
            g.plot_1d(samples, param_name)
            # Fix color and line width manually
            line = g.subplots[0, 0].get_lines()[-1]
            line.set_color(color)
            line.set_linewidth(lw)
            legend_labels.append(label)

        # Axis range and vertical line at 0
        g.subplots[0, 0].set_xlim(-1.5, 1.5)
        g.subplots[0, 0].axvline(0, color='gray', linestyle='--', linewidth=1)
        g.subplots[0, 0].set_xlabel(r"$g / M_\mathrm{pl}^{-1}$", fontsize=12)
        

        custom_lines = [
            Line2D([0], [0], color=color, lw=lw)
            for (_, _, _, color, lw) in plot_data
        ]
        g.subplots[0, 0].legend(custom_lines, legend_labels, loc='upper right', fontsize=10)

        filename = f"{fede_key}.png"
        g.export(os.path.join(output_dir, filename))



def plot_each_chain_separately(chain_dirs: List[str], base_dir: str, output_dir: str):
    os.makedirs(output_dir, exist_ok=True)

    for chain_dir in chain_dirs:
        chain_file = os.path.join(base_dir, chain_dir, "real")
        samples = get_samples(chain_file)
        if not samples:
            continue
        try:
            param_names = filter_params(samples)
            param_name = param_names[0]
        except IndexError:
            continue

        stats = samples.getInlineLatex(param_name, limit=1)
        g = plots.getSubplotPlotter(width_inch=10)
        g.settings.num_plot_contours = 1
        g.settings.alpha_filled_add = 0.4

        # Plot and get axis
        g.plot_1d(samples, param_name, label=stats, shaded=True)
        
        # Annotate mean ± std on the plot
        mean = samples.mean(param_name)
        sigma = samples.std(param_name)
        text = f"{mean:.3f} ± {sigma:.3f}"
        ax = plt.gca()
        ax.text(
            0.98, 0.85, text,
            transform=ax.transAxes,
            fontsize=10, color='black',
            ha='right', va='top',
            bbox=dict(boxstyle="round,pad=0.2", fc="white", alpha=0.7)
        )

        g.add_legend([chain_dir])
        filename = f"{chain_dir}.png"
        g.export(os.path.join(output_dir, filename))

def plot_trace_for_param(chain_dir: str, param_name: str, param_index: int, output_dir: str):
    import numpy as np
    import matplotlib.pyplot as plt
    chains = []
    for i in range(1, 100):  # Look for real.1.txt, real.2.txt, etc.
        chain_file = os.path.join(chain_dir, f"real.{i}.txt")
        if not os.path.exists(chain_file):
            break
        try:
            data = np.loadtxt(chain_file)
            chains.append(data[:, param_index])
        except Exception as e:
            print(f"Failed to read {chain_file}: {e}")
            continue

    if not chains:
        print(f"No chains found in {chain_dir}")
        return

    plt.figure(figsize=(10, 6))
    for i, chain in enumerate(chains):
        plt.plot(chain, label=f"Chain {i+1}", alpha=0.7)
    plt.xlabel("Step")
    plt.ylabel(param_name)
    plt.title(f"Trace plot for {param_name} in {os.path.basename(chain_dir)}")
    plt.legend()
    plt.tight_layout()

    os.makedirs(output_dir, exist_ok=True)
    out_file = os.path.join(output_dir, f"{os.path.basename(chain_dir)}_trace_{param_name}.png")
    plt.savefig(out_file)
    plt.close()

def main():
    parser = argparse.ArgumentParser(description="Plot Cobaya MCMC chains.")
    parser.add_argument('--base_dir', required=True, help='Path to base chain directory')
    parser.add_argument('--output_dir', default='plots', help='Directory to save plots')
    parser.add_argument('--group_by_fede', action='store_true', help='Group chains by fede value')
    parser.add_argument('--plot_traces', action='store_true', help='Generate trace plots to inspect burn-in')

    args = parser.parse_args()

    chain_dirs = find_chain_dirs(args.base_dir)
    '''
    if args.group_by_fede:
        fede_groups = group_samples_by_fede(chain_dirs, args.base_dir)
        plot_grouped_posteriors(fede_groups, args.output_dir)
    else:
        plot_each_chain_separately(chain_dirs, args.base_dir, args.output_dir)
    '''
    if args.plot_traces:
        param_name = "gMpl"
        for chain_dir in chain_dirs:
            chain_path = os.path.join(args.base_dir, chain_dir)
            samples = get_samples(os.path.join(chain_path, "real"))
            if not samples:
                continue
            try:
                param_names = samples.paramNames.list()
                print(param_names)
                if param_name not in param_names:
                    print(f"{param_name} not found in {chain_dir}")
                    continue
                param_index = param_names.index(param_name)
            except Exception as e:
                print(f"Error finding param index in {chain_dir}: {e}")
                continue

            trace_output_dir = os.path.join(args.output_dir, "traces")
            plot_trace_for_param(chain_path, param_name, param_index, trace_output_dir)

if __name__ == "__main__":
    main()

