import os
import re
import argparse
from typing import List, Dict
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from getdist import loadMCSamples, plots

model_config = {
    "BK18lf_nob_bin2-15_all":       ("with_fg", 0, "BK18 EE+EB with foregrounds",         "#1f77b4", 1.5),
    "BK18lf_all_bin2-15_all":       ("with_fg", 1, "BK18 EE+EB+BB with foregrounds",       "#1f77b4", 1.5),
    "BK18lf_eb_bin2-15_gdust":      ("with_fg", 2, "BK18 EE+EB+scaled BB with foregrounds","#1f77b4", 1.5),
    "BK18lf_eb_bin2-15_fixed_dust": ("no_fg",   0, "BK18 EB no foregrounds",               "#ff7f0e", 1.5),
    "eskilt_only":                  ("eskilt",  0, "Eskilt 2023",                           "#2ca02c", 1.5),
    "eskilt_BK18lf":                ("combined",0, "Eskilt 2023 + BK18 EB no foregrounds", "#d62728", 3.0),
    "BK18lf_alens_bin2-15_all":     ("with_fg", 3, "BK18 EE+EB+BB alens",                  "#1f77b4", 1.5)
}


def find_chain_dirs(base_dir: str) -> List[str]:
    return [
        d for d in os.listdir(base_dir)
        if os.path.isdir(os.path.join(base_dir, d))
    ]


def get_samples(chain_path: str):
    try:
        return loadMCSamples(chain_path, settings={"ignore_rows": 0.1})
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
        g.subplots[0, 0].set_xlim(-1, 1)
        g.subplots[0, 0].axvline(0, color='gray', linestyle='--', linewidth=1)

        g.add_legend(legend_labels=legend_labels)
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


def main():
    parser = argparse.ArgumentParser(description="Plot Cobaya MCMC chains.")
    parser.add_argument('--base_dir', required=True, help='Path to base chain directory')
    parser.add_argument('--output_dir', default='plots', help='Directory to save plots')
    parser.add_argument('--group_by_fede', action='store_true', help='Group chains by fede value')

    args = parser.parse_args()

    chain_dirs = find_chain_dirs(args.base_dir)

    if args.group_by_fede:
        fede_groups = group_samples_by_fede(chain_dirs, args.base_dir)
        plot_grouped_posteriors(fede_groups, args.output_dir)
    else:
        plot_each_chain_separately(chain_dirs, args.base_dir, args.output_dir)


if __name__ == "__main__":
    main()

