import os
import re
import argparse
from typing import List, Dict
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from getdist import loadMCSamples, plots, MCSamples
from matplotlib.lines import Line2D
import numpy as np
model_config = {
    "BK18lf_nob_bin2-15_all":       ("with_fg", 1, "BK18 EE+EB with foregrounds",         "#57b9ff", 1),
    "BK18lf_all_bin2-15_all":       ("with_fg", 2, "BK18 EE+EB+BB with foregrounds",       "#a020f0", 5),
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
    return loadMCSamples(chain_path, settings={"ignore_rows":100})
    


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
            if(group == 'combined'):
                print('Skipping: ' + label)
                continue
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
            print(label )
        # Axis range and vertical line at 0
        g.subplots[0, 0].set_xlim(-1.5, 1.5)
        g.subplots[0, 0].axvline(0, color='gray', linestyle='--', linewidth=1)
        g.subplots[0, 0].set_xlabel(r"$g / M_\mathrm{pl}^{-1}$", fontsize=12)
        

        custom_lines = [
            Line2D([0], [0], color=color, lw=lw)
            for (_, _, _, color, lw) in plot_data
        ]
        g.subplots[0, 0].legend(custom_lines, legend_labels, loc='upper left', fontsize=10)

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
    
    chains = []
    for i in range(1, 2):  # Look for real.1.txt, real.2.txt, etc.
        chain_file = os.path.join(chain_dir, f"real.{i}.txt")
        if not os.path.exists(chain_file):
            break
        try:
            data = np.loadtxt(chain_file)
            chains.append(data[:2000, param_index])
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
    
def extract_ldiff_tag(name: str) -> str:
    match = re.search(r'ldiff\d+', name)
    return match.group(0) if match else "no_ldiff_tag"
def get_ldiff_samples(chain_dirs, base_dir):
    ldiff_chains = []
    for chain_dir in chain_dirs:
        chain_file = os.path.join(base_dir, chain_dir, "real")
        ldiff_tag = extract_ldiff_tag(chain_dir)
        if(ldiff_tag == 'no_ldiff_tag'):
            continue 

        samples = get_samples(chain_file)
        if not samples:
            continue
        ldiff_chains.append((samples, chain_dir))
    print(ldiff_chains)
    return ldiff_chains


def plot_ldiff_posteriors(ldiff_chains, output_dir: str):
    os.makedirs(output_dir, exist_ok=True)   
    param_name = "angle_diff"

    # Mapping ldiff tag → (legend label, color)
    ldiff_map = {
        "ldiff8":  ("$\ell_b=265$", "#f30800"),
        "ldiff9":  ("$\ell_b=300$", "#ff7f0e"),
        "ldiff10": ("$\ell_b=335$", "#2ca02c"),
        "ldiff11": ("$\ell_b=370$", "#2727d6"),
        "ldiff12": ("$\ell_b=405$", "#9467bd"),
    }

    plot_data = []

    for samples, dir_name in ldiff_chains:
        ldiff_tag = extract_ldiff_tag(dir_name)
        if ldiff_tag not in ldiff_map:
            continue

        label, color = ldiff_map[ldiff_tag]
        mean = samples.mean(param_name)
        std = samples.std(param_name)
        full_label = f"{label}: {mean:.2f} ± {std:.2f}"
        # Store numeric ldiff for sorting, samples, label, and color
        plot_data.append((int(ldiff_tag.replace("ldiff", "")), samples, full_label, color))

    # Sort by numeric ldiff value
    plot_data.sort(key=lambda x: x[0])

    g = plots.getSubplotPlotter(width_inch=10)
    g.settings.num_plot_contours = 1
    g.settings.alpha_filled_add = 0.4

    legend_labels = []
    custom_lines = []

    for (_, samples, label, color) in plot_data:
        g.plot_1d(samples, param_name)
        line = g.subplots[0, 0].get_lines()[-1]
        line.set_color(color)
        line.set_linewidth(2.0)
        legend_labels.append(label)
        custom_lines.append(Line2D([0], [0], color=color, lw=2))

    ax = g.subplots[0, 0]
    ax.set_xlim(-1.5, 1.5)
    ax.axvline(0, color='gray', linestyle='--', linewidth=1)
    ax.set_xlabel(r"$\Delta\beta_{\ell_b}$ (degrees)", fontsize=14)
    ax.legend(custom_lines, legend_labels, loc='upper left', fontsize=10)

    filename = os.path.join(output_dir, "ldiff.png")
    print('Saving:', filename)
    g.export(filename)

    # Additional plot: mean ± std vs lb, using same colors as first plot
    lbs = []
    means = []
    stds = []
    colors = []

    for (ldiff_number, samples, _, color) in plot_data:
        lb_center = 230 + (ldiff_number - 7) * 35  # Adjust if needed
        lbs.append(lb_center)
        means.append(samples.mean(param_name))
        stds.append(samples.std(param_name))
        colors.append(color)

    fig, ax = plt.subplots(figsize=(8, 5))
    for lb, mean, std, color in zip(lbs, means, stds, colors):
        print('Color: ' + str(color))
        ax.errorbar(lb, mean, yerr=std, fmt='o', capsize=4, color=color, ecolor=color, label=f"$\ell_b={lb}$")

    ax.axhline(0, color='gray', linestyle='--', linewidth=1)
    ax.set_xlabel(r'Multipole breakpoint $\ell_b$', fontsize=14)
    ax.set_ylabel(r'$\Delta\beta$ (deg)', fontsize=14)
    ax.set_title('Rotation Difference', fontsize=14)
    ax.grid(True)

    # Optional: custom legend
    ax.legend(fontsize=10)

    out_path = os.path.join(output_dir, "ldiff_bandpower.png")
    print('Saving:', out_path)
    fig.savefig(out_path, bbox_inches='tight')
    plt.close(fig)



def plot_betacmb_posteriors(chain_dirs: List[str], base_dir: str, output_dir: str):
    os.makedirs(output_dir, exist_ok=True)
    param_name = "alpha_CMB"
    plot_data = []

    # Define custom labels and order
    label_priority = {
        "BK18lf_all_bin2-15_gdust_betacmb": "EE+EB+BB with foregrounds",
        "BK18lf_eb_bin2-15_gdust_betacmb": "EB with foregrounds",
        "BK18lf_eb_bin2-15_det_polrot_betacmb": "EB no foregrounds",
    }

    for chain_dir in chain_dirs:
        if "betacmb" not in chain_dir:
            continue
        print(chain_dir)
        chain_file = os.path.join(base_dir, chain_dir, "real")
        samples = get_samples(chain_file)
        if not samples:
            continue
        print(chain_dir)
        if param_name not in samples.paramNames.list():
            print(f"Skipping {chain_dir}: {param_name} not found")
            continue

        mean = samples.mean(param_name)
        std = samples.std(param_name)

        if chain_dir not in label_priority:
            print(f"Skipping unknown betacmb chain: {chain_dir}")
            continue

        label = f"{label_priority[chain_dir]}: {mean:.2f} ± {std:.2f}"
        plot_data.append((label_priority[chain_dir], samples, label))

    if not plot_data:
        print("No valid betacmb chains found.")
        return

    # Sort in desired order
    desired_order = ["EE+EB+BB with foregrounds", "EB with foregrounds", "EB no foregrounds"]
    plot_data.sort(key=lambda x: desired_order.index(x[0]))

    # Plot
    g = plots.getSubplotPlotter(width_inch=10)
    g.settings.num_plot_contours = 1
    g.settings.alpha_filled_add = 0.4

    legend_labels = []
    colors = plt.cm.viridis(np.linspace(0, 1, len(plot_data)))

    for i, (_, samples, label) in enumerate(plot_data):
        g.plot_1d(samples, param_name)
        line = g.subplots[0, 0].get_lines()[-1]
        line.set_color(colors[i])
        line.set_linewidth(2.0)
        legend_labels.append(label)

    ax = g.subplots[0, 0]
    ax.axvline(0, color='gray', linestyle='--', linewidth=1)
    ax.set_xlabel(r"$\beta_\mathrm{cmb}$", fontsize=14)
    ax.legend(legend_labels, loc='upper left', fontsize=10)

    out_path = os.path.join(output_dir, "betacmb.png")
    print(f"Saving: {out_path}")
    g.export(out_path)


def plot_marginalized_triangle(chain_path, param_names, 
                param_labels=None, outdir='sample_plots', weight_col='weight', burn_in=100):
    """
    Load an MCMC chain and plot a triangle plot for a subset of parameters using getdist.
    Displays mean ± std on the plot.
    """

    # Load the chain
    data = np.genfromtxt(chain_path, names=True)

    # Apply burn-in if requested
    if burn_in > 0:
        data = data[burn_in:]

    # Get weights if present
    weights = data[weight_col] if weight_col in data.dtype.names else None

    # Stack samples of selected parameters
    try:
        samples = np.vstack([data[param] for param in param_names]).T
    except KeyError as e:
        raise ValueError(f"Parameter not found in chain: {e}")

    # Use default labels if not provided
    if param_labels is None:
        param_labels = param_names

    # Create MCSamples object
    samples_obj = MCSamples(samples=samples, names=param_names, labels=param_labels, weights=weights)

    # Get mean ± std for title
    mean_std_strings = []
    for name, label in zip(param_names, param_labels):
        mean = samples_obj.mean(name)
        std = samples_obj.std(name)
        mean_std_strings.append(f"${label}$ = {mean:.3f} ± {std:.3f}")

    # Plot triangle
    g = plots.get_subplot_plotter()
    g.triangle_plot(samples_obj, filled=True)
    # Add text box in top right blank space
    textstr = "\n".join(mean_std_strings)

    # Coordinates are relative to the figure (0 to 1)
    plt.gcf().text(0.93, 0.75, textstr, fontsize=10, verticalalignment='top',
                   bbox=dict(boxstyle='round,pad=0.5', facecolor='white', alpha=0.8))
    plt.tight_layout()

    # Save and show
    os.makedirs(outdir, exist_ok=True)
    out_path = os.path.join(outdir, "isobeta_and_dust.png")
    print('Saving:', out_path)
    g.export(out_path)
    plt.show()


def main():
    parser = argparse.ArgumentParser(description="Plot Cobaya MCMC chains.")
    parser.add_argument('--base_dir', required=True, help='Path to base chain directory')
    parser.add_argument('--output_dir', default='plots', help='Directory to save plots')
    parser.add_argument('--group_by_fede', action='store_true', help='Group chains by fede value')
    parser.add_argument('--group_by_ldiff', action='store_true', help='Group chains by ldiff value')
    parser.add_argument('--plot_traces', action='store_true', help='Generate trace plots to inspect burn-in')
    parser.add_argument('--group_by_cmb', action='store_true', help='Group chains by cmb value')
    parser.add_argument('--isodust', action='store_true', help='only plotting dust and iso angle')
    
    args = parser.parse_args()
    if(args.isodust):
        param_names = ['alpha_CMB', 'A_dust_EB']
        param_labels = [r'\beta_{\rm CMB}', r'A_{\rm dust}^{EB}']
        plot_marginalized_triangle(args.base_dir, param_names, param_labels, outdir=args.output_dir)
        return
    chain_dirs = find_chain_dirs(args.base_dir)

    if args.plot_traces:
        param_name = "gMpl"
        for chain_dir in chain_dirs:
            chain_path = os.path.join(args.base_dir, chain_dir)
            samples = get_samples(os.path.join(chain_path, "real"))
            if not samples:
                continue
            try:
                param_names = samples.paramNames.list()
                if param_name not in param_names:
                    print(f"{param_name} not found in {chain_dir}")
                    continue
                param_index = param_names.index(param_name)
            except Exception as e:
                print(f"Error finding param index in {chain_dir}: {e}")
                continue

            trace_output_dir = os.path.join(args.output_dir, "traces")
            plot_trace_for_param(chain_path, param_name, param_index, trace_output_dir)

    elif args.group_by_ldiff:
        print('Doing ldiff')
        ldiff_chains = get_ldiff_samples(chain_dirs, args.base_dir)
        plot_ldiff_posteriors(ldiff_chains, args.output_dir)

    elif args.group_by_fede:
        fede_groups = group_samples_by_fede(chain_dirs, args.base_dir)
        plot_grouped_posteriors(fede_groups, args.output_dir)
    elif args.group_by_cmb:
        plot_betacmb_posteriors(chain_dirs, args.base_dir, args.output_dir)

    else:
        plot_each_chain_separately(chain_dirs, args.base_dir, args.output_dir)

if __name__ == "__main__":
    main()
