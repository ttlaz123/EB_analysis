import os
import re
import numpy as np
import matplotlib.pyplot as plt
import eb_plot_data as epd

def plot_sim_summary_new(dirpath, params_to_plot=None):
    """
    Plots parameter means and stds vs injected beam uncertainty ε.
    Each parameter gets its own figure.
    For each (g, b) combination, all mapfreq sets (m) appear as curves on the same plot.
    """

    g_list = ['zeroeb', 'fede01']
    b_list = ['bin2-15', 'bin2-10']

    if params_to_plot is None:
        params_to_plot = ['alpha_BK18_B95e', 'alpha_BK18_K95', 
                          'alpha_BK18_150', 'alpha_BK18_220', 'gMpl']

    # Scan all folders once to extract mapfreqs and epsilons
    all_folders = [f for f in os.listdir(dirpath) if os.path.isdir(os.path.join(dirpath, f))]

    folder_info = []
    for f in all_folders:
        match = re.match(r'(.*?)_(.*?)_fixeddust_(.*?)_eb_sigeps_(BK18_\w+)([-\d\.]+)_ebfede0\.07', f)
        if match:
            g, b, m_base, mapfreq, eps_str = match.groups()
            eps = float(eps_str)
            folder_info.append({
                'g': g,
                'b': b,
                'mapfreq': mapfreq,
                'eps': eps,
                'folder': os.path.join(dirpath, f)
            })

    # Loop through parameters
    for param in params_to_plot:
        for g in g_list:
            for b in b_list:
                plt.figure(figsize=(8, 6))
                plt.title(f"{param} beam scaling — {g}, {b}", fontsize=14)

                # Filter folders by g and b
                relevant = [f for f in folder_info if f['g'] == g and f['b'] == b]

                # Get all unique mapfreqs
                mapfreqs = sorted(set(f['mapfreq'] for f in relevant))

                for mapfreq in mapfreqs:
                    subfolders = sorted([f for f in relevant if f['mapfreq'] == mapfreq],
                                        key=lambda x: x['eps'])

                    eps_values, mean_vals, std_vals = [], [], []

                    for entry in subfolders:
                        eps_values.append(entry['eps'])
                        chains_path = os.path.join(entry['folder'], "sim")

                        try:
                            mean_params_dict, std_params_dict = epd.plot_sim_peaks(
                                chains_path, single_sim=1, overwrite=False, do_plots=False
                            )
                            mean_vals.append(mean_params_dict[param])
                            std_vals.append(std_params_dict[param])
                        except Exception:
                            continue

                    if len(mean_vals) > 0:
                        plt.errorbar(
                            eps_values, mean_vals, yerr=std_vals, fmt='-o', label=f"{mapfreq}", capsize=4
                        )

                plt.xlabel("epsilon", fontsize=12)
                plt.ylabel(param, fontsize=12)
                if param == 'gMpl':
                    plt.ylim([-2, 3])
                else:
                    plt.ylim([-0.6, 0.6])
                plt.grid(alpha=0.3)
                plt.legend(title="MapFreq", fontsize=9)
                plt.tight_layout()

                outfile = f"summary_{param}_{g}_{b}.png"
                plt.savefig(outfile, dpi=150)
                print(f"Saved {outfile}")
                plt.close()


if __name__=='__main__':
    dirpath = '/n/holylfs04/LABS/kovac_lab/users/liuto/ede_chains/'
    plot_sim_summary(dirpath)
