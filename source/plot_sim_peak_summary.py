import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')

import os
import eb_plot_data as epd

def plot_sim_summary(dirpath, params_to_plot=None):
    """
    Plots parameter means and stds vs injected beam uncertainty ε.
    Each parameter gets its own figure.
    For each (g, b) combination, all map sets (m) appear as curves on the same plot.
    """

    eps_values = np.arange(-0.05, 0.051, 0.01)  # from -0.05 to 0.05
    g_list = ['zeroeb', 'fede01']
    b_list = ['bin2-15', 'bin2-10']
    m_list = ['BK18', 'BK18_B95e']

    if params_to_plot is None:
        params_to_plot = ['alpha_B95e', 'alpha_K95', 'alpha_150', 'alpha_220', 'gMpl']

    # Loop through parameters
    for param in params_to_plot:
        for g in g_list:
            for b in b_list:
                plt.figure(figsize=(8, 6))
                plt.title(f"{param} beam scaling — {g}, {b}", fontsize=14)

                for m in m_list:
                    mean_vals, std_vals = [], []

                    for eps in eps_values:
                        eps_tag = f"eps{eps:.2f}"
                        if eps_tag == "eps0.00":
                            eps_tag = "eps-0.00"  # handle sign case

                        folder = f"{dirpath}/{g}_{b}_fixeddust_{m}_eb_sig{eps_tag}_ebfede0.07"
                        chains_path = os.path.join(folder, "sim")

                        try:
                            mean_params_dict, std_params_dict = epd.plot_sim_peaks(
                                chains_path, single_sim=1, overwrite=False, do_plots=False
                            )
                            mean_vals.append(mean_params_dict.get(param, np.nan))
                            std_vals.append(std_params_dict.get(param, np.nan))
                        except Exception as e:
                            print(f"Warning: failed for {folder}: {e}")
                            mean_vals.append(np.nan)
                            std_vals.append(np.nan)

                    # Plot means with error bars
                    plt.errorbar(
                        eps_values, mean_vals, yerr=std_vals, fmt='-o', label=f"{m}", capsize=4
                    )

                plt.xlabel("epsilon", fontsize=12)
                plt.ylabel(param, fontsize=12)
                plt.grid(alpha=0.3)
                plt.legend(title="Map", fontsize=9)
                plt.tight_layout()

                outfile = f"summary_{param}_{g}_{b}.png"
                plt.savefig(outfile, dpi=150)
                print(f"Saved {outfile}")
                plt.close()

if __name__=='__main__':
    dirpath = '/n/holylfs04/LABS/kovac_lab/users/liuto/ede_chains/'
    plot_sim_summary(dirpath)
