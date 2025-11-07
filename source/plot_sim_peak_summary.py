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
        fig, axs = plt.subplots(len(g_list), len(b_list), figsize=(10, 6), sharex=True, sharey=True)
        fig.suptitle(f"{param} beam scaling", fontsize=14)

        for i, g in enumerate(g_list):
            for j, b in enumerate(b_list):
                ax = axs[i, j] if len(g_list) > 1 else axs[j]
                for m in m_list:
                    mean_vals, std_vals = [], []

                    for eps in eps_values:
                        eps_tag = f"eps{eps:.2f}"  
                        if eps_tag == "eps0.00":
                            eps_tag = "eps-0.00" 
                        folder = f"{dirpath}/{g}_{b}_fixeddust_{m}_eb_sig{eps_tag}_ebfede0.07"
                        chains_path = os.path.join(folder, "sim")

                        # Call your function that returns parameter means/stds
                        mean_params_dict, std_params_dict = epd.plot_sim_peaks(chains_path, single_sim=1, overwrite=False, do_plots=False)

                        mean_vals.append(mean_params_dict.get(param, np.nan))
                        std_vals.append(std_params_dict.get(param, np.nan))

                    ax.plot(eps_values, mean_vals, marker='o', label=f"{m}")
                    ax.fill_between(eps_values,
                                    np.array(mean_vals) - np.array(std_vals),
                                    np.array(mean_vals) + np.array(std_vals),
                                    alpha=0.2)

                ax.set_title(f"{g}, {b}")
                ax.set_xlabel("epsilon")
                ax.set_ylabel(f"{param}")
                ax.grid(True, alpha=0.3)
                ax.legend(fontsize=8)

        plt.tight_layout(rect=[0, 0, 1, 0.96])
        plt.savefig('summaryplot.png')

if __name__=='__main__':
    dirpath = '/n/holylfs04/LABS/kovac_lab/users/liuto/ede_chains/'
    plot_sim_summary(dirpath)
