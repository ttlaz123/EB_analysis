import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import numpy as np

def plot_sims():
    # Multipole breakpoints
    ell_b = np.array([265, 300, 335, 370, 405])

    # ΛCDM (g = 0) values
    delta_beta_LCDM = np.array([0.01, 0.01, 0.02, 0.01, 0.01])
    delta_beta_LCDM_err = np.array([0.16, 0.14, 0.12, 0.14, 0.17])

    # EDE (g = 1) values
    delta_beta_EDE = np.array([0.13, 0.20, 0.23, 0.26, 0.30])
    delta_beta_EDE_err = np.array([0.17, 0.15, 0.14, 0.14, 0.18])

    plt.rcParams.update({
        "text.usetex": True,
        "mathtext.fontset": "dejavuserif", 
        "font.family": "serif", 
        "font.serif": 'Computer Modern',
        "font.size": 20
    })

    # Create the plot
    fig, ax = plt.subplots(figsize=(8, 5))
    dx = 1
    # Plot ΛCDM (red dots)
    ax.errorbar(
        ell_b-dx, delta_beta_LCDM, yerr=delta_beta_LCDM_err,
        fmt='o', markersize=8,
        markerfacecolor='orange', markeredgecolor='red',
        ecolor='gray', capsize=4, label=r"\texttt{$\Lambda$CDM} ($g = 0$)"
    )

    # Plot EDE (blue dots)
    ax.errorbar(
        ell_b+dx, delta_beta_EDE, yerr=delta_beta_EDE_err,
        fmt='s', markersize=8,
        markerfacecolor='blue', markeredgecolor='blue',
        ecolor='gray', capsize=4, label=r"\texttt{EDE} ($g = 1, f_{\mathrm{EDE}}=0.07$)"
    )

    # Horizontal line at 0
    ax.axhline(0, color='black', linestyle='--', linewidth=1)

    # Axis labels
    ax.set_xlabel(r"Multipole Breakpoint $\ell_b$", fontsize=22)
    ax.set_ylabel(r"Step Size $\Delta \beta_{\ell_b}$ [deg.]", fontsize=22)

    # Set x-ticks to match ell_b
    ax.set_xticks(ell_b)

    # Grid and legend
    ax.grid(True, linestyle=':', alpha=0.5)
    ax.legend(fontsize=16)

    plt.tight_layout()
    print('Saving: ' + "delta_beta_dual_plot.png")
    plt.savefig("delta_beta_dual_plot.png", dpi=300)


def plot_reals():
    # Bandpower breaks and Delta beta values with uncertainties
    ell_b = np.array([265, 300, 335, 370, 405])
    delta_beta = np.array([-0.15, -0.01, 0.05, 0.11, 0.09])
    delta_beta_err = np.array([0.15, 0.14, 0.13, 0.13, 0.15])


    plt.rcParams.update({
        "text.usetex": True,
        "mathtext.fontset": "dejavuserif", 
        "font.family": "serif", 
        "font.serif": 'Computer Modern',
        "font.size": 20
    })

    # Create the plot
    fig, ax = plt.subplots(figsize=(8, 5))

    # Plot red dots with error bars
    ax.errorbar(
        ell_b, delta_beta, yerr=delta_beta_err,
        fmt='o', markersize=10,
        markerfacecolor='red', markeredgecolor='red',
        ecolor='black', capsize=4
    )
    # Horizontal dashed line at 0
    ax.axhline(0, color='gray', linestyle='--', linewidth=1)

    # Labeling
    ax.set_xlabel(r"Multipole Breakpoint $\ell_b$", fontsize=24)
    ax.set_ylabel(r"Step Size $\Delta \beta_{\ell_b}$ [deg.]", fontsize=24)

    # Make x-ticks exactly the data points
    ax.set_xticks(ell_b)

    # Grid and layout
    ax.grid(True, linestyle=':', alpha=0.5)
    plt.tight_layout()

    # Save or show
    plt.savefig("delta_beta_plot.png", dpi=300)  

if __name__ == '__main__':
    plot_sims()