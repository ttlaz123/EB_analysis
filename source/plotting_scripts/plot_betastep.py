import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import numpy as np

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