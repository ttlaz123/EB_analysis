import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
import numpy as np
plt.rcParams.update({
    "text.usetex": True,           # Uses LaTeX to render all text
    "font.family": "serif",        # Sets the font family to serif
    "font.serif": ["Computer Modern Roman"], # Matches standard LaTeX font
    "axes.labelsize": 22,
    "font.size": 18,
    "legend.fontsize": 18,
    "xtick.labelsize": 16,
    "ytick.labelsize": 16
})
# 1. Data initialization
l_b = np.array([265, 300, 335, 370, 405])
delta_beta = np.array([-0.15, -0.01, 0.05, 0.11, 0.09])
errors = np.array([0.15, 0.14, 0.13, 0.13, 0.15])

# 2. Setup the figure
fig, ax = plt.subplots(figsize=(8, 5))

# 3. Create the error plot
# 'capsize' adds the horizontal bars at the ends of the error bars
ax.errorbar(l_b, delta_beta, yerr=errors, fmt='o', color='red', 
            ecolor='black', elinewidth=1.5, capsize=4, markersize=10)

# 4. Styling to match the image
ax.axhline(0, color='gray', linestyle='--', linewidth=1) # The zero line
ax.set_xlabel(r'Multipole Breakpoint $\ell_b$', fontsize=24)
ax.set_ylabel(r'Step Size $\Delta \beta_{\ell_b}$ [deg.]', fontsize=24)

# Set specific ticks to match the breakpoints provided
ax.set_xticks(l_b)
ax.tick_params(axis='both', which='major', labelsize=16)

# Add light grid lines (as seen in the background of your image)
ax.grid(True, linestyle=':', alpha=0.6)

# 5. Save as a resolution-independent PDF
plt.tight_layout()
plt.savefig('multipole_breakpoint_plot.pdf', format='pdf', bbox_inches='tight')

print("Plot saved successfully as multipole_breakpoint_plot.pdf")
plt.show()
