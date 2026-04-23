import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

def plot_eb_fitting_multi():

    # Use Matplotlib's internal math rendering engine
    plt.rcParams.update({
        "text.usetex": False,
        "font.family": "serif",
        "mathtext.fontset": "cm",
        "axes.labelsize": 16,
        "font.size": 14,
        "legend.fontsize": 12,
        "axes.titlesize": 16,
        "xtick.labelsize": 12,
        "ytick.labelsize": 12
    })

    file_path = 'input_data/fEDE0.07_cl.dat'

    try:
        # Load the theoretical spectra
        data = np.loadtxt(file_path, skiprows=1)
        l_full = data[:, 0]
        
        # Columns are assumed to be D_l / T_cmb^2
        C_EE_raw = data[:, 2]
        C_BB_raw = data[:, 4]
        
        T_cmb = 2.7255e6
        D_factor = T_cmb**2
        
        D_EE_full = C_EE_raw * D_factor
        D_BB_full = C_BB_raw * D_factor
        
        # 1. Generate Mock Data (15 points between l = 30 and 500)
        np.random.seed(42) # For reproducibility
        ell_bins = np.linspace(30, 520, 15)
        
        # Interpolate the theoretical spectra at the chosen bins
        D_EE_bins = np.interp(ell_bins, l_full, D_EE_full)
        D_BB_bins = np.interp(ell_bins, l_full, D_BB_full)
        
        beta_true_deg = 0.35
        beta_true_rad = np.radians(beta_true_deg)
        
        # True EB at bins
        D_EB_true = 0.5 * (D_EE_bins - D_BB_bins) * np.sin(4 * beta_true_rad)
        
        # Realistic error bars (noise floor + proxy for cosmic variance)
        sigma_EB = 0.02 + 0.10 * np.abs(D_EB_true) + 0.0001 * ell_bins
        
        # Add noise
        mock_EB = D_EB_true + np.random.normal(0, sigma_EB)
        
        # 2. Setup plotting
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
        
        # Subplot 1: Data Only
        ax1.errorbar(ell_bins, mock_EB, yerr=sigma_EB, fmt='o', color='black', label='Simulated EB Observations', capsize=4, zorder=5)
        ax1.set_xlabel(r'Multipole $\ell$')
        ax1.set_ylabel(r'$\mathcal{D}_\ell^{EB} \quad [\mu\mathrm{K}^2]$')
        ax1.legend()
        ax1.grid(True, linestyle=':', alpha=0.6)
        ax1.set_xlim([0, 550])
        ax1.legend(loc='upper left')
        ax1.set_ylim([-0.2, 0.6])
        # Subplot 2: Data and Fits
        ax2.errorbar(ell_bins, mock_EB, yerr=sigma_EB, fmt='o', color='black', label='Simulated EB Observations', capsize=4, zorder=5)
        
        # Smooth lines for theoretical models
        mask_plot = (l_full >= 20) & (l_full <= 600)
        ell_plot = l_full[mask_plot]
        D_EE_plot = D_EE_full[mask_plot]
        D_BB_plot = D_BB_full[mask_plot]
        
        test_betas = [0.1, 0.35, 0.6]
        colors = ['blue', 'purple', 'red']
        
        print("Chi-squared values for test models:")
        print("-" * 35)
        
        for tb, c in zip(test_betas, colors):
            tb_rad = np.radians(tb)
            
            # Calculate chi-squared at the discrete bins
            model_bins = 0.5 * (D_EE_bins - D_BB_bins) * np.sin(4 * tb_rad)
            chisq = np.sum(((mock_EB - model_bins) / sigma_EB)**2)
            print(f"Beta = {tb:4.2f} degrees : chi^2 = {chisq:.2f}")
            
            # Plot continuous lines using the full theoretical resolution
            model_plot = 0.5 * (D_EE_plot - D_BB_plot) * np.sin(4 * tb_rad)
            ax2.plot(ell_plot, model_plot, label=rf'Model ($\beta = {tb}^\circ$)', color=c, lw=2, alpha=0.8)
            
        ax2.set_xlabel(r'Multipole $\ell$')
        ax2.set_ylabel(r'$\mathcal{D}_\ell^{EB} \quad [\mu\mathrm{K}^2]$')
        # Grab the current handles and labels from the axis
        handles, labels = ax2.get_legend_handles_labels()

        # In ax2, the 'Simulated EB Observations' is the last item appended to the lists, 
        # so we move the last element to the front.
        ordered_handles = [handles[-1]] + handles[:-1]
        ordered_labels = [labels[-1]] + labels[:-1]

        # Apply the explicit ordering
        ax2.legend(ordered_handles, ordered_labels, loc='upper left')
        ax2.set_xlim([0, 550])
        ax2.set_ylim([-0.2, 0.6])
        ax2.grid(True, linestyle=':', alpha=0.6)
        
        plt.tight_layout()
        plt.show()

    except FileNotFoundError:
        print(f"Error: Could not find the file {file_path}. Please check the file name and path.")

def plot_eb_rotations():
    # Use Matplotlib's internal math rendering engine instead of external LaTeX
    plt.rcParams.update({
        "text.usetex": False,
        "font.family": "serif",
        "mathtext.fontset": "cm",  # Computer Modern font for math
        "axes.labelsize": 24,
        "font.size": 24,
        "legend.fontsize": 18,
        "axes.titlesize": 20,
        "xtick.labelsize": 18,
        "ytick.labelsize": 18
    })

    file_path = 'input_data/fEDE0.07_cl.dat'

    try:
        data = np.loadtxt(file_path, skiprows=1)
        
        l_full = data[:, 0]
        
        mask = l_full <= 1000
        l = l_full[mask]
        
        C_EE_raw = data[mask, 2]
        C_BB_raw = data[mask, 4]
        
        T_cmb = 2.7255e6
        D_factor = T_cmb**2
        
        D_EE = C_EE_raw * D_factor
        D_BB = C_BB_raw * D_factor
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        ax.plot(l, D_EE, label='$EE$', color='black', lw=2)
        
        betas = [0.1, 0.4, 0.7, 1.0]
        cmap = plt.get_cmap('Blues')
        colors = [cmap(0.4), cmap(0.6), cmap(0.8), cmap(1.0)]
        
        for b, c in zip(betas, colors):
            beta_rad = np.radians(b)
            D_EB = 0.5 * (D_EE - D_BB) * np.sin(4 * beta_rad)
            
            ax.plot(l, np.abs(D_EB), label=rf'$EB$ ($\beta_{{cmb}} = {b}^\circ$)', color=c, lw=1.5)
            
        ax.set_xscale('linear')
        ax.set_yscale('log')
        ax.set_xlim(2, 1000)
        ax.set_ylim(1e-6, 1e2) 
        
        ax.set_xlabel(r'Multipole $\ell$')
        ax.set_ylabel(r'$\mathcal{D}_\ell \quad [\mu\mathrm{K}^2]$')
        #ax.set_title(r'CMB EE and Induced EB Power Spectra')
        ax.legend()
        ax.grid(True, which='both', linestyle='--', alpha=0.5)
        
        plt.tight_layout()
        plt.show()

    except FileNotFoundError:
        print(f"Error: Could not find the file {file_path}. Please check the file name and path.")

def plot_eb_fitting(target_beta=0.35, frequency="220 GHz"):

    # Use Matplotlib's internal math rendering engine
    plt.rcParams.update({
        "text.usetex": False,
        "font.family": "serif",
        "mathtext.fontset": "cm",
        "axes.labelsize": 16,
        "font.size": 14,
        "legend.fontsize": 12,
        "axes.titlesize": 16,
        "xtick.labelsize": 12,
        "ytick.labelsize": 12
    })

    file_path = 'input_data/fEDE0.07_cl.dat'

    try:
        # Load the theoretical spectra
        data = np.loadtxt(file_path, skiprows=1)
        l_full = data[:, 0]
        
        # Columns are assumed to be D_l / T_cmb^2
        C_EE_raw = data[:, 2]
        C_BB_raw = data[:, 4]
        
        T_cmb = 2.7255e6
        D_factor = T_cmb**2
        
        D_EE_full = C_EE_raw * D_factor
        D_BB_full = C_BB_raw * D_factor
        
        # 1. Generate Mock Data (15 points between l = 30 and 500)
        np.random.seed() # For reproducibility
        ell_bins = np.linspace(30, 520, 15)
        
        # Interpolate the theoretical spectra at the chosen bins
        D_EE_bins = np.interp(ell_bins, l_full, D_EE_full)
        D_BB_bins = np.interp(ell_bins, l_full, D_BB_full)
        
        # Use the requested target_beta
        beta_true_rad = np.radians(target_beta)
        
        # True EB at bins
        D_EB_true = 0.5 * (D_EE_bins - D_BB_bins) * np.sin(4 * beta_true_rad)
        
        # Realistic error bars (noise floor + proxy for cosmic variance)
        sigma_EB = 0.02 + 0.10 * np.abs(D_EB_true) + 0.0001 * ell_bins
        
        # Add noise
        mock_EB = D_EB_true + np.random.normal(0, sigma_EB)
        
        # 2. Find the best fit beta
        beta_grid = np.linspace(-1.0, 1.0, 1000)
        chisq_array = np.zeros_like(beta_grid)
        
        for i, b in enumerate(beta_grid):
            b_rad = np.radians(b)
            model_bins = 0.5 * (D_EE_bins - D_BB_bins) * np.sin(4 * b_rad)
            chisq_array[i] = np.sum(((mock_EB - model_bins) / sigma_EB)**2)
            
        best_beta_idx = np.argmin(chisq_array)
        best_beta = beta_grid[best_beta_idx]
        min_chisq = chisq_array[best_beta_idx]
        
        print(f"Target Beta: {target_beta:.2f} degrees")
        print(f"Best Fit Beta: {best_beta:.2f} degrees (chi^2 = {min_chisq:.2f})")
        
        # 3. Setup plotting
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
        
        # Subplot 1: Data Only
        ax1.errorbar(ell_bins, mock_EB, yerr=sigma_EB, fmt='o', color='black', label=f'Simulated Data ({frequency})', capsize=4, zorder=5)
        ax1.set_xlabel(r'Multipole $\ell$')
        ax1.set_ylabel(r'$\mathcal{D}_\ell^{EB} \quad [\mu\mathrm{K}^2]$')
        ax1.set_title(rf'Mock EB Data ({frequency})')
        ax1.grid(True, linestyle=':', alpha=0.6)
        ax1.set_xlim([0, 550])
        ax1.set_ylim([-0.2, 0.6])
        ax1.legend(loc='upper left')

        # Subplot 2: Data and Best Fit
        ax2.errorbar(ell_bins, mock_EB, yerr=sigma_EB, fmt='o', color='black', label=f'Simulated Data', capsize=4, zorder=5)
        
        # Smooth line for best-fit theoretical model
        mask_plot = (l_full >= 20) & (l_full <= 600)
        ell_plot = l_full[mask_plot]
        D_EE_plot = D_EE_full[mask_plot]
        D_BB_plot = D_BB_full[mask_plot]
        
        best_beta_rad = np.radians(best_beta)
        model_plot = 0.5 * (D_EE_plot - D_BB_plot) * np.sin(4 * best_beta_rad)
        
        # Plot the single best-fit line
        ax2.plot(ell_plot, model_plot, label=rf'Best Fit ($\beta = {best_beta:.2f}^\circ$)', color='purple', lw=2, alpha=0.8)
            
        ax2.set_xlabel(r'Multipole $\ell$')
        ax2.set_ylabel(r'$\mathcal{D}_\ell^{EB} \quad [\mu\mathrm{K}^2]$')
        ax2.set_title(rf'Best Fit $\beta = {best_beta:.2f}^\circ$ ({frequency})')
        
        # Fix legend ordering
        handles, labels = ax2.get_legend_handles_labels()
        ordered_handles = [handles[-1]] + handles[:-1]
        ordered_labels = [labels[-1]] + labels[:-1]
        ax2.legend(ordered_handles, ordered_labels, loc='upper left')
        
        ax2.axhline(0, color='gray', linestyle='--', lw=1.5, alpha=0.7, zorder=1)
        ax2.set_xlim([0, 550])
        ax2.set_ylim([-0.6, 0.6])
        ax2.grid(True, linestyle=':', alpha=0.6)
        
        plt.tight_layout()
        plt.show()

    except FileNotFoundError:
        print(f"Error: Could not find the file {file_path}. Please check the file name and path.")
def plot_beta_likelihood(target_beta=0.35, frequency="220 GHz"):

    plt.rcParams.update({
        "text.usetex": False,
        "font.family": "serif",
        "mathtext.fontset": "cm",
        "axes.labelsize": 16,
        "font.size": 14,
        "legend.fontsize": 12,
        "axes.titlesize": 16,
        "xtick.labelsize": 12,
        "ytick.labelsize": 12
    })

    file_path = 'input_data/fEDE0.07_cl.dat'

    try:
        data = np.loadtxt(file_path, skiprows=1)
        l_full = data[:, 0]
        
        C_EE_raw = data[:, 2]
        C_BB_raw = data[:, 4]
        
        T_cmb = 2.7255e6
        D_factor = T_cmb**2
        
        D_EE_full = C_EE_raw * D_factor
        D_BB_full = C_BB_raw * D_factor
        
        np.random.seed(42) 
        ell_bins = np.linspace(30, 520, 15)
        
        D_EE_bins = np.interp(ell_bins, l_full, D_EE_full)
        D_BB_bins = np.interp(ell_bins, l_full, D_BB_full)
        
        beta_true_rad = np.radians(target_beta)
        
        D_EB_true = 0.5 * (D_EE_bins - D_BB_bins) * np.sin(4 * beta_true_rad)
        
        sigma_EB = 0.02 + 0.10 * np.abs(D_EB_true) + 0.0001 * ell_bins
        
        mock_EB = D_EB_true + np.random.normal(0, sigma_EB)
        
        # Calculate chi-squared over a fine grid centered around the target
        beta_grid = np.linspace(0.0, 0.7, 1000)
        chisq_array = np.zeros_like(beta_grid)
        
        for i, b in enumerate(beta_grid):
            b_rad = np.radians(b)
            model_bins = 0.5 * (D_EE_bins - D_BB_bins) * np.sin(4 * b_rad)
            chisq_array[i] = np.sum(((mock_EB - model_bins) / sigma_EB)**2)
            
        min_chisq = np.min(chisq_array)
        best_beta = beta_grid[np.argmin(chisq_array)]
        
        # Convert chi-squared to normalized likelihood
        likelihood = np.exp(-0.5 * (chisq_array - min_chisq))
        
        fig, ax = plt.subplots(figsize=(8, 6))
        
        ax.plot(beta_grid, likelihood, color='blue', lw=2, label=r'Likelihood $\mathcal{L}(\beta)$')
        #ax.axvline(target_beta, color='black', linestyle='--', label=rf'True $\beta = {target_beta}^\circ$')
        ax.axvline(best_beta, color='red', linestyle=':', label=rf'Best Fit $\beta = {best_beta:.3f}^\circ$')
        
        # Add horizontal lines for standard confidence intervals
        sig1_level = np.exp(-0.5 * 1.0) # Delta chi^2 = 1
        sig2_level = np.exp(-0.5 * 4.0) # Delta chi^2 = 4
        
        ax.axhline(sig1_level, color='gray', linestyle='--', alpha=0.7, label=r'$1\sigma$ ($68\%$ limit)')
        ax.axhline(sig2_level, color='gray', linestyle=':', alpha=0.7, label=r'$2\sigma$ ($95\%$ limit)')
        
        ax.set_xlabel(r'Rotation Angle $\beta$ [degrees]')
        ax.set_ylabel(r'Normalized Likelihood $\mathcal{L}/\mathcal{L}_{max}$')
        ax.set_title(rf'Likelihood Constraint on $\beta$')
        
        # Restrict x-axis to zoom in on the relevant region
        ax.set_xlim([0.1, 0.6])
        ax.set_ylim([0, 1.05])
        ax.grid(True, linestyle=':', alpha=0.6)
        ax.legend(loc='upper right')
        
        plt.tight_layout()
        plt.show()

    except FileNotFoundError:
        print(f"Error: Could not find the file {file_path}. Please check the file name and path.")


def plot_cobaya_triangle(target_sum=0.35):
    
    plt.rcParams.update({
        "text.usetex": False,
        "font.family": "serif",
        "mathtext.fontset": "cm",
        "axes.labelsize": 16,
        "font.size": 14,
        "xtick.labelsize": 12,
        "ytick.labelsize": 12,
        "xtick.direction": "in",
        "ytick.direction": "in"
    })

    file_path = 'input_data/fEDE0.07_cl.dat'
    
    try:
        data = np.loadtxt(file_path, skiprows=1)
        l_full = data[:, 0]
        
        C_EE_raw = data[:, 2]
        C_BB_raw = data[:, 4]
        
        T_cmb = 2.7255e6
        D_factor = T_cmb**2
        
        D_EE_full = C_EE_raw * D_factor
        D_BB_full = C_BB_raw * D_factor
        
        # Generate mock data
        np.random.seed(42) 
        ell_bins = np.linspace(30, 520, 15)
        
        D_EE_bins = np.interp(ell_bins, l_full, D_EE_full)
        D_BB_bins = np.interp(ell_bins, l_full, D_BB_full)
        
        observed_rotation_rad = np.radians(target_sum)
        D_EB_true = 0.5 * (D_EE_bins - D_BB_bins) * np.sin(4 * observed_rotation_rad)
        
        sigma_EB = 0.02 + 0.10 * np.abs(D_EB_true) + 0.0001 * ell_bins
        mock_EB = D_EB_true + np.random.normal(0, sigma_EB)
        
        # Build the 2D grid
        grid_points = 250
        beta_grid = np.linspace(-0.2, 0.9, grid_points)
        alpha_grid = np.linspace(-0.2, 0.9, grid_points)
        
        B, A = np.meshgrid(beta_grid, alpha_grid)
        chisq_2d = np.zeros_like(B)
        
        for i in range(grid_points):
            for j in range(grid_points):
                total_angle_rad = np.radians(B[i,j] + A[i,j])
                model_bins = 0.5 * (D_EE_bins - D_BB_bins) * np.sin(4 * total_angle_rad)
                chisq_2d[i,j] = np.sum(((mock_EB - model_bins) / sigma_EB)**2)
                
        min_chisq = np.min(chisq_2d)
        delta_chisq = chisq_2d - min_chisq
        
        # Calculate Likelihoods
        L_2d = np.exp(-0.5 * delta_chisq)
        
        # Marginalize over alpha to get 1D beta
        L_beta = np.trapz(L_2d, alpha_grid, axis=0)
        L_beta /= np.max(L_beta)
        
        # Marginalize over beta to get 1D alpha
        L_alpha = np.trapz(L_2d, beta_grid, axis=1)
        L_alpha /= np.max(L_alpha)
        
        # Setup Figure and GridSpec
        fig = plt.figure(figsize=(8, 8))
        gs = gridspec.GridSpec(2, 2, hspace=0.05, wspace=0.05)
        
        ax_beta = fig.add_subplot(gs[0, 0])
        ax_joint = fig.add_subplot(gs[1, 0], sharex=ax_beta)
        ax_alpha = fig.add_subplot(gs[1, 1], sharey=ax_joint)
        
        # Cobaya color palette
        dark_blue = '#08519c'
        mid_blue = '#3182bd'
        light_blue = '#9ecae1'
        
        # --- Top Left: 1D Beta ---
        ax_beta.plot(beta_grid, L_beta, color=dark_blue, lw=2)
        ax_beta.fill_between(beta_grid, 0, L_beta, color=light_blue, alpha=0.7)
        ax_beta.set_ylim([0, 1.1])
        ax_beta.set_yticks([])
        plt.setp(ax_beta.get_xticklabels(), visible=False)
        ax_beta.spines['top'].set_visible(False)
        ax_beta.spines['right'].set_visible(False)
        ax_beta.spines['left'].set_visible(False)
        
        # --- Bottom Right: 1D Alpha ---
        # Note: In standard corner plots, the y-axis parameter's 1D plot is rotated.
        # However, plotting it strictly horizontally is also common. We will rotate 
        # it to match the shared y-axis of the 2D contour plot.
        ax_alpha.plot(L_alpha, alpha_grid, color=dark_blue, lw=2)
        ax_alpha.fill_betweenx(alpha_grid, 0, L_alpha, color=light_blue, alpha=0.7)
        ax_alpha.set_xlim([0, 1.1])
        ax_alpha.set_xticks([])
        plt.setp(ax_alpha.get_yticklabels(), visible=False)
        ax_alpha.spines['top'].set_visible(False)
        ax_alpha.spines['right'].set_visible(False)
        ax_alpha.spines['bottom'].set_visible(False)
        ax_alpha.set_xlabel(r'Likelihood')
        
        # --- Bottom Left: 2D Joint Contour ---
        # Delta chi^2 thresholds for 2 degrees of freedom
        levels = [np.exp(-0.5 * 6.18), np.exp(-0.5 * 2.30), 1.1] 
        ax_joint.contourf(B, A, L_2d, levels=levels, colors=[light_blue, mid_blue], alpha=0.9)
        ax_joint.contour(B, A, L_2d, levels=levels[:2], colors=[dark_blue], linewidths=1.5)
        
        # Plot ideal degeneracy line
        beta_line = np.linspace(-0.2, 0.9, 100)
        alpha_line = target_sum - beta_line
        ax_joint.plot(beta_line, alpha_line, color='black', linestyle='--', lw=1.5, alpha=0.8)
        
        ax_joint.set_xlabel(r'Cosmic Birefringence $\beta_{cmb}$ [degrees]')
        ax_joint.set_ylabel(r'Miscalibration $\alpha_\nu$ [degrees]')
        ax_joint.set_xlim([-0.2, 0.9])
        ax_joint.set_ylim([-0.2, 0.9])
        
        plt.show()

    except FileNotFoundError:
        print(f"Error: Could not find the file {file_path}.")



if __name__ == '__main__':
    # Execute the function
    plot_cobaya_triangle(target_sum=0.35)
    #plot_beta_likelihood(target_beta=0.35, frequency="220 GHz")
    #plot_eb_fitting(target_beta=-0.1, frequency="Keck 95 GHz")