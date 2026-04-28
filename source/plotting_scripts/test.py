import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')

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

    file_path = '../../input_data/fEDE0.07_cl.dat'

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
        
        # Use the requested target_beta
        beta_true_rad = np.radians(target_beta)
        
        # True EB at bins
        D_EB_true = 0.5 * (D_EE_bins - D_BB_bins) * np.sin(4 * beta_true_rad)
        
        # Realistic error bars (noise floor + proxy for cosmic variance)
        sigma_EB = 0.02 + 0.10 * np.abs(D_EB_true) + 0.0001 * ell_bins
        
        # Add noise
        mock_EB = D_EB_true + np.random.normal(0, sigma_EB)
        
        # 2. Find the best fit beta for full, l < 300, and l > 300
        beta_grid = np.linspace(-1.0, 1.0, 1000)
        
        chisq_array_full = np.zeros_like(beta_grid)
        chisq_array_low = np.zeros_like(beta_grid)
        chisq_array_high = np.zeros_like(beta_grid)
        
        mask_low = ell_bins < 300
        mask_high = ell_bins >= 300
        
        for i, b in enumerate(beta_grid):
            b_rad = np.radians(b)
            model_bins = 0.5 * (D_EE_bins - D_BB_bins) * np.sin(4 * b_rad)
            
            # Full fit
            chisq_array_full[i] = np.sum(((mock_EB - model_bins) / sigma_EB)**2)
            # Low ell fit
            chisq_array_low[i] = np.sum(((mock_EB[mask_low] - model_bins[mask_low]) / sigma_EB[mask_low])**2)
            # High ell fit
            chisq_array_high[i] = np.sum(((mock_EB[mask_high] - model_bins[mask_high]) / sigma_EB[mask_high])**2)
            
        # Get best fit parameters
        best_beta_full = beta_grid[np.argmin(chisq_array_full)]
        best_beta_low = beta_grid[np.argmin(chisq_array_low)]
        best_beta_high = beta_grid[np.argmin(chisq_array_high)]
        
        print(f"Target Beta: {target_beta:.2f} degrees")
        print(f"Best Fit Beta (All): {best_beta_full:.2f} degrees")
        print(f"Best Fit Beta (l<300): {best_beta_low:.2f} degrees")
        print(f"Best Fit Beta (l>300): {best_beta_high:.2f} degrees")
        
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

        # Subplot 2: Data and Best Fits
        ax2.errorbar(ell_bins, mock_EB, yerr=sigma_EB, fmt='o', color='black', label=f'Simulated Data', capsize=4, zorder=5)
        
        # Smooth line for best-fit theoretical models
        mask_plot = (l_full >= 20) & (l_full <= 600)
        ell_plot = l_full[mask_plot]
        D_EE_plot = D_EE_full[mask_plot]
        D_BB_plot = D_BB_full[mask_plot]
        
        # Generate full-length theoretical curves based on the best fit angles
        model_plot_full = 0.5 * (D_EE_plot - D_BB_plot) * np.sin(4 * np.radians(best_beta_full))
        model_plot_low = 0.5 * (D_EE_plot - D_BB_plot) * np.sin(4 * np.radians(best_beta_low))
        model_plot_high = 0.5 * (D_EE_plot - D_BB_plot) * np.sin(4 * np.radians(best_beta_high))
        
        # Boolean masks to restrict the plotting range
        plot_mask_low = ell_plot <= 300
        plot_mask_high = ell_plot >= 300
        
        # Plot the lines with specific styles and bounds
        ax2.plot(ell_plot, model_plot_full, label=rf'Fit All ($\beta = {best_beta_full:.2f}^\circ$)', color='blue', lw=2, linestyle=':', alpha=0.8)
        ax2.plot(ell_plot[plot_mask_low], model_plot_low[plot_mask_low], label=rf'Fit $\ell<300$ ($\beta = {best_beta_low:.2f}^\circ$)', color='gray', lw=2, linestyle='-', alpha=0.8)
        ax2.plot(ell_plot[plot_mask_high], model_plot_high[plot_mask_high], label=rf'Fit $\ell>300$ ($\beta = {best_beta_high:.2f}^\circ$)', color='red', lw=2, linestyle='-', alpha=0.8)
            
        ax2.set_xlabel(r'Multipole $\ell$')
        ax2.set_ylabel(r'$\mathcal{D}_\ell^{EB} \quad [\mu\mathrm{K}^2]$')
        ax2.set_title(rf'Fits for Multipole Partitions')
        
        # Fix legend ordering (data first)
        handles, labels = ax2.get_legend_handles_labels()
        # Ensure simulated data is the first item in the legend
        sim_data_idx = labels.index('Simulated Data')
        ordered_handles = [handles[sim_data_idx]] + [h for i, h in enumerate(handles) if i != sim_data_idx]
        ordered_labels = [labels[sim_data_idx]] + [l for i, l in enumerate(labels) if i != sim_data_idx]
        
        ax2.legend(ordered_handles, ordered_labels, loc='upper left')
        
        ax2.set_xlim([0, 550])
        ax2.set_ylim([-0.2, 0.6])
        ax2.grid(True, linestyle=':', alpha=0.6)
        
        plt.tight_layout()
        plt.savefig('test.png')

    except FileNotFoundError:
        print(f"Error: Could not find the file {file_path}. Please check the file name and path.")

# Execute the function
plot_eb_fitting(target_beta=0.35, frequency="220 GHz")
