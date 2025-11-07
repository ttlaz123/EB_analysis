
import numpy as np
import pandas as pd
import corner

import matplotlib
import os
matplotlib.use('Agg')  # headless mode
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.colors import BoundaryNorm


from getdist.mcsamples import loadMCSamples


print("Loading Plotting  Modules")
from getdist import plots, MCSamples
from getdist.mcsamples import loadMCSamples
import auxiliary_scripts.mcmc_summary as ms


import bicep_data_consts as bdc
MAP_FREQS = bdc.MAP_FREQS



def plot_covar_matrix(mat, used_maps=None, title='Log of covar matrix',
                        show_plot=False):
    """
    Visualizes a covariance matrix using a symmetric logarithmic color scale.

    Args:
        mat (np.ndarray): The covariance matrix to be plotted.
        used_maps (list of str, optional): List of map labels to use on axes.
        title (str): Title of the plot and the output filename.
        show_plot (bool): Whether to display the plot interactively.

    Saves:
        A PNG file named `{title}.png` showing the covariance matrix.
    """
    #print(max(mat[(mat<0.99)| (mat > 1.01)] ))
    nonzeros = np.abs(mat[(mat!=0) &( ~np.isnan(mat))])
    vpercent =max(np.percentile(nonzeros, 99), 1e-25)
    linthresh = np.percentile(nonzeros, 1)
    #print(nonzeros)
    #print(vpercent)
    cmap = plt.get_cmap('seismic')
    norm = mcolors.SymLogNorm(linthresh=linthresh, 
                                vmin=-vpercent, 
                                vmax=vpercent, base=10)
    plt.figure()
    plt.imshow(mat, cmap=cmap, norm=norm)
    plt.title(title)
    if(used_maps is not None):
        num_bins = int(mat.shape[0]/(len(used_maps)))

        tick_positions = np.arange(0, mat.shape[0], num_bins)
        plt.xticks(tick_positions, used_maps, 
                                rotation=30, ha='right')
        plt.yticks(tick_positions, used_maps)
    plt.colorbar()
    print("Saving: " + title + '.png')
    plt.savefig(title + '.png')
    if(show_plot):
        plt.show()

def plot_overlay_sims(spectra_type, observed_datas_list, outpath, observed_datas_list2=None):
    """
    Plots overlaid simulation data with clipped mean and error bands for a given spectrum type.

    Args:
        spectra_type (str): Type of spectrum to plot ('EE', 'EB', or 'BB').
        observed_datas_list (list of dict): List of observed data dictionaries.
        outpath (str): Base filename for saving the figure.
        observed_datas_list2 (list of dict, optional): Second set of data to subtract from first (e.g., theory).

    Saves:
        A figure named `{outpath}_overlay_confband_{spectra_type}.png`.
    """
    maps_B = set()
    maps_E = set()
    # Use keys from the first observed dataset
    keys = list(observed_datas_list[0].keys())
    for key in keys:
        parts = key.split('x')
        if parts[0].endswith('_B'):
            maps_B.add(parts[0])
        if parts[0].endswith('_E'):
            maps_E.add(parts[0])
        if parts[1].endswith('_B'):
            maps_B.add(parts[1])
        if parts[1].endswith('_E'):
            maps_E.add(parts[1])
    maps_B = sorted(list(maps_B))
    maps_E = sorted(list(maps_E))
    num_columns = len(maps_B)
    num_rows = len(maps_E)
    fig, axes = plt.subplots(num_rows, num_columns, figsize=(num_columns * 4, num_rows * 4))

    try:
        axes = axes.flatten()
    except AttributeError:
        axes = [axes]

    for key in keys:
        spec_type = determine_spectrum_type(key)
        parts = key.split('x')

        if spectra_type in ['EB', 'BE']:
            if spec_type in ['EE', 'BB']:
                continue
            row_idx = maps_E.index(parts[0]) if parts[0].endswith('_E') else maps_E.index(parts[1])
            col_idx = maps_B.index(parts[0]) if parts[0].endswith('_B') else maps_B.index(parts[1])
        elif spectra_type in ['EE', 'BB']:
            if spec_type != spectra_type:
                continue
            row_idx = maps_E.index(parts[0]) if parts[0].endswith('_E') else maps_B.index(parts[0])
            col_idx = maps_E.index(parts[1]) if parts[1].endswith('_E') else maps_B.index(parts[1])
        else:
            continue

        axes_index = row_idx * num_columns + col_idx
        ax = axes[axes_index]
        all_sims = []
        for i in range(len(observed_datas_list)):
            obs1 = observed_datas_list[i][key]
            if observed_datas_list2 is not None:
                if(i > len(observed_datas_list2)):
                    continue
                obs2 = observed_datas_list2[i][key]
                diff = np.array(obs1) - np.array(obs2)
                ax.plot(range(len(diff)), diff, color='gray', alpha=0.08, linewidth=0.5)
                all_sims.append(diff)
            else:
                ax.plot(range(len(obs1)), obs1, color='gray', alpha=0.08, linewidth=0.5)
                all_sims.append(obs1)

        n_sigma = 5  # threshold in units of standard deviation
        clipped_sims = []
        all_sims = np.array(all_sims) 
        for bin_idx in range(all_sims.shape[1]):
            bin_values = all_sims[:, bin_idx]
            mean = np.mean(bin_values)
            std = np.std(bin_values)
        
            # Mask for inliers: within ±n_sigma
            mask = np.abs(bin_values - mean) <= n_sigma * std
        
            # Mask for outliers
            outlier_mask = ~mask
            outlier_indices = np.where(outlier_mask)[0]
        
            if len(outlier_indices) > 0:
                outlier_vals = bin_values[outlier_indices]
                print(f"Bin {bin_idx}:")
                for idx, val in zip(outlier_indices, outlier_vals):
                    delta_sigma = (val - mean) / std
                    print(f"  Sim {idx} -> value = {val:.3e}, Δσ = {delta_sigma:.2f}")
        
            clipped_bin_values = bin_values[mask]
            clipped_sims.append(clipped_bin_values)
    
        # Pad bins with fewer values to uniform length (for np.array conversion)
        max_len = max(len(arr) for arr in clipped_sims)
        clipped_sims_padded = np.full((len(clipped_sims), max_len), np.nan)
        for i, arr in enumerate(clipped_sims):
            clipped_sims_padded[i, :len(arr)] = arr

        mean_vals = np.nanmean(clipped_sims_padded, axis=1)
        std_vals = np.nanstd(clipped_sims_padded, axis=1)

        x = np.arange(len(mean_vals))
        upper = mean_vals + std_vals
        lower = mean_vals - std_vals

        ax.fill_between(x, lower, upper, color='blue', alpha=0.3, label='68% conf. band')
        ax.plot(x, mean_vals, color='red', linewidth=1.2, label='Mean')

        # Set dynamic y-limits based on 1.5 * std from mean
        buffer_scale = 1.5
        y_min = np.min(lower - buffer_scale * std_vals)
        y_max = np.max(upper + buffer_scale * std_vals)
        ax.set_ylim(y_min, y_max)

        ax.set_title(key)
        ax.grid(True, which='both', linestyle='--', linewidth=0.5, alpha=0.6)
        ax.legend(fontsize=8)

    plt.tight_layout(pad=2)
    print("Saving: " + outpath + f'_overlay_confband_{spectra_type}.png')
    plt.savefig(outpath + f'_overlay_confband_{spectra_type}.png')
    plt.close(fig)
    return


def plot_spectra_type(spectra_type, maps_E, maps_B, theory_dict, multicomp_class, observed_datas,
                      outpath, param_stats, chis_sq):
    """
    Plots observed spectra with theory best fits for a specific polarization type.

    Args:
        spectra_type (str): The polarization spectrum type ('EE', 'EB', or 'BB').
        maps_E (list): List of E-polarization map names.
        maps_B (list): List of B-polarization map names.
        theory_dict (dict): Dictionary of best-fit spectra keyed by map pair.
        multicomp_class: Object containing covariance matrix and map metadata.
        observed_datas (dict): Observed spectrum data.
        outpath (str): Base filename for saving the figure.
        param_stats (list of str): List of formatted parameter statistics (e.g., alpha).
        chis_sq (float): Chi-squared value of the best-fit.

    Saves:
        A PNG file `{outpath}_bestfit{spectra_type}.png`.
    """
    plt.rcParams.update({
        "text.usetex": True,
        'font.family':'serif',
        'font.serif':'Computer Modern',
        'font.size': 20,
        'axes.titlesize': 22,
        'axes.labelsize': 22,
        'xtick.labelsize': 18,
        'ytick.labelsize': 18,
        'legend.fontsize': 14
    })

    num_columns = len(maps_B)  # Unique maps for columns
    num_rows = len(maps_E)      # Unique maps for rows
    # Create subplots
    fig, axes = plt.subplots(num_rows, num_columns, 
                    figsize=(num_columns * 4, num_rows * 4),
                    sharex=True, sharey=True)

    try:
        axes = axes.flatten()  # Flatten axes array for easy indexing
    except AttributeError:
        print("Only one axis!")
        axes = [axes]
    keys = list(theory_dict.keys())
    # Plot each spectrum
    bins = multicomp_class.bin_num 
    muls = bdc.L_BIN_CENTERS[[b-1 for b in bins]]
    for idx, key in enumerate(keys):
        observed_data = observed_datas[key]
        best_fit_data = theory_dict[key]
        #print(key)
        #print(observed_data - best_fit_data) 
        # Split key to find row and column indices
        spec_type = determine_spectrum_type(key)
        parts = key.split('x')
        mapname1 = parts[0].split('_')
        mapname2 = parts[1].split('_')
        if(mapname1[1] == 'B95e'):
            mapname1[1] = 'B95'
        if(mapname2[1] == 'B95e'):
            mapname2[1] = 'B95'
        label = rf"{mapname1[1]} $\times$ {mapname2[1]} {spec_type}"
        if(spectra_type in ['EB', 'BE']):
            if(spec_type in ['EE', 'BB']):
                continue
            row_idx = (maps_E).index(parts[0]) if parts[0].endswith('_E') else (maps_E).index(parts[1])
            col_idx = (maps_B).index(parts[0]) if parts[0].endswith('_B') else (maps_B).index(parts[1])
        elif(spectra_type in ['EE', 'BB']):
            if(not spec_type == spectra_type ):
                continue
            row_idx = (maps_E).index(parts[0]) if parts[0].endswith('_E') else (maps_B).index(parts[0])
            col_idx = (maps_E).index(parts[1]) if parts[1].endswith('_E') else (maps_B).index(parts[1])
       
        else: 
            pass
        
        
        map_index = multicomp_class.used_maps.index(key)
        num_bin = len(observed_data)
        covar_mat = multicomp_class.full_covmat
        var = np.diag(covar_mat)[map_index*num_bin:num_bin*(map_index+1)]
        # Plotting observed data
        axes_index = row_idx * num_columns + col_idx
        #print(observed_data)
         
        axes[axes_index].errorbar(
                            x = muls, #range(len(observed_data)),
                            y=(observed_data), 
                            yerr = np.sqrt(var),
                            linestyle='-', marker='o',
                            label=label, color='black')
        # Plotting best fit data
        axes[axes_index].plot(muls, best_fit_data, color='blue', 
                            linestyle='-')#, marker='o')

        #axes[axes_index].set_title(key)
        axes[axes_index].legend(loc='upper left', fontsize=20)
    '''
    for row_idx, map_E in enumerate(maps_E):
        angle = f"alpha_{map_E}"
        axes[row_idx].text(
            0.05, 1.4,  # X and Y position (top-left corner)
            param_stats[row_idx],  # The parameter stats
            transform=axes[row_idx].transAxes,  # Use axes coordinates
            fontsize=10, color='black',
            verticalalignment='top'
        )
    '''
    for ax in axes:
        ax.label_outer()
        ax.set_ylim([-0.02, 0.15])
        ax.axhline(0, color='gray', linestyle='--', linewidth=1)
    '''
    fig.text(
        x=0.5,  # Centered horizontally
        y=0.95,  # Just above bottom edge
        s=f"Chisq: {chis_sq:.2f}",
        fontsize=14,
        color='blue',
        ha='center',
        va='bottom'
    )
    '''
    fig.supxlabel(r"Multipole $\ell$", fontsize=24)
    fig.supylabel(r"$D_b^{EB}(\nu_1\times\nu_2)$ [$\mu K^2$]", fontsize=24)
    plt.tight_layout(pad=1)
    print("Saving: " +outpath + '_bestfit'+ spectra_type +'.png')
    plt.savefig(outpath + '_bestfit' + spectra_type + '.png')
    
    plt.close(fig)
    return 

def determine_spectrum_type(spectrum_name):
    """
    Determines the polarization spectrum type (EE, BB, EB, or BE) 
    from a string formatted like 'something_Exsomething_B'.

    Args:
        spectrum_name (str): The input string containing '_E' or '_B' before and after an 'x'.

    Returns:
        str: A 2-letter string indicating the spectrum type, e.g., 'EB', 'BE', 'EE', or 'BB'.

    Raises:
        AssertionError: If the input format is not as expected.
    """
    spectra = spectrum_name.split('x')
    assert len(spectra) == 2, "spectrum name isn't properly formatted: " + str(spectrum_name)
    
    spec1 = spectra[0][-2:]
    spec2 = spectra[1][-2:]
    
    assert spec1 in ['_E', '_B'], "spectrum name isn't properly formatted: " + str(spectrum_name)
    assert spec2 in ['_E', '_B'], "spectrum name isn't properly formatted: " + str(spectrum_name)

    spec_type = spec1[-1] + spec2[-1]
    return spec_type

def plot_eebbeb(multicomp_class, outpath, param_names, param_bestfit, param_stats, override_maps=None):
    """
    Plots EE, EB, and BB spectra and residual chi-squared blocks for a fitted model.

    Args:
        multicomp_class: Object containing observed data, covariance, and theory calculators.
        outpath (str): Base filename prefix for saving figures.
        param_names (list): List of parameter names.
        param_bestfit (list): Corresponding best-fit parameter values.
        param_stats (list of str): List of strings summarizing parameter stats (e.g., alpha).
        override_maps (list, optional): Subset of maps to override in the theory calculator.

    Saves:
        - A chi-squared heatmap.
        - Spectrum plots for EE, EB, and BB with best-fit overlays.
    """
    #used_maps = multicomp_class.used_maps
    observed_datas = multicomp_class.binned_dl_observed_dict
    param_values = {param_names[i]:param_bestfit[i] 
                            for i in range(len(param_names))}
    if('A_lens' not in param_names):
                param_values['A_lens'] = 1
    if(multicomp_class.theory_comps == 'ldiff'):
        theory_vec=multicomp_class.theory_diff(param_values)
    else:
        theory_vec=multicomp_class.theory(param_values, override_maps=override_maps)
    residuals = multicomp_class.binned_dl_observed_vec - theory_vec
    # Calculate the Mahalanobis distance using the inverse covariance matrix
    chi_squared = residuals.T @ multicomp_class.sim_common_data['full_inv_covmat'] @ residuals
    print('Chi Squared: ' + str(chi_squared))
    theory_dict = multicomp_class.final_detection_dict
    maps_B = set()
    maps_E = set()
    keys = list(theory_dict.keys())
    for key in keys:
        parts = key.split('x')
        if parts[0].endswith('_B'):
            maps_B.add(parts[0])
        if parts[0].endswith('_E'):
            maps_E.add(parts[0])
        if parts[1].endswith('_B'):
            maps_B.add(parts[1])
        if parts[1].endswith('_E'):
            maps_E.add(parts[1])
    maps_B = sorted(list(maps_B))
    maps_E = sorted(list(maps_E))
    param_stats = sorted(param_stats)
    plot_chisq_blocks(multicomp_class=multicomp_class,
                    used_maps=multicomp_class.sim_common_data['used_maps'],
                    observed_datas=observed_datas,
                    final_detection_dict=multicomp_class.final_detection_dict,
                    num_bins=len(multicomp_class.bin_num),
                    outpath = outpath)
    for spectra_type in ['EB', 'EE', 'BB']:
        plot_spectra_type(spectra_type, 
                      maps_E, 
                      maps_B, 
                      theory_dict, multicomp_class, observed_datas,
                      outpath, param_stats, chi_squared)
    

    return 

def plot_chisq_blocks(multicomp_class, used_maps, observed_datas, final_detection_dict, num_bins, outpath):
    """
    Computes and visualizes the block-wise chi-squared contributions between cross-spectra.

    Args:
        multicomp_class: Object containing the inverse covariance matrix.
        used_maps (list): List of cross-spectrum map names.
        observed_datas (dict): Dictionary of observed spectrum values.
        final_detection_dict (dict): Dictionary of best-fit theory values.
        num_bins (int): Number of bins per spectrum.
        outpath (str): Base filename for saving the output figure.

    Saves:
        PNG file `{outpath}_chisqmap.png` showing chi-squared contributions.
    """
    chisq_map = np.zeros((len(used_maps), len(used_maps)))
    chisq_sum = 0 
    for i, cross_map1 in enumerate(used_maps):
        for j, cross_map2 in enumerate(used_maps):
            block = multicomp_class.cov_inv[i*num_bins:(i+1)*num_bins,
                                        j*num_bins:(j+1)*num_bins]
            vector1 = observed_datas[cross_map1] - final_detection_dict[cross_map1]
            vector2 = observed_datas[cross_map2] - final_detection_dict[cross_map2]
            chisq = vector1.T @ block @ vector2
            chisq_map[i,j] = chisq
            chisq_sum += chisq

    # Create figure with proper dimensions and spacing
    fig = plt.figure(figsize=(10, 8))  # Increase figure size
    ax = fig.add_subplot(111)
    
    # Plot the main image
    vrange = np.std(chisq_map)
    im = ax.imshow(chisq_map, cmap='bwr', vmin=-vrange, vmax=vrange)
    
    # Set up ticks and labels
    ax.set_xticks(np.arange(len(used_maps)))
    ax.set_xticklabels(used_maps, 
                      rotation=45, 
                      ha='right',  # Horizontal alignment at right edge
                      rotation_mode='anchor')  # Keep text anchored
    
    ax.set_yticks(np.arange(len(used_maps)))
    ax.set_yticklabels(used_maps)
    
    # Add colorbar with padding
    cbar = fig.colorbar(im, ax=ax, pad=0.02)
    cbar.ax.tick_params(labelsize=8)  # Adjust colorbar tick size if needed
    
    # Adjust layout to prevent cutting off labels
    plt.title("Chisq: " + str(chisq_sum))
    plt.tight_layout(pad=2.0)  # Increase padding around plot
    fig.subplots_adjust(bottom=0.25, left=0.25)  # Adjust these values based on label length
    
    # Save with high resolution and bounding box
    print("Saving: " + outpath + '_chisqmap.png')
    plt.savefig(outpath + '_chisqmap.png', 
               dpi=300, 
               bbox_inches='tight')
    plt.close(fig)



# Calculate safe plot ranges (expand by 20% beyond data range)
def get_safe_ranges(samples_list, padding_factor=0.2):
    """
    Computes plotting ranges for triangle plots by padding param ranges.

    Args:
        samples_list (list of getdist.MCSamples): List of samples objects.
        padding_factor (float): Fraction by which to expand min/max range.

    Returns:
        dict: Mapping of parameter names to (min, max) plot ranges.
    """
    all_samples = np.vstack([s.samples for s in samples_list])
    ranges = {}
    for i, name in enumerate(samples_list[0].getParamNames().names):
        if('chi2' in name.name):
            continue
        vals = all_samples[:, i]
        span = vals.max() - vals.min()
        ranges[name.name] = (
            vals.min() - padding_factor * span,
            vals.max() + padding_factor * span
        )
    return ranges

def plot_triangle(root, replace_dict={}):
    """
    Plots a GetDist triangle plot and summarizes parameter mean ± std.

    Args:
        root (str): Path root to the MCMC chain file.
        replace_dict (dict): Dictionary of parameter names to override mean values.

    Returns:
        tuple: (param_names, means, mean_std_strings)

    Saves:
        PNG file `{root}_triangle_plot.png` with the triangle plot.
    """
    # Load MCMC samples from the specified root
    samples = loadMCSamples(root)
    print([name.name for name in samples.getParamNames().names])
    
    param_names_all = [name.name for name in samples.getParamNames().names
                   if ('chi2' not in name.name and
                       'weight' not in name.name and
                       'minuslogprior' not in name.name)]
    label_dict={
        'gMpl':  r'$g/M_\mathrm{pl}^{-1}$',
        'alpha_BK18_220': r'$\alpha_\mathrm{220}$',
        'alpha_BK18_B95e': r'$\alpha_\mathrm{B95}$',
        'alpha_BK18_K95': r'$\alpha_\mathrm{K95}$',
        'alpha_BK18_150': r'$\alpha_\mathrm{150}$',
    }
    plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    'font.size':18,
    'axes.labelsize':24,
    "font.serif": ["Computer Modern Roman"],
    "axes.unicode_minus": False,
    })
    include_params = ['gMpl', 'alpha_BK18_220', 
                    'alpha_BK18_B95e', 'alpha_BK18_K95',
                    'alpha_BK18_150']
    param_names = [p for p in param_names_all if p in include_params]

    # Compute mean, std, chi2, and prepare labels
    mean_std_strings = []
    means = []
    for param in param_names:
        mean = samples.mean(param)
        if param in replace_dict:
            mean = replace_dict[param]
        std = samples.std(param)
        chisq = samples.mean('chi2')
        mean_std_strings.append(rf"${param}$: ${mean:.2f} \pm {std:.2f}$")
        means.append(mean)

    # Apply parameter labels
    for name in samples.getParamNames().names:
        if name.name in label_dict:
            name.label = label_dict[name.name]

    # Create the triangle plot
    fig = plt.figure()
    g = plots.get_subplot_plotter()
    g.settings.axes_fontsize = 16
    g.settings.lab_fontsize = 20
    g.settings.title_limit_fontsize = 16
    g.settings.legend_fontsize = 14
    g.settings.tight_layout = True

    g.triangle_plot(samples, param_names, filled=True)

    # Get the best-fit (min chi²) point
    chi2_vals = samples.samples[:, samples.paramNames.list().index('chi2')]
    best_fit_index = np.argmin(chi2_vals)
    best_fit_point = samples.samples[best_fit_index, :]

    # Add red mean dots and green best-fit dot
    for i, pi in enumerate(param_names):
        for j, pj in enumerate(param_names):
            if j > i:
                ax = g.subplots[j, i]
                # Red dot: parameter means
                ax.plot(means[i], means[j], 'o', color='red', markersize=6, label='Mean of Posteriors')
                # Green dot: min chi² point
                x_idx = samples.paramNames.list().index(pi)
                y_idx = samples.paramNames.list().index(pj)
                ax.plot(best_fit_point[x_idx], best_fit_point[y_idx], 'o', color='lime', markersize=6, label='Min $\chi^2$ Point in CHain')
    ax_for_legend = g.subplots[1, 0]  # or any subplot where you have the dots
    handles, labels = ax_for_legend.get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper right', bbox_to_anchor=(1.15, 0.9), fontsize=12, frameon=False)

    # Adjust layout to make room for the legend on the right
    plt.subplots_adjust(right=0.85)  # leave 15% space on right for legend

    # Add LaTeX-formatted title in top-right
    title_ax = g.subplots[0, len(param_names) - 1]
    title_text = "\n".join(mean_std_strings)
    #title_ax.text(1.05, 1.05, title_text, transform=title_ax.transAxes,
    #              ha='left', va='top', fontsize=14, family='serif', usetex=True)
    # Save and show
    plt.savefig(f"{root}_triangle_plot.png", bbox_inches='tight')
    print(f"Triangle plot saved as {root}_triangle_plot.png")
    plt.close(fig)
    return param_names_all, means, mean_std_strs



def load_summary_csv(chains_path):
    """
    Loads or regenerates a summary CSV of MCMC stats (mean, std, minchi2).

    Args:
        chains_path (str): Path to chain directory or chain file.

    Returns:
        tuple: (means_df, stds_df, minchisq_df, param_names)
    """
    """Load summary CSV with mean, std, and minchi2 columns."""

    base_dir = os.path.dirname(chains_path)
    base_name = os.path.basename(base_dir)
    csv_file = os.path.join(base_dir, base_name + "_summary.csv")

    if not os.path.exists(csv_file):
        ms.process_single_directory(base_dir)

    df = pd.read_csv(csv_file)

    # Reprocess if minchi2 missing
    min_cols = [col for col in df.columns if col.endswith('_minchi2')]
    if not min_cols:
        print(f"Missing _minchi2 columns in {csv_file}, reprocessing...")
        ms.process_single_directory(base_dir)
        df = pd.read_csv(csv_file)
        min_cols = [col for col in df.columns if col.endswith('_minchi2')]

    mean_cols = [col for col in df.columns if col.endswith('_mean')]
    std_cols = [col for col in df.columns if col.endswith('_std')]
    param_names = [col.replace('_mean', '') for col in mean_cols]

    means_df = df[mean_cols].copy()
    means_df.columns = param_names

    stds_df = df[std_cols].copy()
    stds_df.columns = [col.replace('_std', '') for col in std_cols]

    minchisq_df = df[min_cols].copy()
    minchisq_df.columns = param_names

    return means_df, stds_df, minchisq_df, param_names

def compute_corner_ranges(df, param_names, percentile_clip=(0, 100)):
    """
    Computes plot ranges for parameters using percentile clipping.

    Args:
        df (pandas.DataFrame): DataFrame containing samples.
        param_names (list of str): Names of parameters to clip.
        percentile_clip (tuple): (lower, upper) percentile bounds.

    Returns:
        list: List of (min, max) tuples for each parameter.
    """
    return [np.percentile(df[param], percentile_clip) for param in param_names]

def make_titles(means_df, stds_df, minchisq_df, param_names):
    """
    Creates formatted plot titles with mean ± std and minchi2 ± spread.

    Args:
        means_df (DataFrame): Mean values for each param across sims.
        stds_df (DataFrame): Standard deviations.
        minchisq_df (DataFrame): Min-chi2 values for each sim.
        param_names (list): List of parameter names.

    Returns:
        list of str: Titles for each parameter.
    """
    titles = []
    for p in param_names:
        mean_val = means_df[p].mean()
        std_val = stds_df[p].mean() if p in stds_df.columns else np.nan
        minchi2_mean = minchisq_df[p].mean()
        minchi2_std = minchisq_df[p].std()  # spread of the minchi2 peaks
        title = (f"Peak ± spread:\n{minchi2_mean:.3f} ± {minchi2_std:.3f}\n"
                 f"Mean ± Std:\n{mean_val:.3f} ± {std_val:.3f}\n")
        titles.append(title)
    return titles

def plot_logp_surface(model, param1, param2, range1, range2, 
                    fixed_params, grid_size=100):
    """
    Generates and visualizes a 2D log-likelihood surface over two parameters.

    Args:
        model: An object with a `logp(**params)` method.
        param1 (str): First parameter name to scan.
        param2 (str): Second parameter name to scan.
        range1 (tuple): (min, max) range for param1.
        range2 (tuple): (min, max) range for param2.
        fixed_params (dict): Dictionary of all other parameters held fixed.
        grid_size (int): Resolution of the parameter grid.

    Saves:
        A PNG figure named `2dlikemap_{param1}_{param2}.png`.
    """
    # Create grid
    print('Making 2D likelihood plot')
    p1_vals = np.linspace(*range1, grid_size)
    p2_vals = np.linspace(*range2, grid_size)
    logp_vals = np.zeros((grid_size, grid_size))

    for i, p1 in enumerate(p1_vals):
        for j, p2 in enumerate(p2_vals):
            params = fixed_params.copy()
            params[param1] = p1
            params[param2] = p2
            
            logp_vals[j, i] = model.logp(**params)  # Note: rows = y, cols = x
    print(logp_vals)

    # Plot
    plt.figure(figsize=(8, 6))
    vmin = -np.max(logp_vals)
    vmax = vmin+2#40#-np.min(logp_vals)
    levels = np.linspace(vmin, vmax, 51)
    print(levels)
    norm = BoundaryNorm(boundaries=levels, ncolors=256, extend='both')
    cp = plt.contourf(p1_vals, p2_vals, -logp_vals, levels=levels, 
                norm=norm, cmap='viridis',extend='both')
                    

    cbar = plt.colorbar(cp, label='-log-likelihood')
    plt.xlabel(param1 + ' [deg]')
    plt.ylabel(param2 + ' [deg]')
    plt.title(f'Minus Log-likelihood surface: {param1} vs {param2}')
    plt.tight_layout()
    filename = '2dlikemap_' + param1 + '_' + param2 + '.png'
    print('Saving:' + filename)
    plt.savefig(filename)
    plt.show()

def plot_sim_peaks(chains_path, single_sim, sim_nums=None, single_path=None, 
                   percentile_clip=(0, 100)):
    """
    Create a corner plot visualizing simulation parameter distributions.

    Plots three sets of contours/histograms overlaid:
      - Means and std deviations from all realizations (in red)
      - Parameter values corresponding to minimum chi-squared per realization (in green)
      - Full MCMC chain samples from a single simulation (in blue)

    Args:
        chains_path (str): Path pattern to the directory containing summary CSVs
                           with 'mean', 'std', and 'minchi2' columns.
        single_sim (int): Simulation number used to select the single chain file
                          (used to replace "XXX" in path if single_path not provided).
        sim_nums (list or None): Optional list of simulation numbers to plot.
        single_path (str or None): Explicit path to a single simulation chain file
                                   (overrides chains_path replacement).
        percentile_clip (tuple): Percentile range (min, max) used to clip parameter ranges
                                for plotting (default: full range (0, 100)).

    Prints:
        Dictionary of best-fit parameter values (minimum chi-squared) for the simulations.

    Saves:
        Corner plot PNG file named as `{base_chains_path}{single_sim}_summary.png`.

    Notes:
        Requires functions `load_summary_csv`, `compute_corner_ranges`, and `make_titles`
        which should return appropriate DataFrames and titles for corner plot.
    """
    plt.rcParams.update({
        "text.usetex": True,
        "font.size": 24,
        "font.family": "serif",
        "font.serif": ["Computer Modern"],
        "axes.labelsize": 24,
        "xtick.labelsize": 18,
        "ytick.labelsize": 18,
        "legend.fontsize": 18,
    })

    means_df, stds_df, minchisq_df, param_names = load_summary_csv(chains_path)
    selected_params = None
    
    
    if(True): 
        selected_params = [
            'gMpl',
            'alpha_BK18_B95e',
            'alpha_BK18_220',
            'A_lens'
        ]
    if selected_params is not None:
        param_names = [p for p in param_names if p in selected_params]
    latex_labels = {
        # Polarization angles / rotation parameters
        'alpha_BK18_B95e':       r'$\alpha_{\mathrm{B95}}$',
        'alpha_P353e':        r'$\alpha_{\mathrm{P353}}$',
        'alpha_BK18_220':    r'$\alpha_{\mathrm{220}}$',
        'alpha_BK18_150':    r'$\alpha_{\mathrm{150}}$',
        'alpha_BK18_K95':    r'$\alpha_{\mathrm{K95}}$',
        'alpha_CMB':         r'$\beta_{\mathrm{CMB}}$',  # CMB-frame rotation angle

        # Dust amplitudes
        'A_dust_EE':         r'$A_\mathrm{dust}^{EE}$',
        'A_dust_BB':         r'$A_\mathrm{dust}^{BB}$',
        'A_dust_EB':         r'$A_\mathrm{dust}^{EB}$',
        'alpha_dust_EE':         r'$\alpha_\mathrm{dust}^{EE}$',
        'alpha_dust_BB':         r'$\alpha_\mathrm{dust}^{BB}$',
        'alpha_dust_EB':         r'$\alpha_\mathrm{dust}^{EB}$',

        # Dust spectral indices
        'beta_dust':         r'$\beta_\mathrm{dust}$',

        # Synchrotron amplitudes
        'A_sync_EE':         r'$A_\mathrm{sync}^{EE}$',
        'A_sync_BB':         r'$A_\mathrm{sync}^{BB}$',
        'A_sync_EB':         r'$A_\mathrm{sync}^{EB}$',

        # Synchrotron spectral indices
        'beta_sync':         r'$\beta_\mathrm{sync}$',

        # Multiplicative parameters
        'gMpl':              r'$g / M_\mathrm{pl}^{-1}$',
        'A_lens':            r'$A_{\mathrm{lens}}$',
        'angle_diff':        r'$\Delta\beta_{\ell_b}$',
        'log10_fede':        r'$\log_{10}(f_{\mathrm{EDE}})$',
        'log10z_c':          r'$\log_{10}(z_c)$',
        'theta_i':           r'$\theta_i$',
    }
    labels = [latex_labels.get(p, p) for p in param_names]

    

    ranges = compute_corner_ranges(means_df, param_names, percentile_clip)

    titles = make_titles(means_df, stds_df, minchisq_df, param_names)

    # Plot means (red)
    fig = corner.corner(means_df[param_names],
                        labels=labels,
                        show_titles=False,
                        titles=titles,
                        title_kwargs={"fontsize": 9, "multialignment": "center"},
                        hist_kwargs={'color': 'red', 'density': True},
                        contour_kwargs={'colors': 'red'},
                        range=ranges, 
                        return_fig=True)
    truth_vals = [0,0,0,1, 1]
    #draw_zero_lines_on_corner(fig.axes, param_names,  truth_vals, color='black')
    # Overlay minchi2 (green)
    if(False):
        corner.corner(minchisq_df[param_names],
                  labels=param_names,
                  show_titles=False,
                  hist_kwargs={'color': 'green', 'density': True},
                  contour_kwargs={'colors': 'green'},
                  fig=fig)

    mean_params = means_df[param_names].mean().to_dict()
    std_params = stds_df[param_names].mean().to_dict()

    print("Mean of parameter peaks (across all simulations):")
    for k, v in mean_params.items():
        print(f"  {k}: {v:.4g}")

    print("\nSpread (std) of parameter peaks (across all simulations):")
    for k, v in std_params.items():
        print(f"  {k}: {v:.4g}")

    # Overlay single chain (blue)
    single_chain_path = single_path or chains_path.replace("XXX", f"{single_sim:03d}")
    try:
        with open(single_chain_path, 'r') as f:
            header_line = f.readline().strip().replace('#', '')
        param_header = header_line.split()
        df_chain = pd.read_csv(single_chain_path, delim_whitespace=True, comment='#')
        df_chain.columns = param_header
        df_chain = df_chain[param_names]  # keep only params
        fig = corner.corner(df_chain,
                      labels=labels,
                      show_titles=False,
                      hist_kwargs={'color': 'blue', 'density': True},
                      contour_kwargs={'colors': 'blue'},
                      #truths = df_chain.mean(),
                      #truth_color='red',
                      #truth_kwargs={'linewidth': 0.5, 'ls':'--'},
                      fig=fig,
                      return_fig=True)
        draw_zero_lines_on_corner(fig.axes, param_names, df_chain.mean(), color='red')
    except Exception as e:
        print(f"Could not overlay single sim: {e}")

    # Save and show
    outpath = chains_path.split("XXX")[0] + f"{single_sim}_summary.png"
    n_chains = len(means_df)
    title =  f"(N={n_chains} sims) Sim peaks (red) and Sim {single_sim} single chain (blue)"
    #plt.suptitle(title)
    plt.savefig(outpath, bbox_inches='tight')
    print(f"Saved to {outpath}")
    return mean_params, std_params

def draw_zero_lines_on_corner(flat_axes, param_names, truth_vals, color):
    ndim = len(param_names)
    axes = np.array(flat_axes).reshape((ndim, ndim))
    
    for i in range(ndim):
        for j in range(i + 1):
            ax = axes[i, j]
            line_valx = truth_vals[i]
            line_valy = truth_vals[j]
            if i == j:
                ax.axvline(line_valx, color=color, lw=0.5, ls='--')
            else:
                ax.axvline(line_valy, color=color, lw=0.5, ls='--')
                ax.axhline(line_valx, color=color, lw=0.5, ls='--')
            

def plot_step_example(multicomp_class):
    angle_const = 0.39
    angle_b = 0.27
    angle_diff = 0.52
    lbreak = 370
    min_len = 500
    eb = multicomp_class.dl_theory['EB_EDE'][:min_len]
    ee = multicomp_class.dl_theory['EE'][:min_len]

    ratio = eb / ee
    arcsin_ratio = np.arcsin(2 * ratio) / 4 * 180 / np.pi
    L_BIN_CENTERS = np.array([
        10.0000, 37.5000, 72.5000, 107.5000, 142.5000, 177.5000, 
        212.5000, 247.5000, 282.5000, 317.5000, 352.5000, 387.5000, 
        422.5000, 457.5000, 492.5000, 527.5000, 562.5000
    ])
    ell = L_BIN_CENTERS[1:15]
    
    # Get theory curves
    params_values = {'A_lens': 1, 'alpha_BK18_B95e': angle_const}
    vec_flat = multicomp_class.theory(params_values)

    params_values = {
        'A_lens': 1,
        'alpha_BK18_B95e': angle_b,
        'angle_diff': angle_diff
    }
    vec_diff = multicomp_class.theory_diff(params_values)

    # Get observed data + variance
    real_data = multicomp_class.binned_dl_observed_dict
    used_map = 'BK18_B95e_BxBK18_B95e_E'
    map_index = multicomp_class.used_maps.index(used_map)
    num_bin = len(real_data[used_map])
    data_vals = real_data[used_map]
    
    covar_mat = multicomp_class.sim_common_data['covmat']
    var = np.diag(covar_mat)
    data_err = np.sqrt(var)

    # Plotting
    fig, axs = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

    # Define matching colors
    flat_color = '#1f77b4'  # blue (flat)
    step_color = '#d62728'  # red (step)

    # --- Top plot: theory + real data ---
    axs[1].plot(ell, vec_flat, '--', color=flat_color, label='flat rotation')
    axs[1].plot(ell, vec_flat, 'o', color=flat_color)
    axs[1].plot(ell, vec_diff, '-', color=step_color, label='step rotation')
    axs[1].plot(ell, vec_diff, 'o', color=step_color)
    axs[1].errorbar(ell, data_vals, yerr=data_err, fmt='-o', color='black', label='observed data')

    axs[1].set_ylabel(r'$D_\ell^{EB}$ [$\mu\mathrm{K}^2$]')
    axs[1].legend()
    axs[1].set_title(r'Single Realization Comparing Flat vs $\beta(\ell)$ Step Function')

    # --- Bottom plot: beta(ell) step function ---
    axs[0].hlines(angle_b, ell[0], lbreak, colors=step_color, linewidth=2, label=r'Step function $\beta(\ell)$')
    axs[0].hlines(angle_b + angle_diff, lbreak, ell[-1], colors=step_color, linewidth=2)
    axs[0].axvline(lbreak, color='gray', linestyle=':', label=f'$\ell_{{\\mathrm{{break}}}} = {lbreak}$')
    print(arcsin_ratio)
    axs[0].plot(arcsin_ratio, label=r'True effective $\beta(\ell)$', color='black')
    # Flat reference line (blue, dashed)
    axs[0].hlines(angle_const, ell[0], ell[-1], colors=flat_color, linestyle='--', alpha=0.6, label='Flat rotation')

    axs[0].set_ylabel(r'$\beta(\ell)$ [deg]')
    axs[0].set_xlabel(r'Multipole $\ell$')
    axs[0].set_title(r'Step Function for $\beta(\ell)$')
    axs[0].legend()

    plt.tight_layout()
    filename = 'test.png'
    print('Saving:', filename)
    plt.savefig(filename)
def read_sampler(filepath):
    """
    Reads MCMC sampler data from a file and returns it as a DataFrame.

    Parameters:
    -----------
    filepath : str
        Path to the file containing the sampler data.

    Returns:
    --------
    df : pandas.DataFrame
        DataFrame containing the MCMC sampler data with appropriate column names.
    """
    df = pd.read_csv(filepath, delim_whitespace=True, comment='#', header=None)
        
    # Assign column names based on the data structure
    df.columns = ['weight', 'minuslogpost', 'gMpl', 'aplusb_b95', 'aplusb_b95ext', 
                    'aplusb_k95', 'aplusb_150', 'aplusb_220', 'minuslogprior', 
                    'minuslogprior__0', 'chi2', 'chi2__power']
    return df




def plot_chisq_hist(sim_results_file):
    """
    Plots a histogram of chi-squared values from the simulation results file.

    Parameters:
    -----------
    sim_results_file : str
        Path to the simulation results file.

    Displays:
    ---------
    Histogram plot of chi-squared values with a vertical line for a specific chi-squared value.
    """
    df = pd.read_csv(sim_results_file)
    plt.figure(figsize=(8, 6))
    plt.hist(df['chisq'], bins=30, color='blue', edgecolor='black')
    chisq=162
    plt.axvline(x=chisq, color='red', linestyle='--', 
                linewidth=2, label='Real chisq at ' + str(chisq))
    plt.legend()
    plt.title('Histogram of Chi-squared Values')
    plt.xlabel('Chi-squared')
    plt.ylabel('Frequency')
    plt.grid(True)
    plt.show()



