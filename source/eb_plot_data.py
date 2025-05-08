
import numpy as np
import pandas as pd
import corner

import matplotlib
import os
if "SLURM_JOB_ID" in os.environ:
    matplotlib.use('Agg')  # headless mode
else:
    matplotlib.use('TkAgg')  # if running with GUI (e.g., locally)
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

from getdist.mcsamples import MCSamplesFromCobaya
from getdist.mcsamples import loadMCSamples
import getdist.plots as gdplt
from matplotlib.gridspec import GridSpec
print("Loading Plotting  Modules")
from getdist import plots, MCSamples
from getdist.mcsamples import loadMCSamples



import bicep_data_consts as bdc
MAP_FREQS = bdc.MAP_FREQS

def plot_sample_fit(observed_data, var, mapi, rotated_dict, ebe_dict, tot_dict, params_values):
    plt.errorbar( 
                    x = range(len(observed_data)),
                    y=(observed_data), 
                    yerr = np.sqrt(var),
    )
    ede_cont = ebe_dict[mapi]
    plt.plot(ede_cont, label='EDE contribution')
    parts = mapi.split('x')
    if(parts[0].endswith('_B')):
        ind = 0
    else:
        ind = 1
    result = parts[ind][:-2] + '_E'
    result = result + 'x' + result
    rot_cont = rotated_dict[mapi]
    plt.plot(rot_cont, label = 'Polarization Rotation')
    plt.plot(tot_dict[mapi], label='Both')
    # Convert dictionary to string with a newline after every two keys
    result_lines = []
    for i, (key, value) in enumerate(params_values.items(), start=1):
        result_lines.append(f"{key}: {value}")
        # Add a newline after every two keys
        if i % 2 == 0:
            result_lines.append("\n")  # Blank line for separation

    # Join the lines to form the final string
    dict_as_string = ", ".join(result_lines)
    #dict_as_string = "\n".join(f"{key}: {value}" for key, value in params_values.items())
    plt.title(mapi + '\n ' + str(dict_as_string))
    plt.legend()
    plt.tight_layout()
    plt.show()

def plot_covar_matrix(mat, used_maps=None, title='Log of covar matrix',
                        show_plot=False):
    
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

def plot_spectra_type(spectra_type, maps_E, maps_B, theory_dict, multicomp_class, observed_datas,
                      outpath, param_stats):
    num_columns = len(maps_B)  # Unique maps for columns
    num_rows = len(maps_E)      # Unique maps for rows
    # Create subplots
    fig, axes = plt.subplots(num_rows, num_columns, 
                    figsize=(num_columns * 4, num_rows * 4))

    try:
        axes = axes.flatten()  # Flatten axes array for easy indexing
    except AttributeError:
        print("Only one axis!")
        axes = [axes]
    keys = list(theory_dict.keys())
    # Plot each spectrum
    for idx, key in enumerate(keys):
        observed_data = observed_datas[key]
        best_fit_data = theory_dict[key]
        #print(key)
        #print(observed_data - best_fit_data) 
        # Split key to find row and column indices
        spec_type = determine_spectrum_type(key)
        parts = key.split('x')
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
                            x = range(len(observed_data)),
                            y=(observed_data), 
                            yerr = np.sqrt(var),
                            label='Observed', color='blue')
        # Plotting best fit data
        axes[axes_index].plot(best_fit_data, label='Best Fit', color='red')

        axes[axes_index].set_title(key)
        axes[axes_index].legend()
    for row_idx, map_E in enumerate(maps_E):
        angle = f"alpha_{map_E}"
        axes[row_idx].text(
            0.05, 1.4,  # X and Y position (top-left corner)
            param_stats[row_idx],  # The parameter stats
            transform=axes[row_idx].transAxes,  # Use axes coordinates
            fontsize=10, color='black',
            verticalalignment='top'
        )
    plt.tight_layout(pad=2)
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
    #used_maps = multicomp_class.used_maps
    observed_datas = multicomp_class.binned_dl_observed_dict
    param_values = {param_names[i]:param_bestfit[i] 
                            for i in range(len(param_names))}
    theory_vec=multicomp_class.theory(param_values, override_maps=override_maps)
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
    for spectra_type in ['EE', 'EB', 'BB']:
        plot_spectra_type(spectra_type, 
                      maps_E, 
                      maps_B, 
                      theory_dict, multicomp_class, observed_datas,
                      outpath, param_stats)
    

    return 

def plot_best_crossfit(eb_like_cls, outpath, used_maps, param_names, 
                        param_bestfit, param_stats, signal_params={}):
    
    used_maps = eb_like_cls.used_maps
    #np.savetxt('150220_invcovar.txt', eb_like_cls.cov_inv, delimiter=',')
    observed_datas = eb_like_cls.binned_dl_observed_dict
    theory_spectra = eb_like_cls.dl_theory
    param_values = {param_names[i]:param_bestfit[i] 
                            for i in range(len(param_names))}
    #param_values['alpha_BK18_220'] = 1.2
    #param_values['alpha_BK18_150'] = -0.5
    theory_vec=eb_like_cls.theory(param_values, 
                    theory_spectra, eb_like_cls.used_maps)
    observed_vec = eb_like_cls.binned_dl_observed_vec
    res = theory_vec - observed_vec
    chisq_mat = np.multiply(eb_like_cls.cov_inv, np.outer(res, res))
    chisq_tot = 'chisq:' + str(np.sum(chisq_mat))
    #plot_covar_matrix(chisq_mat, used_maps=used_maps, title=chisq_tot )

    rotated_dict = eb_like_cls.tot_dict
    #print(rotated_dict)
    keys = list(rotated_dict.keys())
    #print(keys)
    # Get block chisqs
    num_bins = len(rotated_dict[used_maps[0]])
    chisq_map = np.zeros((len(used_maps), len(used_maps)))
    print('Num bins:' + str(num_bins))
    for i, cross_map1 in enumerate(used_maps):
        for j, cross_map2 in enumerate(used_maps):
            block = eb_like_cls.cov_inv[i*num_bins:(i+1)*num_bins,
                                        j*num_bins:(j+1)*num_bins]
            vector1 = observed_datas[cross_map1] - rotated_dict[cross_map1]
            vector2 = observed_datas[cross_map2] - rotated_dict[cross_map2]
            chisq = vector1.T @ block @ vector2
            chisq_map[i,j] = chisq
            
    plt.figure()
    print('Chisq:' + str(np.sum(chisq_map)))
    vrange = np.std(chisq_map)
    plt.imshow(chisq_map, cmap='bwr', vmin=-vrange, vmax=vrange)
    plt.colorbar()
    plt.xticks(np.arange(len(used_maps)), used_maps, rotation = 45)
    plt.yticks(np.arange(len(used_maps)), used_maps)
    print("Saving: " +outpath + '_chisqmap.png')
    plt.savefig(outpath + '_chisqmap.png')
    #plt.show()
    # Initialize lists to store unique maps ending with _E and _B
    maps_B = set()
    maps_E = set()

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
    num_columns = len(maps_B)  # Unique maps for columns
    num_rows = len(maps_E)      # Unique maps for rows
    # Create subplots
    fig, axes = plt.subplots(num_rows, num_columns, 
                    figsize=(num_columns * 4, num_rows * 4))

    try:
        axes = axes.flatten()  # Flatten axes array for easy indexing
    except AttributeError:
        print("Only one axis!")
        axes = [axes]
    # Plot each spectrum
    for idx, key in enumerate(keys):
        observed_data = observed_datas[key]
        best_fit_data = rotated_dict[key]
        #print(key)
        #print(observed_data - best_fit_data) 
        # Split key to find row and column indices
        parts = key.split('x')
        row_idx = (maps_E).index(parts[0]) if parts[0].endswith('_E') else (maps_E).index(parts[1])
        col_idx = (maps_B).index(parts[0]) if parts[0].endswith('_B') else (maps_B).index(parts[1])
        map_index = eb_like_cls.used_maps.index(key)
        num_bin = len(observed_data)
        covar_mat = eb_like_cls.filtered_covmat
        var = np.diag(covar_mat)[map_index*num_bin:num_bin*(map_index+1)]
        # Plotting observed data
        axes_index = row_idx * num_columns + col_idx
        #print(observed_data)
        
        axes[axes_index].errorbar(
                            x = range(len(observed_data)),
                            y=(observed_data), 
                            yerr = np.sqrt(var),
                            label='Observed', color='blue')
        # Plotting best fit data
        axes[axes_index].plot(best_fit_data, label='Best Fit', color='red')

        axes[axes_index].set_title(key)
        axes[axes_index].legend()
    for row_idx, map_E in enumerate(maps_E):
        angle = f"alpha_{map_E}"
        axes[row_idx].text(
            0.05, 1.4,  # X and Y position (top-left corner)
            param_stats[row_idx],  # The parameter stats
            transform=axes[row_idx].transAxes,  # Use axes coordinates
            fontsize=10, color='black',
            verticalalignment='top'
        )
    plt.tight_layout(pad=2)
    plt.savefig(outpath + '_bestfit.png')
    print("Saving: " +outpath + '_bestfit.png')
    return 



def plot_triangle(root, replace_dict={}):
    # Load MCMC samples from the specified root
    samples = loadMCSamples(root)
    print([name.name for name in samples.getParamNames().names])
    
    param_names = [name.name for name in samples.getParamNames().names
                   if ('chi2' not in name.name and
                       'weight' not in name.name and
                       'betadust' not in name.name and
                       'betasync' not in name.name and
                       'minuslogprior' not in name.name)]
    
    # Get the mean and std of the parameters for titles
    mean_std_strings = []
    means = []
    count = 0
    for param in param_names:
        
        mean = samples.mean(param)
        if(param in replace_dict):
            mean = replace_dict[param]
        chisq = samples.mean('chi2')
        std = samples.std(param)
        if(count == 0):
            mean_std_strings.append(f"{param}: {mean:.2f} ± {std:.2f} chisq={chisq:.2f}")
            count += 1
        else:
            mean_std_strings.append(f"{param}: {mean:.2f} ± {std:.2f}")

        means.append(mean)

    # Create a triangle plot with all variables
    fig = plt.figure()
    g = plots.get_subplot_plotter()
    g.triangle_plot(samples, param_names, filled=True)

    # Add the mean and std to the plot title
    plt.suptitle("\n".join(mean_std_strings), fontsize=10)
    plt.tight_layout()
    # Save the plot
    plt.savefig(f"{root}_triangle_plot.png")
    print(f"Triangle plot saved as {root}_triangle_plot.png")
    #plt.show()
    plt.close(fig)
    return param_names, means, mean_std_strings






def plot_eigenvalues_eigenvectors(matrix):
    """
    Plots the eigenvalues and eigenvectors of a given square matrix.

    Parameters:
    - matrix (np.ndarray): A square matrix for which to compute and plot eigenvalues and eigenvectors.
    """
    # Compute eigenvalues and eigenvectors
    eigenvalues, eigenvectors = np.linalg.eig(matrix)

    # Create subplots
    plt.figure(figsize=(12, 6))

    # Subplot for eigenvalues
    plt.subplot(1, 2, 1)
    plt.bar(range(len(eigenvalues)), np.log(eigenvalues), color='b')
    plt.title('Eigenvalues')
    plt.xlabel('Index')
    plt.ylabel('Ln Eigenvalue')

    # Subplot for eigenvectors
    plt.subplot(1, 2, 2)
    plt.imshow(np.log(np.abs(eigenvectors)))
    plt.colorbar()
    plt.title('Ln abs Eigenvectors')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.grid()
    plt.legend()

    plt.tight_layout()
    plt.show()
    return

def plot_sim_peaks(chains_path, single_sim, sim_nums, single_path=None, 
                   use_median=True, percentile_clip=(2, 98)):
    modes_dict = {}
    single_df = None
    simcount = 0

    for i in range(1, sim_nums + 1):
        file_path = chains_path.replace('XXX', f'{i:03d}')
        print('loading:' + str(file_path))
        # Read the first line to get the correct header
        try:
            with open(file_path, 'r') as f:
                first_line = f.readline().strip()  # Read the first line
                # Remove the '#' and split to get the correct column names
                corrected_header = first_line.replace('#', '').split()
        except FileNotFoundError:
            print("Skipping " + file_path)
            continue

        chain_df = pd.read_csv(file_path, delim_whitespace=True, comment='#')
        chain_df.columns = corrected_header

        for column in chain_df.columns:
            if column not in modes_dict:
                modes_dict[column] = []
            value = np.median(chain_df[column]) if use_median else np.mean(chain_df[column])
            modes_dict[column].append(value)

        simcount += 1

    modes_df = pd.DataFrame.from_dict(modes_dict)
    default_cols = ['#', 'weight', 'minuslogpost', 'minuslogprior',
                    'minuslogprior__0', 'chi2', 'chi2__my_likelihood']
    param_names = [col for col in modes_df.columns if col not in default_cols]
    print("Parameter names for plotting:", param_names)

    # Compute ranges for corner plot from red summary (modes_df)
    ranges = []
    for param in param_names:
        low, high = np.percentile(modes_df[param], percentile_clip)
        ranges.append((low, high))

    for i in range(single_sim, single_sim + 1):
        fig = corner.corner(modes_df[param_names],
                            labels=param_names,
                            show_titles=True,
                            title_kwargs={"fontsize": 12},
                            hist_kwargs={'color': 'red', 'density': True},
                            contour_kwargs={'colors': 'red'},
                            range=ranges)

        file_path = single_path if single_path is not None else chains_path.replace('XXX', f'{i:03d}')
        single_df = pd.read_csv(file_path, delim_whitespace=True, comment='#')
        single_df.columns = corrected_header

        corner.corner(single_df[param_names],
                      labels=param_names,
                      show_titles=False,
                      hist_kwargs={'color': 'blue', 'density': True},
                      contour_kwargs={'colors': 'blue'},
                      fig=fig)

        supertitle = f'Sim {i} (blue) on top of {simcount} sims (red)'
        plt.suptitle(supertitle)
        outpath = chains_path.split('XXX')[0] + f'{i}_summary.png'
        plt.savefig(outpath)
        print('Saved to ' + outpath)
        plt.show()

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


def get_mcmc_results_to_df(results_file):
    """
    Reads MCMC results from a file and processes it into a DataFrame. If the file
    is a '.txt', it will remove '#' comments and save a cleaned version.

    Parameters:
    -----------
    results_file : str
        Path to the MCMC results file.

    Returns:
    --------
    df : pandas.DataFrame
        DataFrame containing the cleaned MCMC results.
    """
    print('Reading in: ' + results_file)
    if '.txt' in results_file:
        # Open the file and read its contents
        with open(results_file, 'r') as file:
            content = file.read()

        # Remove all '#' symbols from the file content
        cleaned_content = content.replace('#', '')

        # Create a new filename for the cleaned file
        cleaned_file = results_file + '.cleaned'

        # Write the cleaned content to the new file
        with open(cleaned_file, 'w') as file:
            file.write(cleaned_content)

        # Return the DataFrame from the cleaned file
        return pd.read_csv(cleaned_file, delim_whitespace=True, header=0)
    else:
        # Return the DataFrame directly if not a .txt file
        return pd.read_csv(results_file)
    

def scatter_sims(sim_results, mapname='BK18_B95ext'):
    """
    Plots scatter of gMpl vs aplusb with error bars for the given mapname.

    Parameters:
    -----------
    sim_results : str
        Path to the simulation results file.

    mapname : str, optional
        Name of the map to filter by (default is 'BK18_B95ext').

    Saves:
    ------
    Image of the scatter plot as 'dustsims_<mapname>.png'.
    """
    df = pd.read_csv(sim_results)
    map_results = df[df['map_name'] == mapname]
    plt.figure()
    plt.errorbar(map_results['gMpl'], map_results['aplusb'], 
                 xerr=map_results['gMpl_std'], yerr=map_results['aplusb_std'],
                 fmt='o', elinewidth=1, ecolor='red')
    plt.xlabel('gMpl')
    plt.ylabel('aplusb')
    plt.title('gMpl vs aplusb for ' + mapname)
    plt.savefig('dustsims_' + mapname + '.png')
    #plt.show()
    return 


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
def plot_corner(outfile, sim_results_file, real_results_file):
    

    df_sim = get_mcmc_results_to_df(sim_results_file)
    df_real = get_mcmc_results_to_df(real_results_file)
    param_names = ['gMpl', 'aplusb_b95', 'aplusb_k95', 'aplusb_150', 'aplusb_220', 'aplusb_b95ext']
    print(df_real.columns)
    data_sim = df_sim[param_names].values
    data_real = df_real[param_names].values
    # Plot the first corner plot
    print('first plot')
    fig = corner.corner(data_sim, labels=param_names, 
                        show_titles=True, title_fmt=".2f", plot_contours=True, color='red')
    print('second plot')
    print(df_real)
    # Overlay the second corner plot
    corner.corner(data_real, labels=param_names, 
                  show_titles=True, title_fmt=".2f", plot_contours=True, color='blue', fig=fig)

# Add legend
    plt.legend(['Aggregate Sim Dataset', 'Real Dataset'])
    title_str = ('Comparing: ' + sim_results_file.split('/')[-1] + 
                 ' and ' + real_results_file.split('/')[-1])
    plt.suptitle(title_str)
    plt.savefig(outfile)


def plot_cldl(l_bins, spectrum_dict,  output_plots, mapname,scale=100):
    """
    Plots the power spectrum Cls and Dls for EE, BB, and EB spectra.

    Parameters:
    -----------
    l_bins : ndarray
        Array of multipole moments (l) corresponding to the power spectra.

    spectrum_dict : dict
        Dictionary containing the binned spectra data (EE, BB, EB).

    output_plots : str
        Directory path to save the output plot images.

    mapname : str
        Name of the map or dataset being analyzed.

    scale : int, optional
        Scaling factor for the EB spectra.

    Saves:
    ------
    Two output plots (Dls and Cls) to the specified directory.
    """
    vars = np.diag(spectrum_dict['EB_var'+ '_' + mapname])
    plt.figure()
    #plt.plot(GLOBAL_VAR['EE'], label='CAMB theory')
    plt.plot(l_bins, spectrum_dict['EE_binned'+ '_' + mapname], label='Binned EE camb')
    plt.plot(l_bins, spectrum_dict['BB_binned'+ '_' + mapname], label='Binned BB camb')
    plt.plot(l_bins, spectrum_dict['EB_EDE'+ '_' + mapname]*scale, label='Binned EB EDE scaled by ' + str(scale))
    plt.errorbar(l_bins[:], spectrum_dict['EB_observed'+ '_' + mapname]*scale, yerr=np.sqrt(vars)*scale,
            label='C_EB bicep data scaled by ' + str(scale))
    #plt.ylim([-0.00001, 0.00002])
    plt.ylabel(r'$C_{\ell}^{EB}\cdot\ell(\ell+1)/(2\pi)$  [$\mu K^2$]')
    plt.xlabel(r'$\ell$')
    plt.legend()
    plt.title('Map: ' + str(mapname))
    outpath = output_plots + '/' + mapname + '_spectra_Dls.png'
    print('Saving to ' + outpath)
    plt.savefig(outpath) 
    plt.close()

    plt.figure()
    #plt.plot(GLOBAL_VAR['EE'], label='CAMB theory')
    d_to_c_conver = l_bins*(l_bins+1)/(2*np.pi)
    plt.plot(l_bins, spectrum_dict['EE_binned'+ '_' + mapname]/d_to_c_conver, label='Binned EE camb')
    plt.plot(l_bins, spectrum_dict['BB_binned'+ '_' + mapname]/d_to_c_conver, label='Binned BB camb')
    plt.plot(l_bins, spectrum_dict['EB_EDE'+ '_' + mapname]*scale/d_to_c_conver, label='Binned EB EDE scaled by ' + str(scale))
    plt.errorbar(l_bins[:], spectrum_dict['EB_observed'+ '_' + mapname]*scale/d_to_c_conver, yerr=np.sqrt(vars)*scale/d_to_c_conver,
            label='C_EB bicep data scaled by ' + str(scale))
    #plt.ylim([-0.00001, 0.00002])
    plt.ylabel(r'$C_{\ell}^{EB}$  [$\mu K^2$]')
    plt.xlabel(r'$\ell$')
    plt.legend()
    plt.title('Map: ' + str(mapname))
    outpath = output_plots + '/' + mapname + '_spectra_Cls.png'
    print('Saving to ' + outpath)
    plt.savefig(outpath) 
    plt.close()

def plot_info(variables=None, info=None, sampler=None, outfile=None, file_root=None, mapname=None, output_plots='output_plots'):
    """
    Generates and displays or saves a triangle plot of posterior distributions for specified variables.

    Parameters:
    -----------
    variables : list of str, optional
        A list of variable names to include in the plot. If `None`, all available variables are plotted.

    info : dict, optional
        A YAML dictionary containing the configuration information from a Cobaya run.

    sampler : cobaya.run, optional
        The sampling output result from a Cobaya run.

    outfile : str, optional
        Path to save the generated plot. If `None`, the plot is displayed.

    file_root : str, optional
        The root filename (without extension) from which to load MCMC samples.

    Raises:
    -------
    ValueError
        If neither `info`/`sampler` nor `file_root` is provided.
    """
    
    if not (info is None or sampler is None):
        print(sampler.products())
        gdsamples = MCSamplesFromCobaya(info, sampler.products()["sample"])
    elif not file_root is None:
        gdsamples = loadMCSamples(file_root)
    else:
        raise ValueError("No specified MCMC info or file root provided.")

    gdplot = gdplt.get_subplot_plotter(width_inch=5)
    gdplot.triangle_plot(gdsamples, variables, filled=True)
    plt.suptitle(' mapname=' + str(mapname))
    if outfile is None:
        print('showing plot')
        plt.show()
    else:
        outpath = (output_plots + '/' + outfile)
        plt.savefig(outpath)
        print('Saving ' + outpath)
        plt.close()

def plot_best_fit(GLOBAL_VAR, sampler, bin_centers, mapname=None, output_plots='output_plots'):
    """
    Plots the best-fit model for EB spectra based on MCMC sampling results.

    Parameters:
    -----------
    GLOBAL_VAR : dict
        Global dictionary containing the theoretical spectra (EE, BB, EB).

    sampler : cobaya.run
        The sampling output result from a Cobaya run.

    bin_centers : ndarray
        Array of multipole moments (l) bin centers.

    mapname : str, optional
        Name of the map or dataset being analyzed.

    output_plots : str
        Directory path to save the output plot images.

    Returns:
    --------
    gMpl, aplusb, gMpl_std, aplusb_std : float
        Best-fit values of gMpl and aplusb, along with their standard deviations.
    """
    bins = bin_centers

    gd_sample = sampler.products()["sample"]
    n = len(gd_sample['gMpl'])
    gMpl = np.round(gd_sample['gMpl'][n//2:].mean(),3)
    aplusb = np.round(gd_sample['aplusb'][n//2:].mean(),3)
    chisq = np.round(gd_sample['chi2'][n//2:].mean(),3)
    gMpl_std = np.round(gd_sample['gMpl'][n//2:].std(),3)
    aplusb_std = np.round(gd_sample['aplusb'][n//2:].std(),3)
    C_eb_ede = GLOBAL_VAR['EB_EDE']
    C_ee_cmb = GLOBAL_VAR['EE_binned']
    C_bb_cmb = GLOBAL_VAR['BB_binned']
    C_eb_observed = GLOBAL_VAR['EB_observed']
    C_eb_var = GLOBAL_VAR['EB_var']
    cos_term = np.cos(4 * np.deg2rad(aplusb)) * gMpl * C_eb_ede
    sin_term = np.sin(4 * np.deg2rad(aplusb)) / 2 * (C_ee_cmb - C_bb_cmb) 
    
    plt.figure()
    plt.plot(bins, cos_term, label='gMpl contribution')
    plt.plot(bins, sin_term, label='Rotation contribution')
    plt.plot(bins, cos_term+sin_term, label='Combined contribution')
    if(len(C_eb_var.shape)==2 and C_eb_var.shape[0] == C_eb_var.shape[1]):
        C_eb_var = np.diag(C_eb_var)
    plt.errorbar(bins, C_eb_observed, yerr=np.sqrt(C_eb_var), label='observed EB')
    plt.ylabel(r'$C_{\ell}^{EB}\cdot\ell(\ell+1)/(2\pi)$  [$\mu K^2$]')
    plt.xlabel(r'$\ell$')
    plt.legend()
    title_str = ('gMpl=' + str(gMpl) + '+-' + str(gMpl_std) + 
                 ' aplusb=' + str(aplusb) + '+-' + str(aplusb_std) + 
                 '\n chisq=' + str(chisq) + ' mapname=' + str(mapname))
    plt.title(title_str)
    outpath = output_plots + '/' + mapname + '_bestfit.png'
    plt.savefig(outpath)
    print('Saving ' + outpath)
    plt.close()
    return gMpl, aplusb, gMpl_std, aplusb_std

def plot_best_fit_multicomponent(GLOBAL_VAR, sampler_sims, bin_centers, output_plots, 
                                 residuals=False, real_sampler = None, sim_num = None):
    """
    Plots the best-fit model for multiple components (gMpl and aplusb) for different frequencies.

    Parameters:
    -----------
    GLOBAL_VAR : dict
        Global dictionary containing the theoretical spectra (EE, BB, EB) for different frequencies.

    sampler_sims : list
        List of samplers or paths to sampler files for multiple simulation datasets.

    bin_centers : ndarray
        Array of multipole moments (l) bin centers.

    output_plots : str
        Directory path to save the output plot images.

    residuals : bool, optional
        If True, plots the residuals (default is False).

    real_sampler : cobaya.run, optional
        The real dataset sampler to compare against simulations.

    sim_num : int, optional
        Number of simulations.

    Saves:
    ------
    Multicomponent best-fit model plots for each frequency.
    """
    aplusb_dict = {
        'BK18_B95':'aplusb_b95', 
        'BK18_K95':'aplusb_k95', 
        'BK18_150':'aplusb_150', 
        'BK18_220':'aplusb_220', 
        'BK18_B95ext': 'aplusb_b95ext'
    } 
    
    fig = plt.figure(figsize=(15, 10))
    gs = GridSpec(2, 3, figure=fig)

    # Define subplots, leaving one empty spot
    axes = {
        MAP_FREQS[i]: fig.add_subplot(gs[i]) for i in range(len(MAP_FREQS)) 
    }
    alpha = 0.05
    diag_ax = fig.add_subplot(gs[5])  # Diagonal plot spot
    if(len(sampler_sims) > 100):
        with open(output_plots + '/' + 'sim_results_multicomp.csv', 'w') as file:

            header_str = 'sim_num,gMpl,gMpl_std,'
            for freq in MAP_FREQS:
                header_str += aplusb_dict[freq] + ',' + aplusb_dict[freq] + '_std,'
            header_str += 'chisq' 
            file.write(header_str + '\n')
            for i,sampler in enumerate(sampler_sims):
                if(isinstance(sampler, str)):
                    print('Reading file: ' +sampler)
                    gd_sample = read_sampler(sampler)
                else:
                    gd_sample = sampler.products()["sample"]
                var = 'gMpl'
                n = len(gd_sample[var])
                
                gMpl = np.round(gd_sample[var][n//2:].mean(),3)
                gMpl_std = np.round(gd_sample[var][n//2:].std(),3)
                sim_line_str = str(i) + ',' + str(gMpl) + ',' + str(gMpl_std) + ','
                for map_freq in aplusb_dict:
                    ax = axes[map_freq]
                    var = aplusb_dict[map_freq]
                    aplusb = np.round(gd_sample[var][n//2:].mean(),3)
                    aplusb_std = np.round(gd_sample[var][n//2:].std(),3)
                    C_ee_cmb = GLOBAL_VAR['EE_binned' + '_' + map_freq]
                    C_bb_cmb = GLOBAL_VAR['BB_binned'+ '_' + map_freq]
                    C_eb_sim = GLOBAL_VAR['EB_sims_' + map_freq][:, i]
                    
                    
                    C_eb_ede = GLOBAL_VAR['EB_EDE'+ '_' + map_freq]
                    cos_term = np.cos(4 * np.deg2rad(aplusb)) * gMpl * C_eb_ede
                    sin_term = np.sin(4 * np.deg2rad(aplusb)) / 2 * (C_ee_cmb - C_bb_cmb) 
                    if(i == 0):
                        if(residuals):
                            ax.plot(bin_centers, sin_term+cos_term-C_eb_sim, color='red', 
                                alpha=alpha*10, linewidth=1, label = 'residual')
                        else:
                            ax.plot(bin_centers, cos_term, color='blue', 
                                    alpha=alpha, linewidth=1, label = 'gMpl term')
                            ax.plot(bin_centers, sin_term, color='green', 
                                    alpha=alpha, linewidth=1, label = 'aplusb term')
                            
                            ax.plot(bin_centers, sin_term+cos_term, color='purple', 
                                    alpha=alpha*10, linewidth=1, label = 'combined')
                            ax.plot(bin_centers, C_eb_sim, color='red', 
                                    alpha=alpha*10, linewidth=1, label = 'sim curve')
                            
                        
                    else:
                        if(residuals):
                            ax.plot(bin_centers, sin_term+cos_term-C_eb_sim, color='red', 
                                alpha=alpha*10, linewidth=1)
                        else:
                            ax.plot(bin_centers, cos_term, color='blue', 
                                    alpha=alpha, linewidth=1)
                            ax.plot(bin_centers, sin_term, color='green', 
                                    alpha=alpha, linewidth=1)
                            
                            ax.plot(bin_centers, sin_term+cos_term, color='purple', 
                                    alpha=alpha*10, linewidth=1)
                            ax.plot(bin_centers, C_eb_sim, color='red', 
                                    alpha=alpha*10, linewidth=1)
                            
                        
                    
                    sim_line_str += str(aplusb) + ',' + str(aplusb_std) + ',' 
                chisq = np.round(gd_sample['chi2'][n//2:].mean(),3)
                sim_line_str+= str(chisq)
                file.write(sim_line_str + '\n')
        for map_freq in aplusb_dict:
            ax = axes[map_freq]
            C_eb_observed = GLOBAL_VAR['EB_trueobserved'+ '_' + map_freq]
            if(not sim_num is None):
                C_eb_observed = GLOBAL_VAR['EB_sims_' + map_freq][:, sim_num]
            C_eb_var = GLOBAL_VAR['EB_var'+ '_' + map_freq]
            if(len(C_eb_var.shape)==2 and C_eb_var.shape[0] == C_eb_var.shape[1]):
                C_eb_var = np.diag(C_eb_var)
        
            ax.errorbar(bin_centers, C_eb_observed, yerr=np.sqrt(C_eb_var), 
                        linewidth=3, alpha=1, label='observed EB')
            ax.set_title(map_freq)
            ax.legend()
            ax.set_xlabel(r'$\ell$')
            ax.set_ylabel(r'$C_{\ell}^{EB}\cdot\ell(\ell+1)/(2\pi)$  [$\mu K^2$]')
        plt.suptitle('Multicomponent All Sims')
        plt.tight_layout()
        if(residuals):
            outpath = output_plots + '/multicomp_bestfit_residuals_allsims'
        
        else:
            outpath = output_plots + '/multicomp_bestfit_allsims'
        if(sim_num is None):
            outpath = outpath + '_real.png'
        else:
            outpath = outpath + 'sim_num' + str(sim_num) + '.png'
        print('Saving ' + outpath)
        plt.savefig(outpath)
        plt.close()
    
    if(real_sampler is None):
        return 
    if(isinstance(real_sampler, str)):
        print('Reading file: ' +real_sampler)
        gd_sample = read_sampler(real_sampler)
    else:
        gd_sample = real_sampler.products()["sample"]
    var = 'gMpl'
    n = len(gd_sample[var])
    
    gMpl = np.round(gd_sample[var][n//2:].mean(),3)
    gMpl_std = np.round(gd_sample[var][n//2:].std(),3)
    for map_freq in aplusb_dict:
        var = aplusb_dict[map_freq]
        chisq = np.round(gd_sample['chi2'][n//2:].mean(),3)
        aplusb = np.round(gd_sample[var][n//2:].mean(),3)
        aplusb_std = np.round(gd_sample[var][n//2:].std(),3)
        
        C_eb_observed = GLOBAL_VAR['EB_observed'+ '_' + map_freq]
        C_eb_var = GLOBAL_VAR['EB_var'+ '_' + map_freq]
        if(len(C_eb_var.shape)==2 and C_eb_var.shape[0] == C_eb_var.shape[1]):
            C_eb_var = np.diag(C_eb_var)
        C_ee_cmb = GLOBAL_VAR['EE_binned' + '_' + map_freq]
        C_bb_cmb = GLOBAL_VAR['BB_binned'+ '_' + map_freq]
        C_eb_ede = GLOBAL_VAR['EB_EDE'+ '_' + map_freq]
        cos_term = np.cos(4 * np.deg2rad(aplusb)) * gMpl * C_eb_ede
        sin_term = np.sin(4 * np.deg2rad(aplusb)) / 2 * (C_ee_cmb - C_bb_cmb) 

        plt.figure()
        plt.plot(bin_centers, cos_term, color='blue', 
                linewidth=1, label = 'gMpl term')
        plt.plot(bin_centers, sin_term, color='green', 
                linewidth=1, label = 'aplusb term')
        
        plt.plot(bin_centers, sin_term+cos_term, color='purple', 
                 linewidth=3, label = 'combined')
        
        plt.errorbar(bin_centers, C_eb_observed, yerr=np.sqrt(C_eb_var), 
                    linewidth=3, label='observed EB')
        title_str = ('gMpl=' + str(gMpl) + '+-' + str(gMpl_std) + 
                 ' aplusb=' + str(aplusb) + '+-' + str(aplusb_std) + 
                 '\n chisq=' + str(chisq) + ' mapname=' + str(map_freq))
        plt.title(title_str)
        plt.legend()
        plt.xlabel(r'$\ell$')
        plt.ylabel(r'$C_{\ell}^{EB}\cdot\ell(\ell+1)/(2\pi)$  [$\mu K^2$]')
        outpath = output_plots + '/multicomp_bestfit_' + map_freq + '.png'
        print('Saving ' + outpath)
        plt.savefig(outpath)
