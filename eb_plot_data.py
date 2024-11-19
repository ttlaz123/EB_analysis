import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import corner
import re
import glob
from astropy.io import fits
from getdist.mcsamples import MCSamplesFromCobaya
from getdist.mcsamples import loadMCSamples
import getdist.plots as gdplt
from matplotlib.gridspec import GridSpec

import bicep_data_consts as bdc
MAP_FREQS = bdc.MAP_FREQS

def check_file_header(file_path, reference_header):
    with open(file_path, 'r') as f:
        for line in f:
            # Assuming the relevant header is the one with 'BxB' in it
            if line.startswith("#") and "BxB" in line:
                current_header = line.strip()
                if reference_header is None:
                    reference_header = current_header.split()
                elif current_header.split() != reference_header:
                    # Compare the lists element by element and print the differences
                    list1 = current_header.split()
                    list2 = reference_header
                    for i in range(min(len(list1), len(list2))):
                        if list1[i] != list2[i]:
                            print(f"Difference at index {i}: {list1[i]} != {list2[i]}")

                        # If the lists have different lengths, print the extra elements
                        if len(list1) > len(list2):
                            print(f"Extra elements in list1: {list1[len(list2):]}")
                        elif len(list2) > len(list1):
                            print(f"Extra elements in list2: {list2[len(list1):]}")
                    raise ValueError("Header mismatch detected in one or more files.")
                break  # Stop reading further header lines
    return reference_header

def load_observed_spectra(observed_data_path, used_maps, map_reference_header):
    """
    Load observed spectra data from a specified file and filter the data based on the used maps.

    Args:
        observed_data_path (str): The file path to the observed spectra data in a text format.
        used_maps (list of str): A list of map names to be used for filtering the observed data.

    Returns:
        dict: A dict containing the filtered observed spectra data, 
                with each entry representing a spectrum for the specified used maps.

    Raises:
        AssertionError: If the provided map names in `used_maps` do not match the reference header.
        
    Description:
        The function first verifies the header of the observed data file against the 
        reference header (`self.map_reference_header`). It then identifies the indices 
        of the specified `used_maps` within the validated header. After loading the data 
        from the file, the function extracts the relevant columns corresponding to the 
        used maps, adjusting for the fact that the first column in the loaded data is 
        merely an index or identifier (hence the addition of 1 to the indices).
    """
    reference_header = map_reference_header
    print("Loading: " + str(observed_data_path))
    try:
        
        map_reference_header = check_file_header(observed_data_path, reference_header)
        map_reference_header = map_reference_header
    except ValueError:
        with open(observed_data_path, 'r') as f:
            for line in f:
                # Assuming the relevant header is the one with 'BxB' in it
                if line.startswith("#") and "BxB" in line:
                    current_header = line.strip()
                    map_reference_header = current_header.split()
                    break
    used_cols = []
    for cross_map in used_maps:
        try:
            if(cross_map  not in map_reference_header):
                parts = input_str.split('x')

                cross_map = f"{parts[1]}x{parts[0]}"

            used_cols.append(map_reference_header.index(cross_map))
        
        except ValueError:
            # Try to swap _B and _E and check again
            match = re.match(r'(.+)(_B)(x\1)(_E)', cross_map)
            if match:
                # Swap _B and _E if the parts before _B and _E are identical
                swapped_map =  match.group(1) + '_E' + match.group(3) + '_B'
            else:
                raise ValueError("Cross map does not exist")
            used_cols.append(map_reference_header.index(swapped_map))


    obs_data = np.loadtxt(observed_data_path)

    observed_spectra_dict = {}
    for i in range(len(used_cols)):
        input_str = used_maps[i]
        observed_spectra_dict[input_str] = obs_data[:, used_cols[i]]

    
    return observed_spectra_dict, map_reference_header

def load_bpwf(bpwf_directory, map_reference_header):
        """
        Load BPWF (Band Power Window Function) data from the specified directory.

        Args:
            bpwf_directory (str): The file path or pattern specifying the directory where BPWF files are located.

        Returns:
            ndarray: A 3D NumPy array containing the concatenated BPWF data from all files in the specified directory. 
                    The first dimension corresponds to the number of files, and the subsequent dimensions 
                    correspond to the BPWF data.

        Raises:
            ValueError: If no BPWF files are found in the specified directory.

        Description:
            This function searches for BPWF files in the provided directory, ensuring that there is at least 
            one file to load. It checks the consistency of the file headers against a reference header, 
            which is stored in `self.map_reference_header`. Each file's data is read (excluding the first column) 
            and stored in a list, which is then stacked into a 3D NumPy array before being returned.
        """
        
        bpwf_files = sorted(glob.glob(bpwf_directory), 
                        key=lambda x: list(map(int,re.findall("(\d+)", x))))
    
        if len(bpwf_files) < 1:
            raise ValueError("No files found in " + str(bpwf_directory))
        # Initialize variable to store the header line to compare
        reference_header = map_reference_header
        # List to hold all loaded data
        bpwf_data = []

        for file in bpwf_files:
            print("Loading: " + str(file))
            # Read the header and check consistency
            map_reference_header = check_file_header(file, reference_header)
            # Load data, don't ignore the first column
            bpwf_data.append(np.loadtxt(file))

        # Concatenate and return all BPWF data
        return np.stack(bpwf_data, axis=0), map_reference_header


def load_covariance_matrix(covmat_path, map_reference_header):
    """
    Load the covariance matrix from the specified file.

    Args:
        covmat_path (str): The file path to the covariance matrix data in a text format.

    Returns:
        ndarray: A 2D NumPy array containing the covariance matrix.

    Raises:
        AssertionError: If the loaded covariance matrix is not square (i.e., the number of rows does not equal the number of columns).

    Description:
        This function reads a covariance matrix from a specified file path, ensuring that the matrix is square 
        by checking that the number of rows equals the number of columns. It validates the file's header 
        against the existing reference header, stored in `self.map_reference_header`, before loading the matrix data.
    """
    print("Loading: " + str(covmat_path))
    map_reference_header = check_file_header(covmat_path, map_reference_header)
    full_covmat = np.loadtxt(covmat_path)
    shap = full_covmat.shape
    
    assert shap[0] == shap[1], "Covariance matrix must be square."
    return full_covmat

def load_cmb_spectra(lensing_path, dust_paths):
    k_to_uk = 1e6
    theory_dict = {}
    print("Loading: " + str(lensing_path))
    with fits.open(lensing_path) as hdul_lens:
        EE_lens = hdul_lens[1].data['E-mode C_l']
        BB_lens = hdul_lens[1].data['B-mode C_l']
    for map_freq in dust_paths:
        print("Loading: " + str(dust_paths[map_freq]))
        with fits.open(dust_paths[map_freq]) as hdul_dust:
            EE_dust = hdul_dust[1].data['E-mode C_l']
            BB_dust = hdul_dust[1].data['B-mode C_l']
        ee_spectrum = EE_lens + EE_dust
        
        bb_spectrum =  BB_lens + BB_dust
        ee_spectrum *= np.square(k_to_uk)
        bb_spectrum *= np.square(k_to_uk)
        cl_to_dl = np.array([l*(l+1) for l in range(len(ee_spectrum))])/2/np.pi
        theory_dict[map_freq + '_Ex' + map_freq + '_E'] = ee_spectrum*cl_to_dl            
        theory_dict[map_freq + '_Bx' + map_freq + '_B'] = bb_spectrum*cl_to_dl
    #plt.plot(theory_dict[map_freq + '_Ex' + map_freq + '_E'])
    return theory_dict

def include_ede_spectra(ede_path, theory_dict):
        k_to_uk = 1e6
        cmb_temp = 2.726
        data = pd.read_csv(ede_path, delim_whitespace=True, comment='#', header=None)
        data.columns = ['l', 'TT', 'EE', 'TE', 'BB', 'EB', 'TB', 'phiphi', 'TPhi', 'Ephi']
        # Extract 'l' and 'EB' columns
        EB_values = data['EB'].to_numpy()
        EB_ede_dls = -EB_values * np.square(k_to_uk) * np.square(cmb_temp)
        theory_dict['EDE_EB'] = EB_ede_dls
        return theory_dict

'''
Everything after this is deprecated
'''
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
