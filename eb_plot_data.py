import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import corner
from getdist.mcsamples import MCSamplesFromCobaya
from getdist.mcsamples import loadMCSamples
import getdist.plots as gdplt
from matplotlib.gridspec import GridSpec

import bicep_data_consts as bdc
MAP_FREQS = bdc.MAP_FREQS

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
