print('Importing Packages')
import os
from matplotlib import pyplot as plt
import numpy as np
from scipy import stats 

from scipy.optimize import minimize
import time
import pickle
import argparse
import matplotlib
#matplotlib.use('Agg')

import pandas as pd
from cobaya.run import run
from cobaya.model import get_model
from cobaya.yaml import yaml_load
from getdist.mcsamples import MCSamplesFromCobaya
from getdist.mcsamples import loadMCSamples
import getdist.plots as gdplt

import eb_load_data as eld

GLOBAL_VAR = {}


def eb_log_likelihood_vector(C_eb_observed, C_eb_var, C_eb_ede, C_ee_cmb, C_bb_cmb, aplusb, gMpl):
    """
    Computes the total log-likelihood for the EB power spectrum using vectorized operations.

    Parameters:
    -----------
    C_eb_observed : ndarray
        The observed EB power spectrum (Dl) values.
    
    C_eb_var : ndarray
        The variance of the observed EB power spectrum, used as the uncertainty in the likelihood calculation.
    
    C_eb_ede : ndarray
        The predicted EB power spectrum contribution from early dark energy (EDE).
    
    C_ee_cmb : ndarray
        The predicted EE power spectrum from the CMB.
    
    C_bb_cmb : ndarray
        The predicted BB power spectrum from the CMB.
    
    aplusb : float
        The sum of angle parameters `a` and `b` used in the rotation of polarization modes.
    
    gMpl : float
        A scaling factor related to the effective gravitational constant.

    Returns:
    --------
    total_log_likelihood : float
        The total log-likelihood value for the EB power spectrum, computed as the sum of the likelihoods across all bins.
    
    Notes:
    ------
    This function is vectorized to perform all calculations in parallel, providing a more efficient computation compared to looping over each bin individually.
    """
    cos_term = np.cos(4 * np.deg2rad(aplusb)) * gMpl * C_eb_ede
    sin_term = np.sin(4 * np.deg2rad(aplusb)) / 2 * (C_ee_cmb - C_bb_cmb) 
    v = C_eb_observed - cos_term - sin_term
    if(len(C_eb_var.shape)==2 and C_eb_var.shape[0] == C_eb_var.shape[1]):
        C_eb_var_inv = np.linalg.inv(C_eb_var)
        total_log_likelihood = -v @ C_eb_var_inv @ v.T
    else:
        bin_loglike = np.square(v) / C_eb_var
        total_log_likelihood = -np.sum(bin_loglike)
    return total_log_likelihood


def plot_best_fit(sampler, bin_centers, mapname=None, output_plots='output_plots'):
    bins = bin_centers

    gd_sample = sampler.products()["sample"]
    n = len(gd_sample['gMpl'])
    gMpl = np.round(gd_sample['gMpl'][n//2:].mean(),3)
    aplusb = np.round(gd_sample['aplusb'][n//2:].mean(),3)

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
                 '\n mapname=' + str(mapname))
    plt.title(title_str)
    plt.savefig(output_plots + '/' + mapname + '_bestfit.png')
    plt.close()
    return gMpl, aplusb, gMpl_std, aplusb_std

def plot_info(variables=None, info=None, sampler=None, outfile=None, file_root=None, mapname=None, output_plots='output_plots'):
    """
    Generates and displays or saves a triangle plot of posterior distributions for specified variables.

    Parameters:
    -----------
    variables : list of str, optional
        A list of variable names to include in the plot. Each variable should correspond to a parameter in the model.
        Example: ['var1', 'var2', ...]. If `None`, all available variables will be plotted.

    info : dict, optional
        A YAML dictionary containing the configuration information from a Cobaya run. This is used in conjunction
        with the `sampler` output to generate the MCMC samples.

    sampler : cobaya.run, optional
        The sampling output result from a Cobaya run. The `sampler.products()["sample"]` will be used to create the MCMC samples.

    outfile : str, optional
        The path to save the generated plot. If `None`, the plot is displayed on the screen instead of being saved.

    file_root : str, optional
        The root filename (without extension) from which to load MCMC samples if `info` and `sampler` are not provided.

    Raises:
    -------
    ValueError
        If neither `info`/`sampler` nor `file_root` is provided, the function raises an error indicating insufficient information to generate the plot.

    Notes:
    ------
    The function creates a triangle plot using the `getdist` library, which visualizes the 1D and 2D marginalized posterior distributions of the specified variables.
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
    plt.title(' mapname=' + str(mapname))
    if outfile is None:
        print('showing plot')
        plt.show()
    else:
        plt.savefig(output_plots + '/' + outfile)
        plt.close()

def objective_function(aplusb, C_eb_observed, C_eb_var, C_eb_ede, C_ee_cmb, C_bb_cmb, gMpl):
    return -eb_log_likelihood_vector(C_eb_observed, C_eb_var, C_eb_ede, C_ee_cmb, C_bb_cmb, aplusb, gMpl)


def polar_rotation_likelihood():
    C_ee_cmb = GLOBAL_VAR['EE_binned']
    C_bb_cmb = GLOBAL_VAR['BB_binned']
    C_eb_observed = GLOBAL_VAR['EB_observed']
    C_eb_var = GLOBAL_VAR['EB_var']
   
    C_eb_ede = GLOBAL_VAR['EB_EDE']
    # Initial guess for aplusb
    initial_aplusb = 0.0

    # Provide a value for gMpl (if it’s fixed or if it needs to be optimized as well, adjust accordingly)
    gMpl_value = 0
    
    result = minimize(objective_function, initial_aplusb, args=(C_eb_observed, C_eb_var, C_eb_ede, C_ee_cmb, C_bb_cmb, gMpl_value),
                      method='BFGS')
    
    best_fit_aplusb = result.x[0]
    print(f'~~~~~~~~~~~~~~~~~~~Best-fit aplusb: {best_fit_aplusb}')
        # Compute the Hessian matrix
    hessian_inv = result.hess_inv

    # Extract the error bars from the Hessian
    error_bars = np.sqrt(np.diag(hessian_inv))
    print(f'Error bars on aplusb: {error_bars[0]}')
    return best_fit_aplusb, error_bars[0]

def eb_axion_mcmc_runner(aplusb, gMpl):
    # TODO complete this
    
    C_ee_cmb = GLOBAL_VAR['EE_binned']
    C_bb_cmb = GLOBAL_VAR['BB_binned']
    C_eb_observed = GLOBAL_VAR['EB_observed']
    C_eb_var = GLOBAL_VAR['EB_var']
   
    C_eb_ede = GLOBAL_VAR['EB_EDE']
    likelihood = eb_log_likelihood_vector(C_eb_observed, C_eb_var, C_eb_ede, C_ee_cmb, C_bb_cmb, aplusb, gMpl)
    
    
    
    return likelihood

def get_eb_axion_infodict(outpath, variables, priors):
    """
    Generates and returns a dictionary containing the configuration information for an MCMC run, tailored to an EB axion model.

    Parameters:
    -----------
    outpath : str
        The file path where the MCMC output will be saved.

    variables : list of str
        A list of variable names to be included in the MCMC sampling. Each variable corresponds to a model parameter.

    priors : list of tuple
        A list of tuples specifying the prior ranges for each variable. Each tuple contains the minimum and maximum values (min, max) for the corresponding variable.

    Returns:
    --------
    info : dict
        A dictionary structured to define the settings and parameters for running an MCMC sampling. The dictionary contains:
        - `likelihood`: Specifies the likelihood function to be used in the sampling, here defined as `eb_axion_mcmc_runner`.
        - `params`: A dictionary of parameters, where each parameter is associated with its prior range, reference value, and proposal width.
        - `sampler`: Settings for the MCMC sampler, including stopping criteria (`Rminus1_stop`) and the maximum number of attempts (`max_tries`).
        - `output`: The file path where the results will be saved, set to the provided `outpath`.

    Notes:
    ------
    - The reference value (`ref`) and proposal width (`proposal`) for each parameter are set to 0. These values can be adjusted as needed for different models or sampling strategies.
    - The sampler settings are configured for a typical MCMC run but can be fine-tuned according to specific requirements.
    """
    info = {"likelihood": 
        {
            "power": eb_axion_mcmc_runner
        }
    }

    info["params"] = {}

    for i in range(len(variables)):
        var = variables[i]
        prior = priors[i]
        info["params"][var] = {
            "prior": {"min": prior[0], "max": prior[1]},
            "ref": 0,
            "proposal": 0
        }
    info["sampler"] = {"mcmc": {
                                "burn_in":0.2,
                                "Rminus1_stop": 0.03,          
                                "max_tries": 10000}}
    info["output"] = outpath
    return info



def get_priors_and_variables(gMpl_minmax=(-5, 5), aplusb_minmax=(-5, 5)):
    """
    Defines and returns the model variables and their corresponding prior ranges.

    Parameters:
    -----------
    gMpl_minmax : tuple, optional
        The minimum and maximum values (min, max) for the 'gMpl' variable. 
        Defaults to (-5, 5).
    
    aplusb_minmax : tuple, optional
        The minimum and maximum values (min, max) for the 'aplusb' variable.
        Defaults to (-5, 5).

    Returns:
    --------
    variables : list of str
        A list of variable names used in the model:
        - 'gMpl': A scaling factor related to the effective gravitational constant.
        - 'aplusb': The sum of angle parameters `a` and `b` used in the rotation of polarization modes.
    
    priors : list of tuple
        A list of tuples specifying the prior ranges for each corresponding variable:
        - gMpl_minmax : tuple
            Prior range for 'gMpl'.
        - aplusb_minmax : tuple
            Prior range for 'aplusb'.
    """
    variables = ['gMpl', 'aplusb']
    priors = [gMpl_minmax, aplusb_minmax]
    return variables, priors



def bin_spectrum_given_centers(bin_centers, spectrum, ell_min=0):
    '''
    DEPRECATED
    Bins a spectrum into specified bin centers.

    Assumes the spectrum starts at l=ell_min and calculates the bin edges based on the given bin centers.

    Parameters
    ----------
    bin_centers : array_like
        An array of bin center values. The function calculates bin edges based on these centers.
        
    spectrum : array_like
        An array representing the spectrum values. The length of this array should be at least as long as the maximum bin edge.
        
    ell_min : int, optional
        The minimum ell value (default is 0). The spectrum is assumed to start at this value. This parameter is used to adjust the starting point of the bin edges.

    Returns
    -------
    binned_spectrum : ndarray
        An array of the binned spectrum values, where each value represents the average spectrum value within the corresponding bin.

    Raises
    ------
    ValueError
        If the length of the spectrum is insufficient to cover the maximum bin edge.

    Notes
    -----
    The bin edges are computed such that the center of each bin is the midpoint between the start and end of the bin.
    The function assumes that the spectrum starts at `ell_min` and is defined for consecutive ell values.
    '''
    # Calculate bin edges
    binned_spectrum = np.zeros(len(bin_centers))
    bin_starts = np.zeros(len(bin_centers) + 1,dtype=int)
    bin_starts[0] = ell_min

    for i in range(1, len(bin_starts)):
        bin_starts[i] = (2 * bin_centers[i - 1] - bin_starts[i - 1] + 1)

    # Check if the spectrum is long enough
    max_bin_edge = bin_starts[-1]
    if len(spectrum) < max_bin_edge:
        raise ValueError(f"Spectrum length is insufficient. Required: {max_bin_edge}, Provided: {len(spectrum)}")

    # Bin the spectrum
    for i in range(len(bin_centers)):
        bin_cur = 0
        for j in range(bin_starts[i], bin_starts[i + 1]):
            bin_cur += spectrum[j]
        binned_spectrum[i] = bin_cur / (bin_starts[i + 1] - bin_starts[i])

    return binned_spectrum, bin_starts

def rebin(cur_bins, cur_data, new_bin_starts, raw_cl=False, plot=False):
    x_binned = []
    y_binned = []

    # scale from D to C if we're using raw Cls
    if(raw_cl):
        cur_data *= 2*np.pi/cur_bins/(cur_bins+1)
    # Rebin the data
    for i in range(len(new_bin_starts) - 1):
        # Find the indices of x that fall within the current bin
        indices = (cur_bins >= new_bin_starts[i]) & (cur_bins < new_bin_starts[i+1])
        
        if np.any(indices):
            x_bin = np.mean(cur_bins[indices])
            y_bin = np.mean(cur_data[indices])
                
        else:
            x_bin = (new_bin_starts[i] + new_bin_starts[i+1]) / 2
            y_bin = np.interp(x_bin, cur_bins, cur_data)

        x_binned.append(x_bin)
        y_binned.append(y_bin)

    # Convert lists to arrays
    x_binned = np.array(x_binned)
    y_binned = np.array(y_binned)
    
    # Optionally plot the rebinned data
    if(plot):
        plt.figure(figsize=(10, 6))
        plt.plot(cur_bins, cur_data, 'o-', label='Original Data')
        plt.plot(x_binned, y_binned, 's-', label='Rebinned Data', color='red')
        plt.ylabel(r'Arbitrary Scaled $C_{\ell}^{EB}\cdot\ell(\ell+1)/(2\pi)$  [$\mu K^2$]')
        plt.xlabel(r'$\ell$')
        plt.title('Rebinned Data (Averaged y, Ignoring Points Outside Bins)')
        plt.legend()
        plt.grid(True)
        plt.show()

    return y_binned



def scatter_sims(sim_results, mapname='BK18_B95ext'):
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

def main():
    print('~~~~~~~~~~~~~~ Start MCMC ~~~~~~~~~~~~~~')
    
    
    variables, priors = get_priors_and_variables(aplusb_minmax=(-5,5))
    maps = ['BK18_B95', 'BK18_K95', 'BK18_150', 'BK18_220', 'BK18_B95ext']
    zero_ede=False
    table_str = ''
    max_sim = 1
    sim_runs = range(0,max_sim)
    sim_str = 'map_name,sim_num,gMpl,gMpl_std,aplusb,aplusb_std\n'
    csv_resultfile = 'sim_results.csv'
    for map in maps:
        output_plots = 'output_plots_ede' + str(not zero_ede) 
        outpath = 'mcmc_chains_ede' + str(not zero_ede) + '/' + map + '/'
        if(not os.path.exists(output_plots)):
                os.mkdir(output_plots)
        
        bin_centers, spectrum_dict = eld.load_bicep_data(plot=True, mapname=map, output_plots=output_plots, zero_ede=zero_ede)
        GLOBAL_VAR.update(spectrum_dict)
        for sim_num in sim_runs:
            output_plots = 'output_plots_ede' + str(not zero_ede) + '/sim' + str(sim_num) + '/'
            outpath = 'mcmc_chains_ede' + str(not zero_ede) + '/' + map + '/' + 'simnum' + str(sim_num)
            if(not os.path.exists(output_plots)):
                os.mkdir(output_plots)
            GLOBAL_VAR['EB_observed'] = GLOBAL_VAR['EB_sims'][:,sim_num]
            info_dict = get_eb_axion_infodict(outpath, variables, priors)
            init_params = info_dict['params']
            updated_info, sampler = run(info_dict, resume=True)
            gMpl, aplusb, gMpl_std, aplusb_std = plot_best_fit(sampler, bin_centers=bin_centers, mapname=map, output_plots=output_plots)

            plot_info(variables, updated_info, sampler, mapname=map, outfile=map+'_triagplot.png', output_plots=output_plots)
            sim_str+= (map + ',' + str(np.round(sim_num,3)) + ',' + 
                        str(np.round(gMpl,3)) + ',' + str(np.round(gMpl_std,3)) + ',' + 
                        str(np.round(aplusb,3)) + ',' + str(np.round(aplusb_std,3))) + '\n'
        
        output_plots = 'output_plots_ede' + str(not zero_ede) + '/' + 'real' 
        outpath = 'mcmc_chains_ede' + str(not zero_ede) + '/' + map + '/' + 'real' 
        if(not os.path.exists(output_plots)):
                os.mkdir(output_plots)
        info_dict = get_eb_axion_infodict(outpath, variables, priors)
        init_params = info_dict['params']
        # test to make sure the likelihood function works 
        log_test = eb_axion_mcmc_runner(init_params['aplusb']['ref'], 
                                        init_params['gMpl']['ref'])
        print("Initial chisq value: " + str(log_test))
        updated_info, sampler = run(info_dict, resume=True)
        gMpl, aplusb, gMpl_std, aplusb_std = plot_best_fit(sampler, bin_centers=bin_centers, mapname=map, output_plots=output_plots)
   
        plot_info(variables, updated_info, sampler, mapname=map, outfile=map+'_triagplot.png', output_plots=output_plots)
        table_str += map + ': ' + str(aplusb) + ' +- ' + str(aplusb_std) 
        
        aplusb_bestfit, std = polar_rotation_likelihood()
        table_str += ', ' + str(np.round(aplusb_bestfit,3)) + ' +- ' + str(np.round(std,3))
        table_str += '\n'
        
        two_var_chisq = eb_axion_mcmc_runner(aplusb, gMpl)
        one_var_chisq = eb_axion_mcmc_runner(aplusb_bestfit, 0)
        table_str += 'twovar_chisq: ' + str(two_var_chisq) + ' onevar_chisq: ' + str(one_var_chisq) + '\n'
    with open(csv_resultfile, 'w') as file:
        file.write(sim_str)
    print(table_str)
    for mapname in maps:
        scatter_sims(csv_resultfile, mapname=mapname)
    print('~~~~~~~~~~~~~~ End ~~~~~~~~~~~~~~')
if __name__ == '__main__':
   
    main()
    