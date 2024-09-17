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
from matplotlib.gridspec import GridSpec


import pandas as pd
from cobaya.run import run
from cobaya.model import get_model
from cobaya.yaml import yaml_load
from getdist.mcsamples import MCSamplesFromCobaya
from getdist.mcsamples import loadMCSamples
import getdist.plots as gdplt

import eb_load_data as eld

GLOBAL_VAR = {}
MAP_FREQS = ['BK18_B95', 'BK18_K95', 'BK18_150', 'BK18_220', 'BK18_B95ext']

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

def read_sampler(filepath):
    df = pd.read_csv(filepath, delim_whitespace=True, comment='#', header=None)
        
    # Assign column names based on the data structure
    df.columns = ['weight', 'minuslogpost', 'gMpl', 'aplusb_b95', 'aplusb_b95ext', 
                    'aplusb_k95', 'aplusb_150', 'aplusb_220', 'minuslogprior', 
                    'minuslogprior__0', 'chi2', 'chi2__power']
    return df

def plot_best_fit_multicomponent(sampler_sims, bin_centers, output_plots, 
                                 residuals=False, real_sampler = None, sim_num = None):
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
                 '\n mapname=' + str(map_freq))
        plt.title(title_str)
        plt.legend()
        plt.xlabel(r'$\ell$')
        plt.ylabel(r'$C_{\ell}^{EB}\cdot\ell(\ell+1)/(2\pi)$  [$\mu K^2$]')
        outpath = output_plots + '/multicomp_bestfit_' + map_freq + '.png'
        print('Saving ' + outpath)
        plt.savefig(outpath)


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
    outpath = output_plots + '/' + mapname + '_bestfit.png'
    plt.savefig(outpath)
    print('Saving ' + outpath)
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
    plt.suptitle(' mapname=' + str(mapname))
    if outfile is None:
        print('showing plot')
        plt.show()
    else:
        outpath = (output_plots + '/' + outfile)
        plt.savefig(outpath)
        print('Saving ' + outpath)
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

def eb_axion_multicomponent_mcmc_runner(gMpl, aplusb_b95, aplusb_b95ext, aplusb_k95,
                                        aplusb_150, aplusb_220):
    multi_likelihood = 0
    aplusb_dict = {
        'BK18_B95':aplusb_b95, 
        'BK18_K95':aplusb_k95, 
        'BK18_150':aplusb_150, 
        'BK18_220':aplusb_220, 
        'BK18_B95ext': aplusb_b95ext
    } 
    for freq in MAP_FREQS:
        aplusb = aplusb_dict[freq]
        C_ee_cmb = GLOBAL_VAR['EE_binned' + '_' + freq]
        C_bb_cmb = GLOBAL_VAR['BB_binned'+ '_' + freq]
        C_eb_observed = GLOBAL_VAR['EB_observed'+ '_' + freq]
        C_eb_var = GLOBAL_VAR['EB_var'+ '_' + freq]
    
        C_eb_ede = GLOBAL_VAR['EB_EDE'+ '_' + freq]
        likelihood = eb_log_likelihood_vector(C_eb_observed, C_eb_var, C_eb_ede, C_ee_cmb, C_bb_cmb, aplusb, gMpl)
        multi_likelihood += likelihood

    return multi_likelihood
def eb_axion_mcmc_runner(aplusb, gMpl):
    # TODO complete this
    
    C_ee_cmb = GLOBAL_VAR['EE_binned']
    C_bb_cmb = GLOBAL_VAR['BB_binned']
    C_eb_observed = GLOBAL_VAR['EB_observed']
    C_eb_var = GLOBAL_VAR['EB_var']
   
    C_eb_ede = GLOBAL_VAR['EB_EDE']
    likelihood = eb_log_likelihood_vector(C_eb_observed, C_eb_var, C_eb_ede, C_ee_cmb, C_bb_cmb, aplusb, gMpl)
    
    
    
    return likelihood

def get_eb_axion_infodict(outpath, variables, priors, likelihood_func):
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
            "power": likelihood_func
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


def get_priors_and_variables_multi_comp(gMpl_minmax=(-5, 5), aplusb_minmax=(-5, 5)):
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
      
    priors : list of tuple
        A list of tuples specifying the prior ranges for each corresponding variable:
        
    """
    variables = ['gMpl', 'aplusb_b95', 'aplusb_b95ext', 
                 'aplusb_k95', 'aplusb_150', 'aplusb_220']
    priors = [gMpl_minmax, aplusb_minmax, aplusb_minmax, 
              aplusb_minmax, aplusb_minmax, aplusb_minmax]
    return variables, priors


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

def ensure_directory(path):
    """Ensure that the directory exists, create it if it doesn't."""
    if not os.path.exists(path):
        os.makedirs(path)


def run_mcmc_for_simulation(mapname, sim_num, bin_centers, variables, priors, bin_str='', zero_ede=False):
    """Run MCMC analysis for a specific map and simulation number."""
    output_plots = f'output_plots_ede{str(not zero_ede)}{bin_str}/{mapname}/'
    outpath = f'mcmc_chains_ede{str(not zero_ede)}{bin_str}/{mapname}/simnum{sim_num}'
    ensure_directory(output_plots)
    GLOBAL_VAR['EB_observed'] = GLOBAL_VAR['EB_sims'][:, sim_num]
    info_dict = get_eb_axion_infodict(outpath, variables, priors, likelihood_func=eb_axion_mcmc_runner)
    init_params = info_dict['params']
    updated_info, sampler = run(info_dict, resume=True)
    gMpl, aplusb, gMpl_std, aplusb_std = plot_best_fit(sampler, bin_centers=bin_centers, mapname=mapname, output_plots=output_plots)
    
    plot_info(variables, updated_info, sampler, mapname=mapname, outfile=f'{mapname}_triagplot.png', output_plots=output_plots)
    return gMpl, aplusb, gMpl_std, aplusb_std

def run_mcmc_for_real(mapname, bin_centers, variables, priors, bin_str='', zero_ede=False):
    """Run MCMC analysis for the real data."""
    output_plots = f'output_plots_ede{str(not zero_ede)}{bin_str}/real'
    outpath = f'mcmc_chains_ede{str(not zero_ede)}{bin_str}/{mapname}/real'
    ensure_directory(output_plots)
    info_dict = get_eb_axion_infodict(outpath, variables, priors, likelihood_func=eb_axion_mcmc_runner)
    init_params = info_dict['params']
    log_test = eb_axion_mcmc_runner(init_params['aplusb']['ref'], init_params['gMpl']['ref'])
    print("Initial chisq value: " + str(-log_test))
    num_params = len(variables)
    num_dof = len(bin_centers)
    red_chisq = -log_test / (num_dof - num_params)
    print("Reduced chisq:" + str(red_chisq))
    updated_info, sampler = run(info_dict, resume=True)
    gMpl, aplusb, gMpl_std, aplusb_std = plot_best_fit(sampler, bin_centers=bin_centers, mapname=mapname, output_plots=output_plots)
    
    plot_info(variables, updated_info, sampler, mapname=mapname, outfile=f'{mapname}_triagplot.png', output_plots=output_plots)
    return gMpl, aplusb, gMpl_std, aplusb_std

def update_sim_results(sim_str, mapname, sim_num, gMpl, aplusb, gMpl_std, aplusb_std):
    """Update the simulation results string with the latest data."""
    return sim_str + f'{mapname},{np.round(sim_num,3)},{np.round(gMpl,3)},{np.round(gMpl_std,3)},{np.round(aplusb,3)},{np.round(aplusb_std,3)}\n'

def update_table_str(table_str, mapname, aplusb, aplusb_std, aplusb_bestfit, std, two_var_chisq, one_var_chisq):
    """Update the table string with the latest data."""
    table_str += f'{mapname}: {aplusb} +- {aplusb_std}, {np.round(aplusb_bestfit,3)} +- {np.round(std,3)}\n'
    table_str += f'twovar_chisq: {two_var_chisq} onevar_chisq: {one_var_chisq}\n'
    return table_str

def single_freq_analysis(max_sim, bin_num = 10, zero_ede=True):
    """Main function to run the frequency analysis."""
    print('~~~~~~~~~~~~~~ Start MCMC ~~~~~~~~~~~~~~')

    variables, priors = get_priors_and_variables(aplusb_minmax=(-5,5))
    sim_str = 'map_name,sim_num,gMpl,gMpl_std,aplusb,aplusb_std\n'
    table_str = ''
    csv_resultfile = 'sim_results.csv'
    
    bin_str = '_bin' + str(bin_num)
    #bin_str = ''
    for mapname in MAP_FREQS:
        output_plots = f'output_plots_ede{str(not zero_ede)}{bin_str}'
        ensure_directory(output_plots)
        
        bin_centers, spectrum_dict = eld.load_bicep_data(plot=True, 
                                                         mapname=mapname, 
                                                         output_plots=output_plots, 
                                                         zero_ede=zero_ede, 
                                                         bin_end=bin_num)
        GLOBAL_VAR.update(spectrum_dict)
        gMpl, aplusb, gMpl_std, aplusb_std = run_mcmc_for_real(mapname, bin_centers, variables, priors, bin_str=bin_str, zero_ede=zero_ede)
        
        aplusb_bestfit, std = polar_rotation_likelihood()
        two_var_chisq = eb_axion_mcmc_runner(aplusb, gMpl)
        one_var_chisq = eb_axion_mcmc_runner(aplusb_bestfit, 0)
        table_str = update_table_str(table_str, mapname, aplusb, aplusb_std, aplusb_bestfit, std, two_var_chisq, one_var_chisq)
    
        for sim_num in range(max_sim): 
            gMpl, aplusb, gMpl_std, aplusb_std = run_mcmc_for_simulation(mapname, sim_num, bin_centers, variables, priors, bin_str=bin_str, zero_ede=zero_ede)
            sim_str = update_sim_results(sim_str, mapname, sim_num, gMpl, aplusb, gMpl_std, aplusb_std)
        
    with open(csv_resultfile, 'w') as file:
        file.write(sim_str)
    
    print(table_str)
    for mapname in MAP_FREQS:
        scatter_sims(csv_resultfile, mapname=mapname)
    
    print('~~~~~~~~~~~~~~ End ~~~~~~~~~~~~~~')

def multi_freq_analysis(max_sim, do_run=True, bin_num=17, zero_ede=True):
    variables, priors = get_priors_and_variables_multi_comp(aplusb_minmax=(-5,5))
    bin_str = '_bin' + str(bin_num)
    #bin_str = ''
    output_plots = f'output_plots_ede{str(not zero_ede)}_multicomp{bin_str}/real'
    outpath = f'mcmc_chains_ede{str(not zero_ede)}_multicomp{bin_str}/real'

    ensure_directory(output_plots)
    for mapname in MAP_FREQS:
        bin_centers, spectrum_dict = eld.load_bicep_data(plot=True, mapname=mapname, output_plots=output_plots, zero_ede=zero_ede, bin_end=bin_num)
  
        GLOBAL_VAR['EB_sims_' + mapname] = spectrum_dict['EB_sims']
        GLOBAL_VAR['EE_binned' + '_' + mapname] = spectrum_dict['EE_binned']
        GLOBAL_VAR['BB_binned'+ '_' + mapname] = spectrum_dict['BB_binned']
        GLOBAL_VAR['EB_trueobserved'+ '_' + mapname] = spectrum_dict['EB_observed']
        GLOBAL_VAR['EB_observed'+ '_' + mapname] = spectrum_dict['EB_observed']
        GLOBAL_VAR['EB_var'+ '_' + mapname] = spectrum_dict['EB_var']
        GLOBAL_VAR['EB_EDE'+ '_' + mapname] = spectrum_dict['EB_EDE']
        eld.plot_cldl(bin_centers, GLOBAL_VAR,  output_plots, mapname)
    
    info_dict = get_eb_axion_infodict(outpath, variables, priors,
                                      likelihood_func=eb_axion_multicomponent_mcmc_runner)
    init_params = info_dict['params']
    log_test = eb_axion_multicomponent_mcmc_runner( init_params['gMpl']['ref'],
                                                   init_params['aplusb_b95']['ref'],
                                                   init_params['aplusb_b95ext']['ref'],
                                                   init_params['aplusb_k95']['ref'],
                                                   init_params['aplusb_150']['ref'],
                                                   init_params['aplusb_220']['ref'],)
    print("Initial chisq value: " + str(-log_test))
    num_params = len(variables)
    num_dof = len(bin_centers)
    red_chisq = -log_test/(num_dof-num_params)
    print("Reduced chisq:" + str(red_chisq))
    updated_info, real_sampler = run(info_dict, resume=True)
    mapname = 'Multicomponent'
    plot_info(variables, updated_info, real_sampler, mapname=mapname, 
              outfile=f'{mapname}_triagplot.png', output_plots=output_plots)
    all_samplers = []
    for sim_num in range(max_sim): 
        print('~~~~~ Running sim ' + str(sim_num) + '  ~~~~~~~~~~~~~~~~~~~')
        output_plots = f'output_plots_ede{str(not zero_ede)}_multicomp{bin_str}/sim_num{sim_num}/'
        outpath = f'mcmc_chains_ede{str(not zero_ede)}_multicomp{bin_str}/simnum{sim_num}'
        ensure_directory(output_plots) 
        for mapname in MAP_FREQS:
            GLOBAL_VAR['EB_observed'+ '_' + mapname] = GLOBAL_VAR['EB_sims_' + mapname][:, sim_num]
            if(do_run):
                eld.plot_cldl(bin_centers, GLOBAL_VAR,  output_plots, mapname)
        info_dict = get_eb_axion_infodict(outpath, variables, priors,
                                      likelihood_func=eb_axion_multicomponent_mcmc_runner)
        init_params = info_dict['params']
        if(do_run):
            updated_info, sampler = run(info_dict, resume=True)
            all_samplers.append(sampler)
            mapname = 'Multicomponent'
            
            plot_info(variables, updated_info, sampler, mapname=mapname, outfile=f'{mapname}_triagplot.png', output_plots=output_plots)
            plot_best_fit_multicomponent(sampler_sims=all_samplers, bin_centers=bin_centers, 
                                 output_plots=output_plots, residuals=False, real_sampler=sampler) 
        else:
            sampler = outpath + '.1.txt'
            all_samplers.append(sampler)
    
    for sim_num in range(3):
        output_plots = f'output_plots_ede{str(not zero_ede)}_multicomp{bin_str}/sim_num{sim_num}/'
        plot_best_fit_multicomponent(sampler_sims=all_samplers, bin_centers=bin_centers, 
                                 output_plots=output_plots, residuals=False, real_sampler=None, sim_num=sim_num)    

def plot_chisq_hist(sim_results_file):
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

def get_mcmc_results_to_df(results_file):
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
    
def plot_corner(outfile, sim_results_file, real_results_file):
    import corner

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
  

def main():
    for bins in [10, 17]:
        for zero_ede in [False]:
            #multi_freq_analysis(max_sim=499, do_run=False, bin_num=bins, zero_ede=zero_ede)



            mcmc_dir = 'mcmc_chains_ede' + str(not zero_ede) + '_multicomp_bin' + str(bins) + '/'
            plots_dir = 'output_plots_ede' + str(not zero_ede) + '_multicomp_bin' + str(bins) + '/'
            for sims in range(3):
                file_all_sims = plots_dir + 'sim_results_multicomp.csv'

                simnum = 'simnum' + str(sims)
                #simnum = 'real'
                file_real = mcmc_dir + simnum + '.1.txt'
                
                sim_dir = 'sim_num' + str(sims)
                
                outfile = plots_dir + '/' + sim_dir + '/' + simnum + '_and_sims_corner.png'
                
                plot_corner(outfile,  file_all_sims, file_real)
    
    #matplotlib.use('Agg')
    
    #single_freq_analysis(max_sim=0, bin_num=bins)

if __name__ == '__main__':
    main()
    
    