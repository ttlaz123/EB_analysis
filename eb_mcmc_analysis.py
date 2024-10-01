print('Importing Packages')
import os
import numpy as np

from scipy.optimize import minimize
import time
import pickle
import argparse



import pandas as pd
from cobaya.run import run


import eb_load_data as eld
import eb_plot_data as epd
import bicep_data_consts as bdc
GLOBAL_VAR = {}
MAP_FREQS = bdc.MAP_FREQS

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
    return total_log_likelihood/2






def objective_function(aplusb, C_eb_observed, C_eb_var, C_eb_ede, C_ee_cmb, C_bb_cmb, gMpl):
    """
    Objective function to be minimized during the optimization process.

    Parameters:
    -----------
    aplusb : float
        The sum of angle parameters `a` and `b` used in the rotation of polarization modes.
    
    C_eb_observed : ndarray
        The observed EB power spectrum (Dl) values.
    
    C_eb_var : ndarray
        The variance of the observed EB power spectrum.
    
    C_eb_ede : ndarray
        The predicted EB power spectrum contribution from early dark energy (EDE).
    
    C_ee_cmb : ndarray
        The predicted EE power spectrum from the CMB.
    
    C_bb_cmb : ndarray
        The predicted BB power spectrum from the CMB.
    
    gMpl : float
        A scaling factor related to the effective gravitational constant.

    Returns:
    --------
    negative_log_likelihood : float
        The negative log-likelihood value for the EB power spectrum.
    """
    return -eb_log_likelihood_vector(C_eb_observed, C_eb_var, C_eb_ede, C_ee_cmb, C_bb_cmb, aplusb, gMpl)


def polar_rotation_likelihood():
    """
    Optimizes the `aplusb` parameter for the polar rotation likelihood model and computes its error bars.

    Returns:
    --------
    best_fit_aplusb : float
        The best-fit value for the `aplusb` parameter.
    
    error_bars : float
        The error bars for the `aplusb` parameter.
    """
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
    """
    Computes the total log-likelihood for the multi-component EB axion model.

    Parameters:
    -----------
    gMpl : float
        A scaling factor related to the effective gravitational constant.
    
    aplusb_b95, aplusb_b95ext, aplusb_k95, aplusb_150, aplusb_220 : float
        The `aplusb` parameters for different frequency bands.

    Returns:
    --------
    multi_likelihood : float
        The total log-likelihood value for the multi-component model, summed over all frequency bands.
    """
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
    """
    Computes the log-likelihood for the EB axion model with the given `aplusb` and `gMpl` parameters.

    Parameters:
    -----------
    aplusb : float
        The sum of angle parameters `a` and `b` used in the rotation of polarization modes.
    
    gMpl : float
        A scaling factor related to the effective gravitational constant.

    Returns:
    --------
    likelihood : float
        The log-likelihood value for the EB axion model.
    """
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

    likelihood_func : callable
        The function used to compute the likelihood, which should be compatible with the MCMC sampler.

    Returns:
    --------
    info : dict
        A dictionary structured to define the settings and parameters for running an MCMC sampling.
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




def ensure_directory(path):
    """Ensure that the directory exists, create it if it doesn't."""
    if not os.path.exists(path):
        os.makedirs(path)


def run_mcmc_for_simulation(mapname, sim_num, bin_centers, variables, priors, bin_str='', zero_ede=False):
    """Run MCMC analysis for a specific map and simulation number."""
    output_plots = f'output_plots_ede{str(not zero_ede)}{bin_str}/simnum{sim_num}'
    outpath = f'mcmc_chains_ede{str(not zero_ede)}{bin_str}/{mapname}/simnum{sim_num}'
    ensure_directory(output_plots)
    GLOBAL_VAR['EB_observed'] = GLOBAL_VAR['EB_sims'][:, sim_num]
    info_dict = get_eb_axion_infodict(outpath, variables, priors, likelihood_func=eb_axion_mcmc_runner)
    init_params = info_dict['params']
    updated_info, sampler = run(info_dict, resume=True)
    gMpl, aplusb, gMpl_std, aplusb_std = epd.plot_best_fit(GLOBAL_VAR, sampler, bin_centers=bin_centers, mapname=mapname, output_plots=output_plots)
    
    epd.plot_info(variables, updated_info, sampler, mapname=mapname, outfile=f'{mapname}_triagplot.png', output_plots=output_plots)
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
    gMpl, aplusb, gMpl_std, aplusb_std = epd.plot_best_fit(GLOBAL_VAR,sampler, bin_centers=bin_centers, mapname=mapname, output_plots=output_plots)
    
    epd.plot_info(variables, updated_info, sampler, mapname=mapname, outfile=f'{mapname}_triagplot.png', output_plots=output_plots)
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
        epd.scatter_sims(csv_resultfile, mapname=mapname)
    
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
        epd.plot_cldl(bin_centers, GLOBAL_VAR,  output_plots, mapname)
    
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
    epd.plot_info(variables, updated_info, real_sampler, mapname=mapname, 
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
                epd.plot_cldl(bin_centers, GLOBAL_VAR,  output_plots, mapname)
        info_dict = get_eb_axion_infodict(outpath, variables, priors,
                                      likelihood_func=eb_axion_multicomponent_mcmc_runner)
        init_params = info_dict['params']
        if(do_run):
            updated_info, sampler = run(info_dict, resume=True)
            all_samplers.append(sampler)
            mapname = 'Multicomponent'
            
            epd.plot_info(variables, updated_info, sampler, mapname=mapname, outfile=f'{mapname}_triagplot.png', output_plots=output_plots)
            epd.plot_best_fit_multicomponent(GLOBAL_VAR=GLOBAL_VAR, sampler_sims=all_samplers, bin_centers=bin_centers, 
                                 output_plots=output_plots, residuals=False, real_sampler=sampler) 
        else:
            sampler = outpath + '.1.txt'
            all_samplers.append(sampler)
    
    for sim_num in range(3):
        if(not sim_num is None):
            output_plots = f'output_plots_ede{str(not zero_ede)}_multicomp{bin_str}/sim_num{sim_num}/'
            for mapname in MAP_FREQS:
                GLOBAL_VAR['EB_observed'+ '_' + mapname] = GLOBAL_VAR['EB_sims_' + mapname][:, sim_num]
     
        else:
            output_plots = f'output_plots_ede{str(not zero_ede)}_multicomp{bin_str}/real/'
        
        epd.plot_best_fit_multicomponent(GLOBAL_VAR=GLOBAL_VAR,sampler_sims=all_samplers, bin_centers=bin_centers, 
                                 output_plots=output_plots, residuals=False, real_sampler=all_samplers[sim_num], sim_num=sim_num)    



    

  

def main():
    for bins in [17]:
        for zero_ede in [True]:
            single_freq_analysis(max_sim=0, bin_num=bins, zero_ede=zero_ede)
            #multi_freq_analysis(max_sim=1, do_run=True, bin_num=bins, zero_ede=zero_ede)


            '''
            mcmc_dir = 'mcmc_chains_ede' + str(not zero_ede) + '_multicomp_bin' + str(bins) + '/'
            plots_dir = 'output_plots_ede' + str(not zero_ede) + '_multicomp_bin' + str(bins) + '/'
            for sims in range(3):
                file_all_sims = plots_dir + 'sim_results_multicomp.csv'

                simnum = 'simnum' + str(sims)
                #simnum = 'real'
                file_real = mcmc_dir + simnum + '.1.txt'
                
                sim_dir = 'sim_num' + str(sims)
                
                outfile = plots_dir + '/' + sim_dir + '/' + simnum + '_and_sims_corner.png'
                
                epd.plot_corner(outfile,  file_all_sims, file_real)
            '''
    
    
    #matplotlib.use('Agg')
    
    #

if __name__ == '__main__':
    main()
    
    