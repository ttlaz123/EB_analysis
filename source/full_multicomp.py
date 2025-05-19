import argparse
import os
import glob
import shutil
import copy
import numpy as np
#import matplotlib.pyplot as plt

from concurrent.futures import ProcessPoolExecutor, as_completed
from cobaya.run import run
from cobaya.likelihood import Likelihood

import eb_load_data as ld
import eb_file_paths as fp
import eb_calculations as ec
import eb_plot_data as epd
#import BK18_full_multicomp

SHARED_DATA_DICT = {}
FILE_PATHS = {}
class BK18_full_multicomp(Likelihood):
    # define variables
    params_names = [] 
    map_set=None
    dataset= None
    forecast=False 
    bin_num=14 
    used_maps = []
    theory_comps='all' 
    spectra_type='all'
    injected_signal = {}
    sim_common_data = {}
    observe_filepath = None
    def __init__(self, *args, **kwargs):
        if('used_maps' in kwargs):
            self.used_maps = kwargs['used_maps']
            if('zero_offdiag' in kwargs):
                self.zero_offdiag = kwargs['zero_offdiag']
            if('map_set' in kwargs):
                self.map_set = kwargs['map_set']
            if('bin_num' in kwargs):
                self.bin_num = kwargs['bin_num']
            if('theory_comps' in kwargs):
                self.theory_comps = kwargs['theory_comps']
            if('spectra_type' in kwargs):
                self.spectra_type = kwargs['spectra_type']
            if('injected_signal' in kwargs):
                self.injected_signal = kwargs['injected_signal']
            # Passing in the dict slows down the process by a lot
            #if('sim_common_data' in kwargs):
            #    self.sim_common_data = kwargs['sim_common_data']
            if('observe_filepath' in kwargs):
                self.observe_filepath = kwargs['observe_filepath']
            self.initialize()
        else:
            super().__init__(*args, **kwargs)

    def initialize(self):
        self.sim_common_data = SHARED_DATA_DICT
        self.map_reference_header = self.sim_common_data['map_reference_header']
        if(len(self.used_maps) == 0):
            self.used_maps = self.sim_common_data['used_maps']
        self.bandpasses = self.sim_common_data['bandpasses']
        self.bpwf = self.sim_common_data['bpwf']
        self.dl_theory = self.sim_common_data['theory_spectra']
        self.filtered_covmat = self.sim_common_data['covmat']
        self.full_covmat = self.sim_common_data['full_covmat']
        self.cov_inv = self.sim_common_data['inv_covmat']
        self.binned_dl_observed_dict, self.map_reference_header = ld.load_observed_spectra(
                                                            self.observe_filepath,
                                                            self.used_maps,
                                                            self.map_reference_header,
                                                            num_bins = self.bin_num)
        if(len(self.injected_signal) > 1):
            self.binned_dl_observed_dict = ec.inject_signal_prebin(self.used_maps,
                                                        self.injected_signal, 
                                                        self.dl_theory,
                                                        self.binned_dl_observed_dict,
                                                        self.bpwf,
                                                        self.map_reference_header)
        self.initial_theory_dict = ec.apply_initial_conditions(self.dl_theory, self.used_maps)
        self.binned_dl_observed_vec = self.dict_to_vec(self.binned_dl_observed_dict, 
                                                    self.used_maps)
        #print('Showing figs')
        #plt.figure()
        #plt.plot(self.dl_theory['EE']/100, label='EE/100')
        #plt.plot(self.dl_theory['EB_EDE'], label = 'EB EDE')
        #plt.legend()
        #plt.show()

    def get_requirements(self):
        """
        Specify what is needed from other components, such as theory predictions.
        
        Returns:
            dict: A dictionary that specifies the requirements for the likelihood calculation,
                including 'theory' calculations for the parameters in params_names.
        """
        # Specify that this likelihood requires theoretical predictions based on the parameters defined in params_names
        requirements = {
            "theory": self.params_names  # List of parameters that require theoretical predictions
        }
        return self.params_names
    
    def logp(self, **params_values):
        """
        Calculate the log-likelihood based on the current parameter values.
        """

        # Get the theoretical predictions based on the parameter values
        #print(params_values)
        theory_prediction = self.theory(params_values)

        # Calculate the residuals
        residuals = self.binned_dl_observed_vec - theory_prediction
        # Calculate the Mahalanobis distance using the inverse covariance matrix
        chi_squared = residuals.T @ self.cov_inv @ residuals
        if(self.theory_comps == 'eskilt'):
            #chi_squared = 0
            chi_squared += self.theory_eskilt(params_values)
        # Calculate the log-likelihood
        log_likelihood = -0.5 * chi_squared
        #print(log_likelihood)
        return log_likelihood

    def dict_to_vec(self, spectra_dict, used_maps):
        """
        Concatenates spectra from a dictionary into one large vector, following the order in `map_reference_header`.

        Args:
            spectra_dict (dict): A dictionary where keys are map names and values are corresponding spectra (numpy arrays).


        Returns:
            ndarray: A concatenated 1D array containing all the spectra in the given order, 
                    only including maps that exist in `spectra_dict`.
        """
        big_vector = []

        for map_name in self.map_reference_header:
            
            if map_name in used_maps:
                spectype = ec.determine_spectrum_type(map_name)
                
                spec = spectra_dict[map_name].copy()
                if(spectype == 'BB'):
                    spec/=1
                big_vector.append(spec)

        # Concatenate all spectra arrays into a single 1D array
        concat_vec =  np.concatenate(big_vector, axis=0)

        return concat_vec   
    
    def theory_eskilt(self, params_values):
        """
        Compute chi-squared for Eskilt model fit to EB data.

        Parameters:
        -----------
        params_values : dict
            Dictionary containing 'alpha_CMB' (degrees) and 'gMpl' parameters.

        Returns:
        --------
        chisq : float
            Chi-squared value.
        """
        angle_deg = params_values['alpha_CMB']
        g = params_values['gMpl']

        observed = self.sim_common_data['eskilt']['EB_observed']
        var = self.sim_common_data['eskilt']['EB_var']
        ee = self.sim_common_data['eskilt']['EE_binned']
        bb = self.sim_common_data['eskilt']['BB_binned']
        eb = self.sim_common_data['eskilt']['EB_EDE']
    
        sin4theta = np.sin(np.deg2rad(4 * angle_deg))/2
        cos4theta = np.cos(np.deg2rad(4 * angle_deg))
        expected = sin4theta * (ee - bb) + eb * cos4theta * g
        residual = observed - expected
        #plt.figure()
        #plt.plot(expected, label = 'alpha*ee+g*eb')
        #plt.plot(observed, label='obs')
        #plt.plot(ee * 0.01, label='0.01*ee')
        #plt.plot(eb * 0.1, label='0.1*eb')
        #plt.legend()
        #plt.show()
        chisq = np.sum(residual**2 / (var))
        return chisq

    def theory(self, params_values, override_maps=None):
        # define relevant dictionaries
        if(self.theory_comps in ['all', 'fixed_dust', 'eskilt']):
            # do ede shift
            post_inflation_dict = ec.apply_EDE(self.initial_theory_dict,
                                               params_values,
                                               self.dl_theory,
                                               self.used_maps)
        else:
            post_inflation_dict = self.initial_theory_dict
        if(self.theory_comps in ['all', 'no_ede']):
            
            # do cmb rotation
            '''
            post_travel_dict = ec.apply_cmb_rotation(post_inflation_dict,
                                                    params_values,
                                                    self.dl_theory,
                                                    self.used_maps)
            '''
            # do dust
            post_travel_dict = ec.apply_dust(post_inflation_dict, self.bandpasses, params_values)
            
        else: 
            post_travel_dict = post_inflation_dict
        if(self.theory_comps in ['all', 'det_polrot', 'fixed_dust', 'no_ede', 'eskilt']):
            # do detector rotation
            post_detection_dict = ec.apply_det_rotation(post_travel_dict, 
                                                        params_values, 
                                                        self.dl_theory,
                                                        override_maps=override_maps)


        # apply bpwf
        self.final_detection_dict = ec.apply_bpwf(self.map_reference_header,
                                      post_detection_dict,
                                      self.bpwf,
                                      self.used_maps,
                                      do_cross=True)
        theory_vec = self.dict_to_vec(self.final_detection_dict, self.used_maps)
        return theory_vec
    
def load_all_sims(input_args):
    load_shared_data(input_args)
    dirpath = FILE_PATHS['sim_path']
    obs_list = []
    for sim_num in range(input_args.sim_start, input_args.sim_num):
        formatted_simnum = str(sim_num).zfill(3)
        observation_file_path = dirpath.replace('XXX', formatted_simnum)
        used_maps = SHARED_DATA_DICT['used_maps']
        try:
            binned_dl_observed_dict, map_reference_header = ld.load_observed_spectra(
                                                            observation_file_path,
                                                            used_maps,
                                                            None,
                                                            num_bins = input_args.bin_num)
        except FileNotFoundError:
            print("Skipping file: " + str(observation_file_path))
            continue
        obs_list.append(binned_dl_observed_dict)
    return obs_list


def load_shared_data(input_args):
    """
    Loads and initializes shared data structures including bandpasses, theory spectra,
    covariance matrices, and used map configurations. The result is stored in global
    dictionaries SHARED_DATA_DICT and FILE_PATHS.

    Args:
        input_args: Parsed arguments from argparse specifying dataset, bin number, etc.

    Modifies:
        SHARED_DATA_DICT (dict): Shared data for use across likelihoods and simulations.
        FILE_PATHS (dict): Dictionary of file paths to required datasets.
    """
    global SHARED_DATA_DICT, FILE_PATHS
    map_reference_header = None
    FILE_PATHS = fp.set_file_paths(input_args.dataset, input_args.fede)
    
    SHARED_DATA_DICT['bpwf'], map_reference_header = ld.load_bpwf(FILE_PATHS['bpwf'], 
                                            map_reference_header, 
                                            num_bins=input_args.bin_num)
    SHARED_DATA_DICT['theory_spectra'] = ld.load_cmb_theory(FILE_PATHS['camb_lensing'])
                                                             #FILE_PATHS['dust_models'],
                                                             #input_args.theory_comps)
    SHARED_DATA_DICT['theory_spectra'] = ld.load_ede_spectra(FILE_PATHS['EDE_spectrum'], SHARED_DATA_DICT['theory_spectra'])
    SHARED_DATA_DICT['bandpasses'] = ld.read_bandpasses(FILE_PATHS['bandpasses'])
    SHARED_DATA_DICT['map_reference_header'] = map_reference_header
    
    covmat_name = 'covariance_matrix'
    calc_spectra = ec.determine_map_freqs(input_args.map_set)
    do_crosses = True
    used_maps = generate_cross_spectra(calc_spectra, 
                                       do_crosses=do_crosses, 
                                       spectra_type=input_args.spectra_type)
    
    SHARED_DATA_DICT['used_maps'] = ec.filter_used_maps(map_reference_header, used_maps)
    all_maps = generate_cross_spectra(calc_spectra, 
                                       do_crosses=do_crosses, 
                                       spectra_type='all')
    all_maps = ec.filter_used_maps(map_reference_header, all_maps)                          

    full_covmat = ld.load_covariance_matrix(FILE_PATHS[covmat_name],
                                            map_reference_header)
    filtered_covmat = ec.filter_matrix(map_reference_header, 
                                       full_covmat, 
                                       SHARED_DATA_DICT['used_maps'], 
                                       num_bins=input_args.bin_num)

    #plot_covar_matrix(self.filtered_covmat, used_maps=self.used_maps)
    SHARED_DATA_DICT['inv_covmat'] = ec.calc_inverse_covmat(filtered_covmat)
    SHARED_DATA_DICT['covmat'] = filtered_covmat
    SHARED_DATA_DICT['all_maps'] = all_maps
    SHARED_DATA_DICT['full_covmat'] = ec.filter_matrix(map_reference_header, 
                                       full_covmat, 
                                       all_maps, 
                                       num_bins=input_args.bin_num)
    SHARED_DATA_DICT['full_inv_covmat'] = ec.calc_inverse_covmat(SHARED_DATA_DICT['full_covmat'])
    
    bin_starts, raw_cl, SHARED_DATA_DICT['eskilt'] = ld.load_eskilt_data()

def run_bk18_likelihood(params_dict, observation_file_path, input_args, 
                        rstop = 0.03, max_tries=10000):
    """
    Runs the Cobaya MCMC likelihood using BK18_full_multicomp likelihood class.

    Args:
        params_dict (dict): Dictionary of parameter priors for the MCMC.
        observation_file_path (str): Path to observed or simulated map data.
        input_args (Namespace): Input arguments including map set, dataset, etc.
        rstop (float): R-1 convergence threshold for stopping.
        max_tries (int): Maximum number of samples to try.

    Returns:
        Tuple: (updated_info, sampler) from the Cobaya run.
    """
    likelihood_class = BK18_full_multicomp
    likelihood_class.params_names = list(params_dict.keys())

    # Create Cobaya info dictionary
    info = {
        "likelihood": {
            "my_likelihood": {
                "external": likelihood_class,
                "used_maps": SHARED_DATA_DICT['used_maps'],
                "map_set": input_args.map_set,
                "dataset": input_args.dataset,
                "forecast": input_args.forecast,
                "bin_num":  input_args.bin_num,
                "theory_comps": input_args.theory_comps,
                "spectra_type": input_args.spectra_type,
                "injected_signal":input_args.injected_signal,
                #"sim_common_data":SHARED_DATA_DICT,
                "observe_filepath":observation_file_path
            }
        },
        "params": params_dict,
        "sampler":{
            "mcmc": {
                "Rminus1_stop": rstop,
                "max_tries": max_tries,
            }
        },
        "output": input_args.output_path,
        "resume": True
    }

    # Run Cobaya
    print('Running cobaya')
    updated_info, sampler = run(info, stop_at_error=True)
    return updated_info, sampler

def define_priors(calc_spectra, theory_comps, angle_degree=5):
    """
    Defines prior distributions for angle parameters, dust parameters, and EDE params.

    Args:
        calc_spectra (list): List of spectrum types (e.g., ['BK18_95', ...]).
        theory_comps (str): Which theoretical components to include ('all', 'fixed_dust', etc.).
        angle_degree (float): Range of prior on alpha rotation parameters.

    Returns:
        dict: Dictionary defining Cobaya-compatible priors for parameters.
    """
    # define angles based on mapopts
    angle_priors = {"prior": {"min": -angle_degree, "max": angle_degree}, "ref": 0}
    params_dict = {
        'alpha_' + spectrum: {
                **angle_priors,
                'latex': ("\\alpha_{" + 
                            spectrum.replace('_', '\\_') +
                            "}")
                }
                for spectrum in calc_spectra    
    }
    

    # dust priors
    A_dust_priors = {"prior":{"min": -50, "max":150}, 
                            "ref": {"dist":"norm", "loc":6, "scale":1},
                            "proposal":1}
    alpha_dust_priors = {"prior":{"min": -1, "max":1}, 
                                "ref": {"dist":"norm", "loc":-0, "scale":0.01},
                                "proposal":0.1}

    if(theory_comps == 'all'):
        params_dict['gMpl'] = {"prior": {"min": -10, "max": 10}, "ref": 0}
        for spec in ['EE', 'BB']:#, 'EB']:

            params_dict['A_dust_' + spec] = {**A_dust_priors,
                                        "latex":"A_{"+spec+",\mathrm{dust}}"}
            params_dict['alpha_dust_' + spec] = {**alpha_dust_priors,
                                        "latex":"\\alpha_{"+spec+",\mathrm{dust}}"}
        for spec in ['EB']:

            params_dict['A_dust_' + spec] = {"prior":{"min": -3, "max":3}, 
                                            "ref": {"dist":"norm", "loc":0, "scale":0.1},
                                            "proposal":0.1,
                                            "latex":"A_{"+spec+",\mathrm{dust}}"}
            params_dict['alpha_dust_' + spec] = {**alpha_dust_priors,
                                        "latex":"\\alpha_{"+spec+",\mathrm{dust}}"}
        params_dict['beta_dust'] = {"prior":{"min": 0.8, "max":2.4}, 
                                    "ref": {"dist":"norm", "loc":1.6, "scale":0.02},
                                    "proposal":0.02,
                                    "latex":"\\beta_{\mathrm{dust}}"}
    elif(theory_comps == 'eskilt'):
        params_dict['alpha_CMB'] = angle_priors
        params_dict['gMpl'] = {"prior": {"min": -10, "max": 10}, "ref": 0}
    elif(theory_comps == 'det_polrot'):
        pass

    elif(theory_comps == 'fixed_dust'):
        params_dict['gMpl'] = {"prior": {"min": -10, "max": 10}, "ref": 0}

    elif(theory_comps == 'no_ede'):
        params_dict['alpha_CMB'] = angle_priors
        for spec in ['EE', 'BB']:#, 'EB']:

            params_dict['A_dust_' + spec] = {**A_dust_priors,
                                        "latex":"A_{"+spec+",\mathrm{dust}}"}
            params_dict['alpha_dust_' + spec] = {**alpha_dust_priors,
                                        "latex":"\\alpha_{"+spec+",\mathrm{dust}}"}
        for spec in ['EB']:

            params_dict['A_dust_' + spec] = {"prior":{"min": -3, "max":3}, 
                                            "ref": {"dist":"norm", "loc":0, "scale":0.1},
                                            "proposal":0.1,
                                            "latex":"A_{"+spec+",\mathrm{dust}}"}
            params_dict['alpha_dust_' + spec] = {**alpha_dust_priors,
                                        "latex":"\\alpha_{"+spec+",\mathrm{dust}}"}
        params_dict['beta_dust'] = {"prior":{"min": 0.8, "max":2.4}, 
                                    "ref": {"dist":"norm", "loc":1.6, "scale":0.02},
                                    "proposal":0.02,
                                    "latex":"\\beta_{\mathrm{dust}}"}
    else:
        raise ValueError()

    
    return params_dict


def generate_cross_spectra(calc_spectra, do_crosses, spectra_type):
    """
    Generates a list of cross-spectra combinations for E/B modes between maps.

    Args:
        calc_spectra (list): List of maps to use (e.g., ['BK18_95']).
        do_crosses (bool): Whether to include cross-spectra between different maps.
        spectra_type (str): 'all' or 'eb', determines which spectra to include.

    Returns:
        list: List of strings representing spectra types.
    """
    cross_spectra = []
    for spec1 in calc_spectra:
        for spec2 in calc_spectra:
            # don't do cross spectra
            if(not spec1 == spec2 and not do_crosses):
                continue
            if(spectra_type == 'all'):
                cross_spectrum = f"{spec1}_Ex{spec2}_B"
                cross_spectra.append(cross_spectrum)
                cross_spectrum = f"{spec1}_Bx{spec2}_E"
                cross_spectra.append(cross_spectrum)
                cross_spectrum = f"{spec1}_Ex{spec2}_E"
                cross_spectra.append(cross_spectrum)
                cross_spectrum = f"{spec1}_Bx{spec2}_B"
                cross_spectra.append(cross_spectrum)
            elif(spectra_type == 'nob'):
                cross_spectrum = f"{spec1}_Ex{spec2}_B"
                cross_spectra.append(cross_spectrum)
                cross_spectrum = f"{spec1}_Bx{spec2}_E"
                cross_spectra.append(cross_spectrum)
                cross_spectrum = f"{spec1}_Ex{spec2}_E"
                cross_spectra.append(cross_spectrum)
            elif(spectra_type == 'noe'):
                cross_spectrum = f"{spec1}_Ex{spec2}_B"
                cross_spectra.append(cross_spectrum)
                cross_spectrum = f"{spec1}_Bx{spec2}_E"
                cross_spectra.append(cross_spectrum)
                cross_spectrum = f"{spec1}_Bx{spec2}_B"
                cross_spectra.append(cross_spectrum)
            elif(spectra_type == 'eb'):
                cross_spectrum = f"{spec1}_Ex{spec2}_B"
                cross_spectra.append(cross_spectrum)
                cross_spectrum = f"{spec1}_Bx{spec2}_E"
                cross_spectra.append(cross_spectrum)
    return  cross_spectra 

def get_injected_signal(calc_spectra, signal_type):
    injected_signal = {}
    num_spectra = len(calc_spectra)
    count = 0
    for spec in calc_spectra:
        if(signal_type == 'pos'):
            sig = count / num_spectra
        elif(signal_type == 'neg'):
            sig = -count / num_spectra
        elif(signal_type == 'bal'):
            sig = count / num_spectra * 2 - 1

        else:
            print('No signal injected')
            return {} 
        count += 1
        injected_signal['alpha_' + spec] = sig
    print('Injected signal: ' + str(injected_signal))
    return injected_signal

def multicomp_mcmc_driver(input_args):
    """
    Top-level function for managing a full Cobaya MCMC run or batch of simulations.

    Args:
        input_args (Namespace): Parsed command-line arguments defining the run.
    """
    # full multicomp driver
    # define maps based on mapopts
    load_shared_data(input_args)
    calc_spectra = ec.determine_map_freqs(input_args.map_set)
    # define dust params based on dustopts
    params_dict = define_priors(calc_spectra, input_args.theory_comps)
    input_args.injected_signal = get_injected_signal(calc_spectra, 
                                            signal_type=input_args.injected_signal)

    if(input_args.sim_num > 2):
        parallel_simulation(input_args, params_dict)
    else:
        # define relevant files based on opts
        if(input_args.sim_num == -1 or input_args.sim_start == -1):
            observation_file_path = FILE_PATHS['observed_data']
        elif(isinstance(input_args.sim_start, int) and input_args.sim_start >= 1):
            formatted_simnum = str(input_args.sim_start).zfill(3)
            observation_file_path = FILE_PATHS['sim_path'].replace('XXX', formatted_simnum)
    
        
   
        if(input_args.overwrite):
            updated_info, sampler = run_bk18_likelihood(params_dict, 
                                                        observation_file_path, 
                                                        input_args)
            replace_dict = {
                #'alpha_BK18_220': 0,
                #'alpha_BK18_150': 0,
                #'alpha_BK18_K95': 0,
                #'alpha_BK18_B95e': 0,
                #'gMpl': 1,
            }
            param_names, means, mean_std_strs = epd.plot_triangle(input_args.output_path, replace_dict)
            used_maps = SHARED_DATA_DICT["all_maps"]
            multicomp_class = BK18_full_multicomp(
                            used_maps=used_maps,
                            map_set= input_args.map_set,
                            dataset= input_args.dataset,
                            forecast= input_args.forecast,
                            bin_num= input_args.bin_num,
                            theory_comps= input_args.theory_comps,
                            spectra_type= input_args.spectra_type,
                            injected_signal = input_args.injected_signal,
                            #"sim_common_data":SHARED_DATA_DICT,
                            observe_filepath= observation_file_path,
                            sim_common_dat = SHARED_DATA_DICT)
            
            epd.plot_eebbeb(multicomp_class, 
                           input_args.output_path, 
                           param_names, 
                           means, 
                           mean_std_strs,
                           override_maps = used_maps)
    # plot mcmc results
    replace_dict ={}# {"alpha_BK18_220":0.6}
    print(input_args.output_path)
    '''
    
    
    
    '''
    return 

def run_simulation(sim_num, params_dict,input_args):
    """
    Runs a single simulation for a given simulation number.

    Args:
        sim_num (int or str): Simulation index or 'real' for real data.
        params_dict (dict): Priors and parameter definitions.
        input_args (Namespace): Parsed command-line args.
    """ 
    outpath = f"{input_args.output_path}{sim_num:03d}"
    # skip chains that already exist
    if(os.path.exists(outpath + '.1.txt') and os.path.exists(outpath + '_bestfitBB.png')):
        print(f"Skipping existing simulation {sim_num}")
        return
    if(sim_num == 'real'):
            observation_file_path = FILE_PATHS['observed_data']
    elif(isinstance(sim_num, int) and sim_num >= 0):
        formatted_simnum = str(sim_num).zfill(3)
        observation_file_path = FILE_PATHS['sim_path'].replace('XXX', formatted_simnum)

    params_copy = copy.deepcopy(params_dict)
    input_args.output_path = outpath
    updated_info, sampler = run_bk18_likelihood(params_copy, 
                                            observation_file_path, 
                                            input_args)
    param_names, means, mean_std_strs = epd.plot_triangle(input_args.output_path)#, replace_dict)
    used_maps = SHARED_DATA_DICT["all_maps"]
    multicomp_class = BK18_full_multicomp(
                    used_maps=used_maps,
                    map_set= input_args.map_set,
                    dataset= input_args.dataset,
                    forecast= input_args.forecast,
                    bin_num= input_args.bin_num,
                    theory_comps= input_args.theory_comps,
                    spectra_type= input_args.spectra_type,
                    injected_signal = input_args.injected_signal,
                    #"sim_common_data":SHARED_DATA_DICT,
                    observe_filepath= observation_file_path,
                    sim_common_dat = SHARED_DATA_DICT)
    epd.plot_eebbeb(multicomp_class, 
                    input_args.output_path, 
                    param_names, 
                    means, 
                    mean_std_strs,
                    override_maps = used_maps)
    del updated_info, sampler, param_names, means, mean_std_strs, multicomp_class

# Parallel execution with cancellation support
def parallel_simulation(input_args, params_dict):
    """
    Runs multiple simulations in parallel using ProcessPoolExecutor.

    Args:
        input_args (Namespace): Command-line arguments specifying simulation range.
        params_dict (dict): Parameter prior definitions.

    Raises:
        KeyboardInterrupt: If user interrupts execution (gracefully shuts down workers).
    """
    sim_indices = range(input_args.sim_start, input_args.sim_start + input_args.sim_num)
    try:
        maxworkers = 10 
        with ProcessPoolExecutor(max_workers=maxworkers) as executor:
            # Submit all tasks to the executor
            future_to_sim = {
                executor.submit(run_simulation, s,
                                params_dict, 
                                input_args): s
                for s in sim_indices
            }
            for future in as_completed(future_to_sim):
                try:
                    # Wait for task to complete
                    future.result()
                except Exception as e:
                    print(f"Simulation {future_to_sim[future]} failed with error: {e}")
    except KeyboardInterrupt:
        print("Cancelling all simulations...")
        executor.shutdown(cancel_futures=True)  # Terminates all running tasks
        raise  # Re-raise the KeyboardInterrupt to exit the program cleanly

def do_plotting(input_args):
    '''
    for sim in range(input_args.sim_start, input_args.sim_start+input_args.sim_num):
        outpath = f"{input_args.output_path}{sim:03d}"
        try:
            epd.plot_triangle(outpath)
        except OSError:
            print('File note found, skipping: ' + str(outpath))
            continue
    '''
    chains_path = input_args.output_path + "XXX.1.txt"
    epd.plot_sim_peaks(chains_path, input_args.sim_start, input_args.sim_num)

def parse_bin_range(s):
    """
    Parse a string representing a bin range into a list of integers.

    This function supports two formats:
    - A single positive integer "n", which is interpreted as the range [2, n].
    - A hyphenated range "start-end", which is interpreted as the range [start, end], inclusive.

    Args:
        s (str): The input string specifying the bin range.

    Returns:
        list of int: A list of integers representing the parsed bin indices.

    Raises:
        argparse.ArgumentTypeError: If the input is not a valid positive integer or range.

    Examples:
        >>> parse_bin_range("5")
        [2, 3, 4, 5]
        >>> parse_bin_range("3-6")
        [3, 4, 5, 6]
        >>> parse_bin_range("1")
        []
        >>> parse_bin_range("7-3")
        Traceback (most recent call last):
            ...
        argparse.ArgumentTypeError: Must be a positive integer (interpreted as 2-n) or a valid range like 3-7.
    """
    try:
        if '-' in s:
            start, end = map(int, s.split('-'))
            if start > end:
                raise ValueError
            return list(range(start, end + 1))
        else:
            end = int(s)
            if end < 2:
                return []
            return list(range(2, end + 1))
    except ValueError:
        raise argparse.ArgumentTypeError(
            "Must be a positive integer (interpreted as 2-n) or a valid range like 3-7."
        )


def main():
    parser = argparse.ArgumentParser(
        description="Run multicomponent EB MCMC analysis using BICEP/Keck data."
    )

    parser.add_argument('-m', '--map_set', type=str, default='BK18',
    help="""
            Frequency combination to use for computing spectra. Determines which spectra to include in the likelihood calculation.

            Default: BK18

            Available options:
            - BK18: BK18-only maps [220, 150, K95, B95e]
            - BK18_planck: Combines BK18 with Planck maps [includes P030e, P044e, P143e, P217e, P353e]
            - planck: Planck-only maps
            - BK_good: Subset of cleaner BK18 frequencies [150, B95e]
            - BK_bad: Potentially contaminated BK18 maps [220, K95]
            - <custom>: Specify a single map directly (e.g., BK18_150)

            Used to determine which `dust_model` and `bandpass` files are included for each frequency channel.
            """)

    parser.add_argument('-d', '--dataset', type=str, default='BK18lf_fede01',
    help="""
            Name of the dataset directory to use. This sets the data directory and file naming scheme.

            Default: BK18lf_fede01

            Available options include:
            ~~~~ real only ~~~~
            - BK18lf: BICEP/Keck 2018 likelihood baseline dataset.
            - BK18lf_dust: Includes dust modeling.
            - BK18lf_dust_incEE: Includes EE spectra in the analysis.
            - BK18lf_norot: Rotation turned off, excludes EB rotation terms.
            - BK18lf_norot_allbins: Like BK18lf_norot, but includes all bins.
            ~~~~~ sims below ~~~~~~
            - BK18lf_fede01: Simulations with injected fEDE=0.01 signal.
            - BK18lf_fede01_sigl: fEDE=0.01 signal including sig_l scaling.
            - BK18lf_sim: Baseline BK18lf simulations for null testing.
            - BK18lf_mhd    
            - BK18lf_mkd    
            - BK18lf_vansyngel
            - BK18lf_gampmod      
            - BK18lf_pysm1
            - BK18lf_gaussdust    
            - BK18lf_pysm2
            - BK18lf_gdecorr      
            - BK18lf_pysm3

            This choice determines the observed data path, covariance matrix, simulation path, and bandpass files.
            """)

    parser.add_argument(
        '-n', "--sim_num",
        type=int,
        default="1",
        help=(
            "Number of simulations to use. "
            "Set to 500 to run over a batch of simulations in parallel. Default: 1."
        ),
    )

    parser.add_argument(
        '-s', "--sim_start",
        type=lambda x: int(x) if x.isdigit() else x,
        default="real",
        help=(
            "Simulation number to use. Set to 'real' to use real observed data, "
            "or an integer (e.g., 0, 1, ...) to use a specific simulation. "
            "Default: 'real'."
        ),
    )

    
    parser.add_argument(
        '-b', '--bin_num',
        type=parse_bin_range,
        default=list(range(2, 16)),  # Default to 2–15
        help="Bin count (interpreted as 1-n) or a specific range like 3-7. Default: 2-15."
    )

    parser.add_argument(
        '-f', "--forecast",
        action="store_true",
        help="Flag to indicate forecast mode. Default: False (off).",
    )

    parser.add_argument(
        '-o', "--overwrite",
        action="store_true",
        help="Overwrite existing MCMC results. Default: False (off).",
    )
    parser.add_argument(
        '-q', "--plot_peaks",
        action="store_true",
        help="Plot the MCMC sim peaks. Default: False (off).",
    )
    parser.add_argument(
        '-i', "--injected_signal",
        type=str,
        choices=['none', 'pos', 'neg', 'bal'],
        default="none",
        help=(
            "What kind of signal to inject (none, positive, negative, balanced)"
            "Default: 'all'."
        ),
    )
    parser.add_argument(
        '-t', "--spectra_type",
        type=str,
        choices=["all", "eb", "nob", "noe"],
        default="all",
        help=(
            "Which spectra to include. 'all' includes EE, BB, EB, etc., while 'eb' only includes EB-related spectra. "
            "Default: 'all'."
        ),
    )
    parser.add_argument(
        "--fede",
        type=float,
        default="0.07",
        help=(
            "F_ede curve to use for fitting. Set the f_ede parameter"
            "Default: 0.07"
        ),
    )

    parser.add_argument(
        '-c', "--theory_comps",
        type=str,
        choices=["all", "fixed_dust", "det_polrot", "no_ede", "eskilt"],
        default="all",
        help=(
            "Controls which theoretical components are included in the likelihood. "
            "'all': Fit for dust, CMB angle, and EDE (gMpl). "
            "'fixed_dust': Fix dust and CMB angle, fit only gMpl. "
            "'det_polrot': Detectable polarization rotation only. "
            "'no_ede': Only include dust and alpha_CMB, exclude gMpl. "
            "Default: 'all'."
        ),
    )

    parser.add_argument(
        '-p', "--output_path",
        type=str,
        default='chains/default',
        help="Path to directory for storing Cobaya MCMC outputs. Default: chains/default",
    )
    args = parser.parse_args()
    # Check if the overwrite flag is set

    if args.plot_peaks:
        do_plotting(args)
        return
    if args.overwrite:
        # Check if the output path exists
        # Construct the glob pattern to match all files and directories with the specified prefix
        pattern = os.path.join(os.path.dirname(args.output_path), os.path.basename(args.output_path) + '*')

        # Use glob to find all matching files and directories
        matching_items = glob.glob(pattern)
        if len(matching_items) > 1:
            # Prompt user for confirmation
            confirm = input(f"Are you sure you want to delete the existing chains at: {args.output_path}? (y/n): ")
            if confirm.lower() == 'y':
                # Iterate through the matching items and delete them
                for item in matching_items:
                    if os.path.isdir(item):
                        shutil.rmtree(item)  # Remove directory
                    else:
                        os.remove(item)  # Remove file
                print(f"Deleted existing chains at: {args.output_path}")
            else:
                print("Deletion cancelled. Existing chains will be kept.")
        else:
            print(f"No existing chains to overwrite at: {args.output_path}")
    if False:
        observed_datas_list = load_all_sims(input_args=args)
        epd.plot_overlay_sims('BB', observed_datas_list, args.output_path)
    else:
        multicomp_mcmc_driver(args)


if __name__ == '__main__':
    main()
