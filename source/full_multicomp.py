import argparse
import os
import glob
import shutil
import copy
from concurrent.futures import ProcessPoolExecutor, as_completed
from cobaya.run import run

import eb_load_data as ld
import eb_file_paths as fp
import eb_calculations as ec
import eb_plot_data as epd
import BK18_full_multicomp

SHARED_DATA_DICT = {}
FILE_PATHS = {}
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
    FILE_PATHS = fp.set_file_paths(input_args.dataset)
    
    SHARED_DATA_DICT['bpwf'], map_reference_header = ld.load_bpwf(FILE_PATHS['bpwf'], 
                                            map_reference_header, 
                                            num_bins=input_args.bin_num)
    SHARED_DATA_DICT['theory_spectra'] = ld.load_cmb_spectra(FILE_PATHS['camb_lensing'],
                                                             FILE_PATHS['dust_models'],
                                                             input_args.theory_comps)
    SHARED_DATA_DICT['theory_spectra'] = ld.include_ede_spectra(FILE_PATHS['EDE_spectrum'],
                                                                input_args.theory_comps)
    SHARED_DATA_DICT['bandpasses'] = ld.read_bandpasses(FILE_PATHS['bandpasses'])
    SHARED_DATA_DICT['map_reference_header'] = map_reference_header
    
    covmat_name = 'covariance_matrix'
    calc_spectra = ec.determine_map_freqs(input_args.map_set)
    do_crosses = True
    used_maps = generate_cross_spectra(calc_spectra, 
                                       do_crosses=do_crosses, 
                                       spectra_type=input_args.spectra_type)
    SHARED_DATA_DICT['used_maps'] = ec.filter_used_maps(map_reference_header, used_maps)
    full_covmat = ld.load_covariance_matrix(FILE_PATHS[covmat_name])
    filtered_covmat = ec.filter_matrix(map_reference_header, 
                                       full_covmat, 
                                       SHARED_DATA_DICT['used_maps'], 
                                       num_bins=input_args.bin_num)
    #plot_covar_matrix(self.filtered_covmat, used_maps=self.used_maps)
    SHARED_DATA_DICT['inv_covmat'] = ec.calc_inverse_covmat(filtered_covmat)

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
                "sim_common_data":SHARED_DATA_DICT,
                "single_observe_data":observation_file_path
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
    updated_info, sampler = run(info)
    return updated_info, sampler

def define_priors(calc_spectra, theory_comps, angle_degree=3):
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
    A_dust_priors = {"prior":{"min": 0, "max":15}, 
                            "ref": {"dist":"norm", "loc":3, "scale":0.1},
                            "proposal":0.1}
    alpha_dust_priors = {"prior":{"min": -1, "max":0}, 
                                "ref": {"dist":"norm", "loc":-0.5, "scale":0.01},
                                "proposal":0.01}

    if(theory_comps == 'all'):
        params_dict['alpha_CMB'] = angle_priors
        params_dict['gMpl'] = {"prior": {"min": -10, "max": 10}, "ref": 0}
        for spec in ['EE', 'BB', 'EB']:

            params_dict['A_dust_' + spec] = {**A_dust_priors,
                                        "latex":"A_{"+spec+",\mathrm{dust}}"}
            params_dict['alpha_dust_' + spec] = {**alpha_dust_priors,
                                        "latex":"\\alpha_{"+spec+",\mathrm{dust}}"}
        params_dict['beta_dust'] = {"prior":{"min": 0.8, "max":2.4}, 
                                    "ref": {"dist":"norm", "loc":1.6, "scale":0.02},
                                    "proposal":0.02,
                                    "latex":"\\beta_{\mathrm{dust}}"}
    elif(theory_comps == 'det_polrot'):
        pass

    elif(theory_comps == 'fixed_dust'):
        params_dict['gMpl'] = {"prior": {"min": -10, "max": 10}, "ref": 0}

    elif(theory_comps == 'no_ede'):
        params_dict['alpha_CMB'] = angle_priors
        for spec in ['EE', 'BB', 'EB']:

            params_dict['A_dust_' + spec] = {**A_dust_priors,
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

            elif(spectra_type == 'eb'):
                cross_spectrum = f"{spec1}_Ex{spec2}_B"
                cross_spectra.append(cross_spectrum)
                cross_spectrum = f"{spec1}_Bx{spec2}_E"
                cross_spectra.append(cross_spectrum)
    return  cross_spectra 


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
    if(input_args.sim_num == 500):
        parallel_simulation(input_args, params_dict)
    else:
        # define relevant files based on opts
        if(input_args.sim_num == 'real'):
            observation_file_path = FILE_PATHS['observed_data']
        elif(isinstance(input_args.sim_num, int) and input_args.sim_num >= 0):
            formatted_simnum = str(input_args.sim_num).zfill(3)
            observation_file_path = FILE_PATHS['sim_path'].replace('XXX', formatted_simnum)
    
        
   
        if(input_args.overwrite):
            updated_info, sampler = run_bk18_likelihood(params_dict, 
                                                        observation_file_path, 
                                                        input_args)
    # plot mcmc results
    replace_dict ={}# {"alpha_BK18_220":0.6}
    print(input_args.output_path)
    param_names, means, mean_std_strs = epd.plot_triangle(input_args.output_path, replace_dict)
    eb_like_cls = BK18_full_multicomp(used_maps=SHARED_DATA_DICT['used_maps'],
                                      sim_common_dat = SHARED_DATA_DICT)
    epd.plot_best_crossfit(eb_like_cls, 
                           input_args.output_path, 
                           SHARED_DATA_DICT['used_maps'],  
                           param_names, 
                           means, 
                           mean_std_strs)
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
    if(os.path.exists(outpath + '.1.txt') and os.path.exists(outpath + '_bestfit.png')):
        print(f"Skipping existing simulation {sim_num}")
        return
    if(sim_num == 'real'):
            observation_file_path = FILE_PATHS['observed_data']
    elif(isinstance(sim_num, int) and sim_num >= 0):
        formatted_simnum = str(sim_num).zfill(3)
        observation_file_path = FILE_PATHS['sim_path'].replace('XXX', formatted_simnum)

    params_copy = copy.deepcopy(params_dict)
    updated_info, sampler = run_bk18_likelihood(params_copy, 
                                            observation_file_path, 
                                            input_args)



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
    sim_indices = range(input_args.sim_start, input_args.sim_num)
    try:
        with ProcessPoolExecutor() as executor:
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

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-p', '--output_path', default='chains/default',
                        help='directory to save the mcmc chains and plots')
    parser.add_argument('-o', '--overwrite', action='store_true',
                        help='whether to overwrite current chains')
    parser.add_argument('-n', '--sim_num', default=-1, type=int,
                        help='Number of simulations to extract params from')
    parser.add_argument('-s', '--sim_start', default=1, type=int,
                        help='Simulation number start')
    parser.add_argument('-m', '--map_set', default='BK18',
                        help='set of maps to be used for the analysis')
    parser.add_argument('-d', '--dataset', default='BK18lf_fede01',
                        help='dataset to be used for the analysis')
    parser.add_argument('-f', '--forecast', action='store_true',
                        help='whether to do forecast instead of running mcmc chains')
    parser.add_argument('-b', '--bin_num', default=14,
                        help='Number of ell bins used in the analysis')
    parser.add_argument('-u', '--theory_comps', default='all',
                        help='which theory components to include in the analysis')
    parser.add_argument('-t', '--spectra_type', default='eb',
                        help='which spectra type to use in analysis')
    
    args = parser.parse_args()
    # Check if the overwrite flag is set
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
    if(args.sim_num == -1):
        args.sim_num = 'real'
    multicomp_mcmc_driver(args)
    
