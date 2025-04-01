import argparse
import os
import glob
import shutil

from concurrent.futures import ProcessPoolExecutor, as_completed
from cobaya.run import run

import eb_load_data as ld
import eb_file_paths as fp
import BK18_full_multicomp

SHARED_DATA_DICT = {}

def load_shared_data(args):
    map_reference_header = None
    covmat_name = 'covariance_matrix'
    SHARED_DATA_DICT['covmat'] = ld.load_covariance_matrix(fp.FILE_PATHS[covmat_name])
    SHARED_DATA_DICT['bpwf'] = ld.load_bpwf(fp.FILE_PATHS['bpwf'], 
                                            map_reference_header, 
                                            num_bins=args.bin_num)
    SHARED_DATA_DICT['theory_spectra'] = ld.load_cmb_spectra(fp.FILE_PATHS['camb_lensing'],
                                                             fp.FILE_PATHS['dust_models'],
                                                             args.theory_comps)
    SHARED_DATA_DICT['theory_spectra'] = ld.include_ede_spectra(fp.FILE_PATHS['EDE_spectrum'],
                                                                args.theory_comps)
    SHARED_DATA_DICT['bandpasses'] = ld.read_bandpasses(fp.FILE_PATHS['bandpasses'])
    
    

def run_bk18_likelihood(params_dict, used_maps, outpath, observation_file_path,
                        rstop = 0.03, max_tries=10000,
                        map_set='BK18', 
                        dataset='BK18lf_fede01', 
                        forecast=False, 
                        bin_num=14, 
                        theory_comps='all', 
                        spectra_type='all'):
    likelihood_class = BK18_full_multicomp
    likelihood_class.params_names = list(params_dict.keys())

    # Create Cobaya info dictionary
    info = {
        "likelihood": {
            "my_likelihood": {
                "external": likelihood_class,
                "used_maps": used_maps,
                "map_set": map_set,
                "dataset": dataset,
                "forecast": forecast,
                "bin_num":  bin_num,
                "theory_comps": theory_comps,
                "spectra_type": spectra_type,
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
        "output": outpath,
        "resume": True
    }

    # Run Cobaya
    updated_info, sampler = run(info)
    return updated_info, sampler

def define_priors(calc_spectra, theory_comps):
    # define angles based on mapopts
    # define ede opts
    # define other opts based on opts
    return params_dict

def multicomp_mcmc_driver(outpath, overwrite=True, 
                          sim_num='real', 
                          map_set='BK18', 
                          dataset='BK18lf_fede01', 
                          forecast=False, 
                          bin_num=14, 
                          theory_comps='all', 
                          spectra_type='all'):
    # full multicomp driver
    # define maps based on mapopts
    calc_spectra = determine_map_freqs(map_set)
    used_maps = generate_cross_spectra(calc_spectra, do_crosses=do_crosses, spectra_type=spectra_type)
    # define dust params based on dustopts
    params_dict = define_priors(calc_spectra, theory_comps)
    
    # define relevant files based on opts
    # run mcmc
    if(dorun):
        updated_info, sampler = run_bk18_likelihood(
                        params_dict, 
                        used_maps, 
                        outpath, 
                        observation_file_path,
                        rstop = 0.03, 
                        max_tries=10000,
                        map_set=map_set, 
                        dataset=dataset, 
                        forecast=forecast, 
                        bin_num=bin_num, 
                        theory_comps=theory_comps, 
                        spectra_type=spectra_type)
    # plot mcmc results
    replace_dict ={}# {"alpha_BK18_220":0.6}
    print(outpath)
    param_names, means, mean_std_strs = epd.plot_triangle(outpath, replace_dict)
    eb_like_cls = BK18_multicomp(used_maps=used_maps, 
                                signal_params=signal_params)
    epd.plot_best_crossfit(eb_like_cls, 
                           outpath, 
                           used_maps,  
                           param_names, 
                           means, 
                           mean_std_strs, 
                           signal_params=signal_params)
    return 

def run_simulation(s, output_path, overwrite):
    outpath = f"{output_path}{s:03d}"
    # skip chains that already exist
    if(os.path.exists(outpath + '.1.txt') and os.path.exists(outpath + '_bestfit.png')):
        return
    multicomp_mcmc_driver(outpath, overwrite, s)



# Parallel execution with cancellation support
def parallel_simulation(args):
    sim_indices = range(args.sim_start, args.sim_num)
    load_shared_data(args)
    try:
        with ProcessPoolExecutor() as executor:
            # Submit all tasks to the executor
            future_to_sim = {
                executor.submit(run_simulation, s, args.output_path, args.overwrite): s
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
    if(args.sim_num == 500):
        parallel_simulation(args)
    else:
        multicomp_mcmc_driver(args)
    