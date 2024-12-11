print("Loading Standard Modules")
import os
import numpy as np
import argparse
import glob
import shutil
from concurrent.futures import ProcessPoolExecutor, as_completed

print("Loading Local Modules")
import eb_load_data as ld
import eb_plot_data as epd
import eb_calculations as ec

print("Loading Cobaya Modules")
from cobaya.run import run
from cobaya.likelihood import Likelihood



# Global dictionary for file paths
#DATASETNAME = 'BK18lfnorot'
#DATASETNAME = 'BK18lf_fede01'
DATASETNAME = 'BK18lf'
#DATASET_DIRNAME = 'BK18lf_dust_incEE_norot_allbins'
#DATASET_DIRNAME = 'BK18lf_fede01_sigl'
DATASET_DIRNAME = DATASETNAME
#DATASET_DIRNAME = 'BK18lf_sim'

# Define the base directories for the file paths
CAMB_BASE_PATH = '/n/holylfs04/LABS/kovac_lab/general/input_maps/official_cl/'
#BK18_BASE_PATH = '/n/home08/liuto/cosmo_package/data/bicep_keck_2018/BK18_cosmomc/data/BK18lf_dust_incEE_norot_allbins/'
DOMINIC_BASE_PATH = '/n/home01/dbeck/cobaya/data/bicep_keck_2018/BK18_cosmomc/data/'
BK18_BASE_PATH = '/n/home08/liuto/cosmo_package/data/bicep_keck_2018/BK18_cosmomc/data/' + DATASET_DIRNAME + '/'

DOMINIC_SIM_PATH = '/n/home01/dbeck/cobaya/data/bicep_keck_2018/BK18_cosmomc/data/' + DATASETNAME +'/'
BK18_SIM_PATH = BK18_BASE_PATH
BK18_SIM_NAME = DATASETNAME + '_cl_hat_simXXX.dat'
# Consolidate file paths into a dictionary
FILE_PATHS = {
    "camb_lensing": CAMB_BASE_PATH + 'camb_planck2013_r0_lensing.fits',
    "dust_models": {
        "BK18_B95e": CAMB_BASE_PATH + 'dust_B95_3p75.fits',
        "BK18_K95": CAMB_BASE_PATH + 'dust_95_3p75.fits',
        "BK18_150": CAMB_BASE_PATH + 'dust_150_3p75.fits',
        "BK18_220": CAMB_BASE_PATH + 'dust_220_3p75.fits',
        "P030e": CAMB_BASE_PATH + 'dust_30_3p75.fits',
        "P044e": CAMB_BASE_PATH + 'dust_41_3p75.fits',
        "P143e": CAMB_BASE_PATH + 'dust_150_3p75.fits',
        "P353e": CAMB_BASE_PATH + 'dust_270_3p75.fits',
        "P217e": CAMB_BASE_PATH + 'dust_220_3p75.fits',
    },
    "bpwf": BK18_BASE_PATH + 'windows/' + DATASETNAME + '_bpwf_bin*.txt',
    "covariance_matrix": BK18_BASE_PATH + DATASETNAME + '_covmat_dust.dat',
    "observed_data": BK18_BASE_PATH + DATASETNAME + '_cl_hat.dat',
    "EDE_spectrum": '/n/home08/liuto/GitHub/EB_analysis/input_data/fEDE0.07_cl.dat',
 }

class BK18_multicomp(Likelihood):
    params_names = []
    used_maps = []
    include_EDE = True   
    zero_offdiag = False
    signal_params = {}
    def __init__(self,*args,**kwargs):
        if('used_maps' in kwargs):
            self.used_maps = kwargs['used_maps']
            print("New used maps: " + str(self.used_maps))
            if('zero_offdiag' in kwargs):
                self.zero_offdiag = kwargs['zero_offdiag']
            if('signal_params' in kwargs):
                self.signal_params = kwargs['signal_params']
                if('gMpl' in self.signal_params):
                    self.include_EDE = True
            self.initialize()
        else:
            super().__init__(*args,**kwargs)
        
        # Initialize your likelihood class
    def initialize(self):
        # Load any data or set up anything that needs to happen before likelihood calculation
        self.map_reference_header = None
        num_bins =14 
        # BPWF and header check
        self.bpwf, self.map_reference_header = ld.load_bpwf(FILE_PATHS["bpwf"], self.map_reference_header, num_bins = num_bins)
        self.used_maps = self.filter_used_maps(self.used_maps)

        # Theory
        self.dl_theory = ld.load_cmb_spectra(FILE_PATHS['camb_lensing'],
                                               FILE_PATHS['dust_models'])
        print("Include EDE?" + str(self.include_EDE))
        if(self.include_EDE):
            self.dl_theory = ld.include_ede_spectra(FILE_PATHS['EDE_spectrum'],
                                                        self.dl_theory)
        self.binned_dl_theory_dict = ec.apply_bpwf(self.map_reference_header,self.dl_theory, self.bpwf, self.used_maps)
        # Real Data
        self.binned_dl_observed_dict, self.map_reference_header = ld.load_observed_spectra(FILE_PATHS['observed_data'], 
                                    self.used_maps, self.map_reference_header, num_bins=num_bins)
        # inject signal
        print('Inject signal?')
        if(len(self.signal_params) > 0):
            print('Injecting Signal: ')
            print(self.signal_params)
            self.binned_dl_observed_dict = ec.inject_signal(self.used_maps, self.signal_params, 
                    self.binned_dl_theory_dict, self.binned_dl_observed_dict)
        self.binned_dl_observed_vec = self.dict_to_vec(self.binned_dl_observed_dict, 
                                                    self.used_maps)
         
        # Covar matrix
        covmat_name = 'covariance_matrix'
        self.full_covmat = ld.load_covariance_matrix(FILE_PATHS[covmat_name], self.map_reference_header)
        self.filtered_covmat = ec.filter_matrix(self.map_reference_header, self.full_covmat, self.used_maps, num_bins=num_bins, zero_offdiag = self.zero_offdiag)
        #plot_covar_matrix(self.filtered_covmat, used_maps=self.used_maps)
        self.cov_inv = ec.calc_inverse_covmat(self.filtered_covmat)

        test_params = {
                    'gMpl':1,
                    'alpha_BK18_150':0.3,
                    'alpha_BK18_220': 0.3,
                    'alpha_BK18_K95': 0.3,
                    'alpha_BK18_B95e':0.3
        }
        #self.plot_sample_values(test_params)
        
    # plot 
    def plot_sample_values(self, params_values):
        
        theory_prediction = self.theory(params_values, 
                                        self.dl_theory, self.used_maps)
        rotated_dict = ec.apply_bpwf(self.map_reference_header,self.rotated_dict, self.bpwf, self.used_maps,do_cross=True)
        ebe_dict = ec.apply_bpwf(self.map_reference_header,self.ebe_dict, self.bpwf, self.used_maps,do_cross=True)
        tot_dict = self.tot_dict
        num_bin = len(next(iter(rotated_dict.values())))
        for mapi in self.used_maps:
            #plt.plot(self.binned_dl_observed_dict[mapi])
            map_index = self.used_maps.index(mapi)
            covar_mat = self.filtered_covmat
            var = np.diag(covar_mat)[map_index*num_bin:num_bin*(map_index+1)]
            observed_data = self.binned_dl_observed_dict[mapi]
            epd.plot_sample_fit(observed_data, var, mapi, rotated_dict, ebe_dict, tot_dict, params_values)
            

    def filter_used_maps(self, used_maps):
        """
        Remove elements from used_maps that are not in reference_maps.

        Parameters:
        - used_maps: List of maps to filter.
        - reference_maps: List of valid reference maps.

        Returns:
        - A filtered list containing only elements from used_maps that are present in reference_maps.
        """
        maps = [map_ for map_ in self.map_reference_header if map_ in used_maps]
        print(" ~~~~~~~~~~ Using the following maps in analysis: ~~~~~~~~~~")
        print(maps)
        return maps
    
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
                spec = spectra_dict[map_name].copy()
                #spec[7:9] = 0
                big_vector.append(spec)

        # Concatenate all spectra arrays into a single 1D array
        concat_vec =   np.concatenate(big_vector, axis=0)

        return concat_vec   
    
    
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
        theory_prediction = self.theory(params_values, 
                                        self.dl_theory, self.used_maps)
         
        # Calculate the residuals
        residuals = self.binned_dl_observed_vec - theory_prediction
        # Calculate the Mahalanobis distance using the inverse covariance matrix
        chi_squared = residuals.T @ self.cov_inv @ residuals
        # Calculate the log-likelihood
        log_likelihood = -0.5 * chi_squared

        return log_likelihood
    
    

    def theory(self, params_values, dl_theory_dict, used_maps):
        # Compute the model prediction based on the parameter values
        # currently assumes it is only calculating EB
        # all theory based on 
        # https://bicep.rc.fas.harvard.edu/dbeck/20230202_cmbbirefringence/
        self.rotated_dict = {}
        self.ebe_dict = {}
        self.tot_dictt = {}
        self.tot_dict = {}
        for cross_map in used_maps:
            self.rotated_dict[cross_map] = ec.rotate_spectrum(cross_map,
                                            dl_theory_dict, params_values)
            if(self.include_EDE):
                ede_shift = ec.apply_EDE_shift(cross_map,
                                                dl_theory_dict, params_values)
                if self.rotated_dict[cross_map].shape != ede_shift.shape:
                    min_size = min(self.rotated_dict[cross_map].size, 
                                        ede_shift.size)
                    # Truncate both arrays to the minimum size
                    self.rotated_dict[cross_map] = self.rotated_dict[cross_map][:min_size]
                    ede_shift = ede_shift[:min_size]
                    self.ebe_dict[cross_map] = ede_shift
                    self.tot_dictt[cross_map] = self.rotated_dict[cross_map] + self.ebe_dict[cross_map]
        self.tot_dict = ec.apply_bpwf(self.map_reference_header,self.tot_dictt, self.bpwf, self.used_maps,do_cross=True)
  
        theory_vec = self.dict_to_vec(self.tot_dict, used_maps)
        return theory_vec


    

# Function to create and run a Cobaya model with the custom likelihood
def run_bk18_likelihood(params_dict, used_maps, outpath, 
                            include_ede = False, zero_offdiag = False,
                            rstop = 0.02, max_tries=10000, signal_params = {}):

    # Set up the custom likelihood with provided params
    likelihood_class = BK18_multicomp
    likelihood_class.params_names = list(params_dict.keys())
    # Create Cobaya info dictionary
    info = {
        "likelihood": {
            "my_likelihood": {
                "external": likelihood_class,
                "used_maps": used_maps,
                "include_EDE": include_ede,
                "zero_offdiag": zero_offdiag,
                "signal_params": signal_params,
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

def generate_cross_spectra(spectra, do_crosses=True):
    cross_spectra = []
    for spec1 in spectra:
        for spec2 in spectra:
            # don't do cross spectra
            if(not spec1 == spec2 and not do_crosses):
                continue
            cross_spectrum = f"{spec1}_Ex{spec2}_B"
            cross_spectra.append(cross_spectrum)
            cross_spectrum = f"{spec1}_Bx{spec2}_E"
            cross_spectra.append(cross_spectrum)
    return  cross_spectra

def multicomp_mcmc_driver(outpath, dorun, sim_num='real'):

    
    calc_spectra = [
                    'BK18_220', 
                    'BK18_150', 
                    'BK18_K95', 
                    'BK18_B95e',
                    #'P030e', 
                    #'P044e', 
                    #'P143e',
                    #'P217e'
                    ]
    do_crosses =True
    include_ede = True
    if(sim_num != 'real'):
        formatted_simnum = str(sim_num).zfill(3)
        simname = BK18_SIM_NAME.replace("XXX", formatted_simnum)
        FILE_PATHS['observed_data'] = BK18_SIM_PATH + simname
    signal_params = {#}
    #                'gMpl':0.0,
    #                'alpha_BK18_150':-0.5,
    #                'alpha_BK18_220': 1,
    #                'alpha_BK18_K95': -0.1,
    #                'alpha_BK18_B95e':-0.4
                    }
    
    all_cross_spectra = generate_cross_spectra(calc_spectra, do_crosses=do_crosses)
    angle_priors = {"prior": {"min": -3, "max": 3}, "ref": 0}
    params_dict = {
        'alpha_' + spectrum: {
                **angle_priors,
                'latex': ("\\alpha_{" + 
                            spectrum.replace('_', '\\_') +
                            "}")
                }
                for spectrum in calc_spectra    
    }
    
    if(include_ede):
        params_dict['gMpl'] = {"prior": {"min": -10, "max": 10}, "ref": 0}
    
   
    if(dorun):
        updated_info, sampler = run_bk18_likelihood(params_dict, 
                                                all_cross_spectra, 
                                                outpath=outpath,
                                                include_ede = include_ede,
                                                signal_params=signal_params)

    replace_dict ={}# {"alpha_BK18_220":0.6}
    param_names, means, mean_std_strs = epd.plot_triangle(outpath, replace_dict)
    eb_like_cls = BK18_multicomp(used_maps=all_cross_spectra, 
                                signal_params=signal_params)
    epd.plot_best_crossfit(eb_like_cls, outpath, all_cross_spectra,  
                        param_names, means, mean_std_strs, 
                        signal_params=signal_params)
    return 

def run_simulation(s, output_path, overwrite):
    outpath = f"{output_path}{s:03d}"
    if(os.path.exists(outpath + '.1.txt') and os.path.exists(outpath + '_bestfit.png')):
        return
    multicomp_mcmc_driver(outpath, overwrite, s)


# Parallel execution with cancellation support
def parallel_simulation(args):
    sim_indices = range(args.sim_start, args.sim_num)
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
    # Set up argument parsing
    parser = argparse.ArgumentParser()
    parser.add_argument('-p', '--output_path', default='chains/default',
                        help='directory to save the mcmc chains and plots')
    parser.add_argument('-o', '--overwrite', action='store_true',
                        help='whether to overwrite current chains')
    parser.add_argument('-n', '--sim_num', default=-1, type=int,
                        help='Simulation num to extract params from, defaults to real data')
    parser.add_argument('-s', '--sim_start', default=1, type=int,
                        help='Simulation start')

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
        multicomp_mcmc_driver(args.output_path, args.overwrite, args.sim_num)
    
if __name__ == '__main__':
    main()


