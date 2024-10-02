

import os
import numpy as np
import argparse
import glob
from astropy.io import fits
from cobaya.model import get_model
from cobaya.run import run
from cobaya.likelihood import Likelihood

# Global dictionary for file paths
FILE_PATHS = {
    "camb_lensing": '/n/holylfs04/LABS/kovac_lab/general/input_maps/official_cl/camb_planck2013_r0_lensing.fits',
    "dust_models": {
        "BK18_B95": '/n/holylfs04/LABS/kovac_lab/general/input_maps/official_cl/dust_B95_3p75.fits',
        "BK18_K95": '/n/holylfs04/LABS/kovac_lab/general/input_maps/official_cl/dust_95_3p75.fits',
        "BK18_150": '/n/holylfs04/LABS/kovac_lab/general/input_maps/official_cl/dust_150_3p75.fits',
        "BK18_220": '/n/holylfs04/LABS/kovac_lab/general/input_maps/official_cl/dust_220_3p75.fits',
    },
    "bpwf": '/n/home08/liuto/cosmo_package/data/bicep_keck_2018/BK18_cosmomc/data/BK18lf_dust_incEE_norot/windows/BK18lfnorot_bpwf_bin*.txt',
    "covariance_matrix": '/n/home08/liuto/cosmo_package/data/bicep_keck_2018/BK18_cosmomc/data/BK18lf_dust_incEE_norot/BK18lfnorot_covmat_dust.dat',
    "observed_data": '/n/home08/liuto/cosmo_package/data/bicep_keck_2018/BK18_cosmomc/data/BK18lf_dust_incEE_norot/BK18lfnorot_cl_hat.dat'
}
class BK18_multicomp(Likelihood):
    params_names = []
    used_maps = []
    # Initialize your likelihood class
    def initialize(self):
        # Load any data or set up anything that needs to happen before likelihood calculation
        self.frequency_header = None
        self.bpwf = self.load_bpwf(FILE_PATHS["bpwf"])
        self.full_covmat = self.load_covariance_matrix(FILE_PATHS['covariance_matrix'])
        self.dl_theory = self.load_cmb_spectra(FILE_PATHS['camb_lensing'],
                                               FILE_PATHS['dust_models'])
        self.binned_dl_theory = self.apply_bpwf(self.dl_theory, self.bpwf)
        self.binned_dl_observed = self.load_observed_spectra(self, FILE_PATHS['observed_data'], 
                                                             self.used_maps)

        self.filtered_covmat = self.filter_matrix(self.full_covmat)
        self.cov_inv = self.calc_inverse_covmat(self.filtered_covmat)

    def check_file_header(self, file_path, reference_header):
        with open(file_path, 'r') as f:
            for line in f:
                # Assuming the relevant header is the one with 'BxB' in it
                if line.startswith("#") and "BxB" in line:
                    current_header = line.strip()
                    if reference_header is None:
                        reference_header = current_header.split()
                    elif current_header.split() != reference_header:
                        raise ValueError("Header mismatch detected in one or more files.")
                    break  # Stop reading further header lines
        return reference_header
    
    def load_observed_spectra():
        
        return 
    
    def load_bpwf(self, bpwf_directory):
        """
        Load BPWF from the specified directory.
        """
        bpwf_files = sorted(glob.glob(bpwf_directory))
        if(len(bpwf_files) < 1):
            raise ValueError("No files found in " + str(bpwf_directory))
        # Initialize variable to store the header line to compare
        reference_header = self.frequency_header
        # List to hold all loaded data
        bpwf_data = []

        for file in bpwf_files:
            # Read the header and check consistency
            self.frequency_header = self.check_file_header(file, reference_header)
            # Load data, ignoring the first column
            bpwf_data.append(np.loadtxt(file)[:, 1:])

        # Concatenate and return all BPWF data
        return np.concatenate(bpwf_data, axis=0)
    
    def load_covariance_matrix(self, covmat_path):
        self.frequency_header = self.check_file_header(covmat_path, self.frequency_header)
        full_covmat = np.loadtxt(covmat_path)
        shap = full_covmat.shape
        assert shap[0] == shap[1]
        return full_covmat
    
    def load_cmb_spectra(self, lensing_path, dust_paths):
        theory_dict = {}
        with open(lensing_path) as hdul_lens:
            EE_lens = hdul_lens[1].data['E-mode C_l']
            BB_lens = hdul_lens[1].data['B-mode C_l']
        for map_freq in dust_paths:
            with open(dust_paths[map_freq]) as hdul_dust:
                EE_dust = hdul_dust[1].data['E-mode C_l']
                BB_dust = hdul_dust[1].data['B-mode C_l']
            theory_dict[map_freq + '_E'] = EE_lens + EE_dust
            theory_dict[map_freq + '_B'] = BB_lens + BB_dust
        return theory_dict
 
    def apply_bpwf(self, theory_dict, bpwf_mat):
        binned_theory_dict = {}
        # TODO fix this to actually work
        for map in theory_dict:
            col = self.used_maps[map]
            binned_theory_dict[map] = bpwf_mat[col]*theory_dict[map]
        return binned_theory_dict
    
    
    def filter_matrix(matrix, filter_cols):
        # Use np.ix_ to filter both rows and columns in the given indices
        return matrix[np.ix_(filter_cols, filter_cols)]
    
    def calc_inverse_covmat(self, filtered_covmat):
        return np.linalg.inv(filtered_covmat)
    
    def get_requirements(self):
        # Specify what is needed from other components, such as theory predictions
        # Here we're asking for 'theory' calculations for the parameters in params_names
        return {"theory": None}

    def logp(self, **params_values):
        """
        Calculate the log-likelihood based on the current parameter values.
        """
        # Extract parameter values dynamically for Cobaya
        params = [params_values[name] for name in self.params_names]
        # Get the theoretical predictions based on the parameter values
        theory_prediction = self.theory(params_values)
        
        # Calculate the residuals
        residuals = self.binned_dl_observed - theory_prediction
        
        # Calculate the Mahalanobis distance using the inverse covariance matrix
        chi_squared = residuals.T @ self.cov_inv @ residuals
        
        # Calculate the log-likelihood
        log_likelihood = -0.5 * chi_squared

        return log_likelihood

    def theory(self, params_values):
        # Compute the model prediction based on the parameter values
        # This is a placeholder for your theory calculation
        binned_dl_theory = self.binned_dl_theory

        ## TODO figure out how to perform rotation of cross spectra
        return params_values["param1"] + params_values["param2"] + params_values["param3"]

# Function to create and run a Cobaya model with the custom likelihood
def run_bk18_likelihood(params_dict, used_maps, outpath, rstop = 0.02, max_tries=10000):
    # Set up the custom likelihood with provided params
    likelihood_class = BK18_multicomp
    likelihood_class.params_names = list(params_dict.keys())

    # Create Cobaya info dictionary
    info = {
        "likelihood": {
            "my_likelihood": {
                "external": likelihood_class,
                "used_maps": used_maps
            }
        },
        "params": params_dict,
        "theory": None,  # Define theory requirements as needed
        "sampler":{
            "mcmc": {
                "Rminus1_stop": rstop,
                "max_tries": max_tries,
            }
        },
        "output": outpath
    }

    # Run Cobaya
    updated_info, sampler = run(info)
    return updated_info, sampler


def multicomp_mcmc_driver(outpath):
    ### define variables and priors

    ### load in data

    ### define likelihoods

    ### run mcmc

    ### plot results
    # Example of running the function
    
    calc_spectra = ['K95', '150', '220']
    all_cross_spectra = []
    angle_priors = {"prior": {"min": -3, "max": 3}, "ref": 0}
    params_dict = {'alpha_' + spectrum: angle_priors for spectrum in calc_spectra    }

    updated_info, sampler = run_bk18_likelihood(params_dict, all_cross_spectra, 
                                                outpath=outpath)

    # Print results
    print("Updated Info:", updated_info)
    print("Sampler:", sampler)
    return 


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-p', '--output_path', default='chains/default',
                        help='directory to save the mcmc chains and plots')
    args = parser.parse_args()

    
    multicomp_mcmc_driver(args.output_path)

if __name__ == '__main__':
    main()