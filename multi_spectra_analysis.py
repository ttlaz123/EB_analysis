

import os
import numpy as np
import argparse
import glob

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
    
    # Initialize your likelihood class
    def initialize(self):
        # Load any data or set up anything that needs to happen before likelihood calculation
        self.bpwf = self.load_bpwf(FILE_PATHS["bpwf"])
        self.full_covmat = self.load_covariance_matrix(FILE_PATHS['covariance_matrix'])
        self.dl_theory = self.load_cmb_spectra(FILE_PATHS['camb_lensing'],
                                               FILE_PATHS['dust_models'])
        self.binned_dl_theory = self.apply_bpwf(self.dl_theory, self.bpwf)
        self.binned_dl_observed = self.load_data(self, FILE_PATHS['observed_data'])

        self.filtered_covmat = self.filter_covmat(self.full_covmat)
        self.cov_inv = self.calc_inverse_covmat(self.filtered_covmat)

    def load_bpwf(self, bpwf_directory):
        """
        Load BPWF from the specified directory.
        """
        bpwf_files = sorted(glob.glob(bpwf_directory))
        # Load and concatenate BPWF data from all files
        bpwf_data = [np.loadtxt(file) for file in bpwf_files]
        ##TODO still need to remove first column of indices
        return np.concatenate(bpwf_data, axis=0)
    
    def load_covariance_matrix(self, covmat_path):
        full_covmat = np.loadtxt(covmat_path)
        shap = full_covmat.shape
        assert shap[0] == shap[1]
        return full_covmat
    
    def load_cmb_spectra(self, lensing_path, dust_paths):
        return 
 

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
def run_bk18_likelihood(params_dict, priors_dict):
    # Set up the custom likelihood with provided params
    likelihood_class = BK18_multicomp
    likelihood_class.params_names = list(params_dict.keys())

    # Create Cobaya info dictionary
    info = {
        "likelihood": {
            "my_custom_likelihood": likelihood_class,
        },
        "params": {},
        "theory": None,  # Define theory requirements as needed
    }

    # Add params and priors to the info dictionary
    for param, prior in params_dict.items():
        info["params"][param] = priors_dict.get(param, prior)

    # Run Cobaya
    updated_info, sampler = run(info)
    return updated_info, sampler


def multicomp_mcmc_driver():
    ### define variables and priors

    ### load in data

    ### define likelihoods

    ### run mcmc

    ### plot results
    # Example of running the function
    params_dict = {
        "param1": None,
        "param2": None,
        "param3": None,
    }

    priors_dict = {
        "param1": {"prior": {"min": 0, "max": 10}, "ref": 5},
        "param2": {"prior": {"min": -5, "max": 5}, "ref": 0},
        "param3": {"prior": {"dist": "norm", "loc": 0, "scale": 1}},
    }

    updated_info, sampler = run_bk18_likelihood(params_dict, priors_dict)

    # Print results
    print("Updated Info:", updated_info)
    print("Sampler:", sampler)
    return 


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-p', '--output_path', default='chains',
                        help='directory to save the mcmc chains and plots')
    args = parser.parse_args()

    
    multicomp_mcmc_driver(args.output_path)

if __name__ == '__main__':
    main()