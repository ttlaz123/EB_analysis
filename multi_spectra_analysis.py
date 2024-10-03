

import os
import numpy as np
import argparse
import glob
import shutil


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
        self.map_reference_header = None
        
        # BPWF and header check
        self.bpwf = self.load_bpwf(FILE_PATHS["bpwf"])
        self.used_maps = self.filter_used_maps(self.used_maps)

        # Theory
        self.dl_theory = self.load_cmb_spectra(FILE_PATHS['camb_lensing'],
                                               FILE_PATHS['dust_models'])
        self.binned_dl_theory_dict = self.apply_bpwf(self.dl_theory, self.bpwf, self.used_maps)

        # Real Data
        self.binned_dl_observed_dict = self.load_observed_spectra(FILE_PATHS['observed_data'], 
                                                             self.used_maps)
        self.binned_dl_observed_vec = self.dict_to_vec(self.binned_dl_observed_dict, self.used_maps)
        
        # Covar matrix
        self.full_covmat = self.load_covariance_matrix(FILE_PATHS['covariance_matrix'])
        self.filtered_covmat = self.filter_matrix(self.full_covmat, self.used_maps)
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
    
    def filter_used_maps(self, used_maps):
        """
        Remove elements from used_maps that are not in reference_maps.

        Parameters:
        - used_maps: List of maps to filter.
        - reference_maps: List of valid reference maps.

        Returns:
        - A filtered list containing only elements from used_maps that are present in reference_maps.
        """
        maps = [map_ for map_ in used_maps if map_ in self.map_reference_header]
        print(" ~~~~~~~~~~ Using the following maps in analysis: ~~~~~~~~~~")
        print(maps)
        return maps



    def load_observed_spectra(self, observed_data_path, used_maps):
        """
        Load observed spectra data from a specified file and filter the data based on the used maps.

        Args:
            observed_data_path (str): The file path to the observed spectra data in a text format.
            used_maps (list of str): A list of map names to be used for filtering the observed data.

        Returns:
            dict: A dict containing the filtered observed spectra data, 
                    with each entry representing a spectrum for the specified used maps.

        Raises:
            AssertionError: If the provided map names in `used_maps` do not match the reference header.
            
        Description:
            The function first verifies the header of the observed data file against the 
            reference header (`self.map_reference_header`). It then identifies the indices 
            of the specified `used_maps` within the validated header. After loading the data 
            from the file, the function extracts the relevant columns corresponding to the 
            used maps, adjusting for the fact that the first column in the loaded data is 
            merely an index or identifier (hence the addition of 1 to the indices).
        """
        reference_header = self.map_reference_header
        print("Loading: " + str(observed_data_path))
        self.map_reference_header = self.check_file_header(observed_data_path, reference_header)
        used_cols = [self.map_reference_header.index(cross_map) for cross_map in used_maps]
        obs_data = np.loadtxt(observed_data_path)

        observed_spectra_dict = {}
        for i in range(len(used_cols)):
            observed_spectra_dict[used_maps[i]] = obs_data[:, used_cols[i]]
        return observed_spectra_dict
    
    def load_bpwf(self, bpwf_directory):
        """
        Load BPWF (Band Power Window Function) data from the specified directory.

        Args:
            bpwf_directory (str): The file path or pattern specifying the directory where BPWF files are located.

        Returns:
            ndarray: A 3D NumPy array containing the concatenated BPWF data from all files in the specified directory. 
                    The first dimension corresponds to the number of files, and the subsequent dimensions 
                    correspond to the BPWF data.

        Raises:
            ValueError: If no BPWF files are found in the specified directory.

        Description:
            This function searches for BPWF files in the provided directory, ensuring that there is at least 
            one file to load. It checks the consistency of the file headers against a reference header, 
            which is stored in `self.map_reference_header`. Each file's data is read (excluding the first column) 
            and stored in a list, which is then stacked into a 3D NumPy array before being returned.
        """
        bpwf_files = sorted(glob.glob(bpwf_directory))
        if len(bpwf_files) < 1:
            raise ValueError("No files found in " + str(bpwf_directory))
        # Initialize variable to store the header line to compare
        reference_header = self.map_reference_header
        # List to hold all loaded data
        bpwf_data = []

        for file in bpwf_files:
            print("Loading: " + str(file))
            # Read the header and check consistency
            self.map_reference_header = self.check_file_header(file, reference_header)
            # Load data, don't ignore the first column
            bpwf_data.append(np.loadtxt(file))

        # Concatenate and return all BPWF data
        return np.stack(bpwf_data, axis=0)
    
    def load_covariance_matrix(self, covmat_path):
        """
        Load the covariance matrix from the specified file.

        Args:
            covmat_path (str): The file path to the covariance matrix data in a text format.

        Returns:
            ndarray: A 2D NumPy array containing the covariance matrix.

        Raises:
            AssertionError: If the loaded covariance matrix is not square (i.e., the number of rows does not equal the number of columns).

        Description:
            This function reads a covariance matrix from a specified file path, ensuring that the matrix is square 
            by checking that the number of rows equals the number of columns. It validates the file's header 
            against the existing reference header, stored in `self.map_reference_header`, before loading the matrix data.
        """
        print("Loading: " + str(covmat_path))
        self.map_reference_header = self.check_file_header(covmat_path, self.map_reference_header)
        full_covmat = np.loadtxt(covmat_path)
        shap = full_covmat.shape
        
        assert shap[0] == shap[1], "Covariance matrix must be square."
        return full_covmat
    
    def load_cmb_spectra(self, lensing_path, dust_paths):
    
        theory_dict = {}
        print("Loading: " + str(lensing_path))
        with fits.open(lensing_path) as hdul_lens:
            EE_lens = hdul_lens[1].data['E-mode C_l']
            BB_lens = hdul_lens[1].data['B-mode C_l']
        for map_freq in dust_paths:
            print("Loading: " + str(dust_paths[map_freq]))
            with fits.open(dust_paths[map_freq]) as hdul_dust:
                EE_dust = hdul_dust[1].data['E-mode C_l']
                BB_dust = hdul_dust[1].data['B-mode C_l']
            theory_dict[map_freq + '_Ex' + map_freq + '_E'] = EE_lens + EE_dust
            theory_dict[map_freq + '_Bx' + map_freq + '_B'] = BB_lens + BB_dust
        return theory_dict
 
    def apply_bpwf(self, theory_dict, bpwf_mat, used_maps):
        """
        Apply the bandpower window function (BPWF) to a given theory power spectrum.

        Parameters:
        -----------
        theory_dict : dict
            Dictionary containing theoretical power spectra for different map combinations. 
            The keys should be in the format 'mapxmap' and the values are arrays of power spectra.
        bpwf_mat : numpy.ndarray
            3D array representing the bandpower window function matrix.
            The shape is (number_of_ells, number_of_bands, number_of_columns).
        used_maps : list of str
            List of strings representing the cross maps (e.g., 'BK18_150xBK18_220').

        Returns:
        --------
        binned_theory_dict : dict
            Dictionary containing binned theoretical power spectra.
            Keys are the same as in `theory_dict`, and values are the binned power spectra arrays.

        Notes:
        ------
        - The function performs matrix multiplication to apply the BPWF to each element in `theory_dict`.
        - The result is a new dictionary where the theoretical power spectra have been binned according to the BPWF.
        """
        binned_theory_dict = {}
        for cross_map in used_maps:
            maps = cross_map.split('x')
            for freq_map in maps:
                map0 = freq_map + 'x' + freq_map
                col = self.map_reference_header.index(map0)
                num_ells = bpwf_mat.shape[1]
                binned_theory_dict[map0] = np.matmul(bpwf_mat[:,:,col],theory_dict[map0][:num_ells])
        return binned_theory_dict
    
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
                big_vector.append(spectra_dict[map_name])

        # Concatenate all spectra arrays into a single 1D array
        return np.concatenate(big_vector, axis=0)
    
    def filter_matrix(self, matrix, used_maps):
        """
        Filters a given matrix to extract rows and columns that correspond to specific map cross-correlations.

        Args:
            matrix (ndarray): The covariance matrix to be filtered.
            used_maps (list of str): List of cross-correlation map names to be used for filtering.

        Returns:
            ndarray: The filtered covariance matrix, containing only the rows and columns 
                    corresponding to the specified cross-correlation maps in `used_maps`.

        Raises:
            AssertionError: If the number of maps and the size of the covariance matrix do not fit 
                            the expected structure.
        
        Notes:
            This function determines which rows and columns of the covariance matrix should be
            selected based on the `used_maps` provided. It uses `np.ix_` to filter the specified 
            rows and columns simultaneously.
        """
        num_maps = len(self.map_reference_header) - 1
        num_bins = matrix.shape[0] / num_maps

        # Check if num_bins is an integer by checking if the division results in a remainder
        if num_bins != int(num_bins):
            raise ValueError(f"Number of maps {num_maps} and "
                            f"size of covar matrix {matrix.shape[0]} don't fit, "
                            f"num_bins {num_bins} is not an integer.")

        num_bins = int(num_bins) 

        # we subtract 1 because the first element in the reference is a #
        filter_cols = [self.map_reference_header.index(cross_map)-1 for cross_map in used_maps]
        all_bins = [index + i * num_maps for i in range(num_bins) for index in filter_cols]
        # Use np.ix_ to filter both rows and columns in the given indices
        return matrix[np.ix_(all_bins, all_bins)]
    
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
        theory_prediction = self.theory(params_values, 
                                        self.binned_dl_theory_dict, self.used_maps)
        
        # Calculate the residuals
        residuals = self.binned_dl_observed_vec - theory_prediction
        
        # Calculate the Mahalanobis distance using the inverse covariance matrix
        chi_squared = residuals.T @ self.cov_inv @ residuals
        
        # Calculate the log-likelihood
        log_likelihood = -0.5 * chi_squared

        return log_likelihood

    def theory(self, params_values, binned_dl_theory_dict, used_maps):
        # Compute the model prediction based on the parameter values
        # This is a placeholder for your theory calculation
        rotated_dict = {}
        for cross_map in used_maps:
            maps = cross_map.split('x')
            angle1_name = 'alpha_' + maps[0]
            angle2_name = 'alpha_' + maps[1]
            angle1 = params_values[angle1_name]
            angle2 = params_values[angle2_name]
            
            e1e2_name = maps[0] + 'x' + maps[1]

            D_e1e2 = (binned_dl_theory_dict[e1e2_name] * 
                      np.cos(2*np.deg2rad(angle1)) * 
                      np.sin(2*np.deg2rad(angle2)))
            D_b1e2 = 0
            D_e1b2 = 0
            D_b1b2 = 0
            D_eb = D_e1e2 - D_b1b2 + D_e1b2 - D_b1e2  
            rotated_dict[cross_map] = D_eb
        return self.dict_to_vec(rotated_dict, used_maps)

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

def generate_cross_spectra(spectra):
    cross_spectra = []
    for spec1 in spectra:
        for spec2 in spectra:
            cross_spectrum = f"{spec1}_Ex{spec2}_B"
            cross_spectra.append(cross_spectrum)
            cross_spectrum = f"{spec1}_Bx{spec2}_E"
            cross_spectra.append(cross_spectrum)
    return  cross_spectra

def multicomp_mcmc_driver(outpath):
    ### define variables and priors

    ### load in data

    ### define likelihoods

    ### run mcmc

    ### plot results
    # Example of running the function
    
    calc_spectra = ['BK18_K95', 'BK18_150', 'BK18_220']
    all_cross_spectra = generate_cross_spectra(calc_spectra)
    angle_priors = {"prior": {"min": -3, "max": 3}, "ref": 0}
    params_dict = {'alpha_' + spectrum: angle_priors for spectrum in calc_spectra    }

    updated_info, sampler = run_bk18_likelihood(params_dict, all_cross_spectra, 
                                                outpath=outpath)

    # Print results
    print("Updated Info:", updated_info)
    print("Sampler:", sampler)
    return 


def main():
    # Set up argument parsing
    parser = argparse.ArgumentParser()
    parser.add_argument('-p', '--output_path', default='chains/default',
                        help='directory to save the mcmc chains and plots')
    parser.add_argument('-o', '--overwrite', action='store_true',
                        help='whether to overwrite current chains')
    args = parser.parse_args()

    # Check if the overwrite flag is set
    if args.overwrite:
        # Check if the output path exists
        if os.path.exists(args.output_path):
            # Prompt user for confirmation
            confirm = input(f"Are you sure you want to delete the existing chains at: {args.output_path}? (y/n): ")
            if confirm.lower() == 'y':
                # Delete the output path and its contents
                shutil.rmtree(args.output_path)
                print(f"Deleted existing chains at: {args.output_path}")
            else:
                print("Deletion cancelled. Existing chains will be kept.")
        else:
            print(f"No existing chains to overwrite at: {args.output_path}")
    multicomp_mcmc_driver(args.output_path)
if __name__ == '__main__':
    main()