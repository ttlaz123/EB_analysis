print("Loading Modules")
import os
import numpy as np
import random
import argparse
import glob
import shutil
import re

import pandas as pd
from astropy.io import fits
from scipy.stats import gaussian_kde
import scipy.io

print("Loading Cobaya Modules")
from cobaya.model import get_model
from cobaya.run import run
from cobaya.likelihood import Likelihood

print("Loading Plotting  Modules")
from getdist import plots, MCSamples
from getdist.mcsamples import loadMCSamples
import corner
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

# Global dictionary for file paths
#DATASETNAME = 'BK18lfnorot'
DATASETNAME = 'BK18lf_fede01'
#DATASET_DIRNAME = 'BK18lf_dust_incEE_norot_allbins'
DATASET_DIRNAME = DATASETNAME

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
    #"Dominic_invcovmat": '/n/home01/dbeck/keckpipe/Cinv_K95K150K220.dat',
    #"matlab_covmat": '/n/home08/liuto/GitHub/EB_analysis/bk18covmat.mat',
    #"matlab_invcovmat": '/n/home08/liuto/GitHub/EB_analysis/bk18_invcovmat.mat',

    #"signal_only_covmat": DOMINIC_BASE_PATH + 'BK18lf_dust_incEE_norot/BK18lfnorot_covmat_sigtrimmed_dust.dat',
    #'noise_only_covmat': DOMINIC_BASE_PATH + 'BK18lf_dust_incEE_norot/BK18lfnorot_covmat_noi_dust.dat',
}

class BK18_multicomp(Likelihood):
    params_names = []
    used_maps = []
    include_EDE = False    
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
        # BPWF and header check
        self.bpwf = self.load_bpwf(FILE_PATHS["bpwf"])
        self.used_maps = self.filter_used_maps(self.used_maps)

        # Theory
        self.dl_theory = self.load_cmb_spectra(FILE_PATHS['camb_lensing'],
                                               FILE_PATHS['dust_models'])
        print("Include EDE?" + str(self.include_EDE))
        if(self.include_EDE):
            self.dl_theory = self.include_ede_spectra(FILE_PATHS['EDE_spectrum'],
                                                        self.dl_theory)
        self.binned_dl_theory_dict = self.apply_bpwf(self.dl_theory, self.bpwf, self.used_maps)
        # Real Data
        self.binned_dl_observed_dict = self.load_observed_spectra(FILE_PATHS['observed_data'], 
                                    self.used_maps)
        # inject signal
        print('Inject signal?')
        if(len(self.signal_params) > 0):
            print('Injecting Signal: ')
            print(self.signal_params)
            self.binned_dl_observed_dict = self.inject_signal(self.signal_params, 
                    self.binned_dl_theory_dict, self.binned_dl_observed_dict)
        self.binned_dl_observed_vec = self.dict_to_vec(self.binned_dl_observed_dict, 
                                                    self.used_maps)
         
        # Covar matrix
        covmat_name = 'covariance_matrix'
        self.full_covmat = self.load_covariance_matrix(FILE_PATHS[covmat_name])
        self.filtered_covmat = self.filter_matrix(self.full_covmat, self.used_maps)
        #plot_covar_matrix(self.filtered_covmat, used_maps=self.used_maps)
        self.cov_inv = self.calc_inverse_covmat(self.filtered_covmat)
        
        # plot 
        breakpoint()
        num_bin = 16 
        for mapi in self.used_maps:
            #plt.plot(self.binned_dl_observed_dict[mapi])
            map_index = self.used_maps.index(mapi)
            covar_mat = self.filtered_covmat
            var = np.diag(covar_mat)[map_index*num_bin:num_bin*(map_index+1)]
            observed_data = self.binned_dl_observed_dict[mapi]
            plt.errorbar( 
                            x = range(len(observed_data)),
                            y=(observed_data), 
                            yerr = np.sqrt(var),
                            label='Observed', color='blue')
            angles = {'BK18_B95e':0.4,
                        'BK18_K95':0.2, 
                        'BK18_150':0.5, 
                        'BK18_220':-1} 
            ede_cont = self.binned_dl_theory_dict[mapi+'_EDE']
            plt.plot(ede_cont, label='ede curve')
            parts = mapi.split('x')
            if(parts[0].endswith('_B')):
                ind = 0
            else:
                ind = 1
            result = parts[ind][:-2] + '_E'
            angle = angles[parts[ind][:-2]]
            result = result + 'x' + result
            rot_cont = self.binned_dl_theory_dict[result]*np.sin(np.deg2rad(angle))
            plt.plot(rot_cont, label = result + ' times ' + str(np.sin(np.deg2rad(angle))))
            plt.plot(ede_cont + rot_cont, label='both')
            plt.title(mapi)
            plt.legend()
            plt.show()

    def check_file_header(self, file_path, reference_header):
        with open(file_path, 'r') as f:
            for line in f:
                # Assuming the relevant header is the one with 'BxB' in it
                if line.startswith("#") and "BxB" in line:
                    current_header = line.strip()
                    if reference_header is None:
                        reference_header = current_header.split()
                    elif current_header.split() != reference_header:
                        # Compare the lists element by element and print the differences
                        list1 = current_header.split()
                        list2 = reference_header
                        for i in range(min(len(list1), len(list2))):
                            if list1[i] != list2[i]:
                                print(f"Difference at index {i}: {list1[i]} != {list2[i]}")

                            # If the lists have different lengths, print the extra elements
                            if len(list1) > len(list2):
                                print(f"Extra elements in list1: {list1[len(list2):]}")
                            elif len(list2) > len(list1):
                                print(f"Extra elements in list2: {list2[len(list1):]}")
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
        maps = [map_ for map_ in self.map_reference_header if map_ in used_maps]
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
        try:
            
            map_reference_header = self.check_file_header(observed_data_path, reference_header)
            self.map_reference_header = map_reference_header
        except ValueError:
            with open(observed_data_path, 'r') as f:
                for line in f:
                    # Assuming the relevant header is the one with 'BxB' in it
                    if line.startswith("#") and "BxB" in line:
                        current_header = line.strip()
                        map_reference_header = current_header.split()
                        break
        used_cols = []
        for cross_map in used_maps:
            try:
                if(cross_map  not in map_reference_header):
                    parts = input_str.split('x')

                    cross_map = f"{parts[1]}x{parts[0]}"

                used_cols.append(map_reference_header.index(cross_map))
            
            except ValueError:
                # Try to swap _B and _E and check again
                match = re.match(r'(.+)(_B)(x\1)(_E)', cross_map)
                if match:
                    # Swap _B and _E if the parts before _B and _E are identical
                    swapped_map =  match.group(1) + '_E' + match.group(3) + '_B'
                else:
                    raise ValueError("Cross map does not exist")
                used_cols.append(map_reference_header.index(swapped_map))


        obs_data = np.loadtxt(observed_data_path)

        observed_spectra_dict = {}
        for i in range(len(used_cols)):
            input_str = used_maps[i]
            observed_spectra_dict[input_str] = obs_data[:, used_cols[i]]

        
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
        
        bpwf_files = sorted(glob.glob(bpwf_directory), 
                        key=lambda x: list(map(int,re.findall("(\d+)", x))))
    
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
        k_to_uk = 1e6
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
            ee_spectrum = EE_lens + EE_dust
            
            bb_spectrum =  BB_lens + BB_dust
            ee_spectrum *= np.square(k_to_uk)
            bb_spectrum *= np.square(k_to_uk)
            cl_to_dl = np.array([l*(l+1) for l in range(len(ee_spectrum))])/2/np.pi
            theory_dict[map_freq + '_Ex' + map_freq + '_E'] = ee_spectrum*cl_to_dl            
            theory_dict[map_freq + '_Bx' + map_freq + '_B'] = bb_spectrum*cl_to_dl
        #plt.plot(theory_dict[map_freq + '_Ex' + map_freq + '_E'])
        return theory_dict
    
    def include_ede_spectra(self, ede_path, theory_dict):
        k_to_uk = 1e6
        cmb_temp = 2.726
        data = pd.read_csv(ede_path, delim_whitespace=True, comment='#', header=None)
        data.columns = ['l', 'TT', 'EE', 'TE', 'BB', 'EB', 'TB', 'phiphi', 'TPhi', 'Ephi']
        # Extract 'l' and 'EB' columns
        EB_values = data['EB'].to_numpy()
        EB_ede_dls = -EB_values * np.square(k_to_uk) * np.square(cmb_temp)
        theory_dict['EDE_EB'] = EB_ede_dls
        return theory_dict

    def apply_bpwf(self, theory_dict, bpwf_mat, used_maps, do_cross=False):
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
                if(do_cross):
                    map0 = cross_map
                else:
                    map0= freq_map + 'x' + freq_map
                if(map0 not in theory_dict):
                    print("Key " + map0 +" not in dict")
                    print(theory_dict.keys())

                col = self.map_reference_header.index(map0)
                num_ells = bpwf_mat.shape[1]
                binned_theory_dict[map0] = np.matmul(bpwf_mat[:,:,col],
                                            theory_dict[map0][:num_ells])
            if('EDE_EB' in theory_dict):
                binned_theory_dict[cross_map + '_EDE'] = np.matmul(bpwf_mat[:,:,col],
                                                        theory_dict['EDE_EB'][:num_ells])
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
                spec = spectra_dict[map_name].copy()
                #spec[7:9] = 0
                big_vector.append(spec)

        # Concatenate all spectra arrays into a single 1D array
        concat_vec =   np.concatenate(big_vector, axis=0)

        return concat_vec   

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
        filtered_mat = matrix[np.ix_(all_bins, all_bins)]
        reordered_mat = self.reorder_cov_matrix(filtered_mat, 
                                    num_bins, len(self.used_maps))
        if(self.zero_offdiag):
            offdiag = 1
        else:
            offdiag= reordered_mat.shape[0]
        #before_truncate = np.linalg.inv(reordered_mat)
        #trunc_covmat = self.truncate_covariance_matrix(reordered_mat,
        #                                    offdiag=offdiag)
        return reordered_mat

    def reorder_cov_matrix(self, cov_matrix, n_bins, n_maps):
        """
        Reorder a covariance matrix from bin-major order to map-major order.    
    
        Args:
        - cov_matrix (numpy.ndarray): The original covariance matrix (shape: [n_bins * n_maps, n_bins * n_maps]).
        - n_bins (int): Number of bins.
        - n_maps (int): Number of maps.
        
        Returns:
        - numpy.ndarray: The reordered covariance matrix.
        """
        # Calculate the new order of indices
        old_indices = np.arange(n_bins * n_maps)
        new_indices = np.zeros_like(old_indices)
        for map_idx in range(n_maps):
            for bin_idx in range(n_bins):
                old_pos = bin_idx * n_maps + map_idx
                new_pos = map_idx * n_bins + bin_idx
                new_indices[new_pos] = old_indices[old_pos]
        # Reorder rows and columns of the covariance matrix
        reordered_matrix = cov_matrix[np.ix_(new_indices, new_indices)]
    
        return reordered_matrix

    def truncate_covariance_matrix(self, cov_matrix, offdiag=1, block_size=1):
        """
        ##DEPRECATED
        Truncate the covariance matrix by keeping only the diagonal and specified number of off-diagonals.

        Parameters:
        cov_matrix (np.ndarray): The covariance matrix to truncate.
        offdiag (int): The number of off-diagonals to keep.
    
        Returns:
        np.ndarray: The truncated covariance matrix.
        """
        size = cov_matrix.shape[0]
        N = size
        # Create a mask of zeros (False) initially
        mask = np.zeros((N, N), dtype=bool) + 1
        # Iterate over diagonal blocks
        for i in range(0, N, block_size):
            # Set True for the elements in the current diagonal block and its off-diagonal band
            start = i
            end = min(i+(offdiag+1) * block_size, N)
            mask[start:end, start:end] = 1
        
        # Apply the mask to the covariance matrix
        truncated_cov_matrix = cov_matrix * mask
        ### DEPRECATED 
        ####return truncated_cov_matrix
    
    def calc_inverse_covmat(self, filtered_covmat, block_offdiag = 1):
        inverted_mat = np.linalg.inv(filtered_covmat)
        return inverted_mat
    
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
    
    def inject_signal(self, signal_params, 
                    binned_dl_theory_dict, binned_dl_observed_dict):
        """
        Injects a signal into the observed binned power spectrum based on 
        the provided signal parameters, theory spectra, and observed spectra.

        Parameters
        ----------
        signal_params : dict
            A dictionary containing signal parameters such as 'gMpl' (if not defined, set to 0) 
            and rotation angles for different maps (e.g., 'alpha_map'). Each map-specific angle 
            will be extracted or initialized to 0 if not present.
            
        binned_dl_theory_dict : dict
            A dictionary containing the binned theoretical power spectra for 
            different cross-maps, where each entry corresponds to a map pair (e.g., 'map1xmap2').
            
        binned_dl_observed_dict : dict
            A dictionary containing the observed binned power spectra for 
            different cross-maps, to which the injected signal will be added.
            
        Returns
        -------
        binned_dl_observed_dict : dict
            Updated observed binned power spectra with the injected signal 
            for each cross-map. This includes both a rotated spectrum and an Early Dark Energy (EDE) shift.

        Notes
        -----
        - For each cross-map in `self.used_maps`, this method computes the rotation angles and injects 
        the rotated spectrum (D_eb) and EDE shift into the observed power spectrum.
        - The `rotate_spectrum` method computes the EB rotation based on the cross-map and signal parameters.
        - The `apply_EDE_shift` method adds any additional shift due to Early Dark Energy (EDE).
        - If certain signal parameters are missing (e.g., 'gMpl' or map-specific angles), 
        default values of 0 are used.
        """
        # set undefined signal to 0
        #if('gMpl' not in signal_params):
        #    signal_params['gMpl'] = 0
        for cross_map in self.used_maps:
            maps = cross_map.split('x')
            angle_name0 = 'alpha_' + maps[0]
            angle_name1 = 'alpha_' + maps[1]
            angle_name0 = re.sub(r'_[BE]$', '', angle_name0)
            angle_name1 = re.sub(r'_[BE]$', '', angle_name1)
            if(angle_name0 not in signal_params):
                signal_params[angle_name0] = 0
            if(angle_name1 not in signal_params):
                signal_params[angle_name1] = 0
        
            D_eb = self.rotate_spectrum(cross_map, binned_dl_theory_dict, signal_params)
            ede_shift = 0 
            if('gMpl' in signal_params): 
                ede_shift = self.apply_EDE_shift(cross_map, 
                            binned_dl_theory_dict, signal_params)
            binned_dl_observed_dict[cross_map] += D_eb + ede_shift
        return binned_dl_observed_dict


    def rotate_spectrum(self, cross_map, binned_dl_theory_dict, params_values):
        maps = cross_map.split('x')
        angle1_name = 'alpha_' + maps[0]
        angle2_name = 'alpha_' + maps[1]
        # Use regex to remove _B, _E, or any other suffix ending with _ followed by letters
        angle1_name = re.sub(r'_[BE]$', '', angle1_name)
        angle2_name = re.sub(r'_[BE]$', '', angle2_name)
        angle1 = params_values[angle1_name]
        angle2 = params_values[angle2_name]
        
        e1 = maps[0] if maps[0].endswith('_E') else None 
        e2 = maps[1] if maps[1].endswith('_E') else None
        b1 = maps[0] if maps[0].endswith('_B') else None 
        b2 = maps[1] if maps[1].endswith('_B') else None

        if(e1):
            e1e2_name = e1 + 'x' + e1
            b1b2_name = b2 + 'x' + b2
        elif(e2):
            e1e2_name = e2 + 'x' + e2
            b1b2_name = b1 + 'x' + b1
        else:
            raise ValueError("There is no EE spectrum: " + str(cross_map))
        # TODO include the extra terms to improve approximation 
        # spectrum is EB
        if(e1):
            D_e1e2 = (binned_dl_theory_dict[e1e2_name] * 
                  np.cos(2*np.deg2rad(angle1)) * 
                  np.sin(2*np.deg2rad(angle2)))
            D_b1e2 = 0
            D_e1b2 = 0
            D_b1b2 = (binned_dl_theory_dict[b1b2_name] *
                    np.sin(2*np.deg2rad(angle1)) *
                    np.cos(2*np.deg2rad(angle2)))
            D_eb = D_e1e2 - D_b1b2 + D_e1b2 - D_b1e2  
        # spectrum is BE
        if(e2):
            D_e1e2 = (binned_dl_theory_dict[e1e2_name] * 
                  np.cos(2*np.deg2rad(angle2)) * 
                  np.sin(2*np.deg2rad(angle1)))
            D_b1e2 = 0
            D_e1b2 = 0
            D_b1b2 = (binned_dl_theory_dict[b1b2_name] *
                    np.sin(2*np.deg2rad(angle2)) *
                    np.cos(2*np.deg2rad(angle1)))

            D_eb = D_e1e2 - D_b1b2 + D_e1b2 - D_b1e2  
    
        return D_eb
    
    def assemble_eb_crossmaps(self, cross_map, binned_dl_theory_dict):
        maps = cross_map.split('x')

        cross_map1 = cross_map + '_EDE'
        #this should never actually happen
        if(cross_map1 not in binned_dl_theory_dict):
            cross_map1 = maps[1] + 'x' + maps[0] + '_EDE'
        
        # two possible cases 
        # swap out the _E and _B if next char is x or end of string
        cross_map2 = re.sub(r'_B(?=x|$)', '_temp', cross_map)
        cross_map2 = re.sub(r'_E(?=x|$)', '_B', cross_map2)
        cross_map2 = re.sub(r'_temp', '_E', cross_map2)
        cross_map2e = cross_map2 + '_EDE'
        if(cross_map2e not in binned_dl_theory_dict):
            maps2 = cross_map2.split('x')
            cross_map2e = maps2[1] + 'x' + maps2[0] + '_EDE'
        return cross_map1, cross_map2e



    def apply_EDE_shift(self, cross_map, dl_theory_dict, params_values):
        maps = cross_map.split('x')
        map1 = re.sub(r'_{BE}$', '', maps[0])
        map2 = re.sub(r'_{BE}$', '', maps[1])
        angle1_name = 'alpha_' + maps[0]
        angle2_name = 'alpha_' + maps[1]
        # Use regex to remove _B, _E, or any other suffix ending with _ followed by letters
        angle1_name = re.sub(r'_[BE]$', '', angle1_name)
        angle2_name = re.sub(r'_[BE]$', '', angle2_name)
        angle1 = params_values[angle1_name]
        angle2 = params_values[angle2_name]

        #cross_map1, cross_map2 = self.assemble_eb_crossmaps(cross_map,
        #                                   dl_theory_dict)
        cross_map1 = 'EDE_EB'
        cross_map2 = 'EDE_EB'
        #try:
        ede_spec1 = dl_theory_dict[cross_map1]
        ede_spec2 = dl_theory_dict[cross_map2]
        #except KeyError as e:
        #    msg = f"Key '{e.args[0]}' not found. Additional info: {cross_map} not in dict. Available keys: {list(dl_theory_dict.keys())}"
        #    raise KeyError(msg) from e
        gMpl = params_values['gMpl']
        D_e1b2 = (ede_spec1 * np.cos(2*np.deg2rad(angle1)) * 
                                    np.cos(2*np.deg2rad(angle2)))
        D_b1e2 = (ede_spec2 * np.sin(2*np.deg2rad(angle1)) * 
                                    np.sin(2*np.deg2rad(angle2)))

        ede_shift = (D_e1b2 - D_b1e2)
        return ede_shift * gMpl

    def theory(self, params_values, dl_theory_dict, used_maps):
        # Compute the model prediction based on the parameter values
        # currently assumes it is only calculating EB
        # all theory based on 
        # https://bicep.rc.fas.harvard.edu/dbeck/20230202_cmbbirefringence/
        self.rotated_dict = {}
        for cross_map in used_maps:
            self.rotated_dict[cross_map] = self.rotate_spectrum(cross_map,
                                            dl_theory_dict, params_values)
            if(self.include_EDE):
                ede_shift = self.apply_EDE_shift(cross_map,
                                                dl_theory_dict, params_values)
                if self.rotated_dict[cross_map].shape != ede_shift.shape:
                    min_size = min(self.rotated_dict[cross_map].size, 
                                        ede_shift.size)
                    # Truncate both arrays to the minimum size
                    self.rotated_dict[cross_map] = self.rotated_dict[cross_map][:min_size]
                    ede_shift = ede_shift[:min_size]
                    self.rotated_dict[cross_map] += ede_shift

        self.rotated_dict = self.apply_bpwf(self.rotated_dict, self.bpwf, self.used_maps,do_cross=True)
        theory_vec = self.dict_to_vec(self.rotated_dict, used_maps)
        return theory_vec

def plot_covar_matrix(mat, used_maps=None, title='Log of covar matrix'):
    
    #print(max(mat[(mat<0.99)| (mat > 1.01)] ))
    nonzeros = np.abs(mat[(mat!=0) &( ~np.isnan(mat))])
    vpercent =max(np.percentile(nonzeros, 99), 1e-25)
    linthresh = np.percentile(nonzeros, 1)
    #print(nonzeros)
    #print(vpercent)
    cmap = plt.get_cmap('seismic')
    norm = mcolors.SymLogNorm(linthresh=linthresh, 
                                vmin=-vpercent, 
                                vmax=vpercent, base=10)
    plt.figure()
    plt.imshow(mat, cmap=cmap, norm=norm)
    plt.title(title)
    if(used_maps is not None):
        num_bins = int(mat.shape[0]/(len(used_maps)))

        tick_positions = np.arange(0, mat.shape[0], num_bins)
        plt.xticks(tick_positions, used_maps, 
                                rotation=30, ha='right')
        plt.yticks(tick_positions, used_maps)
    plt.colorbar()
    plt.savefig(title + '.png')
    plt.show()

def plot_best_fit(outpath, used_maps, param_names, 
                        param_bestfit, param_stats, signal_params={}):
    eb_like_cls = BK18_multicomp(used_maps=used_maps, 
                                signal_params=signal_params)
    used_maps = eb_like_cls.used_maps
    #np.savetxt('150220_invcovar.txt', eb_like_cls.cov_inv, delimiter=',')
    observed_datas = eb_like_cls.binned_dl_observed_dict
    theory_spectra = eb_like_cls.dl_theory
    param_values = {param_names[i]:param_bestfit[i] 
                            for i in range(len(param_names))}
    #param_values['alpha_BK18_220'] = 1.2
    #param_values['alpha_BK18_150'] = -0.5
    theory_vec=eb_like_cls.theory(param_values, 
                    theory_spectra, eb_like_cls.used_maps)
    observed_vec = eb_like_cls.binned_dl_observed_vec
    res = theory_vec - observed_vec
    chisq_mat = np.multiply(eb_like_cls.cov_inv, np.outer(res, res))
    chisq_tot = 'chisq:' + str(np.sum(chisq_mat))
    #plot_covar_matrix(chisq_mat, used_maps=used_maps, title=chisq_tot )

    rotated_dict = eb_like_cls.rotated_dict
    #print(rotated_dict)
    keys = list(rotated_dict.keys())
    #print(keys)
    # Get block chisqs
    num_bins = len(rotated_dict[used_maps[0]])
    chisq_map = np.zeros((len(used_maps), len(used_maps)))
    print('Num bins:' + str(num_bins))
    for i, cross_map1 in enumerate(used_maps):
        for j, cross_map2 in enumerate(used_maps):
            block = eb_like_cls.cov_inv[i*num_bins:(i+1)*num_bins,
                                        j*num_bins:(j+1)*num_bins]
            vector1 = observed_datas[cross_map1] - rotated_dict[cross_map1]
            vector2 = observed_datas[cross_map2] - rotated_dict[cross_map2]
            chisq = vector1.T @ block @ vector2
            chisq_map[i,j] = chisq
            
    plt.figure()
    print('Chisq:' + str(np.sum(chisq_map)))
    vrange = np.std(chisq_map)
    plt.imshow(chisq_map, cmap='bwr', vmin=-vrange, vmax=vrange)
    plt.colorbar()
    plt.xticks(np.arange(len(used_maps)), used_maps, rotation = 45)
    plt.yticks(np.arange(len(used_maps)), used_maps)
    plt.savefig(outpath + '_chisqmap.png')
    plt.show()
    # Initialize lists to store unique maps ending with _E and _B
    maps_B = set()
    maps_E = set()

    for key in keys:
        parts = key.split('x')
        if parts[0].endswith('_B'):
            maps_B.add(parts[0])
        if parts[0].endswith('_E'):
            maps_E.add(parts[0])
        if parts[1].endswith('_B'):
            maps_B.add(parts[1])
        if parts[1].endswith('_E'):
            maps_E.add(parts[1])
    maps_B = sorted(list(maps_B))
    maps_E = sorted(list(maps_E))
    param_stats = sorted(param_stats)
    num_columns = len(maps_B)  # Unique maps for columns
    num_rows = len(maps_E)      # Unique maps for rows
    # Create subplots
    fig, axes = plt.subplots(num_rows, num_columns, 
                    figsize=(num_columns * 4, num_rows * 4))

    try:
        axes = axes.flatten()  # Flatten axes array for easy indexing
    except AttributeError:
        print("Only one axis!")
        axes = [axes]
    # Plot each spectrum
    for idx, key in enumerate(keys):
        observed_data = observed_datas[key]
        best_fit_data = rotated_dict[key]
        #print(key)
        #print(observed_data - best_fit_data) 
        # Split key to find row and column indices
        parts = key.split('x')
        row_idx = (maps_E).index(parts[0]) if parts[0].endswith('_E') else (maps_E).index(parts[1])
        col_idx = (maps_B).index(parts[0]) if parts[0].endswith('_B') else (maps_B).index(parts[1])
        map_index = eb_like_cls.used_maps.index(key)
        num_bin = len(observed_data)
        covar_mat = eb_like_cls.filtered_covmat
        var = np.diag(covar_mat)[map_index*num_bin:num_bin*(map_index+1)]
        # Plotting observed data
        axes_index = row_idx * num_columns + col_idx
        #print(observed_data)
        
        axes[axes_index].errorbar(
                            x = range(len(observed_data)),
                            y=(observed_data), 
                            yerr = np.sqrt(var),
                            label='Observed', color='blue')
        # Plotting best fit data
        axes[axes_index].plot(best_fit_data, label='Best Fit', color='red')

        axes[axes_index].set_title(key)
        axes[axes_index].legend()
    for row_idx, map_E in enumerate(maps_E):
        angle = f"alpha_{map_E}"
        axes[row_idx].text(
            0.05, 1.4,  # X and Y position (top-left corner)
            param_stats[row_idx],  # The parameter stats
            transform=axes[row_idx].transAxes,  # Use axes coordinates
            fontsize=10, color='black',
            verticalalignment='top'
        )
    plt.tight_layout(pad=2)
    plt.savefig(outpath + '_bestfit.png')
    return 


def plot_triangle(root, replace_dict={}):
    # Load MCMC samples from the specified root
    samples = loadMCSamples(root)
    print([name.name for name in samples.getParamNames().names])
    
    param_names = [name.name for name in samples.getParamNames().names
                   if ('chi2' not in name.name and
                       'weight' not in name.name and
                       'betadust' not in name.name and
                       'betasync' not in name.name and
                       'minuslogprior' not in name.name)]
    
    # Get the mean and std of the parameters for titles
    mean_std_strings = []
    means = []
    count = 0
    for param in param_names:
        
        mean = samples.mean(param)
        if(param in replace_dict):
            mean = replace_dict[param]
        chisq = samples.mean('chi2')
        std = samples.std(param)
        if(count == 0):
            mean_std_strings.append(f"{param}: {mean:.2f} ± {std:.2f} chisq={chisq:.2f}")
            count += 1
        else:
            mean_std_strings.append(f"{param}: {mean:.2f} ± {std:.2f}")

        means.append(mean)

    # Create a triangle plot with all variables
    g = plots.get_subplot_plotter()
    g.triangle_plot(samples, param_names, filled=True)

    # Add the mean and std to the plot title
    plt.suptitle("\n".join(mean_std_strings), fontsize=10)
    plt.tight_layout()
    # Save the plot
    plt.savefig(f"{root}_triangle_plot.png")
    print(f"Triangle plot saved as {root}_triangle_plot.png")
    #plt.show()
    return param_names, means, mean_std_strings

def plot_eigenvalues_eigenvectors(matrix):
    """
    Plots the eigenvalues and eigenvectors of a given square matrix.

    Parameters:
    - matrix (np.ndarray): A square matrix for which to compute and plot eigenvalues and eigenvectors.
    """
    # Compute eigenvalues and eigenvectors
    eigenvalues, eigenvectors = np.linalg.eig(matrix)

    # Create subplots
    plt.figure(figsize=(12, 6))

    # Subplot for eigenvalues
    plt.subplot(1, 2, 1)
    plt.bar(range(len(eigenvalues)), np.log(eigenvalues), color='b')
    plt.title('Eigenvalues')
    plt.xlabel('Index')
    plt.ylabel('Ln Eigenvalue')

    # Subplot for eigenvectors
    plt.subplot(1, 2, 2)
    plt.imshow(np.log(np.abs(eigenvectors)))
    plt.colorbar()
    plt.title('Ln abs Eigenvectors')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.grid()
    plt.legend()

    plt.tight_layout()
    plt.show()
    return

def plot_sim_peaks(chains_path, single_sim, sim_nums, single_path=None):
    modes_dict = {}
    single_df = None
    for i in range(1, sim_nums + 1):
        file_path = chains_path.replace('XXX', f'{i:02d}')
        print('loading:' + str(file_path))
        # Read the first line to get the correct header
        with open(file_path, 'r') as f:
            first_line = f.readline().strip()  # Read the first line
            # Remove the '#' and split to get the correct column names
            corrected_header = first_line.replace('#', '').split()
        
        chain_df = pd.read_csv(file_path, delim_whitespace=True, comment='#')
        chain_df.columns = corrected_header
        if(single_sim == i):
            single_df = chain_df
        for column in chain_df.columns:
            if column not in modes_dict:
                modes_dict[column] = []
            mode = np.mean(chain_df[column])
            modes_dict[column].append(mode)
    # Convert the modes dictionary to a DataFrame
    modes_df = pd.DataFrame.from_dict(modes_dict)
    default_cols = ['#', 'weight', 'minuslogpost', 'minuslogprior',
                   'minuslogprior__0', 'chi2', 'chi2__my_likelihood']
    param_names = [col for col in modes_df.columns if col not in default_cols]
    print(param_names)
    # Create a corner plot of the mean of modes
    #print(modes_df)
    #print(modes_df[param_names])
    
    
    fig = corner.corner(modes_df[param_names], 
                        labels=param_names, 
                    show_titles=True, 
                    title_kwargs={"fontsize": 12},
                    hist_kwargs={'color':'red', 'density':True},
                    contour_kwargs={'colors':'red'})
    
    colors = ['red', 'blue', 'green', 'orange']
    for i in range(1,2):
        if(single_path is None):
            file_path = chains_path.replace('XXX', f'{i:02d}')    
        else:
            file_path = single_path
        single_df = pd.read_csv(file_path, delim_whitespace=True, comment='#')
        single_df.columns = corrected_header
        corner.corner(single_df[param_names], labels=param_names,
                    show_titles=False, 
                    hist_kwargs={'color': colors[i], 'density':True},
                    contour_kwargs={'colors': colors[i]}, 
                    fig=fig)
    
    
    # Show the plot
    supertitle = 'Sim' + str(single_sim) + ' on top of ' + str(sim_nums) + ' sims'
    plt.suptitle(supertitle)
    plt.show()

    return 

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
                    #'BK18_K95', 
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
    param_names, means, mean_std_strs = plot_triangle(outpath, replace_dict)
    plot_best_fit(outpath, all_cross_spectra,  
                        param_names, means, mean_std_strs, 
                        signal_params=signal_params)
    return 



def main():
    # Set up argument parsing
    parser = argparse.ArgumentParser()
    parser.add_argument('-p', '--output_path', default='chains/default',
                        help='directory to save the mcmc chains and plots')
    parser.add_argument('-o', '--overwrite', action='store_true',
                        help='whether to overwrite current chains')
    parser.add_argument('-n', '--sim_num', default=-1, type=int,
                        help='Simulation num to extract params from, defaults to real data')
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
    multicomp_mcmc_driver(args.output_path, args.overwrite, args.sim_num)
    
if __name__ == '__main__':
    #matlab_covmat = '/n/home08/liuto/GitHub/EB_analysis/bk18covmat.mat'
    #plot_sim_peaks('chains/simXXX/mpl00c.1.txt', 1, 10)
    #                single_path='chains/fullcov_real/mpl00b.1.txt')
    main()


