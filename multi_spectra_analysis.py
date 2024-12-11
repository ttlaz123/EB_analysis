print("Loading Modules")
import os
import numpy as np
import random
import argparse
import glob
import shutil
import re
from concurrent.futures import ProcessPoolExecutor, as_completed
import pandas as pd
from scipy.stats import gaussian_kde
import eb_load_data as ld
import eb_plot_data as epd
import matplotlib.pyplot as plt
print("Loading Cobaya Modules")
from cobaya.model import get_model
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
        self.binned_dl_theory_dict = self.apply_bpwf(self.dl_theory, self.bpwf, self.used_maps)
        # Real Data
        self.binned_dl_observed_dict, self.map_reference_header = ld.load_observed_spectra(FILE_PATHS['observed_data'], 
                                    self.used_maps, self.map_reference_header, num_bins=num_bins)
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
        self.full_covmat = ld.load_covariance_matrix(FILE_PATHS[covmat_name], self.map_reference_header)
        self.filtered_covmat = self.filter_matrix(self.full_covmat, self.used_maps, num_bins=num_bins)
        #plot_covar_matrix(self.filtered_covmat, used_maps=self.used_maps)
        self.cov_inv = self.calc_inverse_covmat(self.filtered_covmat)

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
        rotated_dict = self.apply_bpwf(self.rotated_dict, self.bpwf, self.used_maps,do_cross=True)
        ebe_dict = self.apply_bpwf(self.ebe_dict, self.bpwf, self.used_maps,do_cross=True)
        tot_dict = self.tot_dict
        num_bin = len(next(iter(rotated_dict.values())))
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
            )
            ede_cont = ebe_dict[mapi]
            plt.plot(ede_cont, label='EDE contribution')
            parts = mapi.split('x')
            if(parts[0].endswith('_B')):
                ind = 0
            else:
                ind = 1
            result = parts[ind][:-2] + '_E'
            result = result + 'x' + result
            rot_cont = rotated_dict[mapi]
            plt.plot(rot_cont, label = 'Polarization Rotation')
            plt.plot(tot_dict[mapi], label='Both')
            # Convert dictionary to string with a newline after every two keys
            result_lines = []
            for i, (key, value) in enumerate(params_values.items(), start=1):
                result_lines.append(f"{key}: {value}")
                # Add a newline after every two keys
                if i % 2 == 0:
                    result_lines.append("\n")  # Blank line for separation

            # Join the lines to form the final string
            dict_as_string = ", ".join(result_lines)
            #dict_as_string = "\n".join(f"{key}: {value}" for key, value in params_values.items())
            plt.title(mapi + '\n ' + str(dict_as_string))
            plt.legend()
            plt.tight_layout()
            plt.show()

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
                                                        theory_dict['EDE_EB'][:num_ells])*10
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

    def filter_matrix(self, matrix, used_maps, num_bins=None):
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
        
        tot_bins = matrix.shape[0] / num_maps

        # Check if tot_bins is an integer by checking if the division results in a remainder
        if tot_bins != int(tot_bins):
            raise ValueError(f"Number of maps {num_maps} and "
                            f"size of covar matrix {matrix.shape[0]} don't fit, "
                            f"tot_bins {tot_bins} is not an integer.")

        tot_bins = int(tot_bins) 
        if(num_bins is None):
            num_bins = tot_bins
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
        self.ebe_dict = {}
        self.tot_dictt = {}
        self.tot_dict = {}
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
                    self.ebe_dict[cross_map] = ede_shift
                    self.tot_dictt[cross_map] = self.rotated_dict[cross_map] + self.ebe_dict[cross_map]
        self.tot_dict = self.apply_bpwf(self.tot_dictt, self.bpwf, self.used_maps,do_cross=True)
  
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
    epd.plot_best_fit(eb_like_cls, outpath, all_cross_spectra,  
                        param_names, means, mean_std_strs, 
                        signal_params=signal_params)
    return 

def run_simulation(s, output_path, overwrite):
    outpath = f"{output_path}{s:03d}"
    if(os.path.exists(outpath + '.1.txt')):
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


