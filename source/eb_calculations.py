import warnings
import numpy as np
import re
import math
from functools import lru_cache
# Physical constants

h_J_s = 6.62607015e-34
kB_J_K = 1.380649e-23
GHZ_KELVIN = h_J_s / kB_J_K * 1e9
T_CMB_K = 2.7255  # fiducial CMB temperature
D2R=np.pi/180.0
FPIVOT_DUST = 353.0
FPIVOT_SYNC = 23.0
LPIVOT = 80.0
TDUST = 19.6



def determine_map_freqs(mapset):
    if(mapset == 'BK18'):
        calc_spectra = [
                    'BK18_220', 
                    'BK18_150', 
                    'BK18_K95', 
                    'BK18_B95e']
        

    elif(mapset == 'BK18_planck'):
        calc_spectra = [
                    'BK18_220', 
                    'BK18_150', 
                    'BK18_K95', 
                    'BK18_B95e',
                    'P030e', 
                    'P044e', 
                    'P143e',
                    'P217e',
                    'P353e'
                   ]
    elif(mapset == 'planck'):
        calc_spectra = [
                    'P030e', 
                    'P044e', 
                    'P143e',
                    'P217e',
                    'P353e'
                   ]
    elif(mapset == 'BK_good'):
        calc_spectra = ['BK18_150',
                        'BK18_B95e'] 
    elif(mapset == 'BK_bad'):
        calc_spectra = ['BK18_220',
                        'BK18_K95']  
    else:
        calc_spectra = [mapset]
    return calc_spectra

def determine_spectrum_type(spectrum_name):
    """
    Determines the polarization spectrum type (EE, BB, EB, or BE) 
    from a string formatted like 'something_Exsomething_B'.

    Args:
        spectrum_name (str): The input string containing '_E' or '_B' before and after an 'x'.

    Returns:
        str: A 2-letter string indicating the spectrum type, e.g., 'EB', 'BE', 'EE', or 'BB'.

    Raises:
        AssertionError: If the input format is not as expected.
    """
    spectra = spectrum_name.split('x')
    assert len(spectra) == 2, "spectrum name isn't properly formatted: " + str(spectrum_name)
    
    spec1 = spectra[0][-2:]
    spec2 = spectra[1][-2:]
    
    assert spec1 in ['_E', '_B'], "spectrum name isn't properly formatted: " + str(spectrum_name)
    assert spec2 in ['_E', '_B'], "spectrum name isn't properly formatted: " + str(spectrum_name)

    spec_type = spec1[-1] + spec2[-1]
    return spec_type

def apply_initial_conditions(dl_theory_dict, used_maps):
    initial_conditions_dict = {}
    for used_map in used_maps:
        spec_type = determine_spectrum_type(used_map)
        if(spec_type in ['EB', 'BE']):
            continue
        initial_conditions_dict[used_map] = dl_theory_dict[spec_type]
    return initial_conditions_dict

def get_map_freqs(used_map):
    maps = used_map.spilt('x')
    map1 = maps[0]
    map2 = maps[1]
    return map1[:-2], map2[:-2]

def determine_angle_names(used_map):
    maps = used_map.split('x')
    angle1_name = 'alpha_' + maps[0]
    angle2_name = 'alpha_' + maps[1]
    angle1_name = re.sub(r'_[BE]$', '', angle1_name)
    angle2_name = re.sub(r'_[BE]$', '', angle2_name)
    return angle1_name, angle2_name

## TODO: something is off with this
def get_other_spec_map(used_map, all_maps):
    maps = used_map.split('x')
    map1 = maps[0]
    map2 = maps[1]
    ee_map = map1[:-2] + '_Ex' + map2[:-2] + '_E'
    bb_map = map1[:-2] + '_Bx' + map2[:-2] + '_B'
    eb_map = map1[:-2] + '_Ex' + map2[:-2] + '_B'
    be_map = map1[:-2] + '_Bx' + map2[:-2] + '_E'
    if(eb_map in all_maps):
        return ee_map, bb_map, eb_map
    elif(be_map in all_maps):
        return ee_map, bb_map, be_map
    else:
        raise ValueError('Map does not have all three spec types: ' + str(used_map))

# Caching angle lookup
@lru_cache(maxsize=None)
def determine_angle_names_cached(map_name):
    return determine_angle_names(map_name)

def apply_EDE(initial_theory_dict, params_values, dl_theory_dict, used_maps):
    """
    Applies Early Dark Energy (EDE) replacement to EB/BE spectra in a copy of the theory dict.

    Parameters:
        initial_theory_dict (dict): Theory Cls to be copied and modified.
        params_values (dict): Parameters, must include 'gMpl'.
        dl_theory_dict (dict): Must include 'EB_EDE' key with EDE spectrum.
        used_maps (list): List of map names like 'foo_Exbar_B'.

    Returns:
        dict: New theory dictionary with EDE EB/BE spectra replaced.
    """
    ede = dl_theory_dict['EB_EDE']
    g = params_values['gMpl']
    post_dict = {k: v.copy() for k, v in initial_theory_dict.items()}

    for m in used_maps:
        spec = determine_spectrum_type(m)
        if spec in ['EB', 'BE']:
            if m in post_dict:
                post_dict[m] = ede * g
            else:
                print(f"Warning: map {m} not found in theory dict, skipping.")

    return post_dict


def apply_cmb_rotation(post_inflation_dict, params_values, dl_theory_dict, used_maps):
    """
    Applies rotation of CMB polarization spectra due to a global angle alpha_CMB.

    Parameters:
        post_inflation_dict (dict): Theory Cls after inflation but before CMB rotation.
        params_values (dict): Parameters, must include 'alpha_CMB' in degrees.
        dl_theory_dict (dict): (Unused here, kept for compatibility).
        used_maps (list): List of map names like 'foo_Exbar_B'.

    Returns:
        dict: New theory dictionary with rotated spectra.
    """
    angle = np.deg2rad(params_values['alpha_CMB'])

    # Precompute trigonometric terms
    sin2 = np.sin(2 * angle)
    cos2 = np.cos(2 * angle)
    sin4 = np.sin(4 * angle)
    cos4 = np.cos(4 * angle)
    sin2_sq = sin2 ** 2
    cos2_sq = cos2 ** 2

    # Deep copy the theory dict
    post_travel_dict = {k: v.copy() for k, v in post_inflation_dict.items()}

    for m in used_maps:
        spec = determine_spectrum_type(m)
        ee_map, bb_map, eb_map = get_other_spec_map(m, used_maps)

        # Pre-fetch values to avoid repeated dict access
        Cl_EE = post_inflation_dict[ee_map]
        Cl_BB = post_inflation_dict[bb_map]
        Cl_EB = post_inflation_dict[eb_map]

        if spec == 'BB':
            post_travel_dict[m] = Cl_EE * sin2_sq + Cl_BB * cos2_sq + Cl_EB * sin4

        elif spec == 'EE':
            post_travel_dict[m] = Cl_EE * cos2_sq + Cl_BB * sin2_sq - Cl_EB * sin4

        elif spec in ['EB', 'BE']:
            post_travel_dict[m] = 0.5 * (Cl_EE - Cl_BB) * sin4 + Cl_EB / cos4

    return post_travel_dict

def apply_dust(post_travel_dict, bandpasses, params_values):
    """
    Applies dust contamination model to post-CMB-rotation Cls.

    Parameters:
        post_travel_dict (dict): Theory Cls after rotation.
        bandpasses (dict): Dict of bandpass arrays keyed by map names.
        params_values (dict): Must include dust amplitude, tilt, and beta_dust.

    Returns:
        dict: Updated Cls with dust power added.
    """
    beta_dust = params_values['beta_dust']
    dust_cache = {}
    
    for used_map, dls in post_travel_dict.items():
        lmax = dls.shape[0]
        ratio = np.arange(lmax) / LPIVOT

        spec_type = determine_spectrum_type(used_map)
        freq1, freq2 = get_map_freqs(used_map)

        # Use cached dust scaling if available
        if freq1 not in dust_cache:
            dust_cache[freq1] = dust_scaling(beta_dust, TDUST, bandpasses[freq1], FPIVOT_DUST, bandcenter_err=1)
        if freq2 not in dust_cache:
            dust_cache[freq2] = dust_scaling(beta_dust, TDUST, bandpasses[freq2], FPIVOT_DUST, bandcenter_err=1)

        dust_scale1 = dust_cache[freq1]
        dust_scale2 = dust_cache[freq2]

        if spec_type in ['EB', 'BE']:
            A_key = f'A_dust_EB'
            alpha_key = f'alpha_dust_EB'
        else:
            A_key = f'A_dust_{spec_type}'
            alpha_key = f'alpha_dust_{spec_type}'

        A_dust = params_values[A_key]
        alpha_dust = params_values[alpha_key]

        with warnings.catch_warnings():
            warnings.simplefilter('ignore', category=RuntimeWarning)
            dustpow = A_dust * np.power(ratio, alpha_dust)

        if np.isinf(dustpow[0]):
            dustpow[0] = 0

        dls += dustpow * dust_scale1 * dust_scale2

    return post_travel_dict

def apply_det_rotation(post_travel_dict, params_values):
    post_detection_dict = {k: v.copy() for k, v in post_travel_dict.items()}
    for used_map in post_travel_dict.keys():
        ## TODO: finish this
        post_detection_dict[used_map] = 0
    return post_detection_dict
###########################################################
# DEPRECATED FUNCTIONS BELOW
#
###########################################################
def apply_EDE_shift(cross_map, dl_theory_dict, params_values, fixed_dust=True):
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
        if(fixed_dust):
            cross_map1 = 'EDE_EB'
            cross_map2 = 'EDE_EB'
        else:
            cross_map1 = cross_map + '_EDE'
            cross_map2 = cross_map + '_EDE'
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




def dust_scaling(beta, Tdust, bandpass, nu0, bandcenter_err=1):
    """
    Calculates the greybody scaling of a dust signal defined at a pivot frequency (e.g., 353 GHz)
    to a specified bandpass.

    Parameters:
        beta (float): Spectral index for the dust emission.
        Tdust (float): Temperature of the dust in Kelvin.
        bandpass (object): Bandpass object containing:
            - `dnu` (array): Frequency bin widths.
            - `R` (array): Response matrix where R[:, 0] is frequency and R[:, 1] is the bandpass response.
            - `nu_bar` (float): Effective band center frequency.
            - `th_dust` (float): Dust conversion factor for the bandpass.
        nu0 (float): Pivot frequency in GHz (e.g., 353 GHz).
        bandcenter_err (float, optional): Error in the band center frequency. Defaults to 1 (no error).

    Returns:
        float: Dust scaling factor accounting for greybody spectrum and bandpass integration.
    """
    gb_int = np.sum(bandpass.dnu * bandpass.R[:, 1] * bandpass.R[:, 0] ** (3 + beta) /
                    (np.exp(GHZ_KELVIN * bandpass.R[:, 0] / Tdust) - 1))
    # Calculate values at pivot frequency.
    gb0 = nu0 ** (3 + beta) / (np.exp(GHZ_KELVIN * nu0 / Tdust) - 1)
    #  Add correction for band center error
    if bandcenter_err != 1:
        nu_bar = GHZ_KELVIN * bandpass.nu_bar
        # Conversion factor error due to bandcenter error.
        th_err = bandcenter_err ** 4 * (np.exp(GHZ_KELVIN * bandpass % nu_bar *
                                                (bandcenter_err - 1) / T_CMB_K) *
                                        (np.exp(nu_bar / T_CMB_K) - 1) ** 2 /
                                        (np.exp(nu_bar * bandcenter_err
                                                / T_CMB_K) - 1) ** 2)
        # Greybody scaling error due to bandcenter error.
        gb_err = bandcenter_err ** (3 + beta) * (np.exp(nu_bar / Tdust) - 1) / \
                    (np.exp(nu_bar * bandcenter_err / Tdust) - 1)
    else:
        th_err = 1
        gb_err = 1

    # Calculate dust scaling.
    return (gb_int / gb0) / bandpass.th_dust * (gb_err / th_err)

def add_all_dust_foregrounds(dl_theory_dict, data_params, bandpasses):
    """
    Adds dust foregrounds to the theoretical angular power spectrum for each map in the input dictionary.

    Parameters:
        dl_theory_dict (dict): Dictionary of theoretical angular power spectra with keys
            in the format `something_Exsomething_B`.
        data_params (dict): Dictionary containing parameters for dust modeling (e.g., amplitude, slope).
        bandpasses (dict): Dictionary of bandpass objects indexed by frequency.

    Returns:
        dict: Updated dictionary with dust foregrounds added to each spectrum.
    """
    dust_dict = {}
    for spec_map in dl_theory_dict:
        if(spec_map == 'EDE_EB'):
            dust_dict[spec_map] = dl_theory_dict[spec_map]
            for freq in bandpasses:
                for freq2 in bandpasses:
                    dust_foreground = add_foregrounds(bandpasses[freq], 
                            data_params, 'EB', 
                            lmax = dl_theory_dict[spec_map].shape[0],
                            bandpass2 = bandpasses[freq2])
                    freq_name = freq+'_Ex'+freq2+'_B_EDE'
                    freq_name2 = freq+'_Bx'+freq2+'_E_EDE'
                    dust_dict[freq_name] = dl_theory_dict[spec_map] + dust_foreground
                    dust_dict[freq_name2] = dust_dict[freq_name]
            continue
        freq_map = spec_map.split('x')
        assert len(freq_map)==2, 'spec map has more than one x:' + spec_map
        spec0 = freq_map[0][-1]
        spec1 = freq_map[1][-1]
        spec = spec0 + spec1
        freq0 = freq_map[0][:-2]
        freq1 = freq_map[1][:-2]
        assert freq0 == freq1, 'freq map not equal:' + freq0 +','+freq1
        # matches NNNNN_E|BxMMMMM_E|B

        freq = freq1
        bandpass = bandpasses[freq]
        lmax = dl_theory_dict[spec_map].shape[0]
        dust_foreground = add_foregrounds(bandpass, data_params, spec, lmax)
        dust_dict[spec_map] = np.array(dl_theory_dict[spec_map]) + np.array(dust_foreground)
    return dust_dict

def add_foregrounds(bandpass, data_params, spectrum, lmax,lmin=0, bandcenter_err=1, bandpass2=None):
    """
    Computes and returns the dust foreground angular power spectrum.

    Parameters:
        bandpass (object): Bandpass object containing frequency response details.
        data_params (dict): Dictionary of dust parameters, containing:
            - `A_dust_<spectrum>`: Amplitude of dust emission.
            - `alpha_dust_<spectrum>`: Slope of the dust power spectrum.
            - `beta_dust`: Spectral index of the dust.
        spectrum (str): Power spectrum type (`EE`, `EB`, or `BB`).
        lmax (int): Maximum multipole moment (ell).
        lmin (int, optional): Minimum multipole moment (ell). Defaults to 0.
        bandcenter_err (float, optional): Error in the band center frequency. Defaults to 1 (no error).

    Returns:
        ndarray: Array of foreground power spectrum values for each multipole moment.
    """
    try:
        A_dust = data_params['A_dust_' + spectrum]
        alpha_dust = data_params['alpha_dust_' + spectrum]
    except KeyError:
        spectrum = spectrum[1] + spectrum[0]
        A_dust = data_params['A_dust_' + spectrum]
        alpha_dust = data_params['alpha_dust_' + spectrum]
    
    beta_dust = data_params['beta_dust']
    dust_scale = dust_scaling(beta_dust, TDUST, bandpass, FPIVOT_DUST, bandcenter_err)
    if(not bandpass2 is None):
        dust_scale2 = dust_scaling(beta_dust, TDUST, bandpass2, FPIVOT_DUST, bandcenter_err)
    else:
        dust_scale2 = dust_scale
    ratio = np.arange(lmin, lmax)/LPIVOT
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', category=RuntimeWarning)
        dustpow = A_dust * np.power(ratio,alpha_dust)
    # l=0 gives inf, zero it out since shouldn't affect bp
    if(math.isinf(dustpow[0])):
        dustpow[0] = 0
    dl = dustpow * dust_scale * dust_scale2
    return dl

def filter_matrix(map_reference_header, matrix, used_maps, num_bins=None, zero_offdiag=False):
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
        
        num_maps = len(map_reference_header) - 1
        
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
        filter_cols = [map_reference_header.index(cross_map)-1 for cross_map in used_maps]
        all_bins = [index + i * num_maps for i in range(num_bins) for index in filter_cols]
        # Use np.ix_ to filter both rows and columns in the given indices
        filtered_mat = matrix[np.ix_(all_bins, all_bins)]
        reordered_mat = reorder_cov_matrix(filtered_mat, 
                                    num_bins, len(used_maps))
        if(zero_offdiag):
            offdiag = 1
        else:
            offdiag= reordered_mat.shape[0]
        #before_truncate = np.linalg.inv(reordered_mat)
        #trunc_covmat = self.truncate_covariance_matrix(reordered_mat,
        #                                    offdiag=offdiag)
        return reordered_mat

def reorder_cov_matrix(cov_matrix, n_bins, n_maps):
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
def truncate_covariance_matrix(cov_matrix, offdiag=1, block_size=1):
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
    
def calc_inverse_covmat(filtered_covmat, block_offdiag = 1):
    inverted_mat = np.linalg.inv(filtered_covmat)
    return inverted_mat


def apply_bpwf(map_reference_header, theory_dict, bpwf_mat, used_maps, do_cross=False):
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

                col = map_reference_header.index(map0)
                num_ells = bpwf_mat.shape[1]
                binned_theory_dict[map0] = np.matmul(bpwf_mat[:,:,col],
                                            theory_dict[map0][:num_ells])
            if('EDE_EB' in theory_dict):
                binned_theory_dict[cross_map + '_EDE'] = np.matmul(bpwf_mat[:,:,col],
                                                        theory_dict['EDE_EB'][:num_ells])*10
        return binned_theory_dict


def rotate_spectrum(cross_map, binned_dl_theory_dict, params_values):
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

def assemble_eb_crossmaps(cross_map, binned_dl_theory_dict):
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

def apply_EDE_shift(cross_map, dl_theory_dict, params_values, fixed_dust=True):
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
        if(fixed_dust):
            cross_map1 = 'EDE_EB'
            cross_map2 = 'EDE_EB'
        else:
            cross_map1 = cross_map + '_EDE'
            cross_map2 = cross_map + '_EDE'
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

def inject_signal(used_maps, signal_params, 
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
        for cross_map in used_maps:
            maps = cross_map.split('x')
            angle_name0 = 'alpha_' + maps[0]
            angle_name1 = 'alpha_' + maps[1]
            angle_name0 = re.sub(r'_[BE]$', '', angle_name0)
            angle_name1 = re.sub(r'_[BE]$', '', angle_name1)
            if(angle_name0 not in signal_params):
                signal_params[angle_name0] = 0
            if(angle_name1 not in signal_params):
                signal_params[angle_name1] = 0
        
            D_eb = rotate_spectrum(cross_map, binned_dl_theory_dict, signal_params)
            ede_shift = 0 
            if('gMpl' in signal_params): 
                ede_shift = apply_EDE_shift(cross_map, 
                            binned_dl_theory_dict, signal_params)
            binned_dl_observed_dict[cross_map] += D_eb + ede_shift
        return binned_dl_observed_dict
