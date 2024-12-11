import numpy as np
from astropy.io import fits
import sys, os
import pandas as pd
import re
import glob
#Assume installed from github using "git clone --recursive https://github.com/cmbant/CAMB.git"
#This file is then in the docs folders
camb_path = os.path.realpath(os.path.join(os.getcwd(),'..'))
sys.path.insert(0,camb_path)

import camb
from camb import model, initialpower, correlations
print('Using CAMB %s installed at %s'%(camb.__version__,os.path.dirname(camb.__file__)))

import bicep_data_consts

def check_file_header(file_path, reference_header):
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

def load_observed_spectra(observed_data_path, used_maps, map_reference_header, num_bins=None):
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
    
    reference_header = map_reference_header
    print("Loading: " + str(observed_data_path))
    try:
        
        map_reference_header = check_file_header(observed_data_path, reference_header)
        map_reference_header = map_reference_header
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
        observed_spectra_dict[input_str] = obs_data[:num_bins, used_cols[i]]

    
    return observed_spectra_dict, map_reference_header

def load_bpwf(bpwf_directory, map_reference_header, num_bins=None):
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
        reference_header = map_reference_header
        # List to hold all loaded data
        bpwf_data = []
        if(num_bins is None):
            num_bins = len(bpwf_files)
        for n, bfile in enumerate(bpwf_files):
            if(n>=num_bins):
                print('Skipping ' + str(bfile))
                continue
            print("Loading: " + str(bfile))
            # Read the header and check consistency
            map_reference_header = check_file_header(bfile, reference_header)
            # Load data, don't ignore the first column
            bpwf_data.append(np.loadtxt(bfile))

        # Concatenate and return all BPWF data
        return np.stack(bpwf_data, axis=0), map_reference_header


def load_covariance_matrix(covmat_path, map_reference_header):
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
    map_reference_header = check_file_header(covmat_path, map_reference_header)
    full_covmat = np.loadtxt(covmat_path)
    shap = full_covmat.shape
    
    assert shap[0] == shap[1], "Covariance matrix must be square."
    return full_covmat

def load_cmb_spectra(lensing_path, dust_paths):
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

def include_ede_spectra(ede_path, theory_dict):
        k_to_uk = 1e6
        cmb_temp = 2.726
        data = pd.read_csv(ede_path, delim_whitespace=True, comment='#', header=None)
        data.columns = ['l', 'TT', 'EE', 'TE', 'BB', 'EB', 'TB', 'phiphi', 'TPhi', 'Ephi']
        # Extract 'l' and 'EB' columns
        EB_values = data['EB'].to_numpy()
        EB_ede_dls = -EB_values * np.square(k_to_uk) * np.square(cmb_temp)
        theory_dict['EDE_EB'] = EB_ede_dls
        return theory_dict

'''
Everything after this is deprecated
'''
def bin_spectrum(n_bins, delta_ell, ell_min, spectrum):
    """
    Bins a spectrum into a specified number of bins by averaging values within each bin.

    Parameters:
    -----------
    n_bins : int
        The number of bins into which the spectrum will be divided.

    delta_ell : int
        The width of each bin, specifying the number of elements in each bin.

    ell_min : int
        The minimum value of `ell` corresponding to the starting index of the spectrum.

    spectrum : ndarray
        The input spectrum to be binned, with values indexed by `ell`.

    Returns:
    --------
    binned_spectrum : ndarray
        An array of length `n_bins` containing the average values of the spectrum within each bin.

    bin_starts : ndarray
        An array containing the starting index of each bin.
    """
    
    binned_spectrum = np.zeros(n_bins)
    bin_starts = np.zeros(n_bins + 1,dtype=int)
    bin_starts[0]=ell_min
    for i in range(1, len(bin_starts)):
        bin_starts[i] = bin_starts[i-1] + delta_ell
    for ell_b in range(n_bins):
        bin_cur = 0
        
        for ell_0 in range(0, delta_ell):
            ell_cur = ell_min + delta_ell * ell_b + ell_0
            bin_cur += spectrum[ell_cur]

        binned_spectrum[ell_b] = bin_cur / delta_ell
    return binned_spectrum, bin_starts

def read_planck(filepath, spectrum_type):
    """
    Reads Planck power spectrum data from a FITS file and returns the power spectrum (Dl) as a function of multipole moment (l).

    Parameters:
    -----------
    filepath : str
        The path to the FITS file containing the Planck power spectrum data.
    
    spectrum_type : int
        The type of power spectrum to retrieve:
        - 0: TT (Temperature-Temperature) spectrum
        - 1: EE (E-mode Polarization) spectrum
        - 2: BB (B-mode Polarization) spectrum
        - 3: TE (Temperature-E-mode Polarization) spectrum
    
    Returns:
    --------
    ls : ndarray
        Array of multipole moments (l) corresponding to the power spectrum.
    
    dls : ndarray
        Array of power spectrum values (Dl) corresponding to the multipole moments.
    
    errs : ndarray
        Array of errors associated with the power spectrum values.

    Raises:
    -------
    AttributeError:
        If an unsupported `spectrum_type` is provided or if data for the spectrum type is unavailable.
    """
    spectrum_data = fits.open(filepath)

    if(spectrum_type == 0):
        lowind = 1
        highind = 8
    elif(spectrum_type == 1):
        lowind = 3
        highind = 12
    elif(spectrum_type == 2):
        raise AttributeError('No data available for spectrum type ' + str(spectrum_type))
    elif(spectrum_type == 3):
        lowind = 2
        highind = 10
    else:
        raise AttributeError('No data available for spectrum type ' + str(spectrum_type))


    lowls = spectrum_data[lowind].data['ELL']
    lowdls = spectrum_data[lowind].data['D_ELL']
    #TODO have errs be more than just the average
    lowerrs = (spectrum_data[lowind].data['ERRUP'] + spectrum_data[lowind].data['ERRDOWN']) / 2

    hils = spectrum_data[highind].data['ELL']
    hidls = spectrum_data[highind].data['D_ELL']
    hierrs = spectrum_data[highind].data['ERR'] 

    ls = np.concatenate((lowls, hils))
    dls = np.concatenate((lowdls, hidls))
    errs = np.concatenate((lowerrs, hierrs))
    return ls, dls, errs

def load_data(spectrum_type, datafile=None, raw_cls=False):
    """
    Loads power spectrum data from either a Planck data file or generates a theoretical spectrum using CAMB.

    Parameters:
    -----------
    spectrum_type : int
        The type of power spectrum to load:
        - 0: TT (Temperature-Temperature)
        - 1: EE (E-mode Polarization)
        - 2: BB (B-mode Polarization)
        - 3: TE (Temperature-E-mode Polarization)

    datafile : str, optional
        The path to the Planck data file. If provided, the function reads the spectrum from the file.

    raw_cls : bool, optional
        If True, returns raw Cl values from CAMB results.

    Returns:
    --------
    GLOBAL_VAR : dict
        Dictionary containing the power spectrum data and cosmological results.
    """
    GLOBAL_VAR = {}
    # TODO add ability to do multiple spectra
    pars = camb.CAMBparams()
    default_H0 = 67.7
    default_ombh2 = 0.02241
    default_omch2 = 0.1191
    pars.set_cosmology(H0=default_H0, ombh2=default_ombh2, omch2=default_omch2)
    results = camb.get_results(pars)
    GLOBAL_VAR['results'] = results

    if(datafile is None):
        powers = results.get_lensed_scalar_cls(raw_cl=raw_cls,CMB_unit='muK')
        if(spectrum_type=='all'):
            GLOBAL_VAR['TT'] = powers[:, 0]
            GLOBAL_VAR['EE'] = powers[:, 1]
            GLOBAL_VAR['BB'] = powers[:, 2]
            GLOBAL_VAR['TE'] = powers[:, 3]
        else:
            GLOBAL_VAR['measured'] = powers[:, spectrum_type]
        GLOBAL_VAR['ls'] = range(len(powers[:, 0]))
    else:
        ls, dls, errs = read_planck(datafile, spectrum_type)
        GLOBAL_VAR['measured'] = dls
        GLOBAL_VAR['ls'] = ls
        GLOBAL_VAR['errs'] = errs
    return GLOBAL_VAR

def load_dust_lensing_model(bin_start=1, bin_end=10, mapname='BK18_B95', 
                            dust_path='input_data/model_dust.npy', 
                            lensing_path='input_data/model_lens.npy', 
                            bandpowerwindowfunction_path='input_data/bpwf.npy',
                            plot=False):
    """
    Loads the dust and lensing model spectra, applies the bandpower window function, and returns the binned results.

    Parameters:
    -----------
    bin_start : int, optional
        Starting bin for analysis.

    bin_end : int, optional
        Ending bin for analysis.

    mapname : str, optional
        Name of the map or dataset to load the spectra for.

    dust_path : str, optional
        Path to the dust model data file.

    lensing_path : str, optional
        Path to the lensing model data file.

    bandpowerwindowfunction_path : str, optional
        Path to the bandpower window function data file.

    plot : bool, optional
        If True, plots the binned EE and BB spectra.

    Returns:
    --------
    bpwf_ls : ndarray
        The multipole moments after applying the bandpower window function.

    bpwf_cls : ndarray
        The binned spectra after applying the bandpower window function.

    spectrum_dict : dict
        Dictionary containing the binned EE and BB spectra.
    """
    TT = 0; TE = 1; EE = 2;
    BB = 3; TB = 4; EB = 5;
    ET = 6; BT = 7; BE = 8;
    dust_model = np.load(dust_path, allow_pickle=True)
    lensing_model = np.load(lensing_path, allow_pickle=True)
    bpwf = np.load(bandpowerwindowfunction_path, allow_pickle=True)
  
    dataset_index = bicep_data_consts.SPECTRA_DATASETS.index(mapname)

    # a lot of zero indices due to matlab file conversion
    bpwf_ls = bpwf[0][dataset_index]['l'][0]
    dust_cls = dust_model.item()['model_dust'][mapname][0][0]['Cs_l'][0][0][bpwf_ls]
    lensing_cls = lensing_model.item()['model_lens'][0][0]['Cs_l'][bpwf_ls]
    bpwf_cls = bpwf[0][dataset_index]['Cs_l']
    l_bins = bicep_data_consts.L_BIN_CENTERS
    ee_binned = np.matmul(dust_cls[:,EE] + lensing_cls[:,EE], bpwf_cls[:,:,EE])
    
    bb_binned = np.matmul(dust_cls[:,BB] + lensing_cls[:,BB], bpwf_cls[:,:,BB])

    spectrum_dict = {}
    spectrum_dict['EE_binned'] = ee_binned[bin_start:bin_end]
    spectrum_dict['BB_binned'] = bb_binned[bin_start:bin_end]


    if(plot):
        plt.figure()
        plt.title('EE Map: ' + str(mapname))
        plt.plot(l_bins, ee_binned, label='After applying BPWF')
        plt.plot(bpwf_ls,dust_cls[:,EE] + lensing_cls[:,EE], label='Lensing+Dust model')
        plt.ylabel(r'$C_{\ell}^{EE}\cdot\ell(\ell+1)/(2\pi)$  [$\mu K^2$]')
        plt.xlabel(r'$\ell$')
        plt.legend()
        plt.show()

        plt.title('BB Map: ' + str(mapname))
        plt.plot(l_bins, bb_binned, label='After applying BPWF')
        plt.plot(bpwf_ls,dust_cls[:,BB] + lensing_cls[:,BB], label='Lensing+Dust model')
        plt.ylabel(r'$C_{\ell}^{BB}\cdot\ell(\ell+1)/(2\pi)$  [$\mu K^2$]')
        plt.xlabel(r'$\ell$')
        plt.legend()
        plt.show()


    
    return bpwf_ls, bpwf_cls[:,:,EE], spectrum_dict

def load_bicep_sim_data(map_name, bin_start=1, bin_end=10, EB_index=5, data_path='input_data/dust_simulations.npy'):
    """
    Loads simulated BICEP dust data for a specified map and bin range.

    Parameters:
    -----------
    map_name : str
        The name of the map or dataset.

    bin_start : int, optional
        The starting bin for the data.

    bin_end : int, optional
        The ending bin for the data.

    EB_index : int, optional
        The index corresponding to the EB spectra in the simulations.

    data_path : str, optional
        Path to the dust simulation data file.

    Returns:
    --------
    spectrum_dict : dict
        Dictionary containing the EB simulated spectra.
    """
    dust_sims = np.load(data_path, allow_pickle=True)
    dataset_index = bicep_data_consts.SPECTRA_DATASETS.index(map_name)
    eb_dust_spectra = dust_sims[dataset_index][bin_start:bin_end,EB_index,:]
    spectrum_dict = {}
    spectrum_dict['EB_sims'] = eb_dust_spectra
    
    return spectrum_dict


def load_bicep_data(plot=False, mapname=None, output_plots='output_plots', zero_ede=False, bin_end = 17):
    """
    Loads real BICEP data, extracts EB observed data, and computes EB EDE binned spectra.

    Parameters:
    -----------
    plot : bool, optional
        If True, generates plots for the EE, BB, and EB spectra.

    mapname : str, optional
        Name of the map or dataset. Default is 'BK18_B95'.

    output_plots : str, optional
        Directory path for saving output plots.

    zero_ede : bool, optional
        If True, sets the EDE spectrum to zero.

    bin_end : int, optional
        Ending bin for the analysis.

    Returns:
    --------
    l_bins : ndarray
        Array of multipole moments (l) corresponding to the binned spectra.

    spectrum_dict : dict
        Dictionary containing the observed EB data, simulated EB spectra, and binned EE/BB spectra.
    """
    #data_path= 'input_data/real_spectra_bicep.npy'
    #cov_file = 'input_data/bicep_cov.npy'
    offdiag = 2
    data_path= 'input_data/bicep_norot_realspectra.npy'
    cov_file = 'input_data/bicep_cov_simdust.npy'
    bin_start = 1

    spectrum_dict = {}
    if(mapname is None):
        mapname = 'BK18_B95'
    dataset_index = bicep_data_consts.SPECTRA_DATASETS.index(mapname)
    EB_index = 5
    l_bins = bicep_data_consts.L_BIN_CENTERS

    cov_data = np.load(cov_file, allow_pickle=True, encoding='latin1')
    cov_mat = cov_data[dataset_index][0][0]
    cov = cov_mat[EB_index]

    # Create a mask to keep only the diagonal and up to the 2nd off-diagonal terms
    size = cov.shape[0]
    mask = np.abs(np.arange(size)[:, None] - np.arange(size)) <= offdiag

    # Apply the mask to the covariance matrix
    truncated_cov_matrix = cov * mask

    
    np_mat = np.load(data_path, allow_pickle=True, encoding='latin1')
    spectra = np_mat[dataset_index][0] 
    
    temp_dict = load_bicep_sim_data(mapname, bin_start=bin_start, bin_end=bin_end, EB_index=EB_index)
    spectrum_dict.update(temp_dict)
    bpwf_ls, ee_bpwf_cls, temp_dict =load_dust_lensing_model(bin_start=bin_start, bin_end=bin_end, mapname=mapname)
    spectrum_dict.update(temp_dict) 
    l_bins = l_bins[bin_start:bin_end]
    spectrum_dict['EB_observed'] = spectra[bin_start:bin_end,EB_index]
    spectrum_dict['EB_var'] = truncated_cov_matrix[bin_start:bin_end, bin_start: bin_end]
    

    # Extract the data
    if(zero_ede):
        eb_ede_binned = np.zeros(bin_end)
    else:
        eb_ede_theory_provided = read_ede_data()[bpwf_ls]
        eb_ede_binned = np.matmul(eb_ede_theory_provided,ee_bpwf_cls)
    spectrum_dict['EB_EDE'] = eb_ede_binned[bin_start:bin_end]


    return l_bins, spectrum_dict

def read_ede_data(data_path='input_data/fEDE0.07_cl.dat'):
    """
    Reads EDE data from a specified file, extracts the EB spectrum, 
    and returns the processed EB values.


    Parameters:
    data_path (str): Path to the input data file (default is 'input_data/fEDE0.07_cl.dat').

    Returns:
    np.ndarray: Processed EB spectrum values, converted to µK^2 and scaled by 2π.
    """
    k_to_uk = 1e6
    data = pd.read_csv(data_path, delim_whitespace=True, comment='#', header=None)
    data.columns = ['l', 'TT', 'EE', 'TE', 'BB', 'EB', 'TB', 'phiphi', 'TPhi', 'Ephi']
    # Extract 'l' and 'EB' columns
    EB_values = data['EB']
    return -EB_values * np.square(k_to_uk) * 2 * np.pi

def load_eskilt_data(data_path = 'input_data/HFI_f_sky_092_EB_o.npy'):
    """
    Loads Eskilt data, computes and bins EE and BB spectra, and returns the spectrum dictionary.

    Parameters:
    data_path (str): Path to the Eskilt EB data file (default is 'input_data/HFI_f_sky_092_EB_o.npy').

    Returns:
    tuple: Contains the following elements:
        - bin_starts (np.ndarray): Starting values of each bin.
        - raw_cl (bool): Whether raw Cl data is being used.
        - spectrum_dict (dict): Dictionary containing binned and observed spectra.
    """
    raw_cl = True
    spectrum_dict = load_data('all', raw_cls=raw_cl)
    c_l_EB_o_mean_std = np.load(data_path)
    spectrum_dict['EB_observed'] = c_l_EB_o_mean_std[:, 0]
    spectrum_dict['EB_var'] = np.square(c_l_EB_o_mean_std[:, 1])
    ell_min = 51
    ell_max = 1490
    delta_ell = 20
    ell = np.arange(ell_min, ell_max+1, delta_ell)
    n_bins = len(ell)
    ee_binned, bin_starts = bin_spectrum(n_bins, delta_ell, ell_min, spectrum_dict['EE'])
    bb_binned, bin_starts = bin_spectrum(n_bins, delta_ell, ell_min, spectrum_dict['BB'])
    spectrum_dict['EE_binned'] = ee_binned
    spectrum_dict['BB_binned'] = bb_binned
    return bin_starts, raw_cl, spectrum_dict   



