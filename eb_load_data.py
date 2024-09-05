import numpy as np
from astropy.io import fits
import sys, os
import pandas as pd
import matplotlib.pyplot as plt
#Assume installed from github using "git clone --recursive https://github.com/cmbant/CAMB.git"
#This file is then in the docs folders
camb_path = os.path.realpath(os.path.join(os.getcwd(),'..'))
sys.path.insert(0,camb_path)

import camb
from camb import model, initialpower, correlations
print('Using CAMB %s installed at %s'%(camb.__version__,os.path.dirname(camb.__file__)))


import bicep_data_consts

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
        The input spectrum to be binned, with values indexed by `ell`. The length of `spectrum` should be sufficient to cover all bins.

    Returns:
    --------
    binned_spectrum : ndarray
        An array of length `n_bins` containing the average values of the spectrum within each bin.

    Notes:
    ------
    The function assumes that the `spectrum` array is indexed by `ell` values and that the total length of `spectrum` is at least `ell_min + n_bins * delta_ell`.
    Each bin is computed by summing the values of `spectrum` within the bin range and then dividing by the number of elements (`delta_ell`) in that bin.
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
    
    Notes:
    ------
    The function raises an AttributeError if an unsupported `spectrum_type` is provided.
    
    The specific data being extracted from the FITS file corresponds to:
        - low l (unbinned) data for l < ~30 (based on specific spectrum type)
        - high l (binned and unbinned) data for l > ~30
    
    Spectrum types and corresponding data indices:
        - Type 0: TT spectrum
            - low l: index 1
            - high l: index 8
        - Type 1: EE spectrum
            - low l: index 3
            - high l: index 12
        - Type 3: TE spectrum
            - low l: index 2
            - high l: index 10
    
    The BB spectrum (type 2) is not available and will raise an AttributeError.
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
    Loads and sets global variables with the data to be used for fitting the power spectrum.

    Parameters:
    -----------
    spectrum_type : int
        The type of power spectrum to load:
        - 0: TT (Temperature-Temperature) spectrum
        - 1: EE (E-mode Polarization) spectrum
        - 2: BB (B-mode Polarization) spectrum
        - 3: TE (Temperature-E-mode Polarization) spectrum
    datafile : str, optional
        The path to a data file (e.g., Planck data in FITS format). If provided, the function reads the power spectrum
        from this file. If not provided, the function calculates the lensed scalar CMB power spectra using the CAMB 
        results for the specified `spectrum_type`.

    Sets:
    -----
    global_var['results'] : CAMBresults object
        The results from the CAMB cosmology model, used to generate theoretical power spectra.
    
    global_var['measured'] : ndarray
        The measured or calculated power spectrum data (Dl) for the specified `spectrum_type`.
    
    global_var['ls'] : ndarray or range
        The multipole moments (l) corresponding to the `measured` power spectrum data.
    
    global_var['errs'] : ndarray, optional
        The errors associated with the measured power spectrum data. Only set if `datafile` is provided.

    Notes:
    ------
    - If `datafile` is provided, the power spectrum is read from the file, and the corresponding `ls`, `dls`, and `errs`
      are extracted using the `read_planck` function. 
    - If `datafile` is not provided, the function calculates the lensed scalar CMB power spectra using the default 
      cosmological parameters set in the CAMB model and stores the corresponding power spectrum for the given 
      `spectrum_type`.
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

def load_dust_lensing_model(bin_start=2, bin_end=10, mapname='BK18_B95', 
                            dust_path='input_data/model_dust.npy', 
                            lensing_path='input_data/model_lens.npy', 
                            bandpowerwindowfunction_path='input_data/bpwf.npy',
                            plot=False):
    '''
    Assumes specific structure for the npy files
    '''
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

def load_bicep_sim_data(map_name, bin_start=2, bin_end=10, EB_index=5, data_path='input_data/dust_simulations.npy'):
    dust_sims = np.load(data_path, allow_pickle=True)
    dataset_index = bicep_data_consts.SPECTRA_DATASETS.index(map_name)
    eb_dust_spectra = dust_sims[dataset_index][bin_start:bin_end,EB_index,:]
    spectrum_dict = {}
    spectrum_dict['EB_sims'] = eb_dust_spectra
    
    return spectrum_dict



def load_bicep_data(plot=False, mapname=None, output_plots='output_plots', zero_ede=False):
    #data_path= 'input_data/real_spectra_bicep.npy'
    #cov_file = 'input_data/bicep_cov.npy'
    offdiag = 2
    data_path= 'input_data/bicep_norot_realspectra.npy'
    cov_file = 'input_data/bicep_norot_covar.npy'
    bin_start = 1
    bin_end=10
    scale = 100
    raw_cl = False

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

    vars = np.diag(cov)[bin_start:bin_end]
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
        eb_ede_binned = np.zeros(bin_end-bin_start)
    else:
        eb_ede_theory_provided = read_ede_data()[bpwf_ls]
        eb_ede_binned = np.matmul(eb_ede_theory_provided,ee_bpwf_cls)
    spectrum_dict['EB_EDE'] = eb_ede_binned[bin_start:bin_end]

    if(plot):
        plt.figure()
        #plt.plot(GLOBAL_VAR['EE'], label='CAMB theory')
        plt.plot(l_bins, spectrum_dict['EE_binned'], label='Binned EE camb')
        plt.plot(l_bins, spectrum_dict['BB_binned'], label='Binned BB camb')
        plt.plot(l_bins, spectrum_dict['EB_EDE']*scale, label='Binned EB EDE scaled by ' + str(scale))
        plt.errorbar(l_bins[:], spectrum_dict['EB_observed']*scale, yerr=np.sqrt(vars)*scale,
                label='C_EB bicep data scaled by ' + str(scale))
        #plt.ylim([-0.00001, 0.00002])
        plt.ylabel(r'$C_{\ell}^{EB}\cdot\ell(\ell+1)/(2\pi)$  [$\mu K^2$]')
        plt.xlabel(r'$\ell$')
        plt.legend()
        plt.title('Map: ' + str(mapname))
        plt.savefig(output_plots + '/' + mapname + '_spectra.png') 
        plt.close()
    return l_bins, spectrum_dict

def read_ede_data(data_path='input_data/fEDE0.07_cl.dat'):
    k_to_uk = 1e6
    data = pd.read_csv(data_path, delim_whitespace=True, comment='#', header=None)
    data.columns = ['l', 'TT', 'EE', 'TE', 'BB', 'EB', 'TB', 'phiphi', 'TPhi', 'Ephi']
    # Extract 'l' and 'EB' columns
    EB_values = data['EB']
    return -EB_values * np.square(k_to_uk) * 2 * np.pi

def load_eskilt_data(data_path = 'input_data/HFI_f_sky_092_EB_o.npy'):
    raw_cl = True
    load_data('all', raw_cls=raw_cl)
    c_l_EB_o_mean_std = np.load(data_path)
    spectrum_dict = {}
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



