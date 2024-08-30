print('Importing Packages')
import sys, platform, os
from matplotlib import pyplot as plt
import numpy as np
from scipy import stats 
import pandas as pd
import sympy as sp
import time
import pickle
import argparse

#Assume installed from github using "git clone --recursive https://github.com/cmbant/CAMB.git"
#This file is then in the docs folders
camb_path = os.path.realpath(os.path.join(os.getcwd(),'..'))
sys.path.insert(0,camb_path)
from astropy.io import fits
import camb
from camb import model, initialpower, correlations
print('Using CAMB %s installed at %s'%(camb.__version__,os.path.dirname(camb.__file__)))

from cobaya.run import run
from cobaya.model import get_model
from cobaya.yaml import yaml_load
from getdist.mcsamples import MCSamplesFromCobaya
from getdist.mcsamples import loadMCSamples
import getdist.plots as gdplt

import bicep_data_consts

OUTPUT_PLOTS = 'output_plots_zeroede'
GLOBAL_VAR = {}

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


def eb_log_likelihood_vector(C_eb_observed, C_eb_var, C_eb_ede, C_ee_cmb, C_bb_cmb, aplusb, gMpl):
    """
    Computes the total log-likelihood for the EB power spectrum using vectorized operations.

    Parameters:
    -----------
    C_eb_observed : ndarray
        The observed EB power spectrum (Dl) values.
    
    C_eb_var : ndarray
        The variance of the observed EB power spectrum, used as the uncertainty in the likelihood calculation.
    
    C_eb_ede : ndarray
        The predicted EB power spectrum contribution from early dark energy (EDE).
    
    C_ee_cmb : ndarray
        The predicted EE power spectrum from the CMB.
    
    C_bb_cmb : ndarray
        The predicted BB power spectrum from the CMB.
    
    aplusb : float
        The sum of angle parameters `a` and `b` used in the rotation of polarization modes.
    
    gMpl : float
        A scaling factor related to the effective gravitational constant.

    Returns:
    --------
    total_log_likelihood : float
        The total log-likelihood value for the EB power spectrum, computed as the sum of the likelihoods across all bins.
    
    Notes:
    ------
    This function is vectorized to perform all calculations in parallel, providing a more efficient computation compared to looping over each bin individually.
    """
    cos_term = np.cos(4 * np.deg2rad(aplusb)) * gMpl * C_eb_ede
    sin_term = np.sin(4 * np.deg2rad(aplusb)) / 2 * (C_ee_cmb - C_bb_cmb) 
    v = C_eb_observed - cos_term - sin_term
    if(len(C_eb_var.shape)==2 and C_eb_var.shape[0] == C_eb_var.shape[1]):
        C_eb_var_inv = np.linalg.inv(C_eb_var)
        total_log_likelihood = -v @ C_eb_var_inv @ v.T
    else:
        bin_loglike = np.square(v) / C_eb_var
        total_log_likelihood = -np.sum(bin_loglike)
    return total_log_likelihood


def plot_best_fit(sampler, bin_edges, mapname=None):
    bins = (bin_edges[:-1] + bin_edges[1:]) / 2

    gd_sample = sampler.products()["sample"]
    n = len(gd_sample['gMpl'])
    gMpl = np.round(gd_sample['gMpl'][n//2:].mean(),3)
    aplusb = np.round(gd_sample['aplusb'][n//2:].mean(),3)

    gMpl_std = np.round(gd_sample['gMpl'][n//2:].std(),3)
    aplusb_std = np.round(gd_sample['aplusb'][n//2:].std(),3)
    C_eb_ede = GLOBAL_VAR['EB_EDE']
    C_ee_cmb = GLOBAL_VAR['EE_ebinned']
    C_bb_cmb = GLOBAL_VAR['BB_ebinned']
    C_eb_observed = GLOBAL_VAR['EB_observed']
    C_eb_var = GLOBAL_VAR['EB_var']
    cos_term = np.cos(4 * np.deg2rad(aplusb)) * gMpl * C_eb_ede
    sin_term = np.sin(4 * np.deg2rad(aplusb)) / 2 * (C_ee_cmb - C_bb_cmb) 
    
    plt.figure()
    plt.plot(bins, cos_term, label='gMpl contribution')
    plt.plot(bins, sin_term, label='Rotation contribution')
    plt.plot(bins, cos_term+sin_term, label='Combined contribution')
    if(len(C_eb_var.shape)==2 and C_eb_var.shape[0] == C_eb_var.shape[1]):
        C_eb_var = np.diag(C_eb_var)
    plt.errorbar(bins, C_eb_observed, yerr=np.sqrt(C_eb_var), label='observed EB')
    plt.ylabel(r'$C_{\ell}^{EB}\cdot\ell(\ell+1)/(2\pi)$  [$\mu K^2$]')
    plt.xlabel(r'$\ell$')
    plt.legend()
    title_str = ('gMpl=' + str(gMpl) + '+-' + str(gMpl_std) + 
                 ' aplusb=' + str(aplusb) + '+-' + str(aplusb_std) + 
                 '\n mapname=' + str(mapname))
    plt.title(title_str)
    plt.savefig(OUTPUT_PLOTS + '/' + mapname + '_bestfit.png')
    return gMpl, aplusb, gMpl_std, aplusb_std

def plot_info(variables=None, info=None, sampler=None, outfile=None, file_root=None, mapname=None):
    """
    Generates and displays or saves a triangle plot of posterior distributions for specified variables.

    Parameters:
    -----------
    variables : list of str, optional
        A list of variable names to include in the plot. Each variable should correspond to a parameter in the model.
        Example: ['var1', 'var2', ...]. If `None`, all available variables will be plotted.

    info : dict, optional
        A YAML dictionary containing the configuration information from a Cobaya run. This is used in conjunction
        with the `sampler` output to generate the MCMC samples.

    sampler : cobaya.run, optional
        The sampling output result from a Cobaya run. The `sampler.products()["sample"]` will be used to create the MCMC samples.

    outfile : str, optional
        The path to save the generated plot. If `None`, the plot is displayed on the screen instead of being saved.

    file_root : str, optional
        The root filename (without extension) from which to load MCMC samples if `info` and `sampler` are not provided.

    Raises:
    -------
    ValueError
        If neither `info`/`sampler` nor `file_root` is provided, the function raises an error indicating insufficient information to generate the plot.

    Notes:
    ------
    The function creates a triangle plot using the `getdist` library, which visualizes the 1D and 2D marginalized posterior distributions of the specified variables.
    """
    
    if not (info is None or sampler is None):
        print(sampler.products())
        gdsamples = MCSamplesFromCobaya(info, sampler.products()["sample"])
    elif not file_root is None:
        gdsamples = loadMCSamples(file_root)
    else:
        raise ValueError("No specified MCMC info or file root provided.")
    



    gdplot = gdplt.get_subplot_plotter(width_inch=5)
    gdplot.triangle_plot(gdsamples, variables, filled=True)
    plt.title(' mapname=' + str(mapname))
    if outfile is None:
        print('showing plot')
        plt.show()
    else:
        plt.savefig(OUTPUT_PLOTS + '/' + outfile)

def eb_axion_mcmc_runner(aplusb, gMpl):
    # TODO complete this
    
    C_ee_cmb = GLOBAL_VAR['EE_ebinned']
    C_bb_cmb = GLOBAL_VAR['BB_ebinned']
    C_eb_observed = GLOBAL_VAR['EB_observed']
    C_eb_var = GLOBAL_VAR['EB_var']
   
    C_eb_ede = GLOBAL_VAR['EB_EDE']
    likelihood = eb_log_likelihood_vector(C_eb_observed, C_eb_var, C_eb_ede, C_ee_cmb, C_bb_cmb, aplusb, gMpl)
    return likelihood

def get_eb_axion_infodict(outpath, variables, priors):
    """
    Generates and returns a dictionary containing the configuration information for an MCMC run, tailored to an EB axion model.

    Parameters:
    -----------
    outpath : str
        The file path where the MCMC output will be saved.

    variables : list of str
        A list of variable names to be included in the MCMC sampling. Each variable corresponds to a model parameter.

    priors : list of tuple
        A list of tuples specifying the prior ranges for each variable. Each tuple contains the minimum and maximum values (min, max) for the corresponding variable.

    Returns:
    --------
    info : dict
        A dictionary structured to define the settings and parameters for running an MCMC sampling. The dictionary contains:
        - `likelihood`: Specifies the likelihood function to be used in the sampling, here defined as `eb_axion_mcmc_runner`.
        - `params`: A dictionary of parameters, where each parameter is associated with its prior range, reference value, and proposal width.
        - `sampler`: Settings for the MCMC sampler, including stopping criteria (`Rminus1_stop`) and the maximum number of attempts (`max_tries`).
        - `output`: The file path where the results will be saved, set to the provided `outpath`.

    Notes:
    ------
    - The reference value (`ref`) and proposal width (`proposal`) for each parameter are set to 0. These values can be adjusted as needed for different models or sampling strategies.
    - The sampler settings are configured for a typical MCMC run but can be fine-tuned according to specific requirements.
    """
    info = {"likelihood": 
        {
            "power": eb_axion_mcmc_runner
        }
    }

    info["params"] = {}

    for i in range(len(variables)):
        var = variables[i]
        prior = priors[i]
        info["params"][var] = {
            "prior": {"min": prior[0], "max": prior[1]},
            "ref": 0,
            "proposal": 0
        }
    info["sampler"] = {"mcmc": {"Rminus1_stop": 0.1, "max_tries": 100000}}
    info["output"] = outpath
    return info



def get_priors_and_variables(gMpl_minmax=(-5, 5), aplusb_minmax=(-5, 5)):
    """
    Defines and returns the model variables and their corresponding prior ranges.

    Parameters:
    -----------
    gMpl_minmax : tuple, optional
        The minimum and maximum values (min, max) for the 'gMpl' variable. 
        Defaults to (-10, 10).
    
    aplusb_minmax : tuple, optional
        The minimum and maximum values (min, max) for the 'aplusb' variable.
        Defaults to (-10, 10).

    Returns:
    --------
    variables : list of str
        A list of variable names used in the model:
        - 'gMpl': A scaling factor related to the effective gravitational constant.
        - 'aplusb': The sum of angle parameters `a` and `b` used in the rotation of polarization modes.
    
    priors : list of tuple
        A list of tuples specifying the prior ranges for each corresponding variable:
        - gMpl_minmax : tuple
            Prior range for 'gMpl'.
        - aplusb_minmax : tuple
            Prior range for 'aplusb'.
    """
    variables = ['gMpl', 'aplusb']
    priors = [gMpl_minmax, aplusb_minmax]
    return variables, priors

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

def bin_spectrum_given_centers(bin_centers, spectrum, ell_min=0):
    '''
    Bins a spectrum into specified bin centers.

    Assumes the spectrum starts at l=ell_min and calculates the bin edges based on the given bin centers.

    Parameters
    ----------
    bin_centers : array_like
        An array of bin center values. The function calculates bin edges based on these centers.
        
    spectrum : array_like
        An array representing the spectrum values. The length of this array should be at least as long as the maximum bin edge.
        
    ell_min : int, optional
        The minimum ell value (default is 0). The spectrum is assumed to start at this value. This parameter is used to adjust the starting point of the bin edges.

    Returns
    -------
    binned_spectrum : ndarray
        An array of the binned spectrum values, where each value represents the average spectrum value within the corresponding bin.

    Raises
    ------
    ValueError
        If the length of the spectrum is insufficient to cover the maximum bin edge.

    Notes
    -----
    The bin edges are computed such that the center of each bin is the midpoint between the start and end of the bin.
    The function assumes that the spectrum starts at `ell_min` and is defined for consecutive ell values.
    '''
    # Calculate bin edges
    binned_spectrum = np.zeros(len(bin_centers))
    bin_starts = np.zeros(len(bin_centers) + 1,dtype=int)
    bin_starts[0] = ell_min

    for i in range(1, len(bin_starts)):
        bin_starts[i] = (2 * bin_centers[i - 1] - bin_starts[i - 1] + 1)

    # Check if the spectrum is long enough
    max_bin_edge = bin_starts[-1]
    if len(spectrum) < max_bin_edge:
        raise ValueError(f"Spectrum length is insufficient. Required: {max_bin_edge}, Provided: {len(spectrum)}")

    # Bin the spectrum
    for i in range(len(bin_centers)):
        bin_cur = 0
        for j in range(bin_starts[i], bin_starts[i + 1]):
            bin_cur += spectrum[j]
        binned_spectrum[i] = bin_cur / (bin_starts[i + 1] - bin_starts[i])

    return binned_spectrum, bin_starts

def rebin(cur_bins, cur_data, new_bin_starts, raw_cl=False, plot=False):
    x_binned = []
    y_binned = []

    # scale from D to C if we're using raw Cls
    if(raw_cl):
        cur_data *= 2*np.pi/cur_bins/(cur_bins+1)
    # Rebin the data
    for i in range(len(new_bin_starts) - 1):
        # Find the indices of x that fall within the current bin
        indices = (cur_bins >= new_bin_starts[i]) & (cur_bins < new_bin_starts[i+1])
        
        if np.any(indices):
            x_bin = np.mean(cur_bins[indices])
            y_bin = np.mean(cur_data[indices])
                
        else:
            x_bin = (new_bin_starts[i] + new_bin_starts[i+1]) / 2
            y_bin = np.interp(x_bin, cur_bins, cur_data)

        x_binned.append(x_bin)
        y_binned.append(y_bin)

    # Convert lists to arrays
    x_binned = np.array(x_binned)
    y_binned = np.array(y_binned)
    
    # Optionally plot the rebinned data
    if(plot):
        plt.figure(figsize=(10, 6))
        plt.plot(cur_bins, cur_data, 'o-', label='Original Data')
        plt.plot(x_binned, y_binned, 's-', label='Rebinned Data', color='red')
        plt.ylabel(r'Arbitrary Scaled $C_{\ell}^{EB}\cdot\ell(\ell+1)/(2\pi)$  [$\mu K^2$]')
        plt.xlabel(r'$\ell$')
        plt.title('Rebinned Data (Averaged y, Ignoring Points Outside Bins)')
        plt.legend()
        plt.grid(True)
        plt.show()
    
    return y_binned




def load_bicep_data(plot=False, mapname=None):
    #data_path= 'input_data/real_spectra_bicep.npy'
    #cov_file = 'input_data/bicep_cov.npy'

    data_path= 'input_data/bicep_norot_realspectra.npy'
    cov_file = 'input_data/bicep_norot_covar.npy'
    bin_start = 2
    bin_end=10
    scale = 100
    raw_cl = False
    load_data('all', raw_cls=raw_cl)
    if(mapname is None):
        mapname = 'BK18_B95'
    dataset_index = bicep_data_consts.SPECTRA_DATASETS.index(mapname)
    EB_index = 5
    EE_index = 2
    BB_index = 3
    l_bins = bicep_data_consts.L_BIN_CENTERS

    cov_data = np.load(cov_file, allow_pickle=True, encoding='latin1')
    cov_mat = cov_data[dataset_index][0][0]
    cov = cov_mat[EB_index]
    vars = np.diag(cov)[bin_start:bin_end]
    np_mat = np.load(data_path, allow_pickle=True, encoding='latin1')
    spectra = np_mat[dataset_index][0] 
    
    ee_binned, bin_starts = bin_spectrum_given_centers(l_bins, GLOBAL_VAR['EE'])
    bb_binned, bin_starts = bin_spectrum_given_centers(l_bins, GLOBAL_VAR['BB'])
    
    
    
    GLOBAL_VAR['EE_ebinned'] = spectra[bin_start:bin_end,EE_index]
    GLOBAL_VAR['BB_ebinned'] = spectra[bin_start:bin_end,BB_index]
    
    l_bins = l_bins[bin_start:bin_end]
    GLOBAL_VAR['EB_observed'] = spectra[bin_start:bin_end,EB_index]
    GLOBAL_VAR['EB_var'] = cov[bin_start:bin_end, bin_start: bin_end]
    
    if(plot):
        plt.figure()
        #plt.plot(GLOBAL_VAR['EE'], label='CAMB theory')
        plt.plot(l_bins, GLOBAL_VAR['EE_ebinned'], label='Binned EE camb')
        plt.plot(l_bins, GLOBAL_VAR['BB_ebinned'], label='Binned BB camb')
        plt.errorbar(l_bins[:], GLOBAL_VAR['EB_observed']*scale, yerr=np.sqrt(vars)*scale,
                label='C_EB bicep data scaled by ' + str(scale))
        #plt.ylim([-0.00001, 0.00002])
        plt.ylabel(r'$C_{\ell}^{EB}\cdot\ell(\ell+1)/(2\pi)$  [$\mu K^2$]')
        plt.xlabel(r'$\ell$')
        plt.legend()
        plt.title('Map: ' + str(mapname))
        plt.savefig(OUTPUT_PLOTS + '/' + mapname + '_spectra.png') 
    return bin_starts[bin_start:bin_end+1], raw_cl   

def load_ede_data(bin_starts, data_path = 'input_data/f_07.csv', raw_cl=False, plot=False):
    # Read the CSV file
    df = pd.read_csv(data_path, header=None, names=['ell_bin', 'D_eb'])

    # Extract the data
    bin_ell = df['ell_bin'].values
    eb_ede = df['D_eb'].values
    eb_ede = np.zeros(len(bin_ell))
    GLOBAL_VAR['EB_EDE'] = rebin(bin_ell, eb_ede, bin_starts, raw_cl, plot=plot)

def load_eskilt_data(data_path = 'input_data/HFI_f_sky_092_EB_o.npy'):
    raw_cl = True
    load_data('all', raw_cls=raw_cl)
    c_l_EB_o_mean_std = np.load(data_path)
    GLOBAL_VAR['EB_observed'] = c_l_EB_o_mean_std[:, 0]
    GLOBAL_VAR['EB_var'] = np.square(c_l_EB_o_mean_std[:, 1])
    ell_min = 51
    ell_max = 1490
    delta_ell = 20
    ell = np.arange(ell_min, ell_max+1, delta_ell)
    n_bins = len(ell)
    ee_binned, bin_starts = bin_spectrum(n_bins, delta_ell, ell_min, GLOBAL_VAR['EE'])
    bb_binned, bin_starts = bin_spectrum(n_bins, delta_ell, ell_min, GLOBAL_VAR['BB'])
    GLOBAL_VAR['EE_ebinned'] = ee_binned
    GLOBAL_VAR['BB_ebinned'] = bb_binned
    return bin_starts, raw_cl   

def eb_axion_driver(outpath, variables, priors, datafile=None, mapname=None):
    info_dict = get_eb_axion_infodict(outpath, variables, priors)
    init_params = info_dict['params']

    
    bin_starts, raw_cl = load_bicep_data(plot=True, mapname=mapname)
    
    #bin_starts, raw_cl = load_eskilt_data()
    load_ede_data(bin_starts, raw_cl=raw_cl, plot=False)
    # test to make sure the likelihood function works 
    log_test = eb_axion_mcmc_runner(init_params['aplusb']['ref'], 
                                    init_params['gMpl']['ref'])
    print("test value: " + str(log_test))
    updated_info, sampler = run(info_dict, resume=True)
    
    
    return updated_info, sampler, bin_starts


def main():
    print('testin123')
    variables, priors = get_priors_and_variables(aplusb_minmax=(-5,5))
    mapnames = bicep_data_consts.SPECTRA_DATASETS
    for i in range(17):
        map = mapnames[i]
        outpath = 'mcmc_chains/' + map + '_zeroede/'
        updated_info, sampler, bin_starts = eb_axion_driver(outpath, variables, priors, mapname=map)
        gMpl, aplusb, gMpl_std, aplusb_std = plot_best_fit(sampler, bin_edges=bin_starts, mapname=map)
        print(gMpl, aplusb, gMpl_std, aplusb_std)
        plot_info(variables, updated_info, sampler, mapname=map, outfile=map+'_triagplot.png')
    '''
    outpath = 'axion_test/'
    
    
    updated_info, sampler, bin_starts = eb_axion_driver(outpath, variables, priors)
    plot_best_fit(sampler, bin_edges=bin_starts)
    plot_info(variables, updated_info, sampler)
    
    print(variables)
    print(priors)
    '''
if __name__ == '__main__':
    main()