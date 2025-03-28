# Global dictionary for file paths
#DATASETNAME = 'BK18lfnorot'
DATASETNAME = 'BK18lf_fede01'
#DATASETNAME = 'BK18lf'
#DATASET_DIRNAME = 'BK18lf_dust_incEE_norot_allbins'
DATASET_DIRNAME = 'BK18lf_fede01_sigl'
#DATASET_DIRNAME = DATASETNAME
#DATASET_DIRNAME = 'BK18lf_sim'

# Define the base directories for the file paths
CAMB_BASE_PATH = '/n/holylfs04/LABS/kovac_lab/general/input_maps/official_cl/'
#DOMINIC_BASE_PATH = '/n/home01/dbeck/cobaya/data/bicep_keck_2018/BK18_cosmomc/data/'
BK18_BASE_PATH = '/n/home08/liuto/cosmo_package/data/bicep_keck_2018/BK18_cosmomc/data/' + DATASET_DIRNAME + '/'

#DOMINIC_SIM_PATH = '/n/home01/dbeck/cobaya/data/bicep_keck_2018/BK18_cosmomc/data/' + DATASETNAME +'/'
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
    "bandpasses": {
        "BK18_B95e": BK18_BASE_PATH + 'bandpass_BK18_B95e.txt',
        "BK18_K95": BK18_BASE_PATH + 'bandpass_BK18_K95.txt',
        "BK18_150": BK18_BASE_PATH + 'bandpass_BK18_150.txt',
        "BK18_220": BK18_BASE_PATH + 'bandpass_BK18_220.txt',
        "P030e": BK18_BASE_PATH + 'bandpass_P030e.txt',
        "P044e": BK18_BASE_PATH + 'bandpass_P044e.txt',
        "P143e": BK18_BASE_PATH + 'bandpass_P143e.txt',
        "P353e": BK18_BASE_PATH + 'bandpass_P353e.txt',
        "P217e": BK18_BASE_PATH + 'bandpass_P217e.txt',
        
    },
    "bpwf": BK18_BASE_PATH + 'windows/' + DATASETNAME + '_bpwf_bin*.txt',
    "covariance_matrix": BK18_BASE_PATH + DATASETNAME + '_covmat_dust.dat',
    "observed_data": BK18_BASE_PATH + DATASETNAME + '_cl_hat.dat',
    "EDE_spectrum": '/n/home08/liuto/GitHub/EB_analysis/input_data/fEDE0.07_cl.dat',
 }