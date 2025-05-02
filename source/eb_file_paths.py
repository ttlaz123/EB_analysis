

# Define the base directories for the file paths
CAMB_BASE_PATH = '/n/holylfs04/LABS/kovac_lab/general/input_maps/official_cl/'
#DOMINIC_BASE_PATH = '/n/home01/dbeck/cobaya/data/bicep_keck_2018/BK18_cosmomc/data/'
DATA_BASE_PATH = '/n/home08/liuto/cosmo_package/data/bicep_keck_2018/BK18_cosmomc/data/'  



def set_file_paths(dataset, fede=0.07):
    if(dataset == 'BK18lf'):
        DATASETNAME = 'BK18lf'
        DATASET_DIRNAME = 'BK18lf'
    elif(dataset == 'BK18lf_dust'):
        DATASETNAME = 'BK18lf'
        DATASET_DIRNAME = 'BK18lf_dust'
    elif(dataset == 'BK18lf_dust_incEE'):
        DATASETNAME = 'BK18lf'
        DATASET_DIRNAME = 'BK18lf_dust_incEE'
    elif(dataset == 'BK18lf_norot_allbins'):
        DATASETNAME = 'BK18lfnorot'
        DATASET_DIRNAME = 'BK18lf_dust_incEE_norot_allbins'
    elif(dataset == 'BK18lf_norot'):
        DATASETNAME = 'BK18lfnorot'
        DATASET_DIRNAME = 'BK18lf_dust_incEE_norot'
    elif(dataset == 'BK18lf_fede01'):
        DATASETNAME = 'BK18lf_fede01'
        DATASET_DIRNAME = 'BK18lf_fede01'
    elif(dataset == 'BK18lf_fede01_sigl'):
        DATASETNAME = 'BK18lf_fede01'
        DATASET_DIRNAME = 'BK18lf_fede01_sigl'
    elif(dataset == 'BK18lf_sim'):
        DATASETNAME = 'BK18lf'
        DATASET_DIRNAME = 'BK18lf_sim'
    elif(dataset in ['BK18lf_mhd', 'BK18lf_mkd', 'BK18lf_vansyngel',
                     'BK18lf_gampmod', 'BK18lf_gaussdust','BK18lf_gdecorr',
                     'BK18lf_pysm1', 'BK18lf_pysm2','BK18lf_pysm3',
        ]):
        DATASETNAME = 'BK18lf'
        DATASET_DIRNAME = dataset

    else:
        raise ValueError(dataset + ' not one of the dataset options')
    
    BK18_BASE_PATH = DATA_BASE_PATH + DATASET_DIRNAME + '/'
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
        "EDE_spectrum": '/n/home08/liuto/GitHub/EB_analysis/input_data/fEDE' + str(fede) + '_cl.dat',
        "sim_path": BK18_BASE_PATH + DATASETNAME + '_cl_hat_simXXX.dat'
    }
    return FILE_PATHS
