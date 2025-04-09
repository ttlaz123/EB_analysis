from cobaya.likelihood import Likelihood
import numpy as np
import eb_load_data as ld
import eb_calculations as ec

class BK18_full_multicomp(Likelihood):
    # define variables
    map_set=None
    dataset= None
    forecast=False 
    bin_num=14 
    theory_comps='all' 
    spectra_type='all'
    sim_common_data = {}
    observe_filepath = None
    def __init__(self, *args, **kwargs):
    
        if('used_maps' in kwargs):
            self.used_maps = kwargs['used_maps']
            if('zero_offdiag' in kwargs):
                self.zero_offdiag = kwargs['zero_offdiag']
            if('map_set' in kwargs):
                self.map_set = kwargs['map_set']
            if('bin_num' in kwargs):
                self.bin_num = kwargs['bin_num']
            if('theory_comps' in kwargs):
                self.theory_comps = kwargs['theory_comps']
            if('spectra_type' in kwargs):
                self.spectra_type = kwargs['spectra_type']
            if('sim_common_data' in kwargs):
                self.sim_common_data = kwargs['sim_common_data']
            if('single_observe_data' in kwargs):
                self.observe_filepath = kwargs['single_observe_data']
            self.initialize()
        else:
            super().__init__(*args, **kwargs)

    def initialize(self):
        self.map_reference_header = self.sim_common_data['map_reference_header']
        self.used_maps = self.sim_common_data['used_maps']
        self.bandpasses = self.sim_common_data['bandpasses']
        self.bpwf = self.sim_common_data['bpwf']
        self.dl_theory = self.sim_common_data['theory_spectra']
        self.cov_inv = self.sim_common_data['inv_covmat']
        self.binned_dl_observed_dict, self.map_reference_header = ld.load_observed_spectra(
                                                            self.observe_filepath,
                                                            self.used_maps,
                                                            self.map_reference_header,
                                                            num_bins = self.bin_num)
        
        self.initial_theory_dict = ec.apply_initial_conditions(self.dl_theory, self.used_maps, self.spectra_type)
        self.binned_dl_observed_vec = self.dict_to_vec(self.binned_dl_observed_dict, 
                                                    self.used_maps)
         
    
    
    def logp(self, **params_values):
        """
        Calculate the log-likelihood based on the current parameter values.
        """

        # Get the theoretical predictions based on the parameter values
        #print(params_values)
        theory_prediction = self.theory(params_values)

        # Calculate the residuals
        residuals = self.binned_dl_observed_vec - theory_prediction
        # Calculate the Mahalanobis distance using the inverse covariance matrix
        chi_squared = residuals.T @ self.cov_inv @ residuals
        # Calculate the log-likelihood
        log_likelihood = -0.5 * chi_squared
        #print(log_likelihood)
        return log_likelihood

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
        concat_vec =  np.concatenate(big_vector, axis=0)

        return concat_vec   

    def theory(self, params_values):
        # define relevant dictionaries
        if(self.theory_comps in ['all', 'fixed_dust']):
            # do ede shift
            post_inflation_dict = ec.apply_EDE(self.initial_theory_dict, 
                                               params_values,
                                               self.dl_theory,
                                               self.used_maps)
        else:
            post_inflation_dict = self.initial_theory_dict
        if(self.theory_comps in ['all', 'no_ede']):
            
            # do cmb rotation
            post_travel_dict = ec.apply_cmb_rotation(post_inflation_dict,
                                                    params_values,
                                                    self.dl_theory,
                                                    self.used_maps)
            # do dust
            post_travel_dict = ec.apply_dust(post_travel_dict, params_values)
            
        else: 
            post_travel_dict = post_inflation_dict
        if(self.theory_comps in ['all', 'det_polrot', 'fixed_dust', 'no_ede']):
            # do detector rotation
            post_detection_dict = ec.apply_det_rotation(post_travel_dict, params_values)
            


        # apply bpwf
        self.final_detection_dict = ec.apply_bpwf(self.map_reference_header,
                                      post_detection_dict,
                                      self.bpwf,
                                      self.used_maps,
                                      do_cross=True)
        theory_vec = self.dict_to_vec(self.final_detection_dict, self.used_maps)
        return theory_vec
