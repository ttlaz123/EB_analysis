from cobaya.likelihood import Likelihood


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

    def initialize():
        self.map_reference_header = self.sim_common_data['map_reference_header']
        self.used_maps = self.filter_used_maps(self.used_maps)
        self.bandpasses = self.sim_common_data['bandpasses']
        self.dl_theory = self.sim_common_data['theory_spectra']
        self.cov_inv = self.sim_common_data['inv_covmat']
        self.binned_dl_observed_dict, self.map_reference_header = ld.load_observed_spectra(
                                                            observe_filepath,
                                                            self.used_maps,
                                                            self.map_reference_header,
                                                            num_bins = self.bin_num)
    def logp(self, **params_values):
        """
        Calculate the log-likelihood based on the current parameter values.
        """

        # Get the theoretical predictions based on the parameter values
        #print(params_values)
        theory_prediction = self.theory(params_values,
                                        self.dl_theory, self.used_maps)

        # Calculate the residuals
        residuals = self.binned_dl_observed_vec - theory_prediction
        # Calculate the Mahalanobis distance using the inverse covariance matrix
        chi_squared = residuals.T @ self.cov_inv @ residuals
        # Calculate the log-likelihood
        log_likelihood = -0.5 * chi_squared
        #print(log_likelihood)
        return log_likelihood


    def theory():
        # define relevant dictionaries
        if(self.theory_comps in ['all', 'fixed_dust']):
            # do ede shift

                    
        if(self.theory_comps in ['all', 'no_ede']):
            # do dust
            # do cmb rotation

        if(self.theory_comps in ['all', 'det_polrot', 'fixed_dust', 'no_ede'):
            # do detector rotation


        # apply bpwf
        self.tot_dict = ec.apply_bpwf(self.map_reference_header,
                                      self.theory_dict,
                                      self.bpwf,
                                      self.used_maps,
                                      do_cross=True)
        theory_vec = self.dict_to_vec(self.tot_dict, self.used_maps)
        return theory_vec
