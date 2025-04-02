from cobaya.likelihood import Likelihood


class BK18_full_multicomp(Likelihood):
    # define variables
    map_set=None
    dataset= None
    forecast=False 
    bin_num=14 
    theory_comps='all' 
    spectra_type='all'
    def __init__(self, *args, **kwargs):
    
        
        #define variables
        pass
    def initialize():
        # load in bpwf
        # load in theory
        # load in observed
        # load in covmat
        pass 
    def theory():
        # define relevant dictionaries
        # do ede shift
        # do cmb rotation
        # do dust
        # do detector rotation
        # apply bpwf
        pass

