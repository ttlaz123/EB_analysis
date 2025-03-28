import argparse




class BK18_full_multicomp(Likelihood):
    # define variables
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




# run mcmc
    # 

# full multicomp driver
    # define maps based on mapopts
    # define dust params based on dustopts
    # define angles based on mapopts
    # define ede opts
    # define other opts based on opts
    # define relevant files based on opts
    # run mcmc
    # plot mcmc results
# parallel sim running


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument()

    args = parser.parse_args()
    # manage file structure
    # parallel vs not parallel