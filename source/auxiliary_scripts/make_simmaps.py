import numpy as np
import healpy as hp

def write_alms(cls_filepath, alm_outpath, lmax):
    max_alms = 1125750
    print('Reading:' + cls_filepath)
    cls = hp.read_cl(cls_filepath)
    alms = hp.synfast(cls, 512, alm=True, new=True, lmax=lmax)
    print(alms[1].shape)
    print('Writing:' + alm_outpath)
    hp.write_alm(alm_outpath, alms[1], overwrite=True)
    return 


def write_cls(default_cls_path, cls_outpath, ede_path, gMpl):
    k_to_uk = 1e6
    cmb_temp = 2.726
    data = pd.read_csv(ede_path, delim_whitespace=True, comment='#', header=None)
    data.columns = ['l', 'TT', 'EE', 'TE', 'BB', 'EB', 'TB', 'phiphi', 'TPhi', 'Ephi']
    # Extract 'l' and 'EB' columns
    EB_values = data['EB'].to_numpy()
    els = data['l'].to_numpy()

    cl_to_dl = els*(el+1)/2/np.pi
    EB_ede_dls = -EB_values *np.square(k_to_uk) * np.square(cmb_temp)
    ede_cls = EB_ede_dls/cl_to_dl*gMpl
    
    default_cls = hp.read_cl(default_cls_path)
    max_l = min(default_cls.shape[1], els[-1])
    num_spectra=6
    default_num_spectra = 4
    all_cls = np.zeros((num_spectra, max_l))
    for i in range(default_num_spectra):
        all_cls[i,:] = default_cls[i,:]
    all_cls[4,els[0]:] = ede_cls[:max_l-els[0]]
    hp.write_cl(cls_outpath, all_cls)
    return 

def gen_map():
    '''
    gen_map(0,512,1499,'input_maps/official_cl/fede_0p07_g0p1.fits',1,'B3polycorr','',1)
    m=read_fits_map('input_maps/fede_0p07_g0p1/map_lensed_n0512_r0000_sB3polycorr_dNoNoi.fits');
    m=cut2fullsky(m);
    [almT,almE,almB]=healmex.map2alm(full(m.map(:,1)),full(m.map(:,2)),full(m.map(:,3)));
    cl=healmex.alm2cl(almE,almB);
    dl = cl'.*els.*(els+1) ;
    plot(dl)

    '''

def gen_sim_alms(num_sims, alm_outpath, cls_path):
    lmax = 1500
    prefix = 'alm_unlens_l' + str(lmax) + '_r'
    for sim_num in range((num_sims)):
        alm_path = alm_outpath + '/' + prefix + f'{sim_num:04d}.fits'
        write_alms(cls_path, alm_path, lmax=lmax)
        #print(alm_path)

def main():
    input_maps_path = '/n/home08/liuto/bk_analysis/input_maps/'
    gMpl = 0.1
    fede_filename = 'fede_0p07_g0p1'
    num_sims = 500
    
    cls_outpath = input_maps_path + 'official_cl/' + fede_filename + '.fits'
    default_cls_path = input_maps_path + 'official_cl/camb_planck2013_r0.fits'
    alm_outpath = input_maps_path + 'alms/' + fede_filename 
    ede_path = '/n/home08/liuto/GitHub/EB_Analysis/input_data/fEDE0.07_cl.dat'
    #write_cls(default_cls_path, cls_outpath, ede_path, gMpl)
    gen_sim_alms(num_sims, alm_outpath, cls_outpath)
if __name__ == '__main__':
    main()
