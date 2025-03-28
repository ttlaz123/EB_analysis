import numpy as np
import matplotlib.pyplot as plt
import scipy.io
import copy
import eb_calculations as ec
import eb_plot_data as ep



def calc_fisher_derivative(ee_spec, bb_spec, eb_spec, be_spec, angle1, angle2, gMpl, which_derivative=1):
    """
    Computes the partial derivative of a Fisher matrix element with respect to a specified parameter.

    Parameters
    ----------
    ee_spec : ndarray
        The EE power spectrum (polarization E-mode auto-spectrum).
    bb_spec : ndarray
        The BB power spectrum (polarization B-mode auto-spectrum).
    eb_spec : ndarray
        The EB power spectrum (E-mode and B-mode cross-spectrum).
    be_spec : ndarray
        The BE power spectrum (B-mode and E-mode cross-spectrum).
    angle1 : float
        The polarization rotation angle for the first map (in radians).
    angle2 : float
        The polarization rotation angle for the second map (in radians).
    gMpl : float
        A coupling parameter related to cosmic birefringence or early dark energy.
    which_derivative : {1, 2, 'g'}, optional
        Specifies which derivative to compute:
        - `1` for the derivative with respect to the first polarization angle (`angle1`).
        - `2` for the derivative with respect to the second polarization angle (`angle2`).
        - `'g'` for the derivative with respect to the coupling parameter (`gMpl`).
        Defaults to `1`.

    Returns
    -------
    ndarray
        The computed derivative with respect to the specified parameter.

    Raises
    ------
    ValueError
        If `which_derivative` is not one of {1, 2, 'g'}.

    Notes
    -----
    - The derivative formulas are derived based on the dependency of the power spectra 
      on the polarization rotation angles (`angle1` and `angle2`) and the coupling parameter (`gMpl`).
    - The EE, BB, EB, and BE spectra represent different combinations of power spectrum contributions
      and are input as arrays or scalar values depending on the context.

    Examples
    --------
    Compute the derivative of the Fisher matrix with respect to `angle1`:
    >>> fisher_derivative(ee_spec, bb_spec, eb_spec, be_spec, angle1=0.1, angle2=0.2, gMpl=1.0, which_derivative=1)
    array([...])

    Compute the derivative with respect to `gMpl`:
    >>> fisher_derivative(ee_spec, bb_spec, eb_spec, be_spec, angle1=0.1, angle2=0.2, gMpl=1.0, which_derivative='g')
    array([...])
    """
    min_size = min(ee_spec.size, eb_spec.size)
    angle1 = np.deg2rad(angle1)
    angle2 = np.deg2rad(angle2)
    ee_spec = ee_spec[:min_size]
    bb_spec = bb_spec[:min_size]
    eb_spec = eb_spec[:min_size]
    be_spec = be_spec[:min_size]
    if(which_derivative == 1):
        d_da1 = -2 * (ee_spec*np.sin(2*angle1)*np.sin(2*angle2) +
                    bb_spec*np.cos(2*angle2)*np.cos(2*angle1) +
                    gMpl* eb_spec*np.sin(2*angle1)*np.cos(2*angle2) + 
                    gMpl* be_spec*np.cos(2*angle1)*np.sin(2*angle2))
        return d_da1
    elif(which_derivative == 2):
        d_da2 = 2 * (ee_spec*np.cos(2*angle1)*np.cos(2*angle2) +
                    bb_spec*np.sin(2*angle2)*np.sin(2*angle1) -
                    gMpl* eb_spec*np.cos(2*angle1)*np.sin(2*angle2) -
                    gMpl* be_spec*np.sin(2*angle1)*np.cos(2*angle2))
        return d_da2
    elif(which_derivative == 'b'):
        d_da = 2* ((ee_spec-bb_spec)*np.cos(4*angle1) -
               eb_spec*np.sin(4*angle1) - 
               be_spec*np.sin(4*angle1))
        return d_da
    elif(which_derivative == 'g'):
        d_dg = eb_spec*np.cos(2*angle1)*np.cos(2*angle2) - be_spec*np.sin(2*angle1)*np.sin(2*angle2)
        return d_dg
    elif(which_derivative == 0):
        d_da0 = eb_spec*0
        return d_da0
    else:
        raise ValueError("Which derivative must be 1 or 2 or g: "+str(which_derivative)) 

def fisher_matrix(spectra_dict, params_dict, used_maps, 
                    inv_covar_mat, map_reference_header, bpwf):
    num_params = len(params_dict)
    fisher_matrix = np.zeros((num_params, num_params))
    i = 0
    j = 0
    for param1, value1 in sorted(params_dict.items()):
        
        for param2, value2 in sorted(params_dict.items()):
            dmu_dparam1 = get_fisher_derivatives(spectra_dict, param1, 
                            params_dict, used_maps, map_reference_header, bpwf)
            dmu_dparam2 = get_fisher_derivatives(spectra_dict, param2,
                            params_dict, used_maps, map_reference_header, bpwf)
            mat_term = np.matmul(dmu_dparam1, np.matmul(inv_covar_mat, dmu_dparam2))
            fisher_matrix[i,j] = mat_term
            j += 1
        i += 1 
        j = 0 
    return fisher_matrix

def get_fisher_derivatives(spectra_dict, deriv_param, params_dict, 
                            used_maps, map_reference_header, bpwf):
    deriv_dict = {}
    for cross_map in used_maps:
        maps = cross_map.split('x')
        if(maps[0].endswith('_E')):
            emap = maps[0][:-2]
            bmap = maps[1][:-2]
        elif(maps[0].endswith('_B')):
            bmap = maps[0][:-2]
            emap = maps[1][:-2]
        ee_spec_name = emap + '_Ex' + emap + '_E'
        bb_spec_name = bmap + '_Bx' + bmap + '_B'
        eb_spec_name = 'EDE_EB'
        be_spec_name = 'EDE_EB'
        ee_spec = spectra_dict[ee_spec_name]
        bb_spec = spectra_dict[bb_spec_name]
        eb_spec = spectra_dict[eb_spec_name]
        be_spec = spectra_dict[be_spec_name]

        angle1_name = 'alpha_' + emap
        angle2_name = 'alpha_' + bmap
        angle1 = params_dict[angle1_name]
        angle2 = params_dict[angle2_name]
        gMpl = params_dict['gMpl']
        if(deriv_param == 'gMpl'):
            which_derivative='g'

        elif(deriv_param.startswith('alpha_')):
            if(angle1_name == angle2_name and deriv_param == angle1_name):
                which_derivative = 'b'
            elif(deriv_param == angle1_name):
                which_derivative = 1
            elif(deriv_param == angle2_name):
                which_derivative = 2
            else:
                which_derivative = 0
                # derivative is just zero
                #raise ValueError('Derivative param not correct: ' + str(deriv_param) + "  angle1=" + str(angle1_name) + "  angle2=" + str(angle2_name))

        else:
            raise ValueError('Parameter is not part of model: ' + str(deriv_param))
        fisher_deriv=calc_fisher_derivative(ee_spec, bb_spec, eb_spec, be_spec, 
                                        angle1, angle2, gMpl, which_derivative)

        deriv_dict[cross_map] = fisher_deriv
    
    binned_deriv_dict = ec.apply_bpwf(map_reference_header,
                                deriv_dict, bpwf, used_maps, do_cross=True)
    deriv_vec = dict_to_vec(binned_deriv_dict, used_maps, map_reference_header) 
    return deriv_vec


def calc_forecasting_grid(plot_type, filtered_bpcm, scaled_freqs, test_params, 
                 map_reference_header, bpwf, dl_theory, used_maps, do_planck=False):
    sigma_gs = []
    skyfracs = np.logspace(0, np.log10(10), 10)
    if(plot_type == 't_obs'):
    
        noise_scales = np.logspace(0, 1, 10)
    elif(plot_type == 'noise'):
        noise_scales = np.logspace(-1,1,10)
    else:
        raise ValueError("plot_type not one of expected:" + str(plot_type))
    if 1 not in noise_scales:
        noise_scales = np.sort(np.append(noise_scales, 1))
    if 1 not in skyfracs:
        skyfracs = np.sort(np.append(skyfracs, 1))

    # Create a 2D meshgrid
    if(plot_type == 't_obs'):
        SkyfracGrid, NoiseScaleGrid = np.meshgrid(skyfracs, 1/noise_scales)
    elif(plot_type == 'noise'):
        SkyfracGrid, NoiseScaleGrid = np.meshgrid(skyfracs, 1/noise_scales)

    # Initialize 2D array for results
    SigmaGrid = np.zeros_like(SkyfracGrid, dtype=np.float64)
    
    # Iterate over (skyfrac, noise_scale) pairs
    for i in range(len(noise_scales)):
        for j in range(len(skyfracs)):
            skyfrac = SkyfracGrid[i, j]
            noise_scale = NoiseScaleGrid[i, j]

            print(f'Skyfrac: {skyfrac}, Noise Scale: {noise_scale}')

            # Scale covariance with both parameters
            if(not do_planck):
                scaled_cov = scale_covar_mat(filtered_bpcm, skyfrac, noise_scale, 
                                        plot_type, map_reference_header, 
                                        scaled_freqs, used_maps)
            
            else:
                scaled_cov = scale_covar_mat_planck(filtered_bpcm, skyfrac, noise_scale, 
                                        plot_type, map_reference_header, 
                                        scaled_freqs, used_maps)


            scaled_covinv = ec.calc_inverse_covmat(scaled_cov)
            fmat = fisher_matrix(dl_theory, test_params, used_maps, scaled_covinv, map_reference_header, bpwf)

            # Store sigma_g in the 2D grid
            sigma_g = np.sqrt(np.linalg.inv(fmat)[-1, -1])
            print(sigma_g)
            SigmaGrid[i, j] = sigma_g
    return SigmaGrid, skyfracs, noise_scales


def plot_forecast_grid(skyfracs, noise_scales, SigmaGrid, plot_type, mapnames, scaled_maps,outdir):
    mapnames = str(mapnames)
    scaled_maps = str(scaled_maps)
    if(plot_type == 't_obs'):
        plt.figure(figsize=(8, 6))
        sigmas = SigmaGrid[:,0]
        plt.plot(noise_scales, np.squeeze(sigmas), color='blue')
        plt.xlabel('Old Noise/New Noise Level')
        
        plt.ylabel('Sigma g')
        plt.title('Sigma g assuming Constant skyfrac')
        savefile = outdir + '/noise_' + mapnames + '_' + scaled_maps + '.png' 
    elif(plot_type == 'noise'):
        plt.figure(figsize=(8,6))

        unit_noise = np.where(noise_scales == 1)[0]
        sigmas = SigmaGrid[unit_noise,:]
        plt.plot(skyfracs, np.squeeze(sigmas), color='magenta')
        plt.xlabel('Skyfrac_new/Skyfrac_old')
        plt.title('Sigma g assuming Constant Noise Level')
        savefile = outdir + '/skyfrac_' + mapnames + '_' + scaled_maps + '.png' 

    else:
        raise ValueError("plot_type not one of expected:" + str(plot_type))
    plt.savefig(savefile)
    plt.show()

    # Transform the grid values into log space
    skyfracs_log = np.log10(skyfracs)
    noise_scales_log = np.log10(noise_scales)
    SkyfracGrid_log, NoiseScaleGrid_log = np.meshgrid(skyfracs_log, noise_scales_log)
    levels = np.linspace(0, 0.4, 11)
    # Use contourf with the log-spaced grid
    contour = plt.contourf(NoiseScaleGrid_log, SkyfracGrid_log,
            SigmaGrid, levels=levels, cmap='RdPu', vmin=0, vmax=0.4)
    cbar = plt.colorbar(contour)
    cbar.set_label('sigma_g')

    plt.ylabel('Skyfrac_new/skyfrac_old')
    plt.yticks(skyfracs_log, labels=[f'{x:.2f}' for x in skyfracs])

    if(plot_type == 'noise'):
        plt.axvline(x=0.0, linestyle='--', color='magenta',
                linewidth=3, label='Same noise level')
        plt.xlabel('Overall Noise Ratio')
        plt.title('Impact of Sky Fraction and Overall Noise on sigma_g')
        plt.xticks(noise_scales_log, labels=[f'{1/y:.2f}' for y in noise_scales])
        savefile = outdir + '/skyfrac_noise_2D_' + mapnames + '_' + scaled_maps + '.png' 

    else:
        plt.axline((0, 0), slope=1, color='magenta',
                    linestyle='--', linewidth=3,
                    label='Same T_obs per Sky Fraction')
        plt.xlabel('Total Observation Time')
        plt.title('Impact of Sky Fraction and Total Observation Time on sigma_g')
        plt.xticks(noise_scales_log, labels=[f'{y:.2f}' for y in noise_scales])
        savefile = outdir + '/skyfrac_tobs_2D_' + mapnames + '_' + scaled_maps + '.png' 

    plt.legend()
    plt.savefig(savefile)
    plt.show()


def make_forecast_scaling_plot(map_reference_header, used_maps, num_bins, dl_theory, bpwf,
    test_params=None, scaled_freqs = None, mapnames =None):
    outdir = './20250218_ede_forecast/forecast_plots/'
    #plot_type = 'noise'
    plot_type = 't_obs'
    do_planck=False
    print('Making forecast plot')
    if(test_params is None):
        test_params = {
                    'gMpl':0,
                    'alpha_BK18_150':0,
                    'alpha_BK18_220': -0,
                    'alpha_BK18_K95': 0,
                    'alpha_BK18_B95e':-0
        }
    print('Fiducial model:')
    print(test_params)
    bpcm_file = 'bpcm_data.mat'
    supfac_file = 'rwf2.mat'
    bpcm_data = scipy.io.loadmat(bpcm_file)
    supfac_data = scipy.io.loadmat(supfac_file)
    filtered_bpcm = filter_and_process_bpcm(map_reference_header, used_maps, num_bins,
                                        bpcm_data, supfac_data)
    

    print(filtered_bpcm)
    
    # Create a 2D contour plot
    SigmaGrid, skyfracs, noise_scales = calc_forecasting_grid(plot_type, 
                    filtered_bpcm, scaled_freqs, test_params,
                    map_reference_header, bpwf, dl_theory, 
                    used_maps, do_planck=do_planck)
    plot_forecast_grid(skyfracs, noise_scales, SigmaGrid, plot_type, mapnames, scaled_freqs, outdir)
    input("paused")

def filter_and_process_bpcm(map_reference_header, used_maps, num_bins, bpcm_file, supfac_file):
    """Pre-filter and process the BPCM matrices outside the loop"""

    supfac = supfac_file['rwf2']

    bpcm_dict = {
        'sig': bpcm_file['bpcm_sig'],
        'sn1': bpcm_file['bpcm_sn1'],
        'sn2': bpcm_file['bpcm_sn2'],
        'sn3': bpcm_file['bpcm_sn3'],
        'sn4': bpcm_file['bpcm_sn4'],
        'noi': bpcm_file['bpcm_noi']
    }
    print(used_maps)
    # Apply filtering once
    filtered_bpcm = {
        key: ec.filter_matrix(map_reference_header, value / supfac, used_maps, num_bins)
        for key, value in bpcm_dict.items()
    }

    return filtered_bpcm

def get_mapscale_freqs(used_maps, map_reference_header, scaled_freqs):
    order_dict = {element: idx for idx, element in enumerate(map_reference_header)}

    # Sort used_maps based on the order defined in map_reference_header
    ordered_used_maps = sorted(used_maps, key=lambda x: order_dict[x])
    part_index = []
    for s in ordered_used_maps:
        part1, part2 = s.split('x')
        if(scaled_freqs is None):
            part_index.append(0)
            continue
        if(scaled_freqs in part1 and scaled_freqs in part2):
            part_index.append(0)
        elif(scaled_freqs in part2):
            part_index.append(2)
        elif(scaled_freqs in part1):
            part_index.append(1)
        else:
            part_index.append(-1)
    return np.array(part_index)

def scale_covar_mat_planck(bpcm_dict, skyfrac, noise_scale, plot_type, map_reference_header, scaled_freqs, used_maps, bin_num = 14): 
    print('Doing planck scaling')
    scaled_bpcm = copy.deepcopy(bpcm_dict)
    #for key in scaled_bpcm:
    #    ep.plot_covar_matrix(bpcm_dict[key], 
    #                            used_maps=used_maps,
    #                            title='base_' + key, show_plot=True)

    # do some processing here
    if(plot_type == 't_obs'):
        do_t = True
    elif(plot_type == 'noise'):
        do_t = False 
    else:
        ValueError('plot type not correct')

    # sig (SxS, SxS)
    # sn1 (SxN, SxN)
    # sn2 (SxN, NxS)
    # sn3 (NxS, SxN)
    # sn4 (NxS, NxS)
    
    map_indices = get_mapscale_freqs(used_maps, map_reference_header, scaled_freqs)
    if(do_t):
        noise_scale = skyfrac * noise_scale
    skyfrac = np.sqrt(np.sqrt(skyfrac))
    noise_scale = np.sqrt(noise_scale)
    for i in range(len(map_indices)):
        ind_low = i*bin_num
        ind_high = (i+1)*bin_num
        #both B3
        if(map_indices[i] == 0):
            
            scaled_bpcm['sig'][ind_low:ind_high,:] /= skyfrac **2
            scaled_bpcm['sn1'][ind_low:ind_high,:] *= noise_scale/skyfrac
            scaled_bpcm['sn2'][ind_low:ind_high,:] *= noise_scale/skyfrac

            scaled_bpcm['sn3'][ind_low:ind_high,:] *= noise_scale/skyfrac
            scaled_bpcm['sn4'][ind_low:ind_high,:] *= noise_scale/skyfrac
            scaled_bpcm['noi'][ind_low:ind_high,:] *= noise_scale **2
            
            scaled_bpcm['sig'][:,ind_low:ind_high] /= skyfrac **2
            scaled_bpcm['sn1'][:,ind_low:ind_high]*= noise_scale/skyfrac
            scaled_bpcm['sn2'][:,ind_low:ind_high]*= noise_scale/skyfrac
            scaled_bpcm['sn3'][:,ind_low:ind_high]*= noise_scale/skyfrac
            scaled_bpcm['sn4'][:,ind_low:ind_high]*= noise_scale/skyfrac
            scaled_bpcm['noi'][:,ind_low:ind_high] *= noise_scale **2
   
        #both planck, scale only skyfrac
        if(map_indices[i] == -1):
            scaled_bpcm['sig'][ind_low:ind_high,:] /= skyfrac **2
            scaled_bpcm['sn1'][ind_low:ind_high,:] *= skyfrac
            scaled_bpcm['sn2'][ind_low:ind_high,:] *= skyfrac

            scaled_bpcm['sn3'][ind_low:ind_high,:] *= skyfrac
            scaled_bpcm['sn4'][ind_low:ind_high,:] *= skyfrac
            #scaled_bpcm['noi'][ind_low:ind_high,:] *= noise_scale **2
            
            scaled_bpcm['sig'][:,ind_low:ind_high] /= skyfrac **2
            scaled_bpcm['sn1'][:,ind_low:ind_high]*= skyfrac
            scaled_bpcm['sn2'][:,ind_low:ind_high]*= skyfrac
            scaled_bpcm['sn3'][:,ind_low:ind_high]*= skyfrac
            scaled_bpcm['sn4'][:,ind_low:ind_high]*= skyfrac
            #scaled_bpcm['noi'][:,ind_low:ind_high] *= noise_scale **2

            
        #first is B3, second is planck
        if(map_indices[i] == 1):
            scaled_bpcm['sig'][ind_low:ind_high,:] /= skyfrac**2
            scaled_bpcm['sn1'][ind_low:ind_high,:] /= skyfrac
            scaled_bpcm['sn2'][ind_low:ind_high,:] /= skyfrac

            scaled_bpcm['sn3'][ind_low:ind_high,:] *= noise_scale/skyfrac
            scaled_bpcm['sn4'][ind_low:ind_high,:] *= noise_scale/skyfrac
            scaled_bpcm['noi'][ind_low:ind_high,:] *= noise_scale
            
            scaled_bpcm['sig'][:,ind_low:ind_high] /= skyfrac**2
            scaled_bpcm['sn1'][:,ind_low:ind_high] /= skyfrac
            scaled_bpcm['sn2'][:,ind_low:ind_high] *= noise_scale/skyfrac
            scaled_bpcm['sn3'][:,ind_low:ind_high] /= skyfrac
            scaled_bpcm['sn4'][:,ind_low:ind_high] *= noise_scale/skyfrac
            scaled_bpcm['noi'][:,ind_low:ind_high] *= noise_scale

        #first is planck, second is B3
        if(map_indices[i] == 2):
            scaled_bpcm['sig'][ind_low:ind_high,:] /= skyfrac**2
            scaled_bpcm['sn4'][ind_low:ind_high,:] /= skyfrac
            scaled_bpcm['sn3'][ind_low:ind_high,:] /= skyfrac

            scaled_bpcm['sn2'][ind_low:ind_high,:] *= noise_scale/skyfrac
            scaled_bpcm['sn1'][ind_low:ind_high,:] *= noise_scale/skyfrac
            scaled_bpcm['noi'][ind_low:ind_high,:] *= noise_scale
            
            scaled_bpcm['sig'][:,ind_low:ind_high] /= skyfrac**2
            scaled_bpcm['sn4'][:,ind_low:ind_high] /= skyfrac
            scaled_bpcm['sn3'][:,ind_low:ind_high] *= noise_scale/skyfrac
            scaled_bpcm['sn2'][:,ind_low:ind_high] /= skyfrac
            scaled_bpcm['sn1'][:,ind_low:ind_high] *= noise_scale/skyfrac
            scaled_bpcm['noi'][:,ind_low:ind_high] *= noise_scale







    '''
    scaled_bpcm['sig'] /= skyfrac**2
    for key in ['sn1', 'sn2', 'sn3', 'sn4']:
        if(do_t):
            scaled_bpcm[key] *= noise_scale
        else:
            scaled_bpcm[key] *= noise_scale/skyfrac
    if(do_t):
        scaled_bpcm['noi'] *= (noise_scale**2*skyfrac**2)
    else:
        scaled_bpcm['noi'] *= noise_scale**2
    '''
    '''
    for key in scaled_bpcm:
        ep.plot_covar_matrix(scaled_bpcm[key], 
                                used_maps=used_maps,
                                title=key, show_plot=True)

        ep.plot_covar_matrix(scaled_bpcm[key]/bpcm_dict[key], 
                                used_maps=used_maps,
                                title=key, show_plot=True)
    '''
    bpcm = sum(scaled_bpcm.values())
    bpcm = (bpcm+bpcm.T)/2
    #ep.plot_covar_matrix(bpcm/sum(bpcm_dict.values()), 
    #                            used_maps=used_maps,
    #                            title='overall scaled bpcm', show_plot=True)

    return bpcm
 

def scale_covar_mat(bpcm_dict, skyfrac, noise_scale, plot_type, 
                map_reference_header, scaled_freqs, used_maps, bin_num = 14): 
    print('Doing all scaling')
    scaled_bpcm = copy.deepcopy(bpcm_dict)
    #for key in scaled_bpcm:
    #    ep.plot_covar_matrix(bpcm_dict[key], 
    #                            used_maps=used_maps,
    #                            title='base_' + key, show_plot=True)

    # do some processing here
    if(plot_type == 't_obs'):
        do_t = True
    elif(plot_type == 'noise'):
        do_t = False 
    else:
        ValueError('plot type not correct')

    # sig (SxS, SxS)
    # sn1 (SxN, SxN)
    # sn2 (SxN, NxS)
    # sn3 (NxS, SxN)
    # sn4 (NxS, NxS)
    
    map_indices = get_mapscale_freqs(used_maps, map_reference_header, scaled_freqs)
    if(do_t):
        noise_scale = skyfrac * noise_scale
    skyfrac = np.sqrt(np.sqrt(skyfrac))
    noise_scale = np.sqrt(noise_scale)
    for i in range(len(map_indices)):
        ind_low = i*bin_num
        ind_high = (i+1)*bin_num
            
        if(map_indices[i] == 0 and False):
            
            scaled_bpcm['sig'][ind_low:ind_high,:] /= skyfrac **2
            scaled_bpcm['sn1'][ind_low:ind_high,:] *= noise_scale/skyfrac
            scaled_bpcm['sn2'][ind_low:ind_high,:] *= noise_scale/skyfrac

            scaled_bpcm['sn3'][ind_low:ind_high,:] *= noise_scale/skyfrac
            scaled_bpcm['sn4'][ind_low:ind_high,:] *= noise_scale/skyfrac
            scaled_bpcm['noi'][ind_low:ind_high,:] *= noise_scale **2
            
            scaled_bpcm['sig'][:,ind_low:ind_high] /= skyfrac **2
            scaled_bpcm['sn1'][:,ind_low:ind_high]*= noise_scale/skyfrac
            scaled_bpcm['sn2'][:,ind_low:ind_high]*= noise_scale/skyfrac
            scaled_bpcm['sn3'][:,ind_low:ind_high]*= noise_scale/skyfrac
            scaled_bpcm['sn4'][:,ind_low:ind_high]*= noise_scale/skyfrac
            scaled_bpcm['noi'][:,ind_low:ind_high] *= noise_scale **2

        if(map_indices[i] == 1 or map_indices[i] == 0):
            scaled_bpcm['sig'][ind_low:ind_high,:] /= skyfrac
            scaled_bpcm['sn1'][ind_low:ind_high,:] /= skyfrac
            scaled_bpcm['sn2'][ind_low:ind_high,:] /= skyfrac

            scaled_bpcm['sn3'][ind_low:ind_high,:] *= noise_scale
            scaled_bpcm['sn4'][ind_low:ind_high,:] *= noise_scale
            scaled_bpcm['noi'][ind_low:ind_high,:] *= noise_scale
            
            scaled_bpcm['sig'][:,ind_low:ind_high] /= skyfrac
            scaled_bpcm['sn1'][:,ind_low:ind_high] /=skyfrac
            scaled_bpcm['sn2'][:,ind_low:ind_high] *= noise_scale
            scaled_bpcm['sn3'][:,ind_low:ind_high] /=skyfrac
            scaled_bpcm['sn4'][:,ind_low:ind_high] *= noise_scale
            scaled_bpcm['noi'][:,ind_low:ind_high] *= noise_scale

        if(map_indices[i] == 2 or map_indices[i] == 0):
            scaled_bpcm['sig'][ind_low:ind_high,:] /= skyfrac
            scaled_bpcm['sn4'][ind_low:ind_high,:] /= skyfrac
            scaled_bpcm['sn3'][ind_low:ind_high,:] /= skyfrac

            scaled_bpcm['sn2'][ind_low:ind_high,:] *= noise_scale
            scaled_bpcm['sn1'][ind_low:ind_high,:] *= noise_scale
            scaled_bpcm['noi'][ind_low:ind_high,:] *= noise_scale
            
            scaled_bpcm['sig'][:,ind_low:ind_high] /= skyfrac
            scaled_bpcm['sn4'][:,ind_low:ind_high] /=skyfrac
            scaled_bpcm['sn3'][:,ind_low:ind_high] *= noise_scale
            scaled_bpcm['sn2'][:,ind_low:ind_high] /=skyfrac
            scaled_bpcm['sn1'][:,ind_low:ind_high] *= noise_scale
            scaled_bpcm['noi'][:,ind_low:ind_high] *= noise_scale







    '''
    scaled_bpcm['sig'] /= skyfrac**2
    for key in ['sn1', 'sn2', 'sn3', 'sn4']:
        if(do_t):
            scaled_bpcm[key] *= noise_scale
        else:
            scaled_bpcm[key] *= noise_scale/skyfrac
    if(do_t):
        scaled_bpcm['noi'] *= (noise_scale**2*skyfrac**2)
    else:
        scaled_bpcm['noi'] *= noise_scale**2
    '''
    '''
    for key in scaled_bpcm:
        ep.plot_covar_matrix(scaled_bpcm[key], 
                                used_maps=used_maps,
                                title=key, show_plot=True)

        ep.plot_covar_matrix(scaled_bpcm[key]/bpcm_dict[key], 
                                used_maps=used_maps,
                                title=key, show_plot=True)
    '''
    bpcm = sum(scaled_bpcm.values())
    bpcm = (bpcm+bpcm.T)/2
    #ep.plot_covar_matrix(bpcm/sum(bpcm_dict.values()), 
    #                            used_maps=used_maps,
    #                            title='overall scaled bpcm', show_plot=True)


    return bpcm


def dict_to_vec(spectra_dict, used_maps, map_reference_header):
    big_vector = []

    for map_name in map_reference_header:
        if map_name in used_maps:
            spec = spectra_dict[map_name].copy()
            #spec[7:9] = 0
            big_vector.append(spec)
    # Concatenate all spectra arrays into a single 1D array
    concat_vec =   np.concatenate(big_vector, axis=0)

    return concat_vec
