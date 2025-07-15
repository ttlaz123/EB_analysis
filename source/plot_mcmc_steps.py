import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt 
import matplotlib.colors as mcolors
import matplotlib.cm as cm
import argparse
import numpy as np

import eb_load_data as ld
import eb_file_paths as fp
import eb_calculations as ec


import matplotlib.pyplot as plt
import numpy as np

def plot_rotation_values(params_values, eb_maps, dl_theory, used_map, outpath):
    

    fig, ax = plt.subplots(1, len(eb_maps), figsize=(14, 6), sharey=True)
    titles = ['Isotopic Rotation with Initial EB=0', 'Initial EB: fEDE=0.07, g=1']

    # Sort parameters to align with colors
    params_values_sorted = sorted(params_values, key=lambda p: p['alpha_CMB'])

    # Custom colormaps
    neg_cmap = cm.get_cmap('Blues_r')
    pos_cmap = cm.get_cmap('Reds')
    zero_color = 'gray'

    neg_vals = [p['alpha_CMB'] for p in params_values_sorted if p['alpha_CMB'] < 0]
    pos_vals = [p['alpha_CMB'] for p in params_values_sorted if p['alpha_CMB'] > 0]

    # Normalize for colormap
    if neg_vals:
        norm_neg = mcolors.Normalize(vmin=min(neg_vals), vmax=0)
    if pos_vals:
        norm_pos = mcolors.Normalize(vmin=0, vmax=max(pos_vals))

    for i, eb_map in enumerate(eb_maps):
        for params_value in params_values_sorted:
            alpha = params_value['alpha_CMB']
            label = f"beta_CMB = {alpha}"
            post_rot_dict = ec.apply_cmb_rotation(
                eb_map,
                params_value,
                dl_theory,
                [used_map]
            )
            ell = np.arange(len(post_rot_dict[used_map]))

            # Assign color
            if alpha < 0:
                color = neg_cmap(norm_neg(alpha))
            elif alpha > 0:
                color = pos_cmap(norm_pos(alpha))
            else:
                color = zero_color

            ax[i].plot(ell, post_rot_dict[used_map], label=label, linewidth=2, color=color)

        ax[i].set_xlim(0, 700)
        ax[i].set_xlabel(r'Multipole $\ell$', fontsize=14)
        ax[i].set_title(titles[i], fontsize=15)
        ax[i].grid(True, linestyle='--', alpha=0.6)

    ax[0].set_ylabel(r'$D_\ell^{EB}$ [$\mu$K$^2$]', fontsize=14)
    ax[1].legend(fontsize=12, loc='upper left')
    plt.tight_layout()
    
    print('Saving:', outpath)
    plt.savefig(outpath)
    plt.close()

def plot_dust_values(eb_maps, params_values, bandpasses, used_map, dl_theory, outpath):
    plt.figure()
    eb_map = eb_maps[1]
    for param_values in params_values:
        post_rot_dict = ec.apply_cmb_rotation(
                eb_map,
                param_values,
                dl_theory,
                [used_map]
            )
        post_dust_dict =  ec.apply_dust(post_rot_dict, bandpasses, param_values)
        plt.plot(post_dust_dict[used_map], label=param_values)
    print('Saving:', outpath)
    plt.savefig(outpath)
    plt.close()

def get_plotted_values():
    fede=0.07
    FILE_PATHS = fp.set_file_paths('BK18lf', fede=fede)
    used_map = 'BK18_B95e_BxBK18_B95e_E'
    '''
    params_values = [
        {'alpha_CMB': -0.9},
        {'alpha_CMB': -0.6},
        {'alpha_CMB': -0.3},
        {'alpha_CMB': 0},
        {'alpha_CMB': 0.3},
        {'alpha_CMB': 0.6},
        {'alpha_CMB': 0.9},
    ]
    '''
    params_values = [
        {
        'alpha_CMB':-0.3,
        'A_dust_EE': 6, 
        'A_dust_BB': 6, 
        'A_dust_EB': 6, 
        'alpha_dust_EE': -0.5,
        'alpha_dust_BB': -0.5,
        'alpha_dust_EB': -0.5,
        'beta_dust':1.6,
        'A_sync_EE': 6, 
        'A_sync_BB': 6, 
        'A_sync_EB': 6, 
        'alpha_sync_EE': -0.5,
        'alpha_sync_BB': -0.5,
        'alpha_sync_EB': -0.5,
        'beta_sync':-3,
        },
        {
        'alpha_CMB':-0.3,
        'A_dust_EE': 6, 
        'A_dust_BB': 6, 
        'A_dust_EB': 6, 
        'alpha_dust_EE': -0.5,
        'alpha_dust_BB': -0.5,
        'alpha_dust_EB': -0.5,
        'beta_dust':1.6,
        'A_sync_EE': 6, 
        'A_sync_BB': 6, 
        'A_sync_EB': 6, 
        'alpha_sync_EE': -0.3,
        'alpha_sync_BB': -0.5,
        'alpha_sync_EB': -0.5,
        'beta_sync':-3,
        },
        {
        'alpha_CMB':-0.3,
        'A_dust_EE': 10, 
        'A_dust_BB': 6, 
        'A_dust_EB': 6, 
        'alpha_dust_EE': -0.5,
        'alpha_dust_BB': -0.5,
        'alpha_dust_EB': -0.5,
        'beta_dust':1.6,
        'A_sync_EE': 6, 
        'A_sync_BB': 6, 
        'A_sync_EB': 6, 
        'alpha_sync_EE': -0.5,
        'alpha_sync_BB': -0.5,
        'alpha_sync_EB': -0.5,
        'beta_sync':-3,
        },
        {
        'alpha_CMB':-0.3,
        'A_dust_EE': 10, 
        'A_dust_BB': 6, 
        'A_dust_EB': 6, 
        'alpha_dust_EE': -0.3,
        'alpha_dust_BB': -0.5,
        'alpha_dust_EB': -0.5,
        'beta_dust':1.6,
        'A_sync_EE': 6, 
        'A_sync_BB': 6, 
        'A_sync_EB': 6, 
        'alpha_sync_EE': -0.5,
        'alpha_sync_BB': -0.5,
        'alpha_sync_EB': -0.5,
        'beta_sync':-3,
        },
    ]

    dl_theory = ld.load_cmb_theory(FILE_PATHS['camb_lensing'])
    dl_theory = ld.load_ede_spectra(FILE_PATHS['EDE_spectrum'], dl_theory)
    eb_maps = [
        {
        used_map: 0
    },
    {
        used_map: dl_theory['EB_EDE']
    },
    
    ]
    bandpasses = ld.read_bandpasses(FILE_PATHS['bandpasses'])
    return eb_maps, params_values, bandpasses, used_map, dl_theory
    
    
def main():
    parser =argparse.ArgumentParser()
    parser.add_argument('-p', '--outpath')
    args = parser.parse_args()

    eb_maps, params_values, bandpasses, used_map, dl_theory = get_plotted_values()
    plot_dust_values(eb_maps, params_values, bandpasses, used_map, dl_theory, args.outpath)
    #plot_rotation_values(params_values, eb_maps, dl_theory, used_map, args.outpath)
if __name__=='__main__':
    main()