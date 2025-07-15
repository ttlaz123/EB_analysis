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

def plot_param_values(params_values, eb_maps, dl_theory, used_map, outpath):
    

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



def get_plotted_values(outpath):
    fede=0.07
    FILE_PATHS = fp.set_file_paths('BK18lf', fede=fede)
    used_map = 'BK18_B95e_BxBK18_B95e_E'
    params_values = [
        {'alpha_CMB': -0.9},
        {'alpha_CMB': -0.6},
        {'alpha_CMB': -0.3},
        {'alpha_CMB': 0},
        {'alpha_CMB': 0.3},
        {'alpha_CMB': 0.6},
        {'alpha_CMB': 0.9},
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

    plot_param_values(params_values, eb_maps, dl_theory, used_map, outpath)
    
def main():
    parser =argparse.ArgumentParser()
    parser.add_argument('-p', '--outpath')
    args = parser.parse_args()
    get_plotted_values(args.outpath)

if __name__=='__main__':
    main()