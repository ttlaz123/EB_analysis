import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt 
import argparse
import numpy as np

import eb_load_data as ld
import eb_file_paths as fp
import eb_calculations as ec


def plot_param_values(params_values, eb_map, dl_theory, used_map, outpath):
    plt.figure(figsize=(10, 6))
    
    for params_value in params_values:
        label = f"α_CMB = {params_value['alpha_CMB']}"
        post_rot_dict = ec.apply_cmb_rotation(
            eb_map,
            params_value,
            dl_theory,
            [used_map]
        )
        ell = np.arange(len(post_rot_dict[used_map]))
        plt.plot(ell, post_rot_dict[used_map], label=label, linewidth=2)

    plt.xlim(0, 700)
    plt.xlabel(r'Multipole $\ell$', fontsize=14)
    plt.ylabel(r'$D_\ell^{EB}$ [$\mu$K$^2$]', fontsize=14)
    plt.title('EB Spectrum After CMB Rotation', fontsize=16)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend(fontsize=12)
    plt.tight_layout()
    
    print('Saving:', outpath)
    plt.savefig(outpath)
    plt.close()


def get_plotted_values(outpath):
    fede=0.07
    FILE_PATHS = fp.set_file_paths('BK18lf', fede=fede)
    used_map = 'BK18_B95e_BxBK18_B95e_E'
    params_values = [
        {'alpha_CMB': 0},
        {'alpha_CMB': 0.3},
        {'alpha_CMB': 0.6},
        {'alpha_CMB': 0.9},
    ]

    dl_theory = ld.load_cmb_theory(FILE_PATHS['camb_lensing'])
    dl_theory = ld.load_ede_spectra(FILE_PATHS['EDE_spectrum'], dl_theory)
    eb_map = {
        used_map: dl_theory['EB_EDE']
    }
    plot_param_values(params_values, eb_map, dl_theory, used_map, outpath)
    
def main():
    parser =argparse.ArgumentParser()
    parser.add_argument('-p', '--outpath')
    args = parser.parse_args()
    get_plotted_values(args.outpath)

if __name__=='__main__':
    main()