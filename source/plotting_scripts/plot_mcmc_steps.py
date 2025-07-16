import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt 
import matplotlib.colors as mcolors
import matplotlib.cm as cm
import argparse
import numpy as np
import sys 
import os 
source_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(source_dir)

import eb_load_data as ld
import eb_file_paths as fp
import eb_calculations as ec

def sweep_param(base_params, param_name, values):
    return [{**base_params, param_name: v} for v in values]

def plot_dust_values(eb_maps, params_values, bandpasses, used_maps, dl_theory, outpath):
    # Step 1: Identify varying keys
    all_keys = params_values[0].keys()
    varying_keys = [k for k in all_keys if any(p[k] != params_values[0][k] for p in params_values)]
    print("Varying parameters used in legend:", varying_keys)

    # Step 2: Run a dummy pass to get all map keys
    dummy_rot = ec.apply_cmb_rotation(eb_maps[0], params_values[0], dl_theory, used_maps)
    dummy_dust = ec.apply_dust(dummy_rot, bandpasses, params_values[0])
    dummy_detrot = ec.apply_det_rotation(dummy_dust, params_values[0], dl_theory)
    map_keys = list(dummy_detrot.keys())
    print("Map keys:", map_keys)

    # Step 3: Set up subplot grid (rows = keys, cols = eb_maps)
    n_keys, n_maps = len(map_keys), len(eb_maps)
    fig, axs = plt.subplots(n_keys, n_maps, figsize=(6 * n_maps, 3.5 * n_keys), sharex=True)

    if n_keys == 1: axs = np.expand_dims(axs, 0)
    if n_maps == 1: axs = np.expand_dims(axs, 1)

    titles = ['Initial EB = 0', 'Initial EB = EB_EDE']

    for col_idx, eb_map in enumerate(eb_maps):
        for param_values in params_values:
            # Label just the varying parts
            label = ", ".join(f"{k}={param_values[k]}" for k in varying_keys)

            post_rot = ec.apply_cmb_rotation(eb_map, param_values, dl_theory, used_maps)
            post_dust = ec.apply_dust(post_rot, bandpasses, param_values)
            post_detrot = ec.apply_det_rotation(post_dust, param_values, dl_theory)

            for row_idx, key in enumerate(map_keys):
                ell = np.arange(len(post_detrot[key]))
                axs[row_idx, col_idx].plot(ell, post_detrot[key], label=label)

    for row_idx, key in enumerate(map_keys):
        for col_idx in range(n_maps):
            ax = axs[row_idx, col_idx]
            ax.set_xlim(0, 700)
            ax.set_title(titles[col_idx], fontsize=14)
            ax.grid(True, linestyle='--', alpha=0.6)
            if col_idx == 0:
                ax.set_ylabel(f"{key.replace('_', ' ')}\n$D_\\ell$ [$\\mu$K$^2$]", fontsize=12)
            if row_idx == n_keys - 1:
                ax.set_xlabel(r'Multipole $\ell$', fontsize=12)

    axs[0, -1].legend(fontsize=9, loc='upper right')
    plt.tight_layout()
    print("Saving:", outpath)
    plt.savefig(outpath)
    plt.close()

def get_plotted_values():
    fede = 0.07
    FILE_PATHS = fp.set_file_paths('BK18lf', fede=fede)
    used_maps = ['BK18_220_BxBK18_220_E', 'BK18_220_BxBK18_220_B', 'BK18_220_ExBK18_220_E']

    base_params = {
        'alpha_BK18_220': 1,
        'alpha_CMB': -0.3,
        'A_dust_EE': 7,
        'A_dust_BB': 4,
        'A_dust_EB': 0,
        'alpha_dust_EE': -0.5,
        'alpha_dust_BB': -0.5,
        'alpha_dust_EB': -0.5,
        'beta_dust': 1.6,
        'A_sync_EE': 0,
        'A_sync_BB': 0,
        'A_sync_EB': 0,
        'alpha_sync_EE': -0.5,
        'alpha_sync_BB': -0.5,
        'alpha_sync_EB': -0.5,
        'beta_sync': -3,
    }

    # Choose which parameter to sweep here:
    param_to_sweep = 'alpha_BK18_220'
    sweep_values = [1, 0.5, 0, -0.5, -1]

    params_values = sweep_param(base_params, param_to_sweep, sweep_values)

    dl_theory = ld.load_cmb_theory(FILE_PATHS['camb_lensing'])
    dl_theory = ld.load_ede_spectra(FILE_PATHS['EDE_spectrum'], dl_theory)
    eb_maps = [
        {used_maps[0]: 0},  # No EB
        {used_maps[0]: dl_theory['EB_EDE']},  # With EB from EDE
    ]
    bandpasses = ld.read_bandpasses(FILE_PATHS['bandpasses'])
    return eb_maps, params_values, bandpasses, used_maps, dl_theory

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-p', '--outpath')
    args = parser.parse_args()
    eb_maps, params_values, bandpasses, used_maps, dl_theory = get_plotted_values()
    plot_dust_values(eb_maps, params_values, bandpasses, used_maps, dl_theory, args.outpath)

if __name__ == '__main__':
    main()
