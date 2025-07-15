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


def sweep_param(base_params, param_name, values):
    """
    Returns a list of parameter dicts where `param_name` is varied across `values`
    and all other values are kept constant from `base_params`.
    """
    return [
        {**base_params, param_name: v}
        for v in values
    ]


def plot_dust_values(eb_maps, params_values, bandpasses, used_maps, dl_theory, outpath):

    eb_map = eb_maps[1]

    # Step 1: Identify varying keys
    varying_keys = []
    all_keys = params_values[0].keys()
    for key in all_keys:
        values = [p.get(key, None) for p in params_values]
        if not all(v == values[0] for v in values):
            varying_keys.append(key)

    print("Varying parameters used in legend:", varying_keys)

    # Step 2: Prepare a dummy run to extract all map keys
    dummy_rot = ec.apply_cmb_rotation(eb_map, params_values[0], dl_theory, used_maps)
    dummy_dust = ec.apply_dust(dummy_rot, bandpasses, params_values[0])
    dummy_detrot = ec.apply_det_rotation(dummy_dust, params_values[0], dl_theory)
    map_keys = list(dummy_detrot.keys())
    print(map_keys)

    # Step 3: Set up subplots
    n_maps = len(map_keys)
    fig, axs = plt.subplots(n_maps, 1, figsize=(10, 3.5 * n_maps), sharex=True)

    if n_maps == 1:
        axs = [axs]  # Make it iterable even for 1 plot

    # Step 4: Plot each map in its own subplot
    for param_values in params_values:
        post_rot_dict = ec.apply_cmb_rotation(
            eb_map,
            param_values,
            dl_theory,
            map_keys
        )
        post_dust_dict = ec.apply_dust(post_rot_dict, bandpasses, param_values)
        post_detrot_dict = ec.apply_det_rotation(post_dust_dict,
                                                 param_values,
                                                 dl_theory)
        
        # Create concise label
        label_parts = [f"{k}={param_values[k]}" for k in varying_keys]
        label = ", ".join(label_parts)

        for i, key in enumerate(map_keys):
            ell = np.arange(len(post_detrot_dict[key]))
            axs[i].plot(ell, post_detrot_dict[key], label=label)
            axs[i].set_title(key.replace("_", " "), fontsize=14)
            axs[i].grid(True, linestyle='--', alpha=0.6)
            axs[i].set_xlim(0, 700)

    # Step 5: Final formatting
    axs[-1].set_xlabel(r'Multipole $\ell$', fontsize=14)
    for ax in axs:
        ax.set_ylabel(r'$D_\ell$ [$\mu$K$^2$]', fontsize=12)

    axs[0].legend(fontsize=10)
    plt.tight_layout()
    
    print('Saving:', outpath)
    plt.savefig(outpath)
    plt.close()


def get_plotted_values():
    fede = 0.07
    FILE_PATHS = fp.set_file_paths('BK18lf', fede=fede)
    used_maps = ['BK18_220_BxBK18_220_E',
                 'BK18_220_BxBK18_220_B',
                 'BK18_220_ExBK18_220_E',]

    # Shared baseline config
    base_params = {
        'alpha_BK18_220': 1,
        'alpha_CMB': -0.3,
        'A_dust_EE': 7,
        'A_dust_BB': 0,
        'A_dust_EB': 0,
        'alpha_dust_EE': -0.5,
        'alpha_dust_BB': -0.5,
        'alpha_dust_EB': -0.5,
        'beta_dust': 1.6,
        'A_sync_EE': 6,
        'A_sync_BB': 6,
        'A_sync_EB': 6,
        'alpha_sync_EE': -0.5,
        'alpha_sync_BB': -0.5,
        'alpha_sync_EB': -0.5,
        'beta_sync': -3,
    }

    # Choose which parameter to sweep here:
    param_to_sweep = 'A_dust_BB'
    sweep_values = [0, 5, 20, 100]

    # Generate parameter sweep
    params_values = sweep_param(base_params, param_to_sweep, sweep_values)

    # Load spectra and bandpasses
    dl_theory = ld.load_cmb_theory(FILE_PATHS['camb_lensing'])
    dl_theory = ld.load_ede_spectra(FILE_PATHS['EDE_spectrum'], dl_theory)
    eb_maps = [
        {used_maps[0]: 0},  # No EB case
        {used_maps[0]: dl_theory['EB_EDE']},  # EDE case
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
