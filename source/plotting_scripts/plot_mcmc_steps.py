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


def plot_theory_diff_steps_ebonly(dl_theory, initial_eb_map, bpwf, header, used_eb_map, param_combos, outpath):
    """
    Plot EB spectrum evolution under split detector rotation for different (base_angle, angle_diff, l_break).

    Three subplots (stacked vertically):
      - Top: stepwise rotation angle vs. ℓ
      - Middle: EB spectrum before bpwf
      - Bottom: EB spectrum after bpwf (binned)

    Parameters:
        dl_theory: dict of theory spectra (e.g. from CAMB+EDE)
        initial_eb_map: dict with only the EB spectrum, usually 0
        bpwf: bandpower window function
        header: FITS header used in bpwf
        used_eb_map: string key like 'BK18_220_ExBK18_220_B'
        param_combos: list of (base_angle, angle_diff, l_break)
        outpath: where to save the figure
    """
    ell = np.arange(len(dl_theory['EE']))
    L_BIN_CENTERS = np.array([37.5, 72.5, 107.5, 142.5, 177.5, 
                               212.5, 247.5, 282.5, 317.5, 352.5,
                               387.5, 422.5, 457.5, 492.5, 527.5000, 562.5000])

    fig, axs = plt.subplots(3, 1, figsize=(10, 12), sharex=True)

    for base_angle, angle_diff, l_break in param_combos:
        base_params = {'alpha_CMB': base_angle}
        new_params = {'alpha_CMB': base_angle + angle_diff}

        # Angle step function
        angle_arr = np.full_like(ell, base_angle, dtype=float)
        angle_arr[l_break+1:] = base_angle + angle_diff
        axs[0].plot(ell, angle_arr,  linewidth=2)

        # Apply rotation
        rotated_first = ec.apply_cmb_rotation(initial_eb_map, base_params, dl_theory, [used_eb_map])
        rotated_first[used_eb_map][l_break+1:] = 0

        rotated_second = ec.apply_cmb_rotation(initial_eb_map, new_params, dl_theory, [used_eb_map])
        rotated_second[used_eb_map][:l_break+1] = 0

        # Combine
        combined = {
            used_eb_map: rotated_first[used_eb_map] + rotated_second[used_eb_map]
        }

        # Plot before bpwf
        label = fr"$\alpha_\nu={base_angle}$, $\Delta\beta={angle_diff}$, $\ell_b={l_break}$"
        axs[1].plot(ell, combined[used_eb_map], label=label, linewidth=2)
        
        # Apply bpwf
        final = ec.apply_bpwf(header, combined, bpwf, [used_eb_map], do_cross=True)
        axs[2].plot(L_BIN_CENTERS, final[used_eb_map], linewidth=2, marker='o', linestyle='-')

    # Formatting
    axs[-1].set_xlim([0, 600])
    for ax in axs[:-1]:
        ax.label_outer()
    axs[0].set_ylabel(r'$\beta(\ell)$ [deg]', fontsize=14)
    axs[0].set_title('Multipole Dependent Rotation Angle', fontsize=15)
    axs[0].grid(True, linestyle='--', alpha=0.6)
    

    axs[1].set_ylabel(r'$D_\ell^{EB}$ [$\mu$K$^2$]', fontsize=14)
    axs[1].set_title('Rotated EB Spectrum', fontsize=15)
    axs[1].grid(True, linestyle='--', alpha=0.6)
    axs[1].legend(fontsize=10)
    axs[2].set_ylabel(r'$D_b^{EB}$ [$\mu$K$^2$]', fontsize=14)
    axs[2].set_xlabel(r'Multipole $\ell$', fontsize=14)
    axs[2].set_title('EB Spectrum After Bandpower Window Function', fontsize=15)
    axs[2].grid(True, linestyle='dashdot', alpha=0.6)
  
    xticks = np.arange(20, 601, 35)
    for ax in axs:
        ax.set_xticks(xticks)
    bold_ticks = {265, 300, 335, 370, 405}
    for ax in axs:
        for label in ax.get_xticklabels():
            try:
                tick_val = int(label.get_text())
                if tick_val in bold_ticks:
                    label.set_fontweight('bold')
            except ValueError:
                pass
        for line in ax.get_xgridlines():
            if line.get_xdata()[0] in bold_ticks:  # choose tick locations
                line.set_linewidth(1)
                line.set_color('black')
            else:
                line.set_linewidth(0.5)
                line.set_color('gray')
    plt.tight_layout()
    print("Saving:", outpath)
    plt.savefig(outpath)
    plt.close()


def plot_eb_spectra_with_bpwf_comparison(dl_theory, eb_maps, params_values, bpwf, header, used_map, outpath):
    """
    Plot EB spectra before and after BPWF for:
        - pure rotation (using param_values)
        - EDE-injected EB (from theory)
    """
    ell = np.arange(len(dl_theory['EE']))
    L_BIN_CENTERS = np.array([37.5, 72.5, 107.5, 142.5, 177.5, 
                               212.5, 247.5, 282.5, 317.5, 352.5,
                               387.5, 422.5, 457.5, 492.5, 527.5, 562.5])
    
    fig, axs = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
    # Make sure ell and EB maps are truncated to the same length
    maxlen = 700
    ell = ell[:maxlen]
    # --- Plot 1: EB spectra before BPWF ---
    for param_values in params_values:
        rotated = ec.apply_cmb_rotation(eb_maps[0], param_values, dl_theory, [used_map])
        spectrum = rotated[used_map][:maxlen]  # ensure same length
        label = f"Rotation: α={param_values['alpha_CMB']:.2f}"
        axs[0].plot(ell, spectrum, label=label)

    # Add EB from EDE
    ede_spectrum = eb_maps[1][used_map][:maxlen]
    axs[0].plot(ell, ede_spectrum, label="EDE EB", linestyle='--', color='black')

    axs[0].set_title("EB Spectrum Before BPWF", fontsize=14)
    axs[0].set_ylabel(r"$D_\ell^{EB}$ [$\mu$K$^2$]", fontsize=13)
    axs[0].grid(True, linestyle='--', alpha=0.6)
    axs[0].legend(fontsize=10)

    # --- Plot 2: EB spectra after BPWF ---
    for param_values in params_values:
        rotated = ec.apply_cmb_rotation(eb_maps[0], param_values, dl_theory, [used_map])
        binned = ec.apply_bpwf(header, rotated, bpwf, [used_map], do_cross=True)
        axs[1].plot(L_BIN_CENTERS, binned[used_map], marker='o', label=f"Rotation: α={param_values['alpha_CMB']:.2f}")

    ede_binned = ec.apply_bpwf(header, eb_maps[1], bpwf, [used_map], do_cross=True)
    axs[1].plot(L_BIN_CENTERS, ede_binned[used_map], label="fEDE=0.07, g=0.3", linestyle='--', color='black', marker='o')

    axs[1].set_title("EB Spectrum After BPWF", fontsize=14)
    axs[1].set_ylabel(r"$D_b^{EB}$ [$\mu$K$^2$]", fontsize=13)
    axs[1].set_xlabel(r"Multipole $\ell$", fontsize=13)
    axs[1].grid(True, linestyle='-.', alpha=0.6)
    axs[1].legend(fontsize=10)

    for ax in axs:
        ax.set_xlim(0, 600)

    plt.tight_layout()
    print("Saving:", outpath)
    plt.savefig(outpath)
    plt.close()
def plot_dust_eb_spectra_with_bpwf(dl_theory, eb_maps, params_values, bpwf, header, bandpasses, used_maps, outpath):
    """
    Plot EB dust-only spectra before and after BPWF, for each param_values set.
    - Top panel: raw EB from dust (after all processing steps, but before BPWF)
    - Bottom panel: same but after applying BPWF
    """
    ell = np.arange(len(dl_theory['EE']))
    L_BIN_CENTERS = np.array([37.5, 72.5, 107.5, 142.5, 177.5, 
                               212.5, 247.5, 282.5, 317.5, 352.5,
                               387.5, 422.5, 457.5, 492.5, 527.5, 562.5])

    fig, axs = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
    maxlen = 700
    ell = ell[:maxlen]
    used_map = used_maps[0]
    # --- Plot 1: Dust EB before BPWF ---
    for param_values in params_values:
        rot = ec.apply_cmb_rotation(eb_maps[0], param_values, dl_theory, used_maps)
        dust = ec.apply_dust(rot, bandpasses, param_values)
        detrot = ec.apply_det_rotation(dust, param_values, dl_theory)
        spectrum = detrot[used_map][:maxlen]
        print(spectrum)
        label = f"Dust Amplitude EE={param_values['A_dust_EE']:.2f}"
        axs[0].plot(ell, spectrum, label=label)

    axs[0].set_title("Dust EB Spectrum Before BPWF", fontsize=14)
    axs[0].set_ylabel(r"$D_\ell^{EB}$ [$\mu$K$^2$]", fontsize=13)
    axs[0].grid(True, linestyle='--', alpha=0.6)
    axs[0].legend(fontsize=10)

    # --- Plot 2: Dust EB after BPWF ---
    for param_values in params_values:
        rot = ec.apply_cmb_rotation(eb_maps[0], param_values, dl_theory, used_maps)
        dust = ec.apply_dust(rot, bandpasses, param_values)
        detrot = ec.apply_det_rotation(dust, param_values, dl_theory)
        binned = ec.apply_bpwf(header, detrot, bpwf, used_maps, do_cross=True)

        axs[1].plot(L_BIN_CENTERS, binned[used_map], marker='o',
                    label=f"Dust Amplitude EE={param_values['A_dust_EE']:.2f}")

    axs[1].set_title("Dust EB Spectrum After BPWF", fontsize=14)
    axs[1].set_ylabel(r"$D_b^{EB}$ [$\mu$K$^2$]", fontsize=13)
    axs[1].set_xlabel(r"Multipole $\ell$", fontsize=13)
    axs[1].grid(True, linestyle='-.', alpha=0.6)
    axs[1].legend(fontsize=10)

    for ax in axs:
        ax.set_xlim(0, 600)

    plt.tight_layout()
    print("Saving:", outpath)
    plt.savefig(outpath)
    plt.close()

def plot_isotropic_rotations(dl_theory, eb_maps, params_values, bpwf, header, bandpasses, used_maps, outpath):
    ell = np.arange(len(dl_theory['EE']))
    plt.rcParams.update({
        "text.usetex": True,
        'font.size': 20,
        "font.family": "serif", 
        "font.serif": 'Computer Modern',
        'axes.titlesize': 22,
        'axes.labelsize': 22,
        'xtick.labelsize': 18,
        'ytick.labelsize': 18,
        'legend.fontsize': 18
    })
    print('plotting iso rotations')
    fig, axs = plt.subplots(1, 2, figsize=(16, 10), sharey=True)
    maxlen = 650
    ell = ell[:maxlen]
    used_map = used_maps[0]
    max_abs = max(abs(p['alpha_CMB']) for p in params_values)
    cmap = cm.get_cmap('coolwarm')
    norm = mcolors.TwoSlopeNorm(vmin=-max_abs, vcenter=0.0, vmax=+max_abs)
   
    
    for param_values in params_values:
        alpha_val = param_values['alpha_CMB']
        color = cmap(norm(alpha_val))

        rot = ec.apply_cmb_rotation(eb_maps[0], param_values, dl_theory, used_maps)
        label = fr"$\beta_{{\mathrm{{CMB}}}}={param_values['alpha_CMB']:.2f}$"
        spectrum = rot[used_map][:maxlen]
        axs[0].plot(ell, spectrum, label=label, color=color)

        rot = ec.apply_cmb_rotation(eb_maps[1], param_values, dl_theory, used_maps)
        label = fr"$\beta_{{\mathrm{{CMB}}}}={param_values['alpha_CMB']:.2f}$"
        spectrum = rot[used_map][:maxlen]
        axs[1].plot(ell, spectrum, label=label, color=color)
    
    axs[0].set_title(r"Input EB with $g = 0$", fontsize=24)
    axs[0].set_ylabel(r"$\tilde{D}_\ell^{EB,\mathrm{rot}}$ [$\mu$K$^2$]", fontsize=24)
    axs[0].set_xlabel(r"Multipole $\ell$", fontsize=24)
    axs[0].grid(True, linestyle='--', alpha=0.6)
    axs[0].legend(fontsize=20)
    axs[1].set_title(r"Input EB with $g = 1$", fontsize=24)
    axs[1].set_xlabel(r"Multipole $\ell$", fontsize=24)
    axs[1].grid(True, linestyle='-.', alpha=0.6)
    axs[1].legend(fontsize=20)
    plt.tight_layout()
    print("Saving:", outpath)
    plt.savefig(outpath)
    plt.close()
    return 
def plot_bpwf(bpwf, map_reference_header, outpath):
    cross_spec = 29
    plt.rcParams.update({
        "text.usetex": True,
        'font.size': 20,
        "font.family": "serif", 
        "font.serif": 'Computer Modern',
        'axes.titlesize': 22,
        'axes.labelsize': 22,
        'xtick.labelsize': 18,
        'ytick.labelsize': 18,
        'legend.fontsize': 18
    })
    print('Plotting bpwf:')
    max_ell = 650
    ell = range(max_ell)
    print(bpwf.shape)
    for bin in range(2,17):
        print(bin)
        plt.plot(ell, bpwf[bin-2,:max_ell, cross_spec], label='Bin ' + str(bin))
    
    plt.xlabel(r"Multipole $\ell$", fontsize=24)
    plt.ylabel("Suppression Factor")
    plt.title("Band Power Window Function for " + map_reference_header[cross_spec])
    plt.legend()
    print("Saving fig to " + outpath)
    plt.savefig(outpath)
    
    return 

def get_plotted_values():
    fede = 0.07
    FILE_PATHS = fp.set_file_paths('BK18lf', fede=fede)
    used_maps = ['BK18_220_BxBK18_220_E', 'BK18_220_BxBK18_220_B', 'BK18_220_ExBK18_220_E']

    

    dl_theory = ld.load_cmb_theory(FILE_PATHS['camb_lensing'])
    dl_theory = ld.load_ede_spectra(FILE_PATHS['EDE_spectrum'], dl_theory)
    eb_maps = [
        {used_maps[0]: 0},  # No EB
        {used_maps[0]: dl_theory['EB_EDE']},  # With EB from EDE
    ]
    bandpasses = ld.read_bandpasses(FILE_PATHS['bandpasses'])
    bpwf, map_reference_header = ld.load_bpwf(FILE_PATHS['bpwf'], 
                                            None, 
                                            num_bins=np.array(range(16))+2)
    
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
    return eb_maps, base_params, bandpasses, used_maps, dl_theory, bpwf, map_reference_header

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-p', '--outpath')
    parser.add_argument('-a', '--rotate_step', action='store_true')
    parser.add_argument('-b', '--dust_step', action='store_true')
    parser.add_argument('-c', '--miscal_step', action='store_true')
    parser.add_argument('-d', '--bpwf_step', action='store_true')
    parser.add_argument('-e', '--step_function', action='store_true')

    args = parser.parse_args()
    eb_maps, base_params, bandpasses, used_maps, dl_theory, bpwf, header= get_plotted_values()
    initial_eb_map = eb_maps[0]
   
    
    if(args.rotate_step):
        

    # Choose which parameter to sweep here:

        param_to_sweep = 'alpha_CMB'
        sweep_values = [-0.9,-0.6,-0.3,0,0.3,0.6,0.9]
        params_values = sweep_param(base_params, param_to_sweep, sweep_values)
        plot_isotropic_rotations(dl_theory, eb_maps, params_values, bpwf, header, bandpasses, used_maps, args.outpath)
    if(args.dust_step):
        plot_dust_eb_spectra_with_bpwf(dl_theory, eb_maps, params_values, bpwf, header, bandpasses, used_maps, args.outpath)
    if(args.bpwf_step):
        plot_bpwf(bpwf, header, args.outpath)
    if(args.step_function):
        param_combos = [
        (-0.5, 1, 405),
        (0, 0.4, 300),
        (1, 0, 265),
        (0.2, -1, 370),
        (-0.7, -0.3, 335),
        ]
    '''
    plot_eb_spectra_with_bpwf_comparison(
            dl_theory=dl_theory,
            eb_maps=eb_maps,
            params_values=params_values,
            bpwf=bpwf,
            header=header,
            used_map=used_maps[0],
            outpath=args.outpath
        )
    '''
    '''
    plot_theory_diff_steps_ebonly(dl_theory, initial_eb_map, bpwf, header, used_maps[0], 
                                  param_combos, args.outpath)
    plot_dust_values(eb_maps, params_values, bandpasses, used_maps, dl_theory, args.outpath)
    '''
if __name__ == '__main__':
    main()
