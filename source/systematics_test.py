import argparse
import glob
import matplotlib
from matplotlib import cm
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from astropy.io import fits
import numpy as np
import os
import sys
import bicep_data_consts as bdc
BK18_FILENAMES = {
    'BK18_150': 'beamfile_20130222_sum.fits',
    'BK18_K95': 'beamfile_20150101_sum_220.fits',
    'BK18_220': 'beamfile_20150321_sum_100.fits',
    'BK18_95e': 'beamfile_20180206_polycorr_sum_100.fits',
}

def scale_dl_beams(used_maps, binned_dl_dict, injected_signal_dict, bin_nums, output_dir='.',
                   eps_range=0.05, num_eps=11):
    """
    Scales the binned_dl_dict values using the ratio of scaled beams
    interpolated at the binned ell centers, restricted to specified bins.
    
    Args:
        used_maps (list): e.g., ["BK18_150_ExBK18_K95_B"]
        binned_dl_dict (dict): {used_map: array of dl values}
        injected_signal_dict (dict): {'eps': float}
        bin_nums (list or array): indices of bins to use (0-based)
        output_dir (str): directory to save plots
    
    Returns:
        Updated binned_dl_dict with scaled values.
    """
    scaled_beams_dict = load_scaled_beams()  # make sure this function exists
    eps_val = injected_signal_dict.get('eps', 0.0)
    ell_bins_full = bdc.L_BIN_CENTERS
    bin_nums = [b-1 for b in bin_nums]
    ell_bins = ell_bins_full[bin_nums]  # select only requested bins
    epsilons = np.linspace(-eps_range, eps_range, num_eps)
    scaled_dl_all = {}
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)

    for used_map in used_maps:
        map1_full, map2_full = used_map.split('x')
        map1 = map1_full[:-2] if map1_full.endswith(('_E','_B')) else map1_full
        map2 = map2_full[:-2] if map2_full.endswith(('_E','_B')) else map2_full

        if map1 not in scaled_beams_dict or map2 not in scaled_beams_dict:
            print(f"Warning: One of the maps in {used_map} not found in scaled beams.")
            continue

        dl_orig = binned_dl_dict[used_map][bin_nums].copy()
        scaled_dl_all[used_map] = {}

        # Set up colormap
        cmap = cm.get_cmap('coolwarm', num_eps-1)

        plt.figure(figsize=(10,6))

        for idx, eps_val in enumerate(epsilons):
            B1_0 = scaled_beams_dict[map1][0.0]
            B2_0 = scaled_beams_dict[map2][0.0]
            B1_eps = scaled_beams_dict[map1].get(eps_val)
            B2_eps = scaled_beams_dict[map2].get(eps_val)

            if B1_eps is None or B2_eps is None:
                print(f"Warning: epsilon {eps_val} not found for {used_map}. Skipping.")
                continue

            B1_0_interp = np.interp(ell_bins_full, np.arange(len(B1_0)), B1_0)[bin_nums]
            B2_0_interp = np.interp(ell_bins_full, np.arange(len(B2_0)), B2_0)[bin_nums]
            B1_eps_interp = np.interp(ell_bins_full, np.arange(len(B1_eps)), B1_eps)[bin_nums]
            B2_eps_interp = np.interp(ell_bins_full, np.arange(len(B2_eps)), B2_eps)[bin_nums]

            scale_factor = np.sqrt((B1_eps_interp / B1_0_interp) * (B2_eps_interp / B2_0_interp))
            dl_scaled = dl_orig * scale_factor
            scaled_dl_all[used_map][eps_val] = dl_scaled

            # Plot
            label = f"eps={eps_val:+.2f}"
            if eps_val == 0.0:
                plt.plot(ell_bins, dl_scaled, 'o-', color='black', linewidth=2.5, label=label)
            else:
                color = cmap(idx-1)
                plt.plot(ell_bins, dl_scaled, '-', color=color, linewidth=1, label=label)

        plt.title(f"{used_map} binned D_l: Original vs Scaled for all eps")
        plt.xlabel("ell")
        plt.ylabel("D_l")
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()

        output_file = os.path.join(output_dir, f"{used_map}_binned_scaled_all_eps.png")
        plt.savefig(output_file)
        plt.close()
        print(f"Saved binned D_l plot for {used_map} to {output_file}")

    return binned_dl_dict


def load_scaled_beams(file_pattern= '/n/home08/liuto/bicep2_analysis/aux_data/beams/beamfile_*.fits', 
                      eps_range=0.05, n_eps=11):
    """
    Reads FITS files from the given pattern and returns a dict of scaled beams.
    
    Returns:
        dict: {mapname: {epsilon: B_scaled_array}}
    """
    all_files = sorted(glob.glob(file_pattern))
    if not all_files:
        print(f"No files found matching: {file_pattern}")
        return {}

    # Filter files to those we care about
    filtered_files = {}
    for file_path in all_files:
        base_name = os.path.basename(file_path)
        for key, name in BK18_FILENAMES.items():
            if base_name == name:
                filtered_files[key] = file_path

    if not filtered_files:
        print("No desired files found in the pattern.")
        return {}

    results = {}
    epsilons = np.linspace(-eps_range, eps_range, n_eps)

    for mapname, file_path in filtered_files.items():
        try:
            with fits.open(file_path) as hdul:
                if len(hdul) < 2:
                    print(f"Skipping {file_path}: Less than 2 HDUs.")
                    continue
                data = hdul[1].data
                # Pick the correct column
                if 'B_l' in data.dtype.names:
                    beam_values = data['B_l']
                elif 'TEMP AND POL' in data.dtype.names:
                    beam_values = data['TEMP AND POL']
                else:
                    print(f"Skipping {file_path}: No recognizable beam column.")
                    continue

                ell = np.arange(len(beam_values))
                results[mapname] = {}

                for eps in epsilons:
                    scaled_ell = ell * (1 + eps)
                    scaled_beam = np.interp(scaled_ell, ell, beam_values,
                                            left=np.nan, right=np.nan)
                    results[mapname][eps] = scaled_beam

        except Exception as e:
            print(f"Error processing {file_path}: {e}", file=sys.stderr)

    return results



def plot_scaled_beams(scaled_beams_dict, output_dir="."):
    """
    Plots scaled beams from a precomputed dictionary.
    
    Args:
        scaled_beams_dict (dict): {mapname: {epsilon: B_scaled_array}}
        output_dir (str): directory to save plots
    """
    if(not os.path.exists(output_dir)):
        os.mkdir(output_dir)
    for mapname, eps_dict in scaled_beams_dict.items():
        plt.figure(figsize=(10, 6))
        
        epsilons = sorted(eps_dict.keys())
        cmap = cm.get_cmap('coolwarm', len(epsilons)-1)

        ell = np.arange(len(next(iter(eps_dict.values()))))  # assume all same length

        for idx, eps in enumerate(epsilons):
            beam = eps_dict[eps]
            if eps == 0:
                plt.plot(ell, beam, color='black', linewidth=2.5, label=f"ε = {eps:+.2f} (Original)")
            else:
                color = cmap(idx - 1)
                plt.plot(ell, beam, color=color, linewidth=1, label=f"ε = {eps:+.2f}")

        plt.title(f"{mapname} Beam from {BK18_FILENAMES[mapname]} \n(Scaling Error)")
        plt.xlabel("ell")
        plt.ylabel("B(ell*(1+eps))")
        plt.xlim([0, 500])
        plt.legend(loc='upper right', fontsize='small', ncol=2)
        plt.tight_layout()

        output_filename = os.path.join(output_dir, f"{mapname}_scaled_beam_plot.png")
        plt.savefig(output_filename)
        plt.close()
        print(f"Saved plot for {mapname} to {output_filename}")
# --- Main Logic ---

def main():
    parser = argparse.ArgumentParser( )
    parser.add_argument(
        '-p', '--pattern', type=str, 
        default='/n/home08/liuto/bicep2_analysis/aux_data/beams/beamfile_*.fits',
        help="The glob pattern to match FITS files (e.g., 'aux_data/beams/beamfile_*.fits')."
    )

    args = parser.parse_args()
    scaled_beams = load_scaled_beams(args.pattern)
    plot_scaled_beams(scaled_beams, output_dir="beam_plots")

if __name__ == '__main__':
    main()


