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

BK18_FILENAMES = {
    'BK18_150': 'beamfile_20130222_sum.fits',
    'BK18_K95': 'beamfile_20150101_sum_220.fits',
    'BK18_220': 'beamfile_20150321_sum_100.fits',
    'BK18_95e': 'beamfile_20180206_polycorr_sum_100.fits',
}

def scale_dl_beams(used_maps, binned_dl_dict, injected_signal_dict):
    print(binned_dl_dict.keys())
    print(used_maps)
    print(injected_signal_dict)
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


