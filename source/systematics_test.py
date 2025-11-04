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

# --- Helper Functions ---

def plot_fits_data(hdul, file_path, mapname):
    """
    Reads the data from the second HDU (index 1) and attempts to plot it.
    It handles 1D array data (like a table column) and 2D array data (like an image).
    """
    try:
        # Check if the HDU list has a second unit (index 1)
        if len(hdul) < 2:
            print(f"Skipping {file_path}: File only contains {len(hdul)} HDU(s). Index 1 not found.")
            return

        # Access the second HDU
        hdu = hdul[1]
        data = hdu.data

        # Determine the type of data and plot accordingly
        if data is None:
            print(f"Skipping {file_path}: HDU 1 contains no data.")
            return

        print(f"--- HDU 1 Type: {hdu.__class__.__name__}, Shape: {data.shape} ---")

        plt.figure(figsize=(10, 6))
        base_name = os.path.splitext(os.path.basename(file_path))[0]
        output_filename = f"{mapname}_{base_name}_plot.png"
        print(data.dtype.names) 
        if 'B_l' in data.dtype.names:
            beam_values = data['B_l']
        elif('TEMP AND POL' in data.dtype.names):
            beam_values = data['TEMP AND POL']

        else:
            print(f"Skipping {file_path}: Data dimension ({data.ndim}) not supported for simple plotting.")
        ell = np.arange(len(beam_values))
        eps_range = 0.05
        epsilons = np.linspace(-eps_range, eps_range, 11)
        cmap = cm.get_cmap('coolwarm', len(epsilons)-1)
        for idx, eps in enumerate(epsilons):
            scaled_ell = ell * (1+eps)
            scaled_data = np.interp(scaled_ell, ell, beam_values, 
                                    left=np.nan, right=np.nan)
            label = f"eps = {eps:+.2f}"
            if(eps == 0):
                plt.plot(ell, scaled_data, label=label, color='black', linewidth=2.5)
            else:
                color = cmap(idx - 1)
                plt.plot(ell, scaled_data, color=color, linewidth=1, label=label)
        plt.legend()
        plt.title(f"{mapname} Beam from {os.path.basename(file_path)}\n(Scaling Error)")
        plt.xlim([0, 500])
        plt.xlabel("ell")
        plt.ylabel("B(ell(1+eps))")

        plt.tight_layout()
        plt.savefig(output_filename)
        plt.close()
    except Exception as e:
        print(f"An error occurred while plotting data for {file_path}: {e}", file=sys.stderr)


# --- Main Logic ---

def main(file_pattern):
    """
    Finds all files matching the pattern, reads them using astropy, and plots the data
    from the second HDU (index 1) of each file.
    """
    print(f"Searching for files using pattern: {file_pattern}")

    all_files = sorted(glob.glob(file_pattern))

    if not all_files:
        print(f"Error: No files found matching the pattern '{file_pattern}'.")
        return
    
    BK18_FILENAMES = {
        'BK18_150': 'beamfile_20130222_sum.fits',
        'BK18_K95': 'beamfile_20150101_sum_220.fits',
        'BK18_220': 'beamfile_20150321_sum_100.fits',
        'BK18_95e': 'beamfile_20180206_polycorr_sum_100.fits',
    }
    filtered_files = {}
    for file_path in all_files:
        base_name = os.path.basename(file_path)
        for key, name in BK18_FILENAMES.items():
            if(base_name == name):
                filtered_files[key] = file_path
    print(f"Found {len(all_files)} files.")
    print(f"Processing {len(filtered_files)} file(s) after filtering.")
    # 2. Iterate through files, read, and plot
    for key,file_path in filtered_files.items():
        print(f"\nProcessing file {key}: {file_path}")

        try:
            # Use fits.open with a 'with' block for safe file handling
            with fits.open(file_path) as hdul:
                # hdul is the HDU list. Pass it to the plotting function
                plot_fits_data(hdul, file_path, key)

        except OSError as e:
            print(f"Error opening or reading FITS file {file_path}: {e}", file=sys.stderr)
        except Exception as e:
            print(f"An unexpected error occurred with file {file_path}: {e}", file=sys.stderr)

if __name__ == '__main__':
    parser = argparse.ArgumentParser( )
    parser.add_argument(
        '-p', '--pattern', type=str, 
        default='/n/home08/liuto/bicep2_analysis/aux_data/beams/beamfile_*.fits',
        help="The glob pattern to match FITS files (e.g., 'aux_data/beams/beamfile_*.fits')."
    )

    args = parser.parse_args()

    main(args.pattern)


