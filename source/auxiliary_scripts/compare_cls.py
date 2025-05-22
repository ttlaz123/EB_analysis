import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from astropy.io import fits
import argparse
import os

def find_spectrum_columns(column_names):
    """
    Automatically find TT, EE, BB columns from FITS column names.
    Uses case-insensitive substring matching.
    """
    col_lower = [c.lower() for c in column_names]

    def find_col(key_variants):
        for variant in key_variants:
            for col, col_l in zip(column_names, col_lower):
                if variant.lower() in col_l:
                    return col
        return None

    tt_col = find_col(['temperature', 'tt', 'temp'])
    ee_col = find_col(['e-mode', 'gradient', 'ee', 'e'])
    bb_col = find_col(['b-mode', 'curl', 'bb', 'b'])

    if tt_col is None or ee_col is None or bb_col is None:
        raise ValueError(f"Could not find all required spectrum columns.\n"
                         f"Found columns: {column_names}\n"
                         f"TT: {tt_col}, EE: {ee_col}, BB: {bb_col}")

    return tt_col, ee_col, bb_col
def compare_cls_spectra(file1, file2, output_path=None):
    # Open FITS files
    with fits.open(file1) as hdul1, fits.open(file2) as hdul2:
        data1 = hdul1[1].data
        data2 = hdul2[1].data

        cols1 = data1.names
        cols2 = data2.names

        # Find columns automatically
        tt1, ee1, bb1 = find_spectrum_columns(cols1)
        tt2, ee2, bb2 = find_spectrum_columns(cols2)

        ell = np.arange(len(data1))

        # Extract spectra
        cl_tt_1 = data1[tt1]
        cl_ee_1 = data1[ee1]
        cl_bb_1 = data1[bb1]

        cl_tt_2 = data2[tt2]
        cl_ee_2 = data2[ee2]
        cl_bb_2 = data2[bb2]

        # Convert to D_ell
        factor = ell * (ell + 1) / (2 * np.pi)
        dl_tt_1 = factor * cl_tt_1 * 1e12
        dl_ee_1 = factor * cl_ee_1 * 1e12
        dl_bb_1 = factor * cl_bb_1 * 1e12

        dl_tt_2 = factor * cl_tt_2 * 1e12
        dl_ee_2 = factor * cl_ee_2 * 1e12
        dl_bb_2 = factor * cl_bb_2 * 1e12

        # Prepare labels
        label1 = os.path.basename(file1)
        label2 = os.path.basename(file2)

        # Plot
        fig, axes = plt.subplots(3, 1, figsize=(8, 10), sharex=True)

        axes[0].plot(ell, dl_tt_1, label=label1, color='blue')
        axes[0].plot(ell, dl_tt_2, label=label2, color='red', linestyle='--')
        axes[0].set_ylabel(r'$D_\ell^{TT}$ [$\mu K^2$]')
        axes[0].legend()
        axes[0].grid(True)

        axes[1].plot(ell, dl_ee_1, label=label1, color='green')
        axes[1].plot(ell, dl_ee_2, label=label2, color='orange', linestyle='--')
        axes[1].set_ylabel(r'$D_\ell^{EE}$ [$\mu K^2$]')
        axes[1].legend()
        axes[1].grid(True)

        axes[2].plot(ell, dl_bb_1, label=label1, color='purple')
        axes[2].plot(ell, dl_bb_2, label=label2, color='brown', linestyle='--')
        axes[2].set_ylabel(r'$D_\ell^{BB}$ [$\mu K^2$]')
        axes[2].set_xlabel(r'Multipole $\ell$')
        axes[2].legend()
        axes[2].grid(True)

        plt.suptitle(f'Power Spectra Comparison: {label1} vs {label2}')
        plt.tight_layout()
        plt.subplots_adjust(top=0.93)

        if output_path:
            plt.savefig(output_path)
            print(f"Saved figure to {output_path}")
        else:
            plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Compare power spectra from two FITS files.')
    parser.add_argument('file1', type=str, help='Path to first FITS file')
    parser.add_argument('file2', type=str, help='Path to second FITS file')
    parser.add_argument('--output', type=str, default=None, help='Output image filename (optional)')

    args = parser.parse_args()

    compare_cls_spectra(args.file1, args.file2, args.output)
