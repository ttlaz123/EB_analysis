import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as mcolors

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

def read_spectrum_file(filepath):
    """
    Read a power spectrum file (.fits or .dat).
    Returns: ell, cl_ee, cl_bb, cl_eb
    """
    ext = os.path.splitext(filepath)[-1].lower()

    if ext == '.fits':
        with fits.open(filepath) as hdul:
            data = hdul[1].data
            cols = data.names
            ee_col = find_spectrum_columns(cols)[1]
            bb_col = find_spectrum_columns(cols)[2]

            # Try to find EB column, fallback to zeros
            eb_col = None
            for col in cols:
                if 'eb' in col.lower():
                    eb_col = col
                    break

            ell = np.arange(len(data))
            cl_ee = data[ee_col]
            cl_bb = data[bb_col]
            cl_eb = data[eb_col] if eb_col else np.zeros_like(cl_ee)

    elif ext == '.dat':
        arr = np.loadtxt(filepath, comments='#')
        if arr.shape[1] < 5:
            raise ValueError("Expected at least 5 columns (l, TT, EE, TE, BB).")

        ell = arr[:, 0].astype(int)
        cl_ee = arr[:, 2]  # EE = column 3
        cl_bb = arr[:, 4]  # BB = column 5

        # EB = column 6 if available
        cl_eb = arr[:, 5] if arr.shape[1] >= 6 else np.zeros_like(cl_ee)

    else:
        raise ValueError(f"Unsupported file type: {filepath}")

    return ell, cl_ee, cl_bb, cl_eb


def plot_scaled_comparison(ell, cl_dict, output_path=None):
    """
    cl_dict = {
        'file1': {'ee': ..., 'bb': ..., 'eb': ...},
        'file2': {'ee': ..., 'bb': ..., 'eb': ...}
    }
    """
    a_lens_vals = np.arange(0.9, 1.21, 0.1)
    g_vals = np.arange(0.0, 1.0, 0.3)

    files = list(cl_dict.keys())
    base_colors = ['tab:blue', 'tab:red']  # Adjust if comparing more files

    fig, axes = plt.subplots(3, 1, figsize=(9, 12), sharex=True)

    for i, fname in enumerate(files):
        cl_ee = cl_dict[fname]['ee']
        cl_bb = cl_dict[fname]['bb']
        cl_eb = cl_dict[fname]['eb']

        factor = ell * (ell + 1) / (2 * np.pi)
        dl_ee = factor * cl_ee * 1e12
        dl_bb = factor * cl_bb * 1e12
        dl_eb = factor * cl_eb * 1e12

        base_color = base_colors[i]

        # --- EE (just one line per file)
        axes[0].plot(ell, dl_ee, label=f"EE ({fname})", color=base_color)

        # --- BB scaled by A_lens
        for j, a in enumerate(a_lens_vals):
            shade = mcolors.to_rgba(base_color, alpha=0.4 + 0.15 * j)
            axes[1].plot(ell, a * dl_bb, label=fr"$A_\mathrm{{lens}}={a:.1f}$ ({fname})", color=shade)

        # --- EB scaled by g
        for j, g in enumerate(g_vals):
            shade = mcolors.to_rgba(base_color, alpha=0.4 + 0.15 * j)
            axes[2].plot(ell, g * dl_eb, label=fr"$g={g:.1f}$ ({fname})", color=shade)

    # EE plot settings
    axes[0].set_ylabel(r"$D_\ell^{EE}$ [$\mu K^2$]")
    axes[0].legend()
    axes[0].grid(True)

    # BB plot settings
    axes[1].set_ylabel(r"$D_\ell^{BB}$ [$\mu K^2$]")
    axes[1].legend()
    axes[1].grid(True)

    # EB plot settings
    axes[2].set_ylabel(r"$D_\ell^{EB}$ [$\mu K^2$]")
    axes[2].set_xlabel(r"Multipole $\ell$")
    axes[2].legend()
    axes[2].grid(True)

    plt.suptitle("EE, BB (scaled), EB (scaled) Spectra Comparison")
    plt.tight_layout()
    plt.subplots_adjust(top=0.92)

    if output_path:
        plt.savefig(output_path)
        print(f"Saved plot to {output_path}")
    else:
        plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Compare two spectra files.')
    parser.add_argument('file1', type=str)
    parser.add_argument('file2', type=str)
    parser.add_argument('--output', type=str, default=None)
    args = parser.parse_args()

    ell1, ee1, bb1, eb1 = read_spectrum_file(args.file1)
    ell2, ee2, bb2, eb2 = read_spectrum_file(args.file2)

    if not np.array_equal(ell1, ell2):
        raise ValueError("Mismatch in ell arrays")

    cl_dict = {
        os.path.basename(args.file1): {'ee': ee1, 'bb': bb1, 'eb': eb1},
        os.path.basename(args.file2): {'ee': ee2, 'bb': bb2, 'eb': eb2},
    }

    plot_scaled_comparison(ell1, cl_dict, args.output)
