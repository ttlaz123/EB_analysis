import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as mcolors

from astropy.io import fits
import argparse
import os
T_CMB_K = 2.7255
def find_spectrum_columns(column_names):
    """
    Automatically find TT, EE, BB columns from FITS column names.
    Uses case-insensitive substring matching.
    """
    col_lower = [c.lower() for c in column_names]
    print(col_lower)
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
    Returns: ell, cl_ee, cl_bb, cl_eb, scale_factor
    """
    ext = os.path.splitext(filepath)[-1].lower()

    if ext == '.fits':
        with fits.open(filepath) as hdul:
            data = hdul[1].data
            cols = data.names
            ee_col = find_spectrum_columns(cols)[1]
            bb_col = find_spectrum_columns(cols)[2]

            eb_col = None
            for col in cols:
                if 'eb' in col.lower():
                    eb_col = col
                    break

            ell = np.arange(len(data))
            cl_ee = data[ee_col]
            cl_bb = data[bb_col]
            cl_eb = data[eb_col] if eb_col else np.zeros_like(cl_ee)
            scale_factor = 1

    elif ext == '.dat':
        arr = np.loadtxt(filepath, comments='#')
        if arr.shape[1] < 5:
            raise ValueError("Expected at least 5 columns (l, TT, EE, TE, BB).")

        ell = arr[:, 0].astype(int)
        cl_ee = arr[:, 2]
        cl_bb = arr[:, 4]
        cl_eb = arr[:, 5] if arr.shape[1] >= 6 else np.zeros_like(cl_ee)
        scale_factor = 0  # Already dimensionless (l(l+1)/2pi) D_l

    else:
        raise ValueError(f"Unsupported file type: {filepath}")

    return ell, cl_ee, cl_bb, cl_eb, scale_factor

def plot_scaled_comparison(ell_dict, cl_dict, scale_dict, output_path=None):
    a_lens_vals = np.arange(0.9, 1.21, 0.1)
    g_vals = np.arange(0.0, 1.0, 0.3)

    files = list(cl_dict.keys())
    base_colors = ['tab:blue', 'tab:red']
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
    fig, axes = plt.subplots(3, 1, figsize=(9, 12), sharex=False)

    for i, fname in enumerate(files):
    
        ell = ell_dict[fname]
        cl_ee = cl_dict[fname]['ee']
        cl_bb = cl_dict[fname]['bb']
        cl_eb = cl_dict[fname]['eb']
        
        coeff = scale_dict[fname]
        if(coeff == 0):
            factor = 1
            scale = 1e12 * np.square(T_CMB_K)
        elif(coeff == 1):
            factor = ell * (ell + 1) / (2 * np.pi)
            scale = 1e12
        dl_ee = factor * cl_ee * scale
        dl_bb = factor * cl_bb * scale
        dl_eb = factor * cl_eb * scale

        base_color = base_colors[i]
        if('camb' in fname):
            axes[0].plot(ell, dl_ee, color=base_color, label='EE Spectrum')
        if('camb' in fname):
            for j, a in enumerate(a_lens_vals):
                shade = mcolors.to_rgba(base_color, alpha=0.4 + 0.15 * j)
                axes[1].plot(ell, a * dl_bb, label=fr"$A_\mathrm{{lens}}={a:.1f}$", color=shade)
        if('EDE' in fname):
            for j, g in enumerate(g_vals):
                shade = mcolors.to_rgba(base_color, alpha=0.4 + 0.15 * j)
                axes[2].plot(ell, -g * dl_eb, label=fr"$g={g:.1f}$", color=shade)

    for ax in axes:
        ax.set_xlim(0, 700)
        ax.legend()
        ax.grid(True)

    axes[0].set_ylabel(r"$D_\ell^{EE}$ [$\mu K^2$]", fontname='Computer Modern')
    axes[1].set_ylabel(r"$D_\ell^{BB}$ [$\mu K^2$]", fontname='Computer Modern')
    axes[2].set_ylabel(r"$D_\ell^{EB}$ [$\mu K^2$]", fontname='Computer Modern')
    axes[2].set_xlabel(r"Multipole $\ell$", fontname='Computer Modern')

    #plt.suptitle("EE, BB (scaled), EB (scaled) Spectra Comparison")
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

    fname1 = os.path.basename(args.file1)
    fname2 = os.path.basename(args.file2)

    ell1, ee1, bb1, eb1, scale1 = read_spectrum_file(args.file1)
    ell2, ee2, bb2, eb2, scale2 = read_spectrum_file(args.file2)

    ell_dict = {fname1: ell1, fname2: ell2}
    cl_dict = {
        fname1: {"ee": ee1, "bb": bb1, "eb": eb1},
        fname2: {"ee": ee2, "bb": bb2, "eb": eb2},
    }
    scale_dict = {fname1: scale1, fname2: scale2}

    plot_scaled_comparison(ell_dict, cl_dict, scale_dict, args.output)
