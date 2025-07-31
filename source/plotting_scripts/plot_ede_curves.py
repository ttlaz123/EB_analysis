import numpy as np
import matplotlib.pyplot as plt
import argparse
import matplotlib
matplotlib.use('Agg')

def plot_ee_eb_arcsin(filename, scale_ee=1, outpath='test.png'):
    """
    Plot scaled EE, -EB, and arcsin(-2EB/EE)/4 from a CAMB-style Cl file.

    Parameters:
        filename (str): Path to the .dat file with Cls.
        scale_ee (float): Factor to scale EE for visual comparison (default: 20).
    """
    plt.rcParams.update({
        "text.usetex": True,
        'font.size': 20,
        "font.family": "serif",
        "font.serif": 'Computer Modern',
        'axes.titlesize': 22,
        'axes.labelsize': 22,
        'xtick.labelsize': 18,
        'ytick.labelsize': 18,
        'legend.fontsize': 24
    })
    # Load and clean data
    with open(filename, 'r') as f:
        lines = [line for line in f if not line.strip().startswith('#')]
    data = np.loadtxt(lines)

    ell = data[:, 0]
    cl_ee = data[:, 2]
    cl_eb = data[:, 5]

    # Transformations
    scaled_cl_ee = cl_ee / scale_ee
    neg_cl_eb = -cl_eb
    ratio = np.clip(neg_cl_eb / cl_ee, -1, 1)
    arcsin_ratio = np.arcsin(2 * ratio) / 4 * 180 / np.pi
    print(arcsin_ratio[:700])
    # Plot
    print('Plotting')
    fig, ax1 = plt.subplots(figsize=(10, 6))
    ax1.plot(ell, scaled_cl_ee * 1e12, label=r'$\mathrm{EE}$ Planck 2013', color='orange')
    ax1.plot(ell, neg_cl_eb * 1e12, 
                    label=r'$\mathrm{EB}$ $f_{\mathrm{EDE}}=0.07$, $g=1$', color='blue')
    ax1.set_yscale('log')
    ax1.set_xlim([0, 700])

    ax1.set_xlabel(r'Multipole $\ell$', fontsize=24)
    ax1.set_ylabel(r'$D_\ell^{\mathrm{CMB}}$ [$\mu\mathrm{K}^2$]', fontsize=18)
    ax1.grid(True)
    ax1.legend(loc='lower left', fontsize=18)

    # Arcsin overlay
    ax2 = ax1.twinx()
    label = r'$\frac{1}{4} \arcsin\left(\frac{2 D_\ell^{EB,\mathrm{CMB}}}{D_\ell^{EE,\mathrm{CMB}}}\right)$'
    ax2.plot(ell, arcsin_ratio, label=label, 
             color='red', linestyle='--')
    ax2.set_ylabel(r'Effective $\beta(\ell)$ [deg]', fontsize=24)
    ax2.set_ylim([-0.5, 1])
    ax2.legend(loc='lower right', fontsize=18)

    plt.title('EDE Effective Rotation Angle', fontsize=24)
    plt.tight_layout()

    print('Saving ' + outpath)
    plt.savefig(outpath)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot EE, -EB, and arcsin(-EB/EE) from CAMB Cl file.")
    parser.add_argument("file", help="Path to the Cl data file (e.g., fEDE0.07_cl.dat)")
    parser.add_argument("--scale", type=float, default=1, help="Factor to scale EE (default: 20)")
    args = parser.parse_args()

    plot_ee_eb_arcsin(args.file, scale_ee=args.scale)
