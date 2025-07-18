import numpy as np
import matplotlib.pyplot as plt
import argparse
import matplotlib
matplotlib.use('Agg')

def plot_ee_eb_arcsin(filename, scale_ee=1):
    """
    Plot scaled EE, -EB, and arcsin(-2EB/EE)/4 from a CAMB-style Cl file.

    Parameters:
        filename (str): Path to the .dat file with Cls.
        scale_ee (float): Factor to scale EE for visual comparison (default: 20).
    """
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
    fig, ax1 = plt.subplots(figsize=(10, 6))
    ax1.plot(ell, scaled_cl_ee * 1e12, label=r'Planck $\mathrm{EE}$', color='blue')
    ax1.plot(ell, neg_cl_eb * 1e12, label=r'EDE $\mathrm{EB}$', color='green')
    ax1.set_yscale('log')
    ax1.set_xlim([0, 700])
    ax1.set_xlabel(r'Multipole $\ell$')
    ax1.set_ylabel(r'$D_\ell$ [$\mu\mathrm{K}^2$]')
    ax1.grid(True)
    ax1.legend(loc='upper right')

    # Arcsin overlay
    ax2 = ax1.twinx()
    ax2.plot(ell, arcsin_ratio, label=r'$\frac{1}{4} \arcsin\left(\frac{2\,\mathrm{EB}}{\mathrm{EE}}\right)$', 
             color='red', linestyle='--')
    ax2.set_ylabel(r'Effective $\beta(\ell)$ [deg]')
    ax2.legend(loc='lower right')

    plt.title('EE and EB for Best-fit EDE')
    plt.tight_layout()
    plt.savefig('test.png')

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot EE, -EB, and arcsin(-EB/EE) from CAMB Cl file.")
    parser.add_argument("file", help="Path to the Cl data file (e.g., fEDE0.07_cl.dat)")
    parser.add_argument("--scale", type=float, default=1, help="Factor to scale EE (default: 20)")
    args = parser.parse_args()

    plot_ee_eb_arcsin(args.file, scale_ee=args.scale)
