import numpy as np
import matplotlib.pyplot as plt
import argparse
import matplotlib
matplotlib.use('Agg')

def plot_ee_eb_arcsin(filename, scale_ee=1):
    """
    Plot EE/scale_ee, -EB, and arcsin(-EB/EE) from a CAMB-style Cl file.

    Parameters:
        filename (str): Path to the .dat file with Cls.
        scale_ee (float): Factor to scale EE (default: 20).
    """
    with open(filename, 'r') as f:
        lines = [line for line in f if not line.strip().startswith('#')]
    data = np.loadtxt(lines)

    ell = data[:, 0]
    cl_ee = data[:, 2]
    cl_eb = data[:, 5]

    scaled_cl_ee = cl_ee / scale_ee
    neg_cl_eb = -cl_eb

    ratio = np.clip(neg_cl_eb / cl_ee, -1, 1)
    arcsin_ratio = np.arcsin(ratio)

    fig, ax1 = plt.subplots(figsize=(10, 6))
    ax1.plot(ell, scaled_cl_ee, label=f'EE / {scale_ee}', color='blue')
    ax1.plot(ell, neg_cl_eb, label='−EB', color='green')
    ax1.set_yscale('log')
    ax1.set_xlabel(r'$\ell$')
    ax1.set_ylabel('EE / scale and −EB')
    ax1.grid(True)
    ax1.legend(loc='upper right')

    ax2 = ax1.twinx()
    ax2.plot(ell, arcsin_ratio, label=r'$\arcsin(-\mathrm{EB}/\mathrm{EE})$', color='red', linestyle='--')
    ax2.set_ylabel(r'$\arcsin(-\mathrm{EB}/\mathrm{EE})$ [rad]')
    ax2.legend(loc='lower right')

    plt.title('Scaled EE, −EB, and Rotation Angle Proxy')
    plt.tight_layout()
    plt.savefig('test.png')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot EE, -EB, and arcsin(-EB/EE) from CAMB Cl file.")
    parser.add_argument("file", help="Path to the Cl data file (e.g., fEDE0.07_cl.dat)")
    parser.add_argument("--scale", type=float, default=20, help="Factor to scale EE (default: 20)")
    args = parser.parse_args()

    plot_ee_eb_arcsin(args.file, scale_ee=args.scale)
