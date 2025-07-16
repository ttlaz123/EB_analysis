import matplotlib
matplotlib.use('Agg')

import numpy as np
import matplotlib.pyplot as plt
import argparse
import os
import re

def extract_column_indices(header_lines):
    """
    Extracts the column indices for 'l', 'EE', and 'EB' from header lines.
    Returns a dictionary mapping column names to zero-based indices.
    """
    for line in header_lines:
        # Look for line with pattern like: "# 1:l  2:TT  3:EE  ..."
        match = re.findall(r'(\d+)\s*:\s*([a-zA-Z]+)', line)
        if match:
            name_to_index = {}
            for idx_str, name in match:
                name = name.upper()
                index = int(idx_str) - 1  # Convert to zero-based
                if name in ['L', 'ELL']:
                    name_to_index['l'] = index
                elif name == 'EE':
                    name_to_index['EE'] = index
                elif name == 'EB':
                    name_to_index['EB'] = index
            if all(k in name_to_index for k in ['l', 'EE', 'EB']):
                return name_to_index

    raise ValueError("Failed to find column index definitions for l, EE, and EB in header.")

def plot_multiple_eb_ee(filenames, output='eb_ee_plot.png'):
    plt.figure(figsize=(10, 6))
    eb_colors = plt.cm.Blues(np.linspace(0.4, 0.9, len(filenames)))
    ee_colors = plt.cm.Oranges(np.linspace(0.4, 0.9, len(filenames)))

    for idx, filename in enumerate(filenames):
        with open(filename, 'r') as f:
            header_lines = []
            while True:
                pos = f.tell()
                line = f.readline()
                if not line.startswith('#'):
                    f.seek(pos)
                    break
                header_lines.append(line)

        col_idx = extract_column_indices(header_lines)
        data = np.loadtxt(filename, comments='#')

        l = data[:, col_idx['l']]
        EE = data[:, col_idx['EE']] / 20 *1e12
        EB = -data[:, col_idx['EB']] * 1e12

        label_base = os.path.basename(filename)
        plt.plot(l, EB, label=f'EB ({label_base})', color=eb_colors[idx], lw=1.5)
        plt.plot(l, EE, label=f'EE/20 ({label_base})', color=ee_colors[idx], lw=1.5, linestyle='--')

    plt.xlabel('Multipole moment ℓ')
    plt.ylabel(r'$D_\ell$ [$\mu K^2$]')
    plt.title('EB and EE/20 vs ℓ')
    plt.grid(True)
    plt.legend(fontsize='small')
    plt.tight_layout()
    plt.savefig(output)
    print(f"✅ Plot saved to: {output}")

def main():
    parser = argparse.ArgumentParser(description='Plot EB and EE/20 vs ℓ from CAMB-style output files.')
    parser.add_argument('filenames', nargs='+', help='List of CAMB-format .dat files to plot')
    parser.add_argument('--output', '-o', type=str, default='eb_ee_plot.png', help='Filename for output image')
    args = parser.parse_args()

    plot_multiple_eb_ee(args.filenames, args.output)

if __name__ == '__main__':
    main()

