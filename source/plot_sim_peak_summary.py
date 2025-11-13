import os
import re
import numpy as np
import matplotlib.pyplot as plt
import eb_plot_data as epd

def plot_sim_summary_ldiff(dirpath):
    g_list = ['zeroeb', 'fede01']
    params_to_plot = ['alpha_BK18_B95e', 'alpha_BK18_K95', 
                          'alpha_BK18_150', 'alpha_BK18_220', 'angle_diff']
    b = 'bin2-15'
    eps_values = np.arange(-0.05, 0.051, 0.01)
    for param in params_to_plot:
        for g in g_list:
            mean_vals, std_vals = [], []
            for eps in eps_values:
                eps_tag = f"eps{eps:.2f}"
                if eps_tag == "eps0.00":
                    eps_tag = "eps-0.00" 
                folder = f"{dirpath}/{g}_{b}_ldiff12_BK18_eb_sig{eps_tag}_ebfede0.07"
                chains_path = os.path.join(folder, "sim")
                mean_params_dict, std_params_dict = epd.plot_sim_peaks(
                                chains_path, single_sim=1, overwrite=False, do_plots=False
                            )
                mean_vals.append(mean_params_dict[param])
                std_vals.append(std_params_dict[param])
            plt.figure(figsize=(8, 6))
            plt.title(f"{param} beam scaling — {g}", fontsize=14)
            if(len(mean_vals) > 0):
                    plt.errorbar(
                        eps_values, mean_vals, yerr=std_vals, fmt='-o', label=f"{m}", capsize=4
                    )
           

            plt.xlabel("epsilon", fontsize=12)
            plt.ylabel(param, fontsize=12)
            if param == 'gMpl':
                plt.ylim([-2, 3])
            else:
                plt.ylim([-0.6, 0.6])
            if (param == 'gMpl' and b == 'bin2-15' and g == 'fede01'):
                plt.ylim([0.4, 1.6])
            if (param == 'gMpl' and b == 'bin2-15' and g == 'zeroeb'):
                plt.ylim([-0.6, 0.6])
            plt.grid(alpha=0.3)
            plt.legend(title="Scaled Map", fontsize=8)
            plt.tight_layout()

            outfile = f"summary_ldiff12_{param}_{g}_{b}.png"
            plt.savefig(outfile, dpi=150)
            print(f"Saved {outfile}")
            plt.close()


def plot_sim_summary(dirpath, params_to_plot=None):
    """
    Plots parameter means and stds vs injected beam uncertainty ε.
    Each parameter gets its own figure.
    For each (g, b) combination, all mapfreq sets (m) appear as curves on the same plot.
    """

    g_list = ['zeroeb', 'fede01']
    b_list = ['bin2-15', 'bin2-10']

    if params_to_plot is None:
        params_to_plot = ['alpha_BK18_B95e', 'alpha_BK18_K95', 
                          'alpha_BK18_150', 'alpha_BK18_220', 'gMpl']

    # Scan all folders once to extract mapfreqs and epsilons
    map_tokens_re = re.compile(r'(B95e|K95|220|150)')

    # list directories
    all_entries = os.listdir(dirpath)
    all_folders = [f for f in all_entries if os.path.isdir(os.path.join(dirpath, f))]
    folder_info = []
    for fname in all_folders:
        # find the segment between 'sigeps_' and '_ebfede'
        m_seg = re.search(r'sigeps_(.+?)_ebfede', fname)
        if not m_seg:
            # skip folders that don't follow pattern
            # (you can print or log if you'd like)
            # print(f"skip (no sigeps...ebfede): {fname}")
            continue

        seg = m_seg.group(1)  # e.g. 'BK18_2200.04' or 'BK18_K95-0.04'

        # find map token inside the segment
        m_token = map_tokens_re.search(seg)
        if not m_token:
            # couldn't find one of expected map tokens
            # print(f"skip (no known map token): {fname}")
            continue

        token = m_token.group(1)  # '220' or 'K95' or 'B95e' or '150'

        # determine mapfreq string (make it 'BK18_<token>')
        # if seg contains 'BK18_' we keep that prefix; otherwise still use BK18_ for consistency
        if 'BK18_' in seg:
            mapfreq = f"BK18_{token}"
        else:
            mapfreq = f"BK18_{token}"

        # eps should be whatever follows the token in seg (may be '-0.04', '0.04', or '0.00', etc.)
        eps_part = seg[m_token.end():]  # remainder after the token
        # eps_part might start with '-' or nothing (e.g. '-0.04' or '0.04' or '0' etc.)
        # find the last float-like substring in eps_part (robust)
        m_eps = re.search(r'([+-]?\d*\.\d+|[+-]?\d+)$', eps_part)
        if not m_eps:
            # maybe eps is concatenated earlier (rare); try to find any float in seg after the token
            m_eps = re.search(r'([+-]?\d*\.\d+|[+-]?\d+)', seg[m_token.start():])
        if not m_eps:
            # failed to parse epsilon
            # print(f"skip (no eps parsed): {fname}")
            continue

        try:
            eps = float(m_eps.group(1))
        except ValueError:
            # print(f"skip (eps float parse error): {fname}")
            continue

        # extract g and b from filename start (conservative)
        # assume filename starts like: <g>_<b>_fixeddust_...
        m_gb = re.match(r'^([^_]+)_([^_]+)_fixeddust', fname)
        if not m_gb:
            # fallback: unknown g/b, skip
            # print(f"skip (cannot parse g/b): {fname}")
            continue

        g, b = m_gb.group(1), m_gb.group(2)

        folder_info.append({
            'fname': fname,
            'folder': os.path.join(dirpath, fname),
            'g': g,
            'b': b,
            'mapfreq': mapfreq,
            'eps': eps
        })

    # Now plotting (same logic as before but using parsed info)
    for param in params_to_plot:
        for g in g_list:
            for b in b_list:
                plt.figure(figsize=(8, 6))
                plt.title(f"{param} beam scaling — {g}, {b}", fontsize=14)

                # select relevant entries
                relevant = [e for e in folder_info if e['g'] == g and e['b'] == b]
                if not relevant:
                    # nothing found for this (g,b); skip quietly
                    plt.close()
                    continue

                mapfreqs = sorted(set(e['mapfreq'] for e in relevant))

                for mf in mapfreqs:
                    entries = sorted([e for e in relevant if e['mapfreq'] == mf], key=lambda x: x['eps'])
                    eps_vals = []
                    mean_vals = []
                    std_vals = []
                    for ent in entries:
                        eps_vals.append(ent['eps'])
                        chains_path = os.path.join(ent['folder'], "sim")
                        try:
                            mean_params_dict, std_params_dict = epd.plot_sim_peaks(
                                chains_path, single_sim=1, overwrite=False, do_plots=False
                            )
                            # raise if key missing so we skip properly
                            mean_vals.append(mean_params_dict[param])
                            std_vals.append(std_params_dict[param])
                        except Exception:
                            # missing parameter or other error -> skip this entry
                            continue

                    if len(mean_vals) > 0:
                        plt.errorbar(eps_vals, mean_vals, yerr=std_vals, fmt='-o', label=mf, capsize=4)

                plt.xlabel("epsilon", fontsize=12)
                plt.ylabel(param, fontsize=12)
                if param == 'gMpl':
                    plt.ylim([-2, 3])
                else:
                    plt.ylim([-0.6, 0.6])
                if (param == 'gMpl' and b == 'bin2-15' and g == 'fede01'):
                    plt.ylim([0.4, 1.6])
                if (param == 'gMpl' and b == 'bin2-15' and g == 'zeroeb'):
                    plt.ylim([-0.6, 0.6])
                plt.grid(alpha=0.3)
                plt.legend(title="Scaled Map", fontsize=8)
                plt.tight_layout()

                outfile = f"summary_{param}_{g}_{b}.png"
                plt.savefig(outfile, dpi=150)
                print(f"Saved {outfile}")
                plt.close()


if __name__=='__main__':
    dirpath = '/n/holylfs04/LABS/kovac_lab/users/liuto/ede_chains/'
    plot_sim_summary_ldiff(dirpath)
