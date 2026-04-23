"""
eb_slope_mcmc.py
----------------
Streamlined EB-only polarization rotation analysis for 4 BICEP/Keck maps:
    K95, B95e, 150, 220

Rotation model:  alpha_nu + beta * ell
  - alpha_nu : per-map offset angle [degrees], 4 free parameters
  - beta      : shared linear slope [degrees / ell], 1 free parameter

Fits all 4 auto- and 16 cross-EB spectra (20 spectra total) via Cobaya MCMC.
Produces a triangle plot with constraints on all 5 parameters.

Dependencies (same environment as the original code):
    cobaya, numpy, getdist, matplotlib
    + your local: eb_load_data, eb_file_paths, eb_calculations, eb_plot_data, bicep_data_consts
"""

import os
import numpy as np
import argparse
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from cobaya.run import run
from cobaya.likelihood import Likelihood

# ── your existing local modules ───────────────────────────────────────────────
import bicep_data_consts as bc
import eb_load_data as ld
import eb_file_paths as fp
import eb_calculations as ec
import eb_plot_data as epd


# ── constants ─────────────────────────────────────────────────────────────────
# Internal map names as they appear in the data files
MAPS = ["BK18_K95", "BK18_B95e", "BK18_150", "BK18_220"]

# Short labels used for parameter naming (alpha_K95, alpha_B95e, …)
MAP_LABELS = {
    "BK18_K95":  "K95",
    "BK18_B95e": "B95e",
    "BK18_150":  "150",
    "BK18_220":  "220",
}

# Global shared-data container (populated once by load_shared_data)
SHARED = {}


# ══════════════════════════════════════════════════════════════════════════════
#  HELPER: build the list of EB cross-spectra keys
# ══════════════════════════════════════════════════════════════════════════════

def build_eb_spectra_keys(maps):
    """
    Return all EB + BE cross-spectrum keys for every (map_i × map_j) pair,
    including auto-spectra (i == j).

    For maps [A, B, C, D] this gives 4 autos + 12 crosses = 16 pairs,
    each contributing one ExB and one BxE key → 32 keys total (but the
    likelihood treats ExB and BxE together as one EB constraint per pair,
    so effectively 16 independent EB spectra).
    """
    keys = []
    for m1 in maps:
        for m2 in maps:
            keys.append(f"{m1}_Ex{m2}_B")
            keys.append(f"{m1}_Bx{m2}_E")
    return keys


# ══════════════════════════════════════════════════════════════════════════════
#  ROTATION MODEL
# ══════════════════════════════════════════════════════════════════════════════

def rotation_angle_ell(alpha_nu_rad, beta_rad_per_ell, ell_array):
    """
    Total rotation angle as a function of ell for a single map.

    Parameters
    ----------
    alpha_nu_rad : float   per-map constant offset [radians]
    beta_rad_per_ell : float   shared linear slope [radians / ell unit]
    ell_array : ndarray   multipole values

    Returns
    -------
    angle : ndarray  shape (len(ell_array),)
    """
    return alpha_nu_rad + beta_rad_per_ell * ell_array


def rotate_cl_eb(cl_ee, cl_bb, alpha1, alpha2):
    """
    Compute the rotated EB spectrum for a pair of maps given their rotation angles.

    Standard rotation formula (both angles can be ell-dependent arrays):
        C_ell^{EB,rot}_{ij} = (sin2a_i * cos2a_j + cos2a_i * sin2a_j) / 2
                               * (C_ell^{EE} - C_ell^{BB})

    Parameters
    ----------
    cl_ee, cl_bb : ndarray   theory EE and BB spectra (same ell grid)
    alpha1, alpha2 : ndarray or float   rotation angles [radians] for map i, map j

    Returns
    -------
    cl_eb_rot : ndarray
    """
    sin2a1 = np.sin(2.0 * alpha1)
    cos2a1 = np.cos(2.0 * alpha1)
    sin2a2 = np.sin(2.0 * alpha2)
    cos2a2 = np.cos(2.0 * alpha2)

    cl_eb_rot = 0.5 * (sin2a1 * cos2a2 + cos2a1 * sin2a2) * (cl_ee - cl_bb)
    return cl_eb_rot


# ══════════════════════════════════════════════════════════════════════════════
#  DATA LOADING
# ══════════════════════════════════════════════════════════════════════════════

def load_shared_data(dataset, bin_num):
    """
    Load all static data (theory spectra, BPWF, covariance) into SHARED dict.

    Parameters
    ----------
    dataset : str   dataset name (e.g. 'BK18lf')
    bin_num : list  list of bin indices to use
    """
    global SHARED

    file_paths = fp.set_file_paths(dataset, fede=None)

    # Band-power window functions + map reference header
    bpwf, map_ref_header = ld.load_bpwf(
        file_paths["bpwf"],
        map_reference_header=None,
        num_bins=bin_num,
    )

    # CMB theory spectra (EE, BB at minimum)
    theory = ld.load_cmb_theory(file_paths["camb_lensing2018"])

    # Determine which spectrum keys we need from the data file
    eb_keys = build_eb_spectra_keys(MAPS)
    used_maps = ec.filter_used_maps(map_ref_header, eb_keys)

    # Covariance matrix – filter to our EB-only map set
    full_covmat = ld.load_covariance_matrix(
        file_paths["covariance_matrix"], map_ref_header
    )
    filtered_covmat = ec.filter_matrix(
        map_ref_header, full_covmat, used_maps, num_bins=bin_num
    )
    cov_inv = ec.calc_inverse_covmat(filtered_covmat)

    SHARED["file_paths"]      = file_paths
    SHARED["map_ref_header"]  = map_ref_header
    SHARED["bpwf"]            = bpwf
    SHARED["theory"]          = theory
    SHARED["used_maps"]       = used_maps
    SHARED["covmat"]          = filtered_covmat
    SHARED["cov_inv"]         = cov_inv
    SHARED["bin_num"]         = bin_num

    print(f"Loaded {len(used_maps)} EB spectra:")
    for m in used_maps:
        print(f"  {m}")


# ══════════════════════════════════════════════════════════════════════════════
#  LIKELIHOOD CLASS
# ══════════════════════════════════════════════════════════════════════════════

class EBSlopeLikelihood(Likelihood):
    """
    Cobaya Likelihood for EB-only polarization rotation analysis.

    Free parameters
    ---------------
    alpha_K95, alpha_B95e, alpha_150, alpha_220 : float [degrees]
        Per-map constant rotation offsets.
    beta : float [degrees / ell]
        Shared linear slope added to all maps' rotation.

    Model
    -----
    For map pair (i, j):
        theta_i(ell) = alpha_i + beta * ell          [degrees → radians internally]
        C_ell^{EB}_{ij,rot} = 0.5 * (sin2θ_i cos2θ_j + cos2θ_i sin2θ_j)
                               * (C_ell^{EE} - C_ell^{BB})
    """

    # Cobaya will inject these as keyword arguments to logp()
    params = {
        "alpha_K95":  None,
        "alpha_B95e": None,
        "alpha_150":  None,
        "alpha_220":  None,
        "beta":       None,
    }

    def initialize(self):
        """Called once by Cobaya at startup."""
        self.used_maps    = SHARED["used_maps"]
        self.map_ref_hdr  = SHARED["map_ref_header"]
        self.bpwf         = SHARED["bpwf"]
        self.theory       = SHARED["theory"]
        self.cov_inv      = SHARED["cov_inv"]
        self.bin_num      = SHARED["bin_num"]

        # Load observed EB spectra
        obs_dict, _ = ld.load_observed_spectra(
            SHARED["file_paths"]["observed_data"],
            self.used_maps,
            self.map_ref_hdr,
            num_bins=self.bin_num,
        )
        self.obs_vec = self._dict_to_vec(obs_dict)

        # Pre-compute the ell array that corresponds to the BPWF columns
        # (same ell grid as the theory spectra, before BPWF convolution)
        n_ell = len(self.theory["EE"])
        self.ell_array = np.arange(n_ell, dtype=float)  # 0-based index = ell

        # Map name → short label → alpha parameter name
        self._map_to_label = MAP_LABELS          # e.g. "BK18_K95" → "K95"
        self._alpha_param   = {                  # e.g. "BK18_K95" → "alpha_K95"
            m: f"alpha_{lbl}" for m, lbl in MAP_LABELS.items()
        }

    # ── internal helpers ──────────────────────────────────────────────────────

    def _dict_to_vec(self, spectra_dict):
        """Flatten spectra_dict into a 1-D vector, ordered by map_ref_hdr."""
        parts = []
        for key in self.map_ref_hdr:
            if key in self.used_maps:
                parts.append(np.asarray(spectra_dict[key]))
        return np.concatenate(parts)

    def _map_name_to_freq(self, map_name):
        """Strip the 'BK18_' prefix to get the frequency label (e.g. 'K95')."""
        return map_name.replace("BK18_", "")

    def _parse_spectrum_key(self, key):
        """
        Parse a key like 'BK18_K95_ExBK18_150_B' into (map1, pol1, map2, pol2).
        Keys have the form:  {map1}_Ex{map2}_B  or  {map1}_Bx{map2}_E
        """
        # split on '_Ex' or '_Bx'
        if "_Ex" in key:
            left, right = key.split("_Ex", 1)
            map1, pol1 = left, "E"
            map2, pol2 = right.rsplit("_", 1)   # right = "BK18_150_B"
            # map2 actually ends with _B, strip it
            map2 = right[: -(len(pol2) + 1)]    # remove "_B"
        else:  # "_Bx"
            left, right = key.split("_Bx", 1)
            map1, pol1 = left, "B"
            map2, pol2 = right.rsplit("_", 1)
            map2 = right[: -(len(pol2) + 1)]
        return map1, pol1, map2, pol2

    # ── theory prediction ─────────────────────────────────────────────────────

    def _theory_vec(self, alpha_deg, beta_deg_per_ell):
        """
        Compute the theory EB vector for the current parameter values.

        Parameters
        ----------
        alpha_deg : dict  {map_name: alpha [degrees]}
        beta_deg_per_ell : float  shared slope [degrees / ell]

        Returns
        -------
        theory_vec : ndarray  same ordering as obs_vec
        """
        cl_ee = np.asarray(self.theory["EE"])
        cl_bb = np.asarray(self.theory["BB"])

        # Build a dict of pre-rotated (unbinned) EB spectra for each key
        theory_dict = {}
        for key in self.used_maps:
            map1, pol1, map2, pol2 = self._parse_spectrum_key(key)

            # rotation angles as a function of ell [radians]
            a1 = np.deg2rad(alpha_deg[map1] + beta_deg_per_ell * self.ell_array)
            a2 = np.deg2rad(alpha_deg[map2] + beta_deg_per_ell * self.ell_array)

            if pol1 == "E" and pol2 == "B":
                # ExB
                cl_rot = rotate_cl_eb(cl_ee, cl_bb, a1, a2)
            else:
                # BxE  →  same magnitude, but note C_ell^{BE} = C_ell^{EB}
                cl_rot = rotate_cl_eb(cl_ee, cl_bb, a2, a1)

            theory_dict[key] = cl_rot

        # Apply band-power window functions to go from ell → bins
        binned_dict = ec.apply_bpwf(
            self.map_ref_hdr,
            theory_dict,
            self.bpwf,
            self.used_maps,
            do_cross=True,
        )

        return self._dict_to_vec(binned_dict)

    # ── Cobaya interface ──────────────────────────────────────────────────────

    def logp(self, **params_values):
        alpha_deg = {
            "BK18_K95":  params_values["alpha_K95"],
            "BK18_B95e": params_values["alpha_B95e"],
            "BK18_150":  params_values["alpha_150"],
            "BK18_220":  params_values["alpha_220"],
        }
        beta = params_values["beta"]

        theory = self._theory_vec(alpha_deg, beta)
        residuals = self.obs_vec - theory
        chi2 = residuals @ self.cov_inv @ residuals
        return -0.5 * chi2


# ══════════════════════════════════════════════════════════════════════════════
#  PRIORS
# ══════════════════════════════════════════════════════════════════════════════

def build_params_dict(alpha_range=5.0, beta_range=0.01):
    """
    Build the Cobaya params dict with flat priors.

    Parameters
    ----------
    alpha_range : float   ±range for each alpha_nu prior [degrees]
    beta_range  : float   ±range for the beta prior [degrees/ell]
    """
    params = {}

    for label in MAP_LABELS.values():   # K95, B95e, 150, 220
        params[f"alpha_{label}"] = {
            "prior":    {"min": -alpha_range, "max": alpha_range},
            "ref":      0.0,
            "proposal": alpha_range / 10.0,
            "latex":    f"\\alpha_{{\\rm {label}}}",
        }

    params["beta"] = {
        "prior":    {"min": -beta_range, "max": beta_range},
        "ref":      0.0,
        "proposal": beta_range / 20.0,
        "latex":    r"\beta",
    }

    return params


# ══════════════════════════════════════════════════════════════════════════════
#  RUN COBAYA MCMC
# ══════════════════════════════════════════════════════════════════════════════

def run_mcmc(output_path, params_dict, Rminus1_stop=0.03, max_tries=50000):
    """
    Configure and launch the Cobaya MCMC sampler.

    Parameters
    ----------
    output_path   : str   prefix for Cobaya chain files
    params_dict   : dict  Cobaya-format parameter priors
    Rminus1_stop  : float Gelman-Rubin convergence criterion
    max_tries     : int   maximum MCMC steps

    Returns
    -------
    updated_info, sampler
    """
    info = {
        "likelihood": {
            "eb_slope": EBSlopeLikelihood,
        },
        "params":  params_dict,
        "sampler": {
            "mcmc": {
                "Rminus1_stop": Rminus1_stop,
                "max_tries":    max_tries,
            }
        },
        "output": output_path,
        "resume": True,
    }

    print("Starting Cobaya MCMC …")
    updated_info, sampler = run(info, stop_at_error=True)
    return updated_info, sampler


# ══════════════════════════════════════════════════════════════════════════════
#  TRIANGLE PLOT
# ══════════════════════════════════════════════════════════════════════════════

def make_triangle_plot(output_path, params_dict):
    """
    Read Cobaya chains and produce a GetDist triangle plot.

    The plot is saved as <output_path>_triangle.pdf

    Parameters
    ----------
    output_path : str   Cobaya output prefix (same as passed to run_mcmc)
    params_dict : dict  params dict (used to extract latex labels)
    """
    try:
        from getdist import loadMCSamples
        from getdist.plots import getSubplotPlotter
    except ImportError:
        print("getdist not found – skipping triangle plot.")
        return

    samples = loadMCSamples(output_path, settings={"ignore_rows": 0.3})

    param_names = list(params_dict.keys())
    latex_labels = {
        k: v["latex"] for k, v in params_dict.items() if isinstance(v, dict)
    }

    g = getSubplotPlotter(width_inch=10)
    g.triangle_plot(
        [samples],
        param_names,
        filled=True,
        legend_labels=["Posterior"],
        title_limit=1,          # show 68 % limit in title
    )

    plot_path = output_path + "_triangle.pdf"
    g.export(plot_path)
    print(f"Triangle plot saved → {plot_path}")


# ══════════════════════════════════════════════════════════════════════════════
#  CLI
# ══════════════════════════════════════════════════════════════════════════════

def parse_args():
    p = argparse.ArgumentParser(
        description="EB slope MCMC: fit alpha_nu + beta*ell rotation to BK18 EB spectra."
    )
    p.add_argument(
        "-d", "--dataset",
        default="BK18lf",
        help="Dataset name passed to eb_file_paths.set_file_paths(). Default: BK18lf",
    )
    p.add_argument(
        "-b", "--bin_num",
        type=int,
        nargs="+",
        default=list(range(2, 16)),
        help="Bin indices to include. Default: 2 3 … 15",
    )
    p.add_argument(
        "-p", "--output_path",
        default="chains/eb_slope",
        help="Cobaya output prefix. Default: chains/eb_slope",
    )
    p.add_argument(
        "--alpha_range",
        type=float,
        default=5.0,
        help="Half-width of flat prior on each alpha_nu [degrees]. Default: 5",
    )
    p.add_argument(
        "--beta_range",
        type=float,
        default=0.01,
        help="Half-width of flat prior on beta [degrees/ell]. Default: 0.01",
    )
    p.add_argument(
        "--Rminus1_stop",
        type=float,
        default=0.03,
        help="Gelman-Rubin convergence criterion. Default: 0.03",
    )
    p.add_argument(
        "--max_tries",
        type=int,
        default=50000,
        help="Maximum MCMC steps. Default: 50000",
    )
    p.add_argument(
        "--plot_only",
        action="store_true",
        help="Skip MCMC and only remake the triangle plot from existing chains.",
    )
    return p.parse_args()


def main():
    args = parse_args()
    os.makedirs(os.path.dirname(args.output_path) or ".", exist_ok=True)

    # 1. Load data
    load_shared_data(args.dataset, args.bin_num)

    # 2. Define priors
    params_dict = build_params_dict(
        alpha_range=args.alpha_range,
        beta_range=args.beta_range,
    )

    # 3. Run MCMC (unless --plot_only)
    if not args.plot_only:
        run_mcmc(
            output_path=args.output_path,
            params_dict=params_dict,
            Rminus1_stop=args.Rminus1_stop,
            max_tries=args.max_tries,
        )

    # 4. Triangle plot
    make_triangle_plot(args.output_path, params_dict)


if __name__ == "__main__":
    main()