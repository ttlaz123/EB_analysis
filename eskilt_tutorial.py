import numpy as np
import matplotlib.pyplot as plt
import numba as nb
import emcee
import corner

import camb

from eb_mcmc_analysis import bin_spectrum
def eskilt_tutorial():
    # Load the observed EB power spectrum
    c_l_EB_o_mean_std = np.load('HFI_f_sky_092_EB_o.npy')
    # c_l_EB_o_mean_std[:, 0] gives the central values for the binned observed stacked EB power spectrum
    # c_l_EB_o_mean_std[:, 1] gives the corresponding error bars

    # EB is binned starting at ell_min = 51, ell_max = 1490 and delta ell = 20
    # For more details see the article.
    ell_min = 51
    ell_max = 1490
    delta_ell = 20
    ell = np.arange(ell_min, ell_max+1, delta_ell)
    n_bins = len(ell)

    # Plot the observed EB power spectrum
    plt.figure()
    plt.errorbar(ell, c_l_EB_o_mean_std[:, 0], yerr=c_l_EB_o_mean_std[:, 1], fmt='.', color='black')

    plt.axhline(0, linestyle='--', color='black', alpha=0.5)
    plt.xlabel(r'$\ell$')
    plt.ylabel(r'$C^{EB}_{\ell}$ [$\mu K^2$]')
    plt.xlim([0, 1500])
    plt.title('Stacked EB power spectrum')
    plt.show()
    # Get the LCDM spectra from CAMB!
    cp = camb.set_params(tau=0.0544, ns=0.9649, H0=67.36, ombh2=0.02237, omch2=0.12, As=2.1e-09, lmax=ell_max)
    camb_results = camb.get_results(cp)
    all_cls_th = camb_results.get_cmb_power_spectra(lmax=ell_max, raw_cl=True, CMB_unit='muK')['total']

    c_l_th_EE_minus_BB = np.zeros((ell_max+1))
    c_l_th_EE_minus_BB = all_cls_th[:, 1] - all_cls_th[:, 2] # EE minus BB power spectrum

    c_l_th_EE_minus_BB_binned = bin_spectrum(n_bins, delta_ell, ell_min, c_l_th_EE_minus_BB)


    # Now we plot the LCDM EE-BB power spectrum rotated by 0.3 deg vs the observed EB power spectrum
    plt.figure()

    plt.errorbar(ell, c_l_EB_o_mean_std[:, 0], yerr=c_l_EB_o_mean_std[:, 1], fmt='.', color='black', label='Observed EB power spectrum')
    plt.plot(ell, np.sin(4 * 0.3 * np.pi/180)/2 * c_l_th_EE_minus_BB_binned, label = r'$\alpha+\beta = 0.3^\circ$', color='red')

    plt.axhline(0, linestyle='--', color='black', alpha=0.5)
    plt.xlabel(r'$\ell$')
    plt.ylabel(r'$C^{EB}_{\ell}$ [$\mu K^2$]')
    plt.xlim([0, 1500])
    plt.title('Stacked EB power spectrum')
    plt.legend(frameon=False)
    plt.show()
    # Use Numba to speed it up! This is the log-likelihood
    @nb.njit()
    def log_prob(alpha_p_beta):
        v_b = c_l_EB_o_mean_std[:, 0] - np.sin(alpha_p_beta * 4 * np.pi/180)/2 * c_l_th_EE_minus_BB_binned

        var_EB = c_l_EB_o_mean_std[:, 1]**2
        return -0.5 * np.sum(v_b**2 / var_EB)

    # 32 chains
    nwalkers = 32
    # 1 parameter (alpha+beta)
    ndim = 1
    p0 = np.random.rand(nwalkers, ndim)
    sampler = emcee.EnsembleSampler(nwalkers, ndim, log_prob)

    # Run for 200 samples as burnin
    state = sampler.run_mcmc(p0, 200, progress=True)
    sampler.reset()

    # Sample for real by using last chain as start
    sampler.run_mcmc(state, 3000, progress=True)
    samples = sampler.get_chain(flat=True)

    # Plot the posterior distribution
    corner.corner(samples, labels=[r'$\alpha+\beta$'], show_titles=True, fontsize=12, color='black', title_fmt='.2f')
    plt.show()

if __name__ == '__main__':
    eskilt_tutorial()
