import os

import numpy as np
from scipy.stats.qmc import Sobol, discrepancy, scale
import matplotlib.pyplot as plt


def gen_sobol(l_bounds, u_bounds, filename=None, params=None, d=6, m=7, extra=7, seed=0,
              seed_max=65536):

    nicer_seed = seed
    if seed is None:
        disc_min = np.inf
        for s in range(seed_max):
            sampler = Sobol(d, scramble=True, seed=s)  # d is the dimensionality
            sample = sampler.random_base2(m)  # m is the log2 of the number of samples
            disc = discrepancy(sample, method='MD')
            if disc < disc_min:
                nicer_seed = s
                disc_min = disc
        print(f'0 <= seed = {nicer_seed} < {seed_max}, minimizes mixture discrepancy = '
                f'{disc_min}')
        # nicer_seed = 60796, mixture discrepancy = 0.007762706542456144

    sampler = Sobol(d, scramble=True, seed=nicer_seed)
    sample = sampler.random(n=2**m + extra)  # extra is the additional testing samples

    sample = scale(sample, l_bounds, u_bounds)

    if filename is not None:
        np.savetxt(filename, sample, header=' '.join(params))

    return sample


def plt_proj(params, l_bounds, u_bounds, filename, max_rows=None, max_cols=None):
    usecols = range(max_cols) if isinstance(max_cols, int) else max_cols
    sample = np.loadtxt(filename, usecols=usecols, max_rows=max_rows)

    sample = scale(sample, l_bounds, u_bounds, reverse=True)  # undo scaling

    n, d = sample.shape

    axsize = 0.8
    fig, axes = plt.subplots(
        nrows=d,
        ncols=d,
        sharex=True,
        sharey=True,
        squeeze=False,
        subplot_kw={
            'box_aspect': 1,
            'xlim': [0, 1],
            'ylim': [0, 1],
            'xticks': [],
            'yticks': [],
        },
        gridspec_kw={
            'top': 1,
            'left': 0,
            'right': 1,
            'bottom': 0,
            'wspace': 0,
            'hspace': 0,
        },
        figsize=(d * axsize,) * 2,
    )

    for i in range(d):
        for j in range(i):
            axes[i, j].scatter(* sample.T[[j, i]],
                               s=2, marker='o', alpha=0.75, linewidth=0)

        axes[i, i].hist(sample[:, i], bins='sqrt', range=(0, 1),
                        density=True, cumulative=True, histtype='step')
        axes[d-1, i].set_xlabel(params[i])
        axes[i, 0].set_ylabel(params[i])

        for j in range(i+1, d):
            axes[i, j].remove()

    filename = os.path.splitext(filename)[0] + str(n) + '.pdf'
    fig.savefig(filename)
    plt.close(fig)


if __name__ == '__main__':
    params = (r'$\sigma_8$', r'$n_\mathrm{s}$', '$h$',
              r'$\Omega_\mathrm{b}$', r'$\Omega_\mathrm{m}$', r'$\zeta_\mathrm{UV}$')

    l_bounds = (.74, .92, .61, .04, .24, 20)
    u_bounds = (.90, 1.00, .73, .06, .40, 35)
    filename = 'sobol.txt'
    if not os.path.exists(filename):
        gen_sobol(l_bounds, u_bounds, filename=filename, params=params)

    l_bounds = (.7811, .9436, .6466, .04508, .2788, 20)
    u_bounds = (.8411, .9856, .7066, .05358, .3518, 35)
    filename = 'sobol_core.txt'
    if not os.path.exists(filename):
        gen_sobol(l_bounds, u_bounds, filename=filename, params=params)

    plt.style.use('../5par.mplstyle')

    #plt_proj(filename, max_rows=64)
    plt_proj(params, l_bounds, u_bounds, filename, max_rows=128)
