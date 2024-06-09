import numpy as np
import matplotlib.pyplot as plt

from gomppoly_fit import lna, x, rescale_lna, poly, gomppoly, fit


def plot(lna, c, lna_pivot, tilt, x):
    lna_rescaled = rescale_lna(lna, lna_pivot, tilt)
    a_rescaled = np.exp(lna_rescaled)

    ar = np.geomspace(1e-4, 10, num=101)  # for analytic curves

    plt.style.use('../5par.mplstyle')
    plt.rcParams['xtick.bottom'] = False
    plt.rcParams['xtick.labelbottom'] = False
    plt.rcParams['ytick.left'] = False
    plt.rcParams['ytick.labelleft'] = False
    plt.rcParams['xtick.top'] = False
    plt.rcParams['ytick.right'] = False
    fig, axes = plt.subplots(nrows=2, sharex=True,
                             gridspec_kw={'wspace': 0, 'hspace': 0}, figsize=(1.5, 2))
    axes[0].plot(a_rescaled.T, x.T, c='gray', lw=0.3, alpha=0.2)
    axes[0].plot(ar, gomppoly(np.log(ar), c).T, c='C0', lw=0.5)
    axes[0].set_ylim(-0.1, 1.1)

    axes[1].plot(a_rescaled.T, np.log(-np.log(x.T)), c='gray', lw=0.3, alpha=0.2)
    axes[1].plot(ar, poly(np.log(ar), c).T, c='C0', lw=0.5)
    axes[1].set_ylim(-9, 4)
    axes[1].set_xscale('log')
    axes[1].set_xlabel(r'$(a/\alpha)^\beta$')
    axes[1].set_xlim(2e-3, 8)
    plt.savefig('shape_chibi.pdf')


c, lna_pivot, tilt = fit(6, lna, x)
plot(lna, c, lna_pivot, tilt, x)
