import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

from gomppoly_6 import poly6, gomppoly6


# THESAN reionization history is available at
# https://www.thesan-project.com/quantities/reion_history_Thesan1.dat
z, x = np.loadtxt('reion_history_Thesan1.dat', unpack=True)[:2]
a = 1 / (1 + z)
lna = np.log(a)


def fit(lna, x):
    lna_pivot, tilt = 0, 1
    p0 = lna_pivot, tilt

    popt, pcov = curve_fit(gomppoly6, lna, x, p0)

    lna_pivot, tilt = popt
    return lna_pivot, tilt


def plot(lna, lna_pivot, tilt, x):
    a = np.exp(lna)
    a_ = np.geomspace(1e-2, 1, num=101)  # for analytic curves

    plt.style.use('../5par.mplstyle')
    fig, axes = plt.subplots(nrows=2, sharex=True,
                             gridspec_kw={'height_ratios': [2, 1], 'wspace': 0,
                                          'hspace': 0},
                             figsize=(4, 4))

    axes[0].plot(a, x, c='gray', label='THESAN simulation')
    axes[0].plot(a_, gomppoly6(np.log(a_), lna_pivot, tilt), c='C0', ls='--',
                 label='fit with universal shape')
    axes[0].set_ylabel(r'$x_\mathrm{HI}$')
    axes[0].set_ylim(-0.1, 1.1)
    axes[0].legend()

    axes[1].axhline(0, c='gray')
    axes[1].plot(a, gomppoly6(lna, lna_pivot, tilt) - x, c='C0', ls='--')
    axes[1].set_ylabel('abs. diff.')
    axes[1].set_ylim(-0.015, 0.015)
    axes[1].set_xlabel(r'$a$')
    axes[1].set_xscale('log')
    axes[1].set_xlim(a.min(), a.max())

    fig.savefig('thesan_fit.pdf')
    plt.close(fig)


if __name__ == '__main__':
    lna_pivot, tilt = fit(lna, x)
    print(f'{lna_pivot=}, {tilt=}', flush=True)
    plot(lna, lna_pivot, tilt, x)
