from functools import partial

import numpy as np
from numpy.polynomial import Polynomial
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

from gomppoly_6 import poly6, gomppoly6


# THESAN reionization history is available at
# https://www.thesan-project.com/quantities/reion_history_Thesan1.dat

z, x = np.loadtxt('reion_history_Thesan1.dat', unpack=True)[:2]

a = 1 / (1 + z)
lna = np.log(a)
#
#
#def gomppoly6_(lna, lna_pivot, tilt):
#    return gomppoly6(lna, lna_pivot=lna_pivot, tilt=tilt, params=None)


def fit(lna, x):
    lna_pivot, tilt = 0, 1
    p0 = lna_pivot, tilt

    popt, pcov = curve_fit(gomppoly6, lna, x, p0)

    lna_pivot, tilt = popt
    return lna_pivot, tilt


def plot(lna, lna_pivot, tilt, x):
    a_ = np.geomspace(1e-2, 1, num=101)  # for analytic curves

    plt.style.use('../5par.mplstyle')
    fig, axes = plt.subplots(nrows=2, sharex=True,
                             gridspec_kw={'wspace': 0, 'hspace': 0}, figsize=(4, 5))

    axes[0].plot(a, x, c='gray')
    axes[0].plot(a_, gomppoly6(np.log(a_), lna_pivot, tilt), c='C0', ls='--')
    axes[0].set_ylabel(r'$x_\mathrm{HI}$')
    axes[0].set_ylim(-0.1, 1.1)

    axes[1].plot(a, np.log(-np.log(x)), c='gray')
    axes[1].plot(a_, poly6(np.log(a_), lna_pivot, tilt), c='C0', ls='--')
    axes[1].set_ylabel(r'$\ln(-\ln x_\mathrm{HI})$')
    axes[1].set_ylim(-9, 4)
    axes[1].set_xlabel(r'$\tilde{a}$')
    axes[1].set_xscale('log')
    axes[1].set_xlim(a.min(), a.max())

    fig.savefig('thesan_fit.pdf')
    plt.close(fig)


if __name__ == '__main__':
    lna_pivot, tilt = fit(lna, x)
    print(f'{lna_pivot=}, {tilt=}', flush=True)
    plot(lna, lna_pivot, tilt, x)
