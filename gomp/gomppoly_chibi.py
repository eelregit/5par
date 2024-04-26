import numpy as np
from numpy.polynomial import Polynomial
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit


a, x = np.loadtxt('../data/xHI.txt', unpack=True)[:2]

num_sim = 128
num_a = 92  # 122 -> 102 -> 92,  padded zeros removed
a = a.reshape(num_sim, num_a)
x = x.reshape(num_sim, num_a)
lna = np.log(a)
xp = - np.gradient(x, lna[0], axis=1)  # - dx/dlna


def rescale_lna(lna, lna_pivot, tilt):
    if lna_pivot is None or tilt is None:
        return lna
    return (lna - lna_pivot[:, None]) * tilt[:, None]  # shape = num_sim, num_a


def poly(lna, c, lna_pivot=None, tilt=None):
    # the const and linear coeffs can be absorbed by the pivot and tilt of each curve
    c = np.concatenate([[0, 1], c])
    return Polynomial(c)(rescale_lna(lna, lna_pivot, tilt))  # shape = num_sim, num_a


def poly_deriv(lna, c, lna_pivot=None, tilt=None):
    c = np.concatenate([[0, 1], c])
    return Polynomial(c).deriv()(rescale_lna(lna, lna_pivot, tilt))  # shape = num_sim, num_a


def unpack_params(p):
    lna_pivot = np.array(p[:num_sim])  # curve_fit probably cannot handle array params
    tilt = np.array(p[num_sim:2*num_sim])
    c = np.array(p[2*num_sim:])
    return c, lna_pivot, tilt


def gomppoly_ravel(lna, *p):
    c, lna_pivot, tilt = unpack_params(p)
    return gomppoly(lna, c, lna_pivot, tilt).ravel()


def gomppoly(lna, c, lna_pivot=None, tilt=None):
    gompertz = np.exp(- np.exp(poly(lna, c, lna_pivot, tilt)))
    return gompertz  # shape = num_sim, num_a


def gomppoly_deriv(lna, c, lna_pivot=None, tilt=None):
    """- dx / dlna."""
    exponentials = np.exp(poly(lna, c, lna_pivot, tilt))
    gompertz = np.exp(- exponentials)
    derivatives = gompertz * exponentials * poly_deriv(lna, c, lna_pivot, tilt)
    return derivatives  # shape = num_sim, num_a


def fit(degree, lna, x):
    c = np.zeros(degree - 2)
    lna_pivot = np.zeros(num_sim)
    tilt = np.ones(num_sim)
    p0 = np.concatenate([lna_pivot, tilt, c])

    popt, pcov = curve_fit(gomppoly_ravel, lna, x.ravel(), p0)

    c, lna_pivot, tilt = unpack_params(popt)
    return c, lna_pivot, tilt


def plot(lna, c, lna_pivot, tilt, x, xp):
    lna_rescaled = rescale_lna(lna, lna_pivot, tilt)
    a_rescaled = np.exp(lna_rescaled)
    xp_rescaled = xp / tilt[:, None]

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
plot(lna, c, lna_pivot, tilt, x, xp)
