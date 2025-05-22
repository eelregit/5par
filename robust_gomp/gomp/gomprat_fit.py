from functools import partial

import numpy as np
from scipy.optimize import minimize
import matplotlib.pyplot as plt


a_edge, x_edge = np.loadtxt('../data/xHI.txt', unpack=True)[:2]
a_core, x_core = np.loadtxt('../data/xHI_core.txt', unpack=True)[:2]
a = np.concatenate((a_edge, a_core))
x = np.concatenate((x_edge, x_core))

num_sim = 512 + 512
num_a = 127
a = a.reshape(num_sim, num_a)
x = x.reshape(num_sim, num_a)
lna = np.log(a)
print(f'{lna.min()=}, {lna.max()=}')
xp = - np.gradient(x, lna[0], axis=1)  # - dx/dlna
print(f'prepending x_HI=1 at lna=(-200, -100, -50, -20, -10) and appending x_HI=0 at lna=(1, 2, 5, 10)')
#_lna_, _x_ = lna, x  # was easier for me to find the working initial guess without the paddings
_lna_ = np.concatenate([np.full((num_sim, 5), (-200, -100, -50, -20, -10)), lna,
                        np.full((num_sim, 4), (1, 2, 5, 10))], axis=1)
_x_ = np.concatenate([np.ones((num_sim, 5)), x, np.zeros((num_sim, 4))], axis=1)
num_a += 9

# THESAN reionization history is available at
# https://www.thesan-project.com/quantities/reion_history_Thesan1.dat
z_ts, x_ts = np.loadtxt('reion_history_Thesan1.dat', unpack=True)[:2]
z_ts = z_ts[None]  # prepending the num_sim axis, to be able to reuse the functions
x_ts = x_ts[None]
a_ts = 1 / (1 + z_ts)
lna_ts = np.log(a_ts)


def rescale_lna(lna, lna_pivot, tilt):
    if lna_pivot is None or tilt is None:
        return lna
    if np.ndim(lna) == 2:  # shape = num_sim, num_a
        if np.ndim(lna_pivot) == 1:  # shape = num_sim
            lna_pivot = lna_pivot[:, None]
        if np.ndim(tilt) == 1:  # shape = num_sim
            tilt = tilt[:, None]
    return (lna - lna_pivot) * tilt


def rat(lna, c, lna_pivot=None, tilt=None, deriv=False):
    # the const and linear coeffs can be absorbed by the pivot and tilt of each curve
    lna_rescaled = rescale_lna(lna, lna_pivot, tilt)
    numerator = lna_rescaled * (1 + lna_rescaled * (c[0] + lna_rescaled * c[1]))
    denominator = 1 + lna_rescaled * (c[2] + lna_rescaled * c[3])

    if not deriv:
        return numerator / denominator  # shape = num_sim, num_a

    # rational function derivative wrt c
    if deriv is Ellipsis:
        return np.stack([
            lna_rescaled ** 2 / denominator,
            lna_rescaled ** 3 / denominator,
            - numerator * lna_rescaled / denominator ** 2,
            - numerator * (lna_rescaled / denominator) ** 2,
        ], axis=0)

    # rational function derivative wrt rescaled lna
    numerator_deriv = 1 + lna_rescaled * (2 * c[0] + lna_rescaled * (3 * c[1]))
    denominator_deriv = c[2] + lna_rescaled * (2 * c[3])
    return ((numerator_deriv - numerator * denominator_deriv / denominator)
            / denominator)


def unpack_params(x):
    """Unpack concatenated local pivots, local tilts, and global rational function coeffs."""
    lna_pivot = np.array(x[:num_sim])
    tilt = np.array(x[num_sim:2*num_sim])
    c = np.array(x[2*num_sim:])
    return c, lna_pivot, tilt


def gomprat_obj(x, *args):
    c, lna_pivot, tilt = unpack_params(x)
    lna, x = args
    return np.square(gomprat(lna, c, lna_pivot, tilt) - x).sum()


def gomprat_objjac(x, *args):
    c, lna_pivot, tilt = unpack_params(x)
    lna, x = args

    exponentials = np.exp(rat(lna, c, lna_pivot, tilt))
    gompertz = np.exp(- exponentials)
    fac = 2 * (x - gomprat(lna, c, lna_pivot, tilt))
    fac *= gompertz * exponentials
    fac[np.isnan(fac)] = 0

    rat_deriv = rat(lna, c, lna_pivot, tilt, deriv=True)
    lna_pivot_deriv = fac * rat_deriv * rescale_lna(np.zeros((1, 1)), 1, tilt)  # HACK
    lna_pivot_deriv = lna_pivot_deriv.sum(axis=1)
    tilt_deriv = fac * rat_deriv * rescale_lna(lna, lna_pivot, 1)  # HACK
    tilt_deriv = tilt_deriv.sum(axis=1)

    c_deriv = fac * rat(lna, c, lna_pivot, tilt, deriv=...)
    c_deriv = c_deriv.sum(axis=(1, 2))

    derivatives = np.concatenate([lna_pivot_deriv, tilt_deriv, c_deriv])
    return derivatives


def gomprat_obj_fixing_c(c):
    """A factory returning gomprat_obj's similar to above but allowing c to be fixed
    and only fitting to the other 2. Useful for fitting to THESAN-1 simulation."""
    def gomprat_obj(x, *args):
        lna_pivot, tilt = x
        lna, x = args
        return np.square(gomprat(lna, c, lna_pivot, tilt) - x).sum()
    return gomprat_obj


def gomprat(lna, c, lna_pivot=None, tilt=None):
    gompertz = np.exp(- np.exp(rat(lna, c, lna_pivot, tilt)))
    return gompertz  # shape = num_sim, num_a


def gomprat_deriv(lna, c, lna_pivot=None, tilt=None):
    """- dx / dlna_rescaled."""
    exponentials = np.exp(rat(lna, c, lna_pivot, tilt))
    gompertz = np.exp(- exponentials)
    derivatives = gompertz * exponentials * rat(lna, c, lna_pivot, tilt, deriv=True)
    derivatives[np.isnan(exponentials)] = 0
    return derivatives  # shape = num_sim, num_a


def fit(_lna_, _x_):
    # fit to 21cmFAST simulations
    lna_pivot = np.full(num_sim, -2, dtype=float)
    tilt = np.full(num_sim, 7, dtype=float)
    # quadratic and cubic coeffs for numerator
    # followed by linear and quadratic ones for denominator
    c = np.array([0.1, 0, 0.1, 0.1])
    x0 = np.concatenate([lna_pivot, tilt, c])
    res = minimize(gomprat_obj, x0, args=(_lna_, _x_), method='BFGS', jac=gomprat_objjac)
    c, lna_pivot, tilt = unpack_params(res.x)

    print(f'{c = }')
    print(f'{lna_pivot = }')
    print(f'{tilt = }')
    print(f'{res.success = }')
    print(f'{res.status = }')
    print(f'{res.message = }')
    print(f'{res.fun = }  # chi-squared')
    print(f'{res.nfev = }')
    print(f'{res.nit = }')

    # fit to THESAN-1 simulation
    lna_pivot_ts, tilt_ts = 0, 1
    x0 = np.array([lna_pivot_ts, tilt_ts], dtype=float)
    res_ts = minimize(gomprat_obj_fixing_c(c), x0, args=(lna_ts, x_ts), method='Nelder-Mead')
    lna_pivot_ts, tilt_ts = res_ts.x

    print(f'{lna_pivot_ts = }')
    print(f'{tilt_ts = }')
    print(f'{res_ts.success = }')
    print(f'{res_ts.status = }')
    print(f'{res_ts.message = }')
    print(f'{res_ts.fun = }  # chi-squared')
    print(f'{res_ts.nfev = }')
    print(f'{res_ts.nit = }')

    return c, lna_pivot, tilt, lna_pivot_ts, tilt_ts


def plot(lna, c, lna_pivot, tilt, lna_pivot_ts, tilt_ts, x, xp):
    lna_rescaled = rescale_lna(lna, lna_pivot, tilt)
    a_rescaled = np.exp(lna_rescaled)
    xp_rescaled = xp / tilt[:, None]

    lna_ts_rescaled = rescale_lna(lna_ts, lna_pivot_ts, tilt_ts)
    a_ts_rescaled = np.exp(lna_ts_rescaled)

    ar = np.geomspace(1e-4, 10, num=101)  # for analytic curves

    plt.style.use('../5par.mplstyle')
    fig, axes = plt.subplots(nrows=3, sharex=True,
                             gridspec_kw={'wspace': 0, 'hspace': 0}, figsize=(3, 4))

    axes[0].plot(a_rescaled.T, x.T, c='gray', lw=0.3, alpha=0.2, zorder=1.8)
    axes[0].plot(ar, gomprat(np.log(ar), c).T, c='C0', lw=1, zorder=2)
    #axes[0].plot(a_ts_rescaled[0], x_ts[0], c='C1', ls=':', lw=2, alpha=0.5, zorder=1.9)
    axes[0].set_ylabel(r'$x_\mathrm{HI}$')
    axes[0].set_ylim(-0.1, 1.1)

    axes[1].plot(a_rescaled.T, xp_rescaled.T, c='gray', lw=0.3, alpha=0.2)
    axes[1].plot(ar, gomprat_deriv(np.log(ar), c).T, c='C0', lw=1)
    axes[1].set_ylabel(r'$- \mathrm{d}x_\mathrm{HI}/\mathrm{d}\ln\tilde{a}$')
    axes[1].set_yscale('log')
    axes[1].set_ylim(1e-3, 0.9)

    axes[2].plot(a_rescaled.T, np.log(-np.log(x.T)), c='gray', lw=0.3, alpha=0.2)
    axes[2].plot(ar, rat(np.log(ar), c).T, c='C0', lw=1)
    axes[2].set_ylabel(r'$\ln(-\ln x_\mathrm{HI})$')
    axes[2].set_ylim(-9, 4)
    axes[2].set_xlabel(r'$\tilde{a}$')
    axes[2].set_xscale('log')
    axes[2].set_xlim(2e-3, 8)

    fig.savefig('shape_rat.pdf')
    plt.close(fig)


if __name__ == '__main__':
    print('#'*32,
          'rational function of the form (x + c0 x^2 + c1 x^3) / (1 + c2 x + c3 x^2)',
          '#'*32)

    c, lna_pivot, tilt, lna_pivot_ts, tilt_ts = fit(_lna_, _x_)

    np.savetxt('pivottilt_rat.txt', np.stack([lna_pivot, tilt], axis=1))

    plot(lna, c, lna_pivot, tilt, lna_pivot_ts, tilt_ts, x, xp)
