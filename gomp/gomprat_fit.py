from functools import partial

import numpy as np
import jax.numpy as jnp
import jax
from jax.scipy.optimize import minimize
import matplotlib.pyplot as plt


jax.config.update('jax_enable_x64', True)


a_edge, x_edge = np.loadtxt('../data/xHI.txt', unpack=True)[:2]
a_core, x_core = np.loadtxt('../data/xHI_core.txt', unpack=True)[:2]
a = np.concatenate((a_edge, a_core))
x = np.concatenate((x_edge, x_core))
a = jnp.array(a)
x = jnp.array(x)

num_sim = 128 + 128
num_a = 127
a = a.reshape(num_sim, num_a)
x = x.reshape(num_sim, num_a)
lna = jnp.log(a)
print(f'{lna.min()=}, {lna.max()=}')
xp = - jnp.gradient(x, lna[0], axis=1)  # - dx/dlna
print(f'prepending x_HI=1 at lna=(-200, -100, -50, -20, -10) and appending x_HI=0 at lna=(1, 2, 5, 10)')
_lna_ = jnp.concatenate(
    [
        jnp.full((num_sim, 5), jnp.array([-200, -100, -50, -20, -10])),
        lna,
        jnp.full((num_sim, 4), jnp.array([1, 2, 5, 10])),
    ],
axis=1)
_x_ = jnp.concatenate([jnp.ones((num_sim, 5)), x, jnp.zeros((num_sim, 4))], axis=1)
num_a += 9

# THESAN reionization history is available at
# https://www.thesan-project.com/quantities/reion_history_Thesan1.dat
z_ts, x_ts = np.loadtxt('reion_history_Thesan1.dat', unpack=True)[:2]
z_ts, x_ts = jnp.asarray(z_ts), jnp.asarray(x_ts)
z_ts = z_ts[None]  # prepending the num_sim axis, to be able to reuse the functions
x_ts = x_ts[None]
a_ts = 1 / (1 + z_ts)
lna_ts = jnp.log(a_ts)


def rescale_lna(lna, lna_pivot, tilt):
    if lna_pivot is None or tilt is None:
        return lna
    if lna.ndim == 2:  # shape = num_sim, num_a
        if lna_pivot.ndim == 1:  # shape = num_sim
            lna_pivot = lna_pivot[:, None]
        if tilt.ndim == 1:  # shape = num_sim
            tilt = tilt[:, None]
    return (lna - lna_pivot) * tilt


def rat(lna, c, lna_pivot=None, tilt=None):
    # the const and linear coeffs can be absorbed by the pivot and tilt of each curve
    lna_rescaled = rescale_lna(lna, lna_pivot, tilt)
    numerator = lna_rescaled * (1 + lna_rescaled * (c[0] + lna_rescaled * c[1]))
    denominator = 1 + lna_rescaled * (c[2] + lna_rescaled * c[3])
    return numerator / denominator  # shape = num_sim, num_a


def unpack_params(x):
    """Unpack concatenated local pivots, local tilts, and global rational function coeffs."""
    lna_pivot = jnp.array(x[:num_sim])
    tilt = jnp.array(x[num_sim:2*num_sim])
    c = jnp.array(x[2*num_sim:])
    return c, lna_pivot, tilt


def gomprat_obj(x, *args):
    c, lna_pivot, tilt = unpack_params(x)
    lna, x = args
    return jnp.square(gomprat(lna, c, lna_pivot, tilt) - x).sum()


def gomprat_obj_fixing_c(c):
    """A factory returning gomprat_obj's similar to above but allowing c to be fixed
    and only fitting to the other 2. Useful for fitting to THESAN-1 simulation."""
    def gomprat_obj(x, *args):
        lna_pivot, tilt = x
        lna, x = args
        return jnp.square(gomprat(lna, c, lna_pivot, tilt) - x).sum()
    return gomprat_obj


def gomprat(lna, c, lna_pivot=None, tilt=None):
    gompertz = jnp.exp(- jnp.exp(rat(lna, c, lna_pivot, tilt)))
    return gompertz  # shape = num_sim, num_a


@partial(jnp.vectorize, excluded=(1, 2, 3))
def gomprat_deriv(lna, c, lna_pivot, tilt):
    """- dx / dlna_rescaled."""
    deriv = jax.grad(gomprat)(lna, c, lna_pivot, tilt)
    return deriv / tilt


def fit(_lna_, _x_):
    # fit to 21cmFAST simulations
    lna_pivot = jnp.full(num_sim, -2, dtype=float)
    tilt = jnp.full(num_sim, 7, dtype=float)
    # quadratic and cubic coeffs for numerator
    # followed by linear and quadratic ones for denominator
    c = jnp.zeros(4)
    x0 = jnp.concatenate([lna_pivot, tilt, c])
    res = minimize(gomprat_obj, x0, args=(_lna_, _x_), method='BFGS')
    c, lna_pivot, tilt = unpack_params(res.x)

    print(f'{c = }')
    print(f'{lna_pivot = }')
    print(f'{tilt = }')
    print(f'{res.success = }')
    print(f'{res.status = }')
    print(f'{res.fun = }  # chi-squared')
    print(f'{res.nfev = }')
    print(f'{res.nit = }')

    # fit to THESAN-1 simulation
    lna_pivot_ts, tilt_ts = 0, 1
    x0 = jnp.array([lna_pivot_ts, tilt_ts], dtype=float)
    res_ts = minimize(gomprat_obj_fixing_c(c), x0, args=(lna_ts, x_ts), method='BFGS')
    lna_pivot_ts, tilt_ts = res_ts.x

    print(f'{lna_pivot_ts = }')
    print(f'{tilt_ts = }')
    print(f'{res_ts.success = }')
    print(f'{res_ts.status = }')
    print(f'{res_ts.fun = }  # chi-squared')
    print(f'{res_ts.nfev = }')
    print(f'{res_ts.nit = }')

    return c, lna_pivot, tilt, lna_pivot_ts, tilt_ts


def plot(lna, c, lna_pivot, tilt, lna_pivot_ts, tilt_ts, x, xp):
    lna_rescaled = rescale_lna(lna, lna_pivot, tilt)
    a_rescaled = jnp.exp(lna_rescaled)
    xp_rescaled = xp / tilt[:, None]

    lna_ts_rescaled = rescale_lna(lna_ts, lna_pivot_ts, tilt_ts)
    a_ts_rescaled = jnp.exp(lna_ts_rescaled)

    ar = jnp.geomspace(1e-4, 10, num=101)  # for analytic curves

    plt.style.use('../5par.mplstyle')
    fig, axes = plt.subplots(nrows=3, sharex=True,
                             gridspec_kw={'wspace': 0, 'hspace': 0}, figsize=(3, 4))

    axes[0].plot(a_rescaled.T, x.T, c='gray', lw=0.3, alpha=0.2, zorder=1.8)
    axes[0].plot(ar, gomprat(jnp.log(ar), c).T, c='C0', lw=1, zorder=2)
    #axes[0].plot(a_ts_rescaled[0], x_ts[0], c='C1', ls=':', lw=2, alpha=0.5, zorder=1.9)
    axes[0].set_ylabel(r'$x_\mathrm{HI}$')
    axes[0].set_ylim(-0.1, 1.1)

    axes[1].plot(a_rescaled.T, xp_rescaled.T, c='gray', lw=0.3, alpha=0.2)
    axes[1].plot(ar, gomprat_deriv(jnp.log(ar), c, 0, 1).T, c='C0', lw=1)
    axes[1].set_ylabel(r'$- \mathrm{d}x_\mathrm{HI}/\mathrm{d}\ln\tilde{a}$')
    axes[1].set_yscale('log')
    axes[1].set_ylim(1e-3, 0.9)

    axes[2].plot(a_rescaled.T, jnp.log(-jnp.log(x.T)), c='gray', lw=0.3, alpha=0.2)
    axes[2].plot(ar, rat(jnp.log(ar), c).T, c='C0', lw=1)
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

    np.savetxt('pivottilt_rat.txt', np.array(jnp.stack([lna_pivot, tilt], axis=1)))

    plot(lna, c, lna_pivot, tilt, lna_pivot_ts, tilt_ts, x, xp)
