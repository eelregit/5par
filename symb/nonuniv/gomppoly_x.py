import numpy as np
from numpy.polynomial import Polynomial
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.colors import Normalize


#var_names = ['s8', 'ns', 'h', 'Ob', 'Om', 'zt']
#var_cols = [2, 3, 4, 5, 6, 7]
a, x, *params = np.loadtxt('../xHI.txt', unpack=True)[:8]
params = np.stack(params, axis=0)

num_sim = 128
num_a = 92  # 122 -> 102 -> 92,  padded zeros removed
a = a.reshape(num_sim, num_a)
x = x.reshape(num_sim, num_a)
params = params.reshape(6, num_sim, num_a)[..., 0]
lna = np.log(a)
xp = - np.gradient(x, lna[0], axis=1)  # - dx/dlna


lna_pivot, tilt = np.loadtxt('../pivottilt_6.txt', unpack=True)


def rescale_lna(lna, lna_pivot, tilt):
    return (lna - lna_pivot[:, None]) * tilt[:, None]  # shape = num_sim, num_a


def plot(lna, lna_pivot, tilt, x, xp, param):
    lna_rescaled = rescale_lna(lna, lna_pivot, tilt)
    a_rescaled = np.exp(lna_rescaled)
    xp_rescaled = xp / tilt[:, None]

    s8, ns, h, Ob, Om, zt = params
    param = Ob / Om
    #param_label = '$\sigma_8$'
    #param_label = '$n_s$'
    #param_label = '$h$'
    #param_label = '$\Omega_\mathrm{b}$'
    param_label = '$\Omega_\mathrm{b} / \Omega_\mathrm{m}$'
    #param_label = '$\zeta_\mathrm{eff}$'

    #norm = Normalize(0.74, 0.9)   # s8
    #norm = Normalize(0.92, 1)     # ns
    #norm = Normalize(0.61, 0.73)  # h
    #norm = Normalize(0.04, 0.06)  # Ob
    #norm = Normalize(0.24, 0.4)   # Om
    #norm = Normalize(15, 30)      # zt
    norm = Normalize(param.min(), param.max())

    plt.style.use('../5par.mplstyle')
    fig, axes = plt.subplots(nrows=3, sharex=True,
                             gridspec_kw={'wspace': 0, 'hspace': 0}, figsize=(3.5, 4))

    lc = LineCollection(np.stack([a_rescaled, x], axis=-1), cmap='RdBu_r', norm=norm, lw=0.3, alpha=0.3)
    lc.set_array(param)
    lines = axes[0].add_collection(lc)
    axes[0].set_ylabel(r'$x_\mathrm{HI}$')
    axes[0].set_ylim(-0.1, 1.1)

    lc = LineCollection(np.stack([a_rescaled, xp_rescaled], axis=-1), cmap='RdBu_r', norm=norm, lw=0.3, alpha=0.3)
    lc.set_array(param)
    lines = axes[1].add_collection(lc)
    axes[1].set_ylabel(r'$- \mathrm{d}x_\mathrm{HI}/\mathrm{d}\ln a_\mathrm{r}$')
    axes[1].set_yscale('log')
    axes[1].set_ylim(1e-3, 0.9)

    lc = LineCollection(np.stack([a_rescaled, np.log(-np.log(x))], axis=-1), cmap='RdBu_r', norm=norm, lw=0.3, alpha=0.3)
    lc.set_array(param)
    lines = axes[2].add_collection(lc)
    axes[2].set_ylabel(r'$\ln(-\ln x_\mathrm{HI})$')
    axes[2].set_ylim(-9, 4)
    axes[2].set_xlabel(r'$a_\mathrm{r}$')
    axes[2].set_xscale('log')
    axes[2].set_xlim(2e-3, 8)

    cbar = fig.colorbar(lines, ax=axes)
    cbar.ax.set_xlabel(param_label)

    fig.savefig(f'shape_Lambda.pdf')
    plt.close(fig)


degree = 6
plot(lna, lna_pivot, tilt, x, xp, params)
