import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import CubicSpline

a, x = np.loadtxt('../data/xHI.txt', unpack=True)[:2]

num_sim = 128
num_a = 92  # 122 -> 102 -> 92,  padded zeros removed
a = a[:num_a]
x = x.reshape(num_sim, num_a)

lna = np.log(a)
xp = - np.gradient(x, lna, axis=1)  # - dx/dlna

xp_spl = CubicSpline(lna, xp, axis=1)
norm = xp_spl.integrate(lna[0], lna[-1])  # this should ideally be one
#print(norm)

xp_lna_mean_spl = CubicSpline(lna, lna * xp, axis=1)
lna_mean = xp_lna_mean_spl.integrate(lna[0], lna[-1])
lna_mean /= norm
print(lna_mean)
lna_mean = lna_mean.reshape(num_sim, 1)

xp_lna_var_spl = CubicSpline(lna, (lna - lna_mean)**2 * xp, axis=1)
lna_std = xp_lna_var_spl.integrate(lna[0], lna[-1])
lna_std /= norm
print(lna_std)
lna_std = np.sqrt(lna_std)
lna_std = lna_std.reshape(num_sim, 1)

#norm = np.trapz(xp, x=lna, axis=1)  # this should ideally be one
#lna_mean = np.trapz(lna * xp, x=lna, axis=1)
#lna_mean /= norm
#lna_mean = lna_mean.reshape(num_sim, 1)
#lna_std = np.trapz((lna - lna_mean)**2 * xp, x=lna, axis=1)
#lna_std /= norm
#lna_std = np.sqrt(lna_std)
#lna_std = lna_std.reshape(num_sim, 1)

lna_rescaled = (lna - lna_mean) / lna_std
a_rescaled = np.exp(lna_rescaled)
xp_rescaled = xp * lna_std
ar = np.geomspace(a_rescaled.min(), a_rescaled.max())

plt.style.use('../5par.mplstyle')
fig, axes = plt.subplots(nrows=3, sharex=True, gridspec_kw={'wspace': 0, 'hspace': 0},
                         figsize=(3, 4))
axes[0].plot(a_rescaled.T, x.T, c='gray', lw=0.3, alpha=0.2)
axes[0].set_ylabel(r'$x_\mathrm{HI}$')
axes[0].set_ylim(-0.1, 1.1)
axes[1].plot(a_rescaled.T, xp_rescaled.T, c='gray', lw=0.3, alpha=0.2)
axes[1].set_ylabel(r'$\mathrm{d}x_\mathrm{HI}/\mathrm{d}\ln\tilde{a}$')
axes[1].set_yscale('log')
axes[1].set_ylim(1e-3, 0.9)
axes[2].plot(a_rescaled.T, np.log(-np.log(x.T)), c='gray', lw=0.3, alpha=0.2)
#axes[2].plot(ar, np.log(-np.log(0.12985623)) + np.log(ar) * np.log((1 + ar**2) / 0.18488112), lw=0.5)
axes[2].set_ylabel(r'$\ln(-\ln x_\mathrm{HI})$')
axes[2].set_ylim(-10, 4)
axes[2].set_xlabel(r'$\tilde{a}$')
axes[2].set_xscale('log')
axes[2].set_xlim(1e-2, 9)
fig.savefig('shape.pdf')
plt.close(fig)
