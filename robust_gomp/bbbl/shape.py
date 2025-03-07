import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import CubicSpline


def plot_shape(infiles, outfile):
    plt.style.use('../5par.mplstyle')
    fig, axes = plt.subplots(nrows=3, sharex=True,
                            gridspec_kw={'wspace': 0, 'hspace': 0},
                            figsize=(3, 4))

    for infile in infiles:
        x = np.loadtxt(infile)[3:]  # skipping a_e, b, x(R=0)=1
        R = np.arange(1, 1+len(x))
        lnR = np.log(R)
        neg_x_spl = CubicSpline(lnR, - x)
        xp = neg_x_spl.derivative()(lnR)  # - dx/dlnR

        xp_spl = CubicSpline(lnR, xp)
        norm = xp_spl.integrate(lnR[0], lnR[-1])
        print(norm)  # this should be very close to one

        xp_lnR_mean_spl = CubicSpline(lnR, lnR * xp)
        lnR_mean = xp_lnR_mean_spl.integrate(lnR[0], lnR[-1])
        lnR_mean /= norm
        print(lnR_mean)

        xp_lnR_var_spl = CubicSpline(lnR, (lnR - lnR_mean)**2 * xp)
        lnR_std = xp_lnR_var_spl.integrate(lnR[0], lnR[-1])
        lnR_std /= norm
        print(lnR_std)
        lnR_std = np.sqrt(lnR_std)

        lna_rescaled = (lnR - lnR_mean) / lnR_std
        a_rescaled = np.exp(lna_rescaled)
        xp_rescaled = xp * lnR_std

        axes[0].plot(a_rescaled, x, c='gray', lw=0.5, alpha=0.5)
        axes[1].plot(a_rescaled, xp_rescaled, c='gray', lw=0.5, alpha=0.5)
        axes[2].plot(a_rescaled, np.log(-np.log(x)), c='gray', lw=0.5, alpha=0.5)

    axes[0].set_ylabel(r'$x_\mathrm{HI}$')
    axes[0].set_ylim(-0.1, 1.1)
    axes[1].set_ylabel(r'$\mathrm{d}x_\mathrm{HI}/\mathrm{d}\ln\tilde{a}$')
    axes[1].set_yscale('log')
    axes[1].set_ylim(1e-3, 0.9)
    axes[2].set_ylabel(r'$\ln(-\ln x_\mathrm{HI})$')
    axes[2].set_ylim(-10, 4)
    axes[2].set_xlabel(r'$\tilde{a}$')
    axes[2].set_xscale('log')
    axes[2].set_xlim(1e-2, 9)

    fig.savefig(outfile)
    plt.close(fig)


if __name__ == '__main__':
    import glob
    plot_shape(glob.glob('512_MC_*.txt'), 'shape_MC.pdf')
    plot_shape(glob.glob('512_LSS_*.txt'), 'shape_LSS.pdf')
    plot_shape(glob.glob('512_Grid_*.txt'), 'shape_Grid.pdf')
    plot_shape(glob.glob('512_QMC_*.txt'), 'shape_QMC.pdf')
