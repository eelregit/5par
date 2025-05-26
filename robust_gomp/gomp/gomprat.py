from functools import partial

import numpy as np


def reparam(h, omega_b, omega_cdm):
    Ob = omega_b / h**2
    Om = omega_cdm / h**2 + Ob
    return Ob, Om


def rat(lna, lna_pivot=None, tilt=None, params=None):
    # the const and linear coeffs can be absorbed by the pivot and tilt of each curve
    def R(lna_rescaled):
        return (
            lna_rescaled * (1 + lna_rescaled * (0.12748899 + lna_rescaled * 0.10220648))
            / (1 + (lna_rescaled * (-0.00604965 + lna_rescaled * 0.0815222)))
        )

    if params is not None:
        lna_pivot = pivot_sr(**params)
        tilt = tilt_sr(**params)

    if lna_pivot is None or tilt is None:
        return R(lna)

    return R((lna - lna_pivot) * tilt)


def gomprat(lna, lna_pivot=None, tilt=None, params=None):
    return np.exp(- np.exp(rat(lna, lna_pivot=lna_pivot, tilt=tilt, params=params)))


def pivot_sr(h, omega_b, omega_cdm, sigma8, n_s, zt, Tv, LX, fit):
    Ob, Om = reparam(h, omega_b, omega_cdm)
    s8, ns = sigma8, n_s

    raise NotImplementedError('Discuss about if we want to or should do SR again')
    if fit == 'rgomp1': # Robust gomp complexity 34
        return ((((Tv - np.log((((Tv ** (-40.065258 + LX)) + zt) + np.exp(s8)) ** s8)) / 2.5946999) - np.exp(ns - (Ob ** Om))) - ((h ** (Om ** 0.79247624)) / (0.8632819 ** ns)))

    if fit == 'rgomp2': # Robust gomp complexity 27
        return (((Ob / (Om * s8)) ** (s8 * h)) - (((((19.920519 + zt) + (0.08890569 ** (40.43043 - LX))) * s8) ** (ns / Tv)) + Om))

    raise ValueError(f'{fit=} not recognized')


def tilt_sr(h, omega_b, omega_cdm, sigma8, n_s, zt, Tv, LX, fit):
    Ob, Om = reparam(h, omega_b, omega_cdm)
    s8, ns = sigma8, n_s

    raise NotImplementedError('Discuss about if we want to or should do SR again')
    if fit == 'rgomp1': # Robust gomp complexity 24
        return (((zt / h) ** 0.4991587) - (((((Ob + -0.20859116) * LX) - -6.0811667) - (Om * (Tv - np.exp(ns ** 2.8035064)))) / s8))

    if fit == 'rgomp2': # Robust gomp complexity 26
        return (((((0.12804393 - Ob) / h) * (zt + ((Tv - ((s8 / Om) + 1.0161631)) / Om))) - (ns**3.052106)) + np.exp(0.039550677 * LX))

    raise ValueError(f'{fit=} not recognized')


def tanh(z, z_reio):
    """Planck 2018 VI footnote 15"""
    y, y_reio = (1 + z) ** 1.5, (1 + z_reio) ** 1.5
    delta_z = 0.5
    delta_y = 1.5 * np.sqrt(1 + z_reio) * delta_z
    x_HI = 0.5 * (1 + np.tanh((y - y_reio) / delta_y))
    return x_HI


if __name__ == '__main__':
    params = dict(
        h=0.67810,
        omega_b=0.02238280,
        omega_cdm=0.1201075,
        sigma8=0.8159,
        n_s=0.9660499,
        zt=24,
        Tv=4.5,
        LX=40,
    )
    fits = ['rgomp1', 'rgomp2']

    import matplotlib.pyplot as plt
    z = np.linspace(5, 16, num=111, endpoint=True)
    lna = - np.log(1 + z)
    for fit in fits:
        params['fit'] = fit
        plt.plot(z, gomprat(lna, params=params), label=fit, lw=1, alpha=.5)
    raise NotImplementedError('Discuss about if we want to or should do SR again')
    plt.plot(z, gomprat(lna, lna_pivot=-2.1038, tilt=7.49), label='gomp complexity 1',
             lw=1, alpha=0.7)
    plt.plot(z, tanh(z, z_reio=7.67), label='tanh', lw=1, alpha=.5)
    plt.legend()
    plt.xlabel('$z$')
    plt.ylabel(r'$x_\mathrm{HI}$')
    plt.show()
