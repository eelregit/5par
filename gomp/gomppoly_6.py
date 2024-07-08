from functools import partial

import numpy as np
from numpy import exp, log
from numpy.polynomial import Polynomial


def reparam(h, omega_b, omega_cdm):
    Ob = omega_b / h**2
    Om = omega_cdm / h**2 + Ob
    return Ob, Om


def poly6(lna, lna_pivot=None, tilt=None, params=None):
    # the const and linear coeffs can be absorbed by the pivot and tilt of each curve
    c = np.array([0, 1, 1.12988669e-1, 2.59887866e-2, 5.49077282e-4, -6.51779652e-5])
    poly = Polynomial(c)

    if params is not None:
        lna_pivot = pivot_sr(**params)
        tilt = tilt_sr(**params)

    if lna_pivot is None or tilt is None:
        return poly(lna)

    return poly((lna - lna_pivot) * tilt)


def gomppoly6(lna, lna_pivot=None, tilt=None, params=None):
    return exp(- exp(poly6(lna, lna_pivot=lna_pivot, tilt=tilt, params=params)))


def pivot_sr(h, omega_b, omega_cdm, sigma8, n_s, zt, fit):
    Ob, Om = reparam(h, omega_b, omega_cdm)
    s8, ns = sigma8, n_s

    if fit == 'gomp1':  # complexity 22
        return (ns + 0.35580978 * log(0.11230898 * zt)) * (0.048352774 - s8) - Om - ns + (Ob / Om) ** h
    if fit == 'gomp1_raw':
        return ((((ns - (log(0.11230898 * zt) * -0.35580978)) * (0.048352774 - s8)) - (Om + ns)) + ((Ob / Om) ** h))

    #if fit == 'gomp2':  # complexity ??
    #if fit == 'gomp2_raw':

    raise ValueError(f'fit = {fit} not recognized')


def tilt_sr(h, omega_b, omega_cdm, sigma8, n_s, zt, fit):
    Ob, Om = reparam(h, omega_b, omega_cdm)
    s8, ns = sigma8, n_s

    if fit == 'gomp1':  # complexity 25
        return log(Ob) * (0.005659511 ** Om / 0.601493 - log(zt - (Om + ns * h) ** 15.051933) + h) + h / s8
    if fit == 'gomp1_raw':
        return ((log(Ob) * (((0.005659511 ** Om) / 0.601493) - (log(zt - ((Om + (ns * h)) ** 15.051933)) - h))) + (h / s8))

    #if fit == 'gomp2':  # complexity ??
    #if fit == 'gomp2_raw':

    raise ValueError(f'fit = {fit} not recognized')


def tanh(z, z_reio):
    """Planck 2018 VI footnote 15"""
    y, y_reio = (1 + z) ** 1.5, (1 + z_reio) ** 1.5
    delta_z = 0.5
    delta_y = 1.5 * np.sqrt(1 + z_reio) * delta_z
    x_HI = 0.5 * (1 + np.tanh((y - y_reio) / delta_y))
    return x_HI


def xHI_like(_self=None, type='dw', fit='gomp1'):
    data = np.array([
        [7.29, 0.49, -0.11, 0.11],  # combined quasars daming wing
        [6.15, 0.20, -0.12, 0.14],  # 2404.12585 damping wing
        [6.35, 0.29, -0.13, 0.14],
        [5.60, 0.19, -0.16, 0.11],  # 2405.12273 damping wing
        [6.10, 0.21, -0.07, 0.17],  # 2401.10328 damping wing
        [6.46, 0.21, -0.07, 0.33],
        [6.87, 0.37, -0.17, 0.17],
        [6.60, 0.08, -0.05, 0.08],  # 2101.01205 luminosity function
        [7.00, 0.28, -0.05, 0.05],
        #[7.30, 0.83, -0.07, 0.06],  # concerning interpolation of UV luminosity
    ])
    if type == 'dwlf':  # both damping wing and luminosity function
        pass
    elif type == 'dw':  # only damping wing
        data = data[:-2]
    else:
        raise ValueError(f'{type=} not recognized')

    z, m, l, h = data.T
    lna = - log(1 + z)
    mean = m + (l + h) / 2  # symmetrized
    var = ((h - l) / 2) ** 2  # symmetrized

    if fit == 'tanh':
        z_reio = _self.provider.get_param('z_reio')
        xHI = tanh(z, z_reio)
    else:
        params = dict(
            h=_self.provider.get_param('H0') / 100,
            omega_b=_self.provider.get_param('omega_b'),
            omega_cdm=_self.provider.get_param('omega_cdm'),
            sigma8=_self.provider.get_param('sigma8'),
            n_s=_self.provider.get_param('n_s'),
            zt=_self.provider.get_param('zt'),
            fit=fit,
        )
        xHI = gomppoly6(lna, params=params)

    half_neg_chi2 = -0.5 * ((xHI - mean) ** 2 / var).sum()
    return half_neg_chi2

tanh_dw = partial(xHI_like, type='dw', fit='tanh')
tanh_dwlf = partial(xHI_like, type='dwlf', fit='tanh')
gomp_1dw = partial(xHI_like, type='dw', fit='gomp1')
gomp_1dwlf = partial(xHI_like, type='dwlf', fit='gomp1')
gomp_2dw = partial(xHI_like, type='dw', fit='gomp2')
gomp_2dwlf = partial(xHI_like, type='dwlf', fit='gomp2')


if __name__ == '__main__':
    params = dict(
        h=0.67810,
        omega_b=0.02238280,
        omega_cdm=0.1201075,
        sigma8=0.8159,
        n_s=0.9660499,
        zt=24,
    )
    #fits = ['gomp1', 'gomp1_raw']
    fits = ['gomp1']

    import matplotlib.pyplot as plt
    z = np.linspace(5, 16, num=111, endpoint=True)
    lna = - log(1 + z)
    for fit in fits:
        params['fit'] = fit
        plt.plot(z, gomppoly6(lna, params=params), label=fit, lw=1, alpha=.5)
    plt.plot(z, tanh(z, z_reio=7.67), label='tanh', lw=1, alpha=.5)
    plt.legend()
    plt.xlabel('$z$')
    plt.ylabel('$x_\mathrm{HI}$')
    plt.show()
