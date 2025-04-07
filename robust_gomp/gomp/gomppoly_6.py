from functools import partial

import numpy as np
from numpy import exp, log
from numpy.polynomial import Polynomial


def reparam(h, omega_b, omega_cdm):
    Ob = omega_b / h**2
    Om = omega_cdm / h**2 + Ob
    return Ob, Om


def poly6(lna, lna_pivot=None, tilt=None, params=None, robu_flag=False):
    # the const and linear coeffs can be absorbed by the pivot and tilt of each curve
    c = np.array([0, 1, 1.25218906e-01, 3.53290242e-02, 2.20265427e-03, 7.48303918e-06])
    poly = Polynomial(c)

    if params is not None and robu_flag is False:
        lna_pivot = pivot_sr(**params)
        tilt = tilt_sr(**params)

    if params is not None and robu_flag is True:
        lna_pivot = robust_pivot_sr(**params)
        tilt = robust_tilt_sr(**params)

    if lna_pivot is None or tilt is None:
        return poly(lna)

    return poly((lna - lna_pivot) * tilt)


def gomppoly6(lna, lna_pivot=None, tilt=None, params=None, robu_flag=False):
    return exp(- exp(poly6(lna, lna_pivot=lna_pivot, tilt=tilt, params=params, robu_flag=robu_flag)))


def pivot_sr(h, omega_b, omega_cdm, sigma8, n_s, zt, fit):
    Ob, Om = reparam(h, omega_b, omega_cdm)
    s8, ns = sigma8, n_s

    if fit == 'gomp1':  # complexity 22
        return (ns + 0.35580978 * log(0.11230898 * zt)) * (0.048352774 - s8) - Om - ns + (Ob / Om) ** h
    if fit == 'gomp1_raw':
        return ((((ns - (log(0.11230898 * zt) * -0.35580978)) * (0.048352774 - s8)) - (Om + ns)) + ((Ob / Om) ** h))

    if fit == 'gomp2':  # complexity 22
        return (Ob / Om) ** Om - log((zt + Ob ** -0.49822742) ** s8 * h) ** 0.5721157 - ns ** 1.8340757
    if fit == 'gomp2_raw':
        return ((((Ob / Om) ** Om) - (log(((zt + (Ob ** -0.49822742)) ** s8) * h) ** 0.5721157)) - (ns ** 1.8340757))
    
    raise ValueError(f'fit = {fit} not recognized')

def robust_pivot_sr(h, omega_b, omega_cdm, sigma8, n_s, zt, Tv, LX, fit):
    Ob, Om = reparam(h, omega_b, omega_cdm)
    s8, ns = sigma8, n_s
    if fit == 'robustgomp1': # complexity 34
        return ((((Tv - np.log((((Tv ** (-40.065258 + LX)) + zt) + np.exp(s8)) ** s8)) / 2.5946999) - np.exp(ns - (Ob ** Om))) - ((h ** (Om ** 0.79247624)) / (0.8632819 ** ns)))

    if fit == 'robustgomp2': # complexity 27
        return (((Ob / (Om * s8)) ** (s8 * h)) - (((((19.920519 + zt) + (0.08890569 ** (40.43043 - LX))) * s8) ** (ns / Tv)) + Om))

    raise ValueError(f'fit = {fit} not recognized')


def tilt_sr(h, omega_b, omega_cdm, sigma8, n_s, zt, fit):
    Ob, Om = reparam(h, omega_b, omega_cdm)
    s8, ns = sigma8, n_s

    if fit == 'gomp1':  # complexity 25
        return log(Ob) * (0.005659511 ** Om / 0.601493 - log(zt - (Om + ns * h) ** 15.051933) + h) + h / s8
    if fit == 'gomp1_raw':
        return ((log(Ob) * (((0.005659511 ** Om) / 0.601493) - (log(zt - ((Om + (ns * h)) ** 15.051933)) - h))) + (h / s8))

    if fit == 'gomp2':  # complexity 11, 1e-5 difference in constants from those in csv
        return ((zt - Om ** -1.583228) / (Ob * h)) ** 0.31627414
    if fit == 'gomp2_raw':
        return (((zt - (Om ** -1.583228)) / (Ob * h)) ** 0.31627414)

    raise ValueError(f'fit = {fit} not recognized')

def robust_tilt_sr(h, omega_b, omega_cdm, sigma8, n_s, zt, Tv, LX, fit):
    Ob, Om = reparam(h, omega_b, omega_cdm)
    s8, ns = sigma8, n_s

    if fit == 'robustgomp1': # complexity 24
        return (((zt / h) ** 0.4991587) - (((((Ob + -0.20859116) * LX) - -6.0811667) - (Om * (Tv - np.exp(ns ** 2.8035064)))) / s8))

    if fit == 'robustgomp2': # complexity 26
        return (((((0.12804393 - Ob) / h) * (zt + ((Tv - ((s8 / Om) + 1.0161631)) / Om))) - (ns**3.052106)) + np.exp(0.039550677 * LX))
    
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
    elif fit == 'robustgomp1' or fit == 'robustgomp2':
        params = dict(
            h=_self.provider.get_param('H0') / 100,
            omega_b=_self.provider.get_param('omega_b'),
            omega_cdm=_self.provider.get_param('omega_cdm'),
            sigma8=_self.provider.get_param('sigma8'),
            n_s=_self.provider.get_param('n_s'),
            zt=_self.provider.get_param('zt'),
            Tv=_self.provider.get_param('Tv'),
            LX=_self.provider.get_param('LX'),
            fit=fit,
        )
        # aww, making a flag was a bad idea: wdm_flag, axion flag...
        # fine for now
        xHI = gomppoly6(lna, params=params, robu_flag=True)
    elif fit == 'reio_gomp_noSR':
        lna_pivot = _self.provider.get_param('alpha_gomp')
        tilt = _self.provider.get_param('beta_gomp')
        xHI = gomppoly6(lna, lna_pivot=lna_pivot, tilt=tilt, robu_flag=False)
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
        xHI = gomppoly6(lna, params=params, robu_flag=False)

    half_neg_chi2 = -0.5 * ((xHI - mean) ** 2 / var).sum()
    return half_neg_chi2

tanh_dw = partial(xHI_like, type='dw', fit='tanh')
tanh_dwlf = partial(xHI_like, type='dwlf', fit='tanh')
gomp_1dw = partial(xHI_like, type='dw', fit='gomp1')
gomp_1dwlf = partial(xHI_like, type='dwlf', fit='gomp1')
gomp_2dw = partial(xHI_like, type='dw', fit='gomp2')
gomp_2dwlf = partial(xHI_like, type='dwlf', fit='gomp2')
robustgomp_1dw = partial(xHI_like, type='dw', fit='robustgomp1')
robustgomp_2dw = partial(xHI_like, type='dw', fit='robustgomp2')
freegomp_dw = partial(xHI_like, type='dw', fit='reio_gomp_noSR')


if __name__ == '__main__':
    params = dict(
        h=0.67810,
        omega_b=0.02238280,
        omega_cdm=0.1201075,
        sigma8=0.8159,
        n_s=0.9660499,
        zt=24,
    )
    #fits = ['gomp1', 'gomp1_raw', 'gomp2', 'gomp2_raw']
    fits = ['gomp1', 'gomp2']

    import matplotlib.pyplot as plt
    z = np.linspace(5, 16, num=111, endpoint=True)
    lna = - log(1 + z)
    for fit in fits:
        params['fit'] = fit
        plt.plot(z, gomppoly6(lna, params=params, robu_flag=False), label=fit, lw=1, alpha=.5)
    plt.plot(z, gomppoly6(lna, lna_pivot=-2.1038, tilt=7.49, robu_flag=False), label='free', lw=1, alpha=0.7)
    plt.plot(z, tanh(z, z_reio=7.67), label='tanh', lw=1, alpha=.5)
    plt.legend()
    plt.xlabel('$z$')
    plt.ylabel('$x_\mathrm{HI}$')
    plt.show()
