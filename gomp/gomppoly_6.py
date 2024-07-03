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
    for fit in fits:
        params['fit'] = fit
        plt.plot(z, gomppoly6(log(1/(1+z)), params=params), label=fit, lw=1, alpha=.5)
    plt.legend()
    plt.xlabel('$z$')
    plt.ylabel('$x_\mathrm{HI}$')
    plt.show()
