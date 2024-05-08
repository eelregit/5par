import numpy as np
from numpy import exp, log
from numpy.polynomial import Polynomial

import gomppoly_6 as manyfig


def reparam(h, omega_b, omega_cdm):
    Ob = omega_b / h**2
    Om = omega_cdm / h**2 + Ob
    return Ob, Om


def poly6(lna, lna_pivot=None, tilt=None, params=None):
    # the const and linear coeffs can be absorbed by the pivot and tilt of each curve
    c = np.array([0, 1, 0.1503, 0.04850, 0.005261, 0.0002182])
    poly = Polynomial(c)

    if params is not None:
        lna_pivot = pivot_sr(**params)
        tilt = tilt_sr(**params)

    if lna_pivot is None or tilt is None:
        return poly(lna)

    return poly((lna - lna_pivot) * tilt)


def gomppoly6(lna, lna_pivot=None, tilt=None, params=None):
    return exp(- exp(poly6(lna, lna_pivot=lna_pivot, tilt=tilt, params=params)))


def pivot_sr(h, omega_b, omega_cdm, sigma8, n_s, fit):
    Ob, Om = reparam(h, omega_b, omega_cdm)
    s8, ns = sigma8, n_s

    if fit == '0226_simple':  # complexity 11
        return ((-1.039 - s8) * (((Om * h) - Ob) + ns))

    if fit == '0227_simple':  # complexity 11
        return (((h ** Om) + s8) * (Ob - (ns + Om)))

    raise ValueError(f'fit = {fit} not recognized')


def tilt_sr(h, omega_b, omega_cdm, sigma8, n_s, fit):
    Ob, Om = reparam(h, omega_b, omega_cdm)
    s8, ns = sigma8, n_s

    if fit.startswith('0226'):
        return 8.331

    if fit.startswith('0227'):
        return 8.290

    raise ValueError(f'fit = {fit} not recognized')


if __name__ == '__main__':
    params = dict(
        h=0.67810,
        omega_b=0.02238280,
        omega_cdm=0.1201075,
        sigma8=0.8159,
        n_s=0.9660499,
    )
    fits = ['0226_simple', '0227_simple']

    z = np.linspace(5, 16, num=10000)
    for fit in fits:
        params['fit'] = fit
        x_many = manyfig.gomppoly6(log(1/(1+z)), params=params)
        x_4 = gomppoly6(log(1/(1+z)), params=params)
        print(fit)
        print(f'  min  = {(x_4 - x_many).min()}')
        print(f'  max  = {(x_4 - x_many).max()}')
        print(f'  mean = {(x_4 - x_many).mean()}')
        print(f'  std  = {(x_4 - x_many).std()}')
