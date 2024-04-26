import numpy as np
from numpy import exp, log
from numpy.polynomial import Polynomial


def reparam(h, omega_b, omega_cdm):
    Ob = omega_b / h**2
    Om = omega_cdm / h**2 + Ob
    return Ob, Om


def poly6(lna, lna_pivot=None, tilt=None, params=None):
    # the const and linear coeffs can be absorbed by the pivot and tilt of each curve
    c = np.array([0, 1, 0.15034337, 0.04849586, 0.00526138, 0.0002182])
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
        return ((-1.0389123 - s8) * (((Om * h) - Ob) + ns))

    if fit == '0226_complex':  # complexity 27
        return (((((Ob ** h) + (-0.118975304 / ((((ns + -0.3362399) * 1.591861) ** ((log(h) / Ob) + 6.789797)) / s8))) - ns) * exp(Om)) - s8)

    if fit == '0227_simple':  # complexity 11
        return (((h ** Om) + s8) * (Ob - (ns + Om)))

    if fit == '0227_complex':  # complexity 28
        return (((Om * h) + ns) * ((-1.0621809 - s8) + (Ob * exp(((0.040550426 + ns) ** ((3.622915 - (h ** log(Ob))) / 0.2182732)) / exp(Om)))))

    raise ValueError(f'fit = {fit} not recognized')


def tilt_sr(h, omega_b, omega_cdm, sigma8, n_s, fit):
    Ob, Om = reparam(h, omega_b, omega_cdm)
    s8, ns = sigma8, n_s

    if fit.startswith('0226'):
        return 8.331045

    if fit.startswith('0227'):
        return 8.289981

    raise ValueError(f'fit = {fit} not recognized')


if __name__ == '__main__':
    params = dict(
        h=0.67810,
        omega_b=0.02238280,
        omega_cdm=0.1201075,
        sigma8=0.8159,
        n_s=0.9660499,
    )
    fits = ['0226_simple', '0226_complex', '0227_simple', '0227_complex']

    import matplotlib.pyplot as plt
    z = np.linspace(5, 16, num=111)
    for fit in fits:
        params['fit'] = fit
        plt.plot(z, gomppoly6(log(1/(1+z)), params=params), label=fit, lw=0.3)
    plt.legend()
    plt.xlabel('$z$')
    plt.ylabel('$x_\mathrm{HI}$')
    plt.show()
