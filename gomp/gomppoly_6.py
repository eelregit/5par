import numpy as np
from numpy import exp, log
from numpy.polynomial import Polynomial
try:
    from cobaya.likelihood import Likelihood
except:
    pass


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


def pivot_sr(h, omega_b, omega_cdm, sigma8, n_s, zt, sr_label,
             **kwargs):  # kwargs absorbs other params of Cobaya
    Ob, Om = reparam(h, omega_b, omega_cdm)
    s8, ns = sigma8, n_s

    if sr_label == 'gomp1':  # complexity 22
        return (ns + 0.35580978 * log(0.11230898 * zt)) * (0.048352774 - s8) - Om - ns + (Ob / Om) ** h
    if sr_label == 'gomp1_raw':
        return ((((ns - (log(0.11230898 * zt) * -0.35580978)) * (0.048352774 - s8)) - (Om + ns)) + ((Ob / Om) ** h))

    #if sr_label == 'gomp2':  # complexity ??
    #if sr_label == 'gomp2_raw':

    raise ValueError(f'sr_label = {sr_label} not recognized')


def tilt_sr(h, omega_b, omega_cdm, sigma8, n_s, zt, sr_label,
            **kwargs):  # kwargs absorbs other params of Cobaya
    Ob, Om = reparam(h, omega_b, omega_cdm)
    s8, ns = sigma8, n_s

    if sr_label == 'gomp1':  # complexity 25
        return log(Ob) * (0.005659511 ** Om / 0.601493 - log(zt - (Om + ns * h) ** 15.051933) + h) + h / s8
    if sr_label == 'gomp1_raw':
        return ((log(Ob) * (((0.005659511 ** Om) / 0.601493) - (log(zt - ((Om + (ns * h)) ** 15.051933)) - h))) + (h / s8))

    #if sr_label == 'gomp2':  # complexity ??
    #if sr_label == 'gomp2_raw':

    raise ValueError(f'sr_label = {sr_label} not recognized')


class PlanA(Likelihood):

    def initialize(self):
        data = np.array([
            [7.29, 0.49, -0.11, 0.11],  # combined quasars
            [6.15, 0.20, -0.12, 0.14],  # 2404.12585 damping wing
            [6.35, 0.29, -0.13, 0.14],
            [5.60, 0.19, -0.16, 0.11],  # 2405.12273 damping wing
            [6.10, 0.21, -0.07, 0.17],  # 2401.10328 damping wing
            [6.46, 0.21, -0.07, 0.33],
            [6.87, 0.37, -0.17, 0.17],
        ])
        self.get_my_data(data)

    def get_my_data(self, data):
        z, mean, lo, hi = data.T
        self.lna = - np.log(1 + z)
        self.mean = mean + (lo + hi) / 2  # symmetrized
        self.var = ((hi - lo) / 2) ** 2  # symmetrized

    def get_requirements(self):
        return dict(H0=None)

    def logp(self, **param_values):
        param_values.update(dict(
            h = self.provider.get_param('H0') / 100,
            omega_b = self.provider.get_param('omega_b'),
            omega_cdm = self.provider.get_param('omega_cdm'),
            sigma8 = self.provider.get_param('sigma8'),
            n_s = self.provider.get_param('n_s'),
            zt = self.provider.get_param('zt'),
            sr_label = 'gomp1',
        ))
        xHI = gomppoly6(self.lna, params=param_values)
        return -0.5 * np.sum(np.log(2 * np.pi * self.var)
                             + (xHI - self.mean) ** 2 / self.var)


class PlanB(PlanA):

    def initialize(self):
        data = np.array([
            [7.29, 0.49, -0.11, 0.11],  # combined quasars
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
        self.get_my_data(data)


if __name__ == '__main__':
    params = dict(
        h=0.67810,
        omega_b=0.02238280,
        omega_cdm=0.1201075,
        sigma8=0.8159,
        n_s=0.9660499,
        zt=24,
    )
    #sr_labels = ['gomp1', 'gomp1_raw']
    sr_labels = ['gomp1']

    import matplotlib.pyplot as plt
    z = np.linspace(5, 16, num=111, endpoint=True)
    for sr_label in sr_labels:
        params['sr_label'] = sr_label
        plt.plot(z, gomppoly6(log(1/(1+z)), params=params), label=sr_label, lw=1, alpha=.5)
    plt.legend()
    plt.xlabel('$z$')
    plt.ylabel('$x_\mathrm{HI}$')
    plt.show()
