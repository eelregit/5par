import numpy as np
from cobaya.likelihood import Likelihood

from reio_like.gomprat import gomprat, tanh


class Gomp(Likelihood):
    """Likelihood on Gompertzian reionization history."""

    def get_requirements(self):
        return {'alpha_gomp': None, 'beta_gomp': None}

    def get_xHI(self, params_values):
        lna_pivot, tilt = params_values['alpha_gomp'], params_values['beta_gomp']
        xHI = gomprat(self.lna, lna_pivot=lna_pivot, tilt=tilt)
        return xHI


class RGomp(Likelihood):
    """Likelihood on Gompertzian reionization history, in symbolic regressed 21cmFAST
    parameters."""

    def get_requirements(self):
        return {
            'H0': None,
            'omega_b': None,
            'omega_cdm': None,
            'sigma8': None,
            'n_s': None,
            'zt': None,
            'Tv': None,
            'LX': None,
        }

    SR_fit = 'rgomp1'

    def get_xHI(self, params_values):
        params = {
            'h': params_values['H0'] / 100,
            'omega_b': params_values['omega_b'],
            'omega_cdm': params_values['omega_cdm'],
            'sigma8': params_values['sigma8'],
            'n_s': params_values['n_s'],
            'zt': params_values['zt'],
            'Tv': params_values['Tv'],
            'LX': params_values['LX'],
            'fit': self.SR_fit,
        }
        xHI = gomprat(self.lna, params=params)
        return xHI


RGomp1 = RGomp


class RGomp2(RGomp):
    """Likelihood on Gompertzian reionization history, in symbolic regressed 21cmFAST
    parameters using half of training set."""

    SR_fit = 'rgomp2'


class Tanh(Likelihood):
    """Likelihood on logistic reionization history."""

    def get_requirements(self):
        return {'z_reio': None}

    def get_xHI(self, params_values):
        z_reio = self.provider.get_param('z_reio')
        xHI = tanh(self.z, z_reio)
        return xHI


class QuasarDampingWing(Likelihood):
    """Quasar damping wing likelihood."""

    def initialize(self):
        data = np.array([
            [7.29, 0.49, -0.11, 0.11],  # combined quasars daming wing
            [6.15, 0.20, -0.12, 0.14],  # 2404.12585
            [6.35, 0.29, -0.13, 0.14],
            [5.60, 0.19, -0.16, 0.11],  # 2405.12273
            [6.10, 0.21, -0.07, 0.17],  # 2401.10328
            [6.46, 0.21, -0.07, 0.33],
            [6.87, 0.37, -0.17, 0.17],
        ])

        z, m, l, h = data.T
        self.z = z
        self.lna = - np.log(1 + z)
        self.mean = m + (l + h) / 2  # symmetrized
        self.var = ((h - l) / 2) ** 2  # symmetrized

    def logp(self, **params_values):
        xHI = self.get_xHI(params_values)
        half_neg_chi2 = -.5 * ((xHI - self.mean) ** 2 / self.var).sum()
        return half_neg_chi2


class LymanbetaDarkGap(Likelihood):
    """Lyman-beta dark gap likelihood, as extra conservative upper bounds."""

    def initialize(self):
        # points from Lyman-beta forest dark gaps 2205.04569
        # *plus* their errors to be extra conservative
        self.z = np.array([5.55, 5.75, 5.95])
        self.lna = - np.log(1 + self.z)
        self.upper_bound = np.array([0.05 + 0.04, 0.17 + 0.05, 0.29 + 0.09])  # for HI


    def logp(self, **params_values):
        xHI = self.get_xHI(params_values)
        return 0 if np.all(xHI <= self.upper_bound) else - np.inf


class GompQDW(Gomp, QuasarDampingWing):
    """Quasar damping wing likelihood on Gompertzian reionization history."""


class RGompQDW(RGomp, QuasarDampingWing):
    """Quasar damping wing likelihood on Gompertzian reionization history, in symbolic
    regressed 21cmFAST parameters."""


RGomp1QDW = RGompQDW


class RGomp2QDW(RGomp2, QuasarDampingWing):
    """Quasar damping wing likelihood on Gompertzian reionization history, in symbolic
    regressed 21cmFAST parameters using half of training set."""


class TanhQDW(Tanh, QuasarDampingWing):
    """Quasar damping wing likelihood on logistic reionization history."""


class GompLybDG(Gomp, LymanbetaDarkGap):
    """Lyman-beta dark gap likelihood on Gompertzian reionization history."""


class RGompLybDG(RGomp, LymanbetaDarkGap):
    """Lyman-beta dark gap likelihood on Gompertzian reionization history, in symbolic
    regressed 21cmFAST parameters."""


RGomp1LybDG = RGompLybDG


class RGomp2LybDG(RGomp2, LymanbetaDarkGap):
    """Lyman-beta dark gap likelihood on Gompertzian reionization history, in symbolic
    regressed 21cmFAST parameters using half of training set."""


class TanhLybDG(Tanh, LymanbetaDarkGap):
    """Lyman-beta dark gap likelihood on logistic reionization history."""
