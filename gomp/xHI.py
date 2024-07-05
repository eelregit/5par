import numpy as np
from cobaya.likelihood import Likelihood

from gomppoly_6 import gomppoly6


class xHI(Likelihood):
    """Combined quasars."""

    lna = - np.log(1 + 7.29)
    mean = 0.49
    var = 0.11 ** 2  # symmetric

    def logp(self, **param_values):
        # forgot there are other entries  in param dictionary, I can define a new one without extras for easy access
        param = {
                 'n_s': param_values.get('n_s'),
                 'H0': param_values.get('H0'),
                 'sigma8': param_values.get('sigma8'),
                 'omega_b': param_values.get('omega_b'),
                 'omega_cdm': param_values.get('omega_cdm'),
                 'zt': param_values.get('zt')
                 }
        param['fit'] = 'gomp1'
#        xHI = gomppoly6(self.lna, **param_values)
        xHI = gomppoly6(self.lna, param)
        print('Check xHI ', xHI)
        return -0.5 * (np.log(2 * np.pi * self.var) + (xHI - self.mean) ** 2 / self.var)

    def get_requirements(self):
        # maybe change requirements, interface with classy broken
        return {'n_s': None,
                'H0': None,
                'sigma8': None,
                'zt': None,
                'omega_b': None,
                'omega_cdm': None}


"""
    Current error: params successfully passed to xhi likelihood [x]
                   xhi produces sensible value [x]
                   likelihood gets computed [x]
                   [classy] starts computation of new state [x]
                   [classy] receives incomplete list of parameters [ ]

    Specifically, classy receives {'H0': 66.68275231533434, 'z_reio': 20.0, 'm_ncdm': 0.06,
                                   'reio_parametrization': 'reio_gomp1', 'nonlinear_min_k_max': 20,
                                   'N_ncdm': 1, 'N_ur': 2.0328, 'l_max_scalars': 2700, 'output': ''}

    Hence fails to compute a sensible new state.

    Reason: [model] assigns parameters as follows:
            - xHI: Input: ['H0', 'n_s', 'sigma8', 'zt', 'omega_b', 'omega_cdm']
            - xHI: Output: [ ]
            - classy: Input: ['H0', 'z_reio', 'm_ncdm']
            - classy: Output: ['A_s', 'tau_reio', 'theta_s_100', 'Omega_m', 'Omega_Lambda', 'YHe', 'age', 'rs_drag']

"""
