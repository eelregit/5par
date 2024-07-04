import numpy as np
from cobaya.likelihood import Likelihood

from gomppoly_6 import gomppoly6


class OneQuasarOneLikelihood(Likelihood):
    """Combined quasars."""

    lna = - np.log(1 + 7.29)
    mean = 0.49
    var = 0.11 ** 2  # symmetric

    def logp(self, **param_values):
        param_values['fit'] = 'gomp1'
        xHI = gomppoly6(self.lna, **param_values)
        return -0.5 * (np.log(2 * np.pi * self.var) + (xHI - self.mean) ** 2 / self.var)
