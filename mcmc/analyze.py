#!/usr/bin/env python

import math
import sys

from getdist import loadMCSamples
from getdist import plots
import matplotlib.pyplot as plt


def triangle_plot(chains):
    samples = loadMCSamples(chains, settings={'ignore_rows': 0.2})
    samples.addDerived(samples.getParams().s8omegamp5 / math.sqrt(0.3), name='S8',
                       label='S_8')

    params = ['A_s', 'n_s', 'theta_s_100', 'omega_cdm', 'omega_b', 'Omega_m',
              'omegamh2', 'H0', 'sigma8', 'S8', 'tau_reio',
              'chi2', 'chi2__planck_2018_lowl.EE']
    for param in ['z_reio', 'zt', 'chi2__xHI']:
        if param in samples.getParamNames().list():
            params.append(param)
    #print(f'{chains} samples:')
    #print(samples.getParamNames().list())

    plt.figure(figsize=(8, 6))
    g = plots.get_subplot_plotter()
    g.settings.figure_legend_frame = False
    g.settings.alpha_filled_add = 0.4
    g.settings.title_limit_fontsize = 14
    g.triangle_plot(
        samples,
        params,
        filled=True,
        #shaded=True,
        legend_labels=chains.split(sep='/')[1],
        legend_loc='upper right',
        line_args=[{'ls': '--', 'color': 'green'}],
        contour_colors=['green'],
        title_limit=1, # first title limit (for 1D plots) is 68% by default
        markers={
            'A_s': 2.1e-9, 'n_s': 0.9649, 'theta_s_100': 1.04092, 'omega_cdm': 0.1200,
            'omega_b': 0.02237, 'Omega_m': 0.3153, 'omegamh2': 0.1430, 'H0': 67.36,
            'sigma8': 0.8111, 'S8': 0.832, 'tau_reio': 0.0544,
        }
    )
    g.export(chains + '_triangle.pdf')


if __name__ == '__main__':
    plt.style.use('../5par.mplstyle')
    for chains in sys.argv[1:]:
        triangle_plot(chains)
