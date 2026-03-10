#!/usr/bin/env python

import math
import os
import sys
from functools import reduce
from operator import and_ as intersection

from getdist import loadMCSamples
from getdist import plots
import matplotlib.pyplot as plt


plt.switch_backend('pgf')


def triangle_plot(*roots):
    assert len(roots) > 0

    samples, labels, paths, models, data = [], [], [], [], []
    for root in roots:
        s = loadMCSamples(root, settings={'ignore_rows': 0.2,
                                          'credible_interval_threshold': float('inf')})
        p = s.getParamNames().list()
        if 'S8' not in p:
            s.addDerived(s.getParams().s8omegamp5 / math.sqrt(0.3), name='S8',
                         label='S_8')
        samples.append(s)

        path = os.path.dirname(os.path.realpath(root))
        paths.append(path)
        label = path.split('/')[-2:]  # HACK assuming a ".../model/data" dir structure
        labels.append('/'.join(label))
        m = label[0]
        models.append(m)
        d = set(label[1].split('_'))  # HACK assuming datasets are joined by _
        data.append(d)

        print(path)
        print('  params:', p)
        print('  data:', sorted(d))
    assert len(paths) == len(set(paths)), 'duplicate detected'

    if len(roots) == 1:
        filepath = roots[0] + '_triangle.pdf'
    else:
        commonpath = os.path.commonpath(paths)
        multimodel = len(set(models)) > 1
        commondata = reduce(intersection, data)
        modeldata = []
        for m, d in zip(models, data):
            md = [m] if multimodel else []
            md += sorted(d - commondata)
            md = '_'.join(md) if md else 'base'
            modeldata.append('_' + md)
        filepath = (commonpath + '/' + '_'.join(sorted(commondata)) + '_triangle'
                    + '_V'.join(modeldata) + '.pdf')

    params = ['logA', 'n_s', 'theta_s_100', 'omega_cdm', 'omega_b', 'Omega_m',
              'omegamh2', 'H0', 'sigma8', 'S8']
    for param in ['alpha_gomp', 'threehalves_over_beta', 'beta_gomp', 'tau_reio',
                  'w0_fld', 'w_half', 'wa_fld', 'm_ncdm',
                  'chi2__lowlTT', 'chi2__lowlEE', 'chi2__Planck', 'chi2__ACT',
                  'chi2__QDW', 'chi2__DP', 'chi2__lens', 'chi2__bao']:
        for s in samples:
            if param in s.getParamNames().list():
                params.append(param)
                break

    g = plots.get_subplot_plotter(subplot_size=1)
    g.settings.line_styles = ['-C0', '--C1', '-.C3', ':C2']
    g.settings.linewidth = 3
    g.settings.solid_colors = ['C0', 'C1', 'C3', 'C2']
    g.settings.alpha_filled_add = 0.5
    g.settings.figure_legend_frame = False
    g.settings.figure_legend_loc = 'upper right'
    g.settings.legend_colored_text = True
    g.settings.legend_fontsize = 28
    g.settings.title_limit = 1
    g.settings.title_limit_fontsize = 14

    g.triangle_plot(
        samples,
        params,
        #filled=True,
        legend_labels=labels,
        markers={
            'logA': 3.060, 'n_s': 0.9743, 'theta_s_100': 1.04086, 'omega_cdm': 0.1179,
            'omega_b': 0.02256, 'Omega_m': 0.3032, 'omegamh2': 0.1411, 'H0': 68.22,
            'sigma8': 0.8126, 'S8': 0.8169, 'tau_reio': 0.0632,
        },  # ACT DR6 2503.14452 Table 5 P-ACT-LB
    )

    g.export(filepath)


if __name__ == '__main__':
    roots = sys.argv[1:]

    plt.style.use('../5par.mplstyle')
    triangle_plot(*roots)
