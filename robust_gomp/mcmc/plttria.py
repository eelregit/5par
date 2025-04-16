#!/usr/bin/env python

import math
import os
import sys
from functools import reduce
from operator import and_ as intersection

from getdist import loadMCSamples
from getdist import plots
import matplotlib.pyplot as plt


def triangle_plot(*roots):
    assert len(roots) > 0

    samples, labels, paths, models, data = [], [], [], [], []
    for root in roots:
        s = loadMCSamples(root, settings={'ignore_rows': 0.2})
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

    params = ['A_s', 'n_s', 'theta_s_100', 'omega_cdm', 'omega_b', 'Omega_m',
              'omegamh2', 'H0', 'sigma8', 'S8', 'tau_reio']
    for param in ['m_ncdm', 'w0_fld', 'wa_fld', 'z_reio', 'alpha_gomp', 'beta_gomp',
                  'zt', 'Tv', 'LX',
                  'chi2__planck_2018_lowl.EE_sroll2',
                  'chi2__bao.desi_dr2.desi_bao_all',
                  'chi2__xHI']:
        for s in samples:
            if param in s.getParamNames().list():
                params.append(param)
                break

    g = plots.get_subplot_plotter(subplot_size=1)
    g.settings.line_styles = ['-C2', '--C0', '-.C1', ':C3']
    g.settings.linewidth = 3
    g.settings.solid_colors = ['C2', 'C0', 'C1', 'C3']
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
        filled=True,
        legend_labels=labels,
        markers={
            'A_s': 2.100e-9, 'n_s': 0.9649, 'theta_s_100': 1.04092, 'omega_cdm': 0.1200,
            'omega_b': 0.02237, 'Omega_m': 0.3153, 'omegamh2': 0.1430, 'H0': 67.36,
            'sigma8': 0.8111, 'S8': 0.832, 'tau_reio': 0.0544,
        },
    )

    g.export(filepath)


if __name__ == '__main__':
    roots = sys.argv[1:]

    plt.style.use('../5par.mplstyle')
    triangle_plot(*roots)
