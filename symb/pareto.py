#!/usr/bin/env python

import os
import sys

import numpy as np
import scipy
import pysr
import matplotlib.pyplot as plt


def _get_lower(polygon):
    """https://stackoverflow.com/questions/76838415/lower-convex-hull"""
    minx = np.argmin(polygon[:, 0])
    maxx = np.argmax(polygon[:, 0]) + 1
    if minx >= maxx:
        lower_points = np.concatenate([polygon[minx:], polygon[:maxx]])
    else:
        lower_points = polygon[minx:maxx]
    return lower_points


def pareto_plot(equation_file, savefig=True, lower_convex_hull=True):
    """Plot Pareto front."""
    model = pysr.PySRRegressor.from_file(equation_file)
    hof = model.get_hof()

    plt.style.use('../5par.mplstyle')

    hof = hof.rename(columns={'loss': 'Pareto front'})  # HACK to change legend label
    hof.plot(x='complexity', y='Pareto front', figsize=(4.8, 3.6), loglog=True,
             xlim=(1, None), ylabel='loss', drawstyle='steps-post')
    hof = hof.rename(columns={'Pareto front': 'loss'})  # HACK change it back

    if lower_convex_hull:
        points = hof[['complexity', 'loss']].to_numpy()
        points = points[np.isfinite(points.sum(axis=1))]  # remove inf

        hull = scipy.spatial.ConvexHull(np.log(points))
        lower_points = _get_lower(points[hull.vertices])

        plt.plot(lower_points[:, 0], lower_points[:, 1], ls=':', label='convex hull')
        plt.legend()

    if savefig:
        fig_file = os.path.splitext(equation_file)[0] + '.pdf'
        plt.savefig(fig_file)
    else:
        return plt.gcf()


if __name__ == '__main__':
    for equation_file in sys.argv[1:]:
        pareto_plot(equation_file)
