#!/usr/bin/env python

import sys

from cobaya.samplers.mcmc import plot_progress
import matplotlib.pyplot as plt

for chains in sys.argv[1:]:
    plot_progress(chains, figure_kwargs={'figsize': (6,4)})
    plt.savefig(chains + '_progress.pdf')
