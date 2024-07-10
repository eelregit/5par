import numpy as np
import matplotlib.pyplot as plt

"""
    Generate current tau constraints taken from Planck data (different analyses most have reccent CMB data some even include WMAP)

"""

taus = [.053, .054, .054, .058, .059, .0619, .063, .066, .069, .080]
errors = [.001, .002, .007, .006, .006, ([.0068], [.0056]), ([.0058], [.005]), .013, .011, .012]
ys = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
markers = ['|', 'x', 'd', '8', 'p', '*', 'H', 's', 'o', '^']
labels = ['This work', 'Paoletti et al. 2024', 'Planck et al. 2020',
          'Tristram et al. 2023', 'Pagano et al. 2020', 'Heinrich \& Hu 2021',
          'de Belsunce et al. 2021', 'Paradiso et al. 2023', 'Natale et al. 2020',
          'Giar\`e et al. 2023']
colors = ['C2', 'C7', 'C0', 'C1', 'C3', 'C4', 'C9', 'C5', 'C6', 'C8']

plt.style.use('../5par.mplstyle')
fig, ax = plt.subplots(figsize=(2.7, 2.7))
for tau, y, error, marker, color, label in zip(
    taus, ys, errors, markers, colors, labels):
    ax.errorbar(tau, y, xerr=error, marker=marker, color=color, label=label)
# handles
handles, labels = ax.get_legend_handles_labels()
ax.legend(reversed(handles), reversed(labels), loc='center left', bbox_to_anchor=(1, .5))
ax.set_xlabel(r'$\tau_{\rm reio}$')
ax.set_yticklabels([])
ax.set_yticks([])
fig.savefig('../figs/tau_fig.pdf')
