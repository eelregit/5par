import numpy as np
import matplotlib.pyplot as plt

"""
    Generate current tau constraints taken from Planck data (different analyses most have reccent CMB data some even include WMAP)

"""

tau = np.array([0.051, 0.054, 0.054, 0.058, 0.059, 0.066, 0.069, 0.080])
tau_e = [0.001, 0.002, 0.007, 0.006, 0.006, 0.013, 0.011, 0.012]
y = [1, 2, 3, 4, 5, 6, 7, 8]
markers = ['|', 'x', 'd', '8', 'p', 's', 'o', '^']
labels = ['This work', 'Paoletti', 'Planck et al. 2020', 'Tristram et al. 2023', 'Pagano et al. 2020', 'Paradiso et al. 2023', 'Natale et al. 2020', r'Giar\`e et al. 2023']
colors = ['green', 'gray', 'darkblue', 'orange', 'purple', 'brown', 'pink', 'magenta']

plt.style.use('../5par.mplstyle')
fig = plt.figure(figsize=(2,2))
ax1 = fig.add_subplot(111)
for i in range(len(tau)):
    ax1.errorbar(tau[i], y[i], xerr=tau_e[i], yerr=0, marker=markers[i], color=colors[i], label=labels[i])
# handles
handles, p_labels = ax1.get_legend_handles_labels()
handles = [h[0] for h in handles]
ax1.set_xlabel(r'$\tau_{\rm reio}$')
ax1.set_yticklabels([])
ax1.set_yticks([])
ax1.legend(handles, labels, loc='center left',bbox_to_anchor=(1.,0.5))
plt.savefig('../figs/tau_fig.pdf')
