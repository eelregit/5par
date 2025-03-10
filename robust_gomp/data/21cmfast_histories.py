import numpy as np
import matplotlib.pyplot as plt


a_edge, x_edge = np.loadtxt('../data/xHI.txt', unpack=True)[:2]
a_core, x_core = np.loadtxt('../data/xHI_core.txt', unpack=True)[:2]
a = np.concatenate((a_edge, a_core))
x = np.concatenate((x_edge, x_core))
num_sim = 512 + 512
num_a = 127
a = a[:num_a]
x = x.reshape(num_sim, num_a)

plt.style.use('../5par.mplstyle')
fig = plt.figure(figsize=(3, 2))
ax = fig.add_subplot()
ax.plot(a.T, x.T, c='gray', lw=0.3, alpha=0.2)
ax.set_xlim(.02, 0.25)
ax.set_ylim(0, 1)
ax.set_xlabel(r'$a$')
ax.set_ylabel(r'$x_\mathrm{HI}$')

plt.savefig('21cmfast_histories.pdf')
