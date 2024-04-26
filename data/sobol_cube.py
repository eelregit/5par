import numpy as np
from scipy.stats.qmc import Sobol
import matplotlib.pyplot as plt

from sobol import gen_sobol


# generate the same Sobol sequence used in simulations, without rescaling
m = 7
sample = gen_sobol(m=m, l_bounds=0, u_bounds=1)
x, y, z = sample[:2**m, 2::-1].T  # tweak order for appearance

plt.style.use('../5par.mplstyle')

fig = plt.figure(figsize=(2, 2))
ax = fig.add_subplot(projection='3d')

ax.set_box_aspect((1,) * 3)
ax.set_proj_type('persp', focal_length=0.3)
elev, azim, roll = 15, -16.5, 0  # elevation, azimuth, and roll
ax.view_init(elev, azim, roll)
ax.set_axis_off()
ax.set_xlim([0, 1])
ax.set_ylim([0, 1])
ax.set_zlim([0, 1])

verts = np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0],
                  [0, 0, 1], [1, 0, 1], [1, 1, 1], [0, 1, 1]])
edges = np.array([[0, 1], [1, 2], [2, 3], [3, 0],
                  [0, 4], [1, 5], [2, 6], [3, 7],
                  [4, 5], [5, 6], [6, 7], [7, 4]])
c = 'gray'
alphas = [1, 1, 0.5, 0.5, 1, 1, 1, 0.5, 1, 1, 1, 1]
zorders = [0, 0, 0, 0, 0, 9, 0, 0, 9, 9, 0, 0]
for i, (edge, alpha, zorder) in enumerate(zip(edges, alphas, zorders)):
    ax.plot(verts[edge, 0], verts[edge, 1], verts[edge, 2], c=c, alpha=alpha,
            zorder=zorder)

ax.scatter(x, y, z, marker='.')

plt.savefig('sobol_cube.pdf')
