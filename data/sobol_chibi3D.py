from scipy.stats.qmc import Sobol, LatinHypercube
import matplotlib.pyplot as plt


if __name__ == '__main__':
#    params = (r'$\sigma_8$', r'$n_\mathrm{s}$', '$h$')
#    l_bounds = (.74, .92, .61)
#    u_bounds = (.90, 1.00, .73)
#    d = 3 # num params
#    m = 3 # 7 = 128 points
#    sampler = Sobol(d, scramble=True)
#    sample = sampler.random(n=2**m)
#    sample = scale(sample, l_bounds, u_bounds)

#    sampler = LatinHypercube(d=3) # aww latin just as ugly...
#    sample = sampler.random(n=100)

    plt.style.use('../5par.mplstyle')


#    x = sample[:,0]
#    y = sample[:,1]
#    z = sample[:,2]


#    x = [2, 2, 2, 2, 2, 2, 2, 2, 2,
#         1, 1, 1, 1, 1, 1, 1, 1, 1,
#         0, 0, 0, 0, 0, 0, 0, 0, 0
#         ]
#    y = [0, 1, 2, 0, 1, 2, 0, 1, 2,
#         0, 1, 2, 0, 1, 2, 0, 1, 2,
#         0, 1, 2, 0, 1, 2, 0, 1, 2
#         ]
#    z = [0, 0, 0, 1, 1, 1, 2, 2, 2,
#         0, 0, 0, 1, 1, 1, 2, 2, 2,
#         0, 0, 0, 1, 1, 1, 2, 2, 2
#         ]
#
    x = [2, 2, 2, 2, 2, 2,
         1, 1, 1, 1, 1, 1,
         0, 0, 0, 0, 0, 0]
    y = [0, 1, 2, 0, 1, 2,
         0, 1, 2, 0, 1, 2,
         0, 1, 2, 0, 1, 2]
    z = [0, 0, 0, 1, 1, 1,
         0, 0, 0, 1, 1, 1,
         0, 0, 0, 1, 1, 1]

    plt.rcParams['xtick.bottom'] = False
    plt.rcParams['xtick.labelbottom'] = False
    plt.rcParams['ytick.left'] = False
    plt.rcParams['ytick.labelleft'] = False
    plt.rcParams['axes.grid'] = False
    fig = plt.figure(figsize=(3,2))
    ax = fig.add_subplot(projection='3d')
    ax.grid(False)

    ax.scatter(x,y,z, marker='*')
    plt.savefig('cube_chibi.pdf')
