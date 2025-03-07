from functools import partial
from itertools import count
from math import ceil, cbrt

import jax
import jax.numpy as jnp


# Bubble growing in radius, during matter domination
# grid spacing, l
# sqrt(a) da = sqrt(Omega_m) H0 dt
# d R = d (chi / l) = c dt / a / l = {2 c / [sqrt(Omega_m) H0 l]} d sqrt(a)
#   = d[(a/a_g)^(1/2)]
#   for bubble radius to grow by equal amount per step
#   here we use chi to denote particle horizon instead of the usual radial comoving
#   distance, differing only in their integral limits
#   a_g^(-1/2) = 2 * 2997.92458 / sqrt(Omega_m h^2) / l
#       Omega_m h^2 = 0.1424 from Planck 2018
# R = (a/a_g)^(1/2), particle horizon in unit of grid spacing, the size of the largest
#   possible bubble
# r = R - r_e, with each bubble grows to radius r after emerging at r_e
#   r_e should follow the distribution described below
#
# Bubble emerging from seeds, during matter domination
# box size, L = N l
# seed emergence rate, SFR psi * galaxy luminosity function phi?
#   say f a^b, per e-fold in a and per Mpc^3
#   b>=0, or the integral diverges at a=0. Take b>0 below
# f a^b l^3 d lna = (f l^3 / b) d a^b = d[(R^2 / a_e)^b]
#   so we can use a_e and b as the code parameters
#   a_e^-b = f l^3 a_g^b / b. Note the volume scaling in l^3 and in a_g
# FIXME how is a_e, b related to alpha, beta used in Gompertz?
#
# Code "time" in particle horizon (in unit of grid spacing)
# if b > 1/2, bubble seeds emerge faster
# if b < 1/2, bubble radii grow faster
# TODO b ~ ?
# but for the purpose of neutral fraction, enough to set the code time unit to R because
# we don't care which bubble ionizes which cell
#
# Algorithm
# Monte Carlo (FIXME or QMC, shuffled grid points (Grid), mass-ordered halos (LSS)) the
# spatial distribution of bubble seeds time stepping loop till reionization completes
#   bubbles emerge from seeds
#   bubbles grows spherically around their seeds
#       each bubble grow by one shell per step, from r to r+1
#       mark the grid points in [r, r+1) as ionized
#       jitted function keeps a list of relative indices of cells in the shell
#   count neutral fraction


def sow_bubble_seeds(size, kind, dens, b):
    """Init randomly bubble spatial distribution and emerging time."""
    # easier to parameterize with number density than with a_e
    a_e = 3 * (size//2 + 1)**2 / dens ** (1/b)
    # heuristic, this many should always be enough because the first bubble should have
    # ionized the whole field when the last bubble emerges
    # FIXME/HACK first bubble can have r_e>0, maybe not worth fixing unless problematic
    num = 1 + ceil(size**3 * (3 * (size//2 + 1)**2 / a_e)**b)
    print(f'sowing {num} bubble seeds with {a_e=}')

    # Monte Carlo ~ Poisson point process
    if kind == 'MC':
        seed = 0  # size = 256 or 512 seems big enough for low variance
        key = jax.random.PRNGKey(seed)
        pos = jax.random.randint(key, (num, 3), 0, size, dtype=jnp.int16)

    # Mass ordered halos representing LSS
    elif kind == 'LSS':
        import numpy as np
        masspos = np.loadtxt('quijote_hr_rockstar_0_z2.txt')
        mass = masspos[:, 0]
        ind = jnp.argsort(mass, descending=True)
        pos = masspos[ind, 1:][:num]
        pos = pos / 1000 * size  # HACK rescale Quijote box size to our box size
        pos = jnp.rint(pos).astype(jnp.int16) % size

    # Grid points, randomly ordered
    elif kind == 'Grid':
        z = jnp.linspace(0, size, num=ceil(cbrt(num)), endpoint=False)
        z = jnp.rint(z).astype(jnp.int16)
        pos = jnp.meshgrid(z, z, z, indexing='ij')
        pos = jnp.stack([z.ravel() for z in pos], axis=1)

        seed = 0
        key = jax.random.PRNGKey(seed)
        pos = jax.random.permutation(key, pos)
        pos = pos[:num]

    # Quasi-Monte Carlo, more uniform than Poisson
    elif kind == 'QMC':
        from scipy.stats.qmc import Sobol

        seed = 0
        sampler = Sobol(3, scramble=True, seed=seed)
        pos = sampler.random(num) * size
        pos = jnp.rint(pos).astype(jnp.int16) % size

    else:
        raise ValueError(f'{kind=} not supported')

    # when bubbles emerge, rounded to integers
    r_e = jnp.sqrt(a_e * (jnp.arange(1, 1+num) / size**3) ** (1/b))
    r_e = jnp.rint(r_e).astype(jnp.int16).tolist()

    if r_e[0] > 128:
        raise RuntimeError('first bubble emerges too slowly')

    return pos, r_e


@partial(jax.jit, static_argnames=('r'))
def add_shell(field, pos, r):
    """Add a spherical shell for one bubble at one step."""
    with jax.ensure_compile_time_eval():
        z = jnp.arange(-r-1, r+2, dtype=jnp.int16)
        xyz = jnp.meshgrid(z, z, z, indexing='ij')
        xyz = jnp.stack([z.ravel() for z in xyz], axis=1)
        r2 = (xyz**2).sum(axis=1)
        xyz = xyz[(r**2 <= r2) & (r2 < (r+1)**2)]

        shape = jnp.array(field.shape)

    xyz = (pos + xyz) % shape  # relative to absolute cell positions in shell
    x, y, z = xyz[:, 0], xyz[:, 1], xyz[:, 2]

    return field.at[x, y, z].set(False)


def grow_bubbles(field, bubbles, R):
    """Grow a spherical shell for each bubble at one step."""
    for pos, r_e in zip(*bubbles):
        if r_e >= R:
            break  # because r_e is monotonically increasing
        field = add_shell(field, pos, R - r_e)
    return field


def bubble_sim(size, kind, dens, b):
    field = jnp.ones((size,) * 3, dtype=jnp.bool)
    print(f'{field.shape=}, {kind=}, {dens=}, {b=}')

    bubbles = sow_bubble_seeds(size, kind, dens, b)
    r_e_min, r_e_max = bubbles[1][0], bubbles[1][-1]
    print(f'{r_e_min=}, {r_e_max=}', flush=True)

    f = open(f'{size}_{kind}_{dens:.0e}_{b:.0e}.txt', 'w')
    f.write(f'{dens:.3e} {b:.3e}')

    for R in count():
        field = grow_bubbles(field, bubbles, R)

        if R > r_e_max:
            raise RuntimeError('running out of bubble seeds, this should not happen: '
                               f'{R=}, {r_e_max=}')

        if R > r_e_min and (R - r_e_min)**2 > 3 * (size//2 + 1)**2:
            raise RuntimeError('long enough for the first bubble to fill the box: '
                               f'{R=}, {r_e_min=}, {size=}',
                               f'{(R - r_e_min)**2=}, {3 * (size//2 + 1)**2}')

        field_sum = field.sum()
        x_HI = field_sum / field.size
        f.write(f' {x_HI:.7e}')

        if field_sum <= 0:
            break

    f.write('\n')
    f.close()


if __name__ == '__main__':
    import sys
    size = int(sys.argv[1])
    kind = sys.argv[2]
    dens = float(sys.argv[3])
    b = float(sys.argv[4])

    bubble_sim(size=size, kind=kind, dens=dens, b=b)
    print()
