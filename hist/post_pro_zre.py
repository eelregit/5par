import numpy as np

from getdist import loadMCSamples
from tqdm import tqdm
from numpy.polynomial import Polynomial
from numpy import exp, log
import sys

"""
    Code to post-process the chains and find the z_re distribution using linear interp
"""


def XHI_gomp1(z, sigma8, n_s, H0, Omega_b, Omega_m, zt):
    a = 1./(1. + z)
    h = H0 / 100.
    ns = n_s
    Ob = Omega_b
    Om = Omega_m
    pivot = ((((ns - (np.log(0.11230898 * zt) * -0.35580978)) * (0.048352774 - sigma8)) - (Om + ns)) + ((Ob / Om)**h))
    tilt = ((np.log(Ob) * (((0.005659511**Om) / 0.601493) - (np.log(zt - ((Om + (ns * h))**15.051933)) - h))) + (h / sigma8))
    lnar = (np.log(a) - pivot) * tilt
    c = np.array([0.0, 1.0, 1.12988593e-01, 2.59887121e-02, 5.49059964e-04, -6.51788022e-05])
    poly = Polynomial(c)
    P5 = poly(lnar)
    return exp(- exp(P5))

def XHI_gomp2(z, sigma8, n_s, H0, Omega_b, Omega_m, zt):
    a = 1./(1. + z)
    h = H0 / 100.
    ns = n_s
    Ob = Omega_b
    Om = Omega_m
    pivot = ((((Ob / Om)**Om) - (np.log(((zt + (Ob**-0.49822742))**sigma8) * h)**0.5721157)) - (ns**1.8340757))
    tilt = (((zt - (Om**-1.583228)) / (Ob * h))**0.31627414)
    lnar = (np.log(a) - pivot) * tilt
    c = np.array([0.0, 1.0, 1.12988593e-01, 2.59887121e-02, 5.49059964e-04, -6.51788022e-05])
    poly = Polynomial(c)
    P5 = poly(lnar)
    return exp(- exp(P5))


# the gomp + SRFull + CMB + DW
chain_root = "../mcmc/gomp_1dw/gomp1"

# Load chain with GetDist, disable base statistics
samples = loadMCSamples(chain_root, settings={'ignore_rows': 0.2})

param_names = samples.getParamNames().list()
print("Parameters in chain:", param_names)

# Extract samples safely
params_chain = ["sigma8", "n_s", "H0", "Omega_b", "Omega_m", "zt"]

samples_array = np.column_stack([samples[param] for param in params_chain])

n_samples = samples_array.shape[0]

# Compute x_HI(z)
z = np.linspace(5., 30, 200)
xHI_samples = np.array([
    XHI_gomp1(z, **dict(zip(params_chain, row))) for row in tqdm(samples_array, total=n_samples)
])

id = np.argmax(xHI_samples > 0.5, axis=1)
z0, z1 = z[id - 1], z[id]
x0 = xHI_samples[np.arange(n_samples), id - 1]
x1 = xHI_samples[np.arange(n_samples), id]

z_re_samples = z0 + (0.5 - x0) * (z1 - z0) / (x1 - x0)

median_gomp1 = np.percentile(z_re_samples, 50, axis=0)
p68_gomp1 = np.percentile(z_re_samples, [16, 84], axis=0)
p95_gomp1 = np.percentile(z_re_samples, [2.5, 97.5], axis=0)

# the gomp + SRFull + CMB + DW
chain_root2 = "../mcmc/gomp_2dw/gomp2"

# Load chain with GetDist, disable base statistics
samples2 = loadMCSamples(chain_root2, settings={'ignore_rows': 0.2})

param_names2 = samples2.getParamNames().list()
print("Parameters in chain:", param_names2)

# Extract samples safely
params_chain2 = ["sigma8", "n_s", "H0", "Omega_b", "Omega_m", "zt"]

samples_array2 = np.column_stack([samples2[param] for param in params_chain2])

n_samples2 = samples_array2.shape[0]

# Compute x_HI(z)
xHI_samples2 = np.array([
    XHI_gomp2(z, **dict(zip(params_chain2, row))) for row in tqdm(samples_array2, total=n_samples2)
])


id_2 = np.argmax(xHI_samples2 > 0.5, axis=1)
print(id_2)
z0_2, z1_2 = z[id_2 - 1], z[id_2]
print(z0_2, z1_2)
x0_2 = xHI_samples2[np.arange(n_samples2), id_2 - 1]
x1_2 = xHI_samples2[np.arange(n_samples2), id_2]
print(x0_2)
print(x1_2)

z_re_samples2 = z0_2 + (0.5 - x0_2) * (z1_2 - z0_2) / (x1_2 - x0_2)

median_gomp2 = np.percentile(z_re_samples2, 50, axis=0)
p68_gomp2 = np.percentile(z_re_samples2, [16, 84], axis=0)
p95_gomp2 = np.percentile(z_re_samples2, [2.5, 97.5], axis=0)


print('The gomp1 distro: \n')
print('The median: ', median_gomp1)
print('The 1-sigma: ', p68_gomp1)
print('###')
print('The gomp2 distro: \n')
print('The median: ', median_gomp2)
print('The 1-sigma: ', p68_gomp2)
