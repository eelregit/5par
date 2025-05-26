import numpy as np

p6, t6 = np.loadtxt('pivottilt_6.txt', unpack=True)
pr, tr = np.loadtxt('pivottilt_rat.txt', unpack=True)
print('Comparing', len(p6), 'pivots and tilts, from fitting using polynomial vs rational function')

print('max of (ln) pivot diff is', np.abs(pr - p6).max())
print('std of (ln) pivot diff is', (pr - p6).std())
print('max of tilt abs diff is', np.abs(tr - t6).max())
print('std of tilt abs diff is', (tr - t6).std())
