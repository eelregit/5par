"""Get mass ordered halo catalogs from Quijote."""

import numpy as np


# snapshot 001 means z=2
fname = '/mnt/ceph/users/fvillaescusa/Quijote/Halos/Rockstar/fiducial_HR/0/out_1_pid.list'
catalog = np.loadtxt(fname, usecols=(2, 8, 9, 10))
print(f'{catalog.shape=}, {catalog.min(axis=0)=}, {catalog.max(axis=0)=}')
#catalog[:, 1:] /= 1000
#print('normalized:')
#print(f'{catalog.min(axis=0)=}, {catalog.max(axis=0)=}')
np.savetxt('quijote_hr_rockstar_0_z2.txt', catalog)
