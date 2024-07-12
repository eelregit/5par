import numpy as np


var_names = ['s8', 'ns', 'h', 'Ob', 'Om', 'zt']
var_cols = [2, 3, 4, 5, 6, 7]
var_edge = np.loadtxt('xHI.txt', dtype='f4', usecols=var_cols)
var_core = np.loadtxt('xHI_core.txt', dtype='f4', usecols=var_cols)
var = np.concatenate((var_edge, var_core), axis=0)
num_sim = 128 + 128
num_a = 127
var = var.reshape(num_sim, num_a, len(var_cols))
var = var[:, 0, :]  # ignore the time dimension

pivot, tilt = np.loadtxt('pivottilt_6.txt', dtype='f4', unpack=True)

var_train = var.reshape(2, num_sim//2, len(var_cols))[:, :64].reshape(128, -1)
pivot_train = pivot.reshape(2, num_sim//2)[:, :64].ravel()
tilt_train = tilt.reshape(2, num_sim//2)[:, :64].ravel()
var_test = var.reshape(2, num_sim//2, len(var_cols))[:, 64:].reshape(128, -1)
pivot_test = pivot.reshape(2, num_sim//2)[:, 64:].ravel()
tilt_test = tilt.reshape(2, num_sim//2)[:, 64:].ravel()


def pivot_sr(s8, ns, h, Ob, Om, zt):
    return (Ob / Om) ** Om - np.log((zt + Ob ** -0.49822742) ** s8 * h) ** 0.5721157 - ns ** 1.8340757


def tilt_sr(s8, ns, h, Ob, Om, zt):
    return ((zt - Om ** -1.583228) / (Ob * h)) ** 0.31627414


print('pivot:')
print(np.var(pivot_train - pivot_sr(*var_train.T)))
print(np.var(pivot_test - pivot_sr(*var_test.T)))
print('tilt:')
print(np.var(tilt_train - tilt_sr(*var_train.T)))
print(np.var(tilt_test - tilt_sr(*var_test.T)))
