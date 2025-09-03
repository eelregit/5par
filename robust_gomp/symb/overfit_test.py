import numpy as np


var_names = ['s8', 'ns', 'h', 'Ob', 'Om', 'zt', 'Tv', 'LX']
var_cols = [2, 3, 4, 5, 6, 7, 8, 9]
var_edge = np.loadtxt('xHI.txt', dtype='f4', usecols=var_cols)
var_core = np.loadtxt('xHI_core.txt', dtype='f4', usecols=var_cols)
var = np.concatenate((var_edge, var_core), axis=0)
num_sim = 512 + 512
num_a = 127
var = var.reshape(num_sim, num_a, len(var_cols))
var = var[:, 0, :]  # ignore the time dimension

pivot, tilt = np.loadtxt('pivottilt_6.txt', dtype='f4', unpack=True)

var_train = var.reshape(2, num_sim//2, len(var_cols))[:, :256].reshape(512, -1)
pivot_train = pivot.reshape(2, num_sim//2)[:, :256].ravel()
tilt_train = tilt.reshape(2, num_sim//2)[:, :256].ravel()
var_test = var.reshape(2, num_sim//2, len(var_cols))[:, 256:].reshape(512, -1)
pivot_test = pivot.reshape(2, num_sim//2)[:, 256:].ravel()
tilt_test = tilt.reshape(2, num_sim//2)[:, 256:].ravel()


def pivot_sr(s8, ns, h, Ob, Om, zt, Tv, LX):
    return (((Ob / (Om * s8)) ** (s8 * h)) - (((((19.920519 + zt) + (0.08890569 ** (40.43043 - LX))) * s8) ** (ns / Tv)) + Om))




def tilt_sr(s8, ns, h, Ob, Om, zt, Tv, LX):
    return ((((((0.1253896 - Ob) * zt) / h) - ((s8 / Tv) / Om)) / ns) + np.exp(0.03960936 * LX))



print('pivot:')
print(np.var(pivot_train - pivot_sr(*var_train.T)))
print(np.var(pivot_test - pivot_sr(*var_test.T)))
print('tilt:')
print(np.var(tilt_train - tilt_sr(*var_train.T)))
print(np.var(tilt_test - tilt_sr(*var_test.T)))
