import numpy as np


var_names = ['s8', 'ns', 'h', 'Ob', 'Om', 'zt']
var_cols = [2, 3, 4, 5, 6, 7]
var = np.loadtxt('xHI.txt', dtype='f4', usecols=var_cols)
num_sim = 128
var = var.reshape(num_sim, -1, len(var_cols))
var = var[:, 0, :]  # ignore the time dimension

pivot = np.loadtxt('pivottilt_6.txt', dtype='f4', usecols=0)

var_train = var[:64]
pivot_train = pivot[:64]
var_test = var[64:]
pivot_test = pivot[64:]


def pivot_sr(s8, ns, h, Ob, Om, zt):
    return ((-1.0389123 - s8) * (((Om * h) - Ob) + ns))  # 0226 complexity 11
    #return (((((Ob ** h) + (-0.118975304 / ((((ns + -0.3362399) * 1.591861) ** ((np.log(h) / Ob) + 6.789797)) / s8))) - ns) * np.exp(Om)) - s8)  # 0226 complexity 27


print(np.var(pivot_train - pivot_sr(*var_train.T)))
print(np.var(pivot_test - pivot_sr(*var_test.T)))
