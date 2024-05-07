import numpy as np


var_names = ['s8', 'ns', 'h', 'Ob', 'Om', 'zt']
var_cols = [2, 3, 4, 5, 6, 7]
var = np.loadtxt('xHI.txt', dtype='f4', usecols=var_cols)
num_sim = 128
var = var.reshape(num_sim, -1, len(var_cols))
var = var[:, 0, :]  # ignore the time dimension

tilt = np.loadtxt('pivottilt_6.txt', dtype='f4', usecols=1)

var_train = var[:64]
tilt_train = tilt[:64]
var_test = var[64:]
tilt_test = tilt[64:]


def tilt_sr(s8, ns, h, Ob, Om, zt):
    #return 8.331045  # 0226 complexity 1
    #return (11.389431 - np.exp((Ob / Om) + ns))  # 0226 complexity 8
    #return (7.5477395 + (Om * np.exp((ns ** (((((((Ob + 0.037888385) + ns) ** 8.331033) * h) ** zt) - np.exp(Ob / 0.02986493)) + (h ** -5.923527))) * 1.384385)))  # 0226 complexity 29
    return ((((Om ** Ob) * 9.6885805) - s8) / ((ns + 0.036959298) ** (((0.010214449 ** h) - Ob) / -0.008327238)))  # 0301 complexity 19
    #return np.exp((1.0982229 / (((Om + h) / np.exp(-0.13151546)) ** np.log(s8))) + (((ns + 0.046335895) ** (((((s8 ** ((ns / s8) - 1.3218222)) - h) ** 1.3147393) - (Ob / 0.18633473)) / (Om * 0.077069536))) ** (Ob / 0.0729048)))  # 0301 complexity 40


print(np.var(tilt_train - tilt_sr(*var_train.T)))
print(np.var(tilt_test - tilt_sr(*var_test.T)))
