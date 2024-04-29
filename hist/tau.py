import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from scipy.integrate import quad
from scipy.interpolate import interp1d

a, x, sigma8, ns, h, OMb, OMm = np.loadtxt('../data/xHI.txt', unpack=True, usecols=[0,1,2,3,4,5,6])

num_sim = 128
num_a = 102
a = a[:num_a]
x = x.reshape(num_sim, num_a)
sigma8 = sigma8.reshape(num_sim, num_a)
ns = ns.reshape(num_sim, num_a)
h = h.reshape(num_sim, num_a)
OMb = OMb.reshape(num_sim, num_a)
OMm = OMm.reshape(num_sim, num_a)


# new approach, we compute tau without trapezoidal
MU = 1. + 0.7428 * 0.2454
def dtau_z(z, a_array, x_array, h, OMb, OMm):
    # interpolate to redshift we can change to scale factor
    # however we can compare more readily with literature if we use z for this
    x_z = interp1d(np.flip(1/a_array - 1), np.flip(x_array), kind='linear', bounds_error=False, fill_value=(0.,1.))
    E_z = np.sqrt(OMm * (1. + z)**3 + (1 - OMm))
    return (1. - x_z(z)) * 0.0691 * (OMb * h**2) * (1. + z)**2 / (MU * h * E_z)

def tau_to_z(z_high, a_array, x_array, h, OMb, OMm):
    return quad(dtau_z, 0.000, z_high, args=(a_array, x_array, h, OMb, OMm), limit=100, epsabs=1.49e-03, epsrel=1.49e-03)[0]


# if we want the difference with tanh
def useful_y(z):
    """ used in the hyperbolic tangent model"""
    return 1.0 * pow(1.0 + z, 1.5)

def tanH_model(z_re,z):
    """ hyperbolyc tangent model, without Helium contribution """
    delta_z = 0.5 # as defined by Planck
    y_z_re = useful_y(z_re)
    y_z = useful_y(z)
    delta_y = 1.5 * pow(1.0 + z_re, 0.5) * delta_z
    temp = y_z_re - y_z
    temp = temp / (1.0 * delta_y)
    # now we would include the helium correction doing this
#    Y_factor = 0.2454
#    He_ratio = Y_factor / (1.0 - Y_factor) / 4.0
    He_ratio = 0
    x_e = (1.0 + He_ratio) * (1.0 + np.tanh(temp)) / 2.0
    return x_e

def integrand_planck(z):
    # could allow for different cosmologies if wanted
    Omega_m = 0.3088
    Omega_l = 1 - Omega_m
    omega_b = 0.0223
    h = 0.6774
    z_mid = 7.69
    E_z = np.sqrt(Omega_m*(1. + z)**3 + Omega_l)
    # from 2205.11504
    return 0.0691 * omega_b / (h * MU) * (1. + z)**2 * tanH_model(z_mid, z) / E_z

def tau_to_z_planck(z_high):
    # gives the optical depth up to a z_high redshift
    return quad(integrand_planck, 0, z_high, limit=100, epsabs=1.49e-03, epsrel=1.49e-03)[0]

# let's get tau planck first
z_tau = np.linspace(0.001, 15, num_a)
tau_planck = np.zeros(len(z_tau))
for i in range(0, len(z_tau)):
    tau_planck[i] = tau_to_z_planck(z_tau[i])

# next our sobols
taus = np.zeros((len(x),len(z_tau)))
diff_taus = np.zeros((len(x),len(z_tau)))
for i in range(0, len(x)):
    for j in range(0, len(z_tau)):
        taus[i][j] = tau_to_z(z_tau[j], a, x[i], h[i][0], OMb[i][0], OMm[i][0])
        diff_taus[i][j] = taus[i][j] - tau_planck[j]

# we want to color them by their A_s this will require mapping from sigma8 to As, luckily
# we can use the recent SR paper
def sigma8_to_As(sigma8, ns, h, OMb, OMm):
    a = np.array([1.61320734729e8, 0.343134609906, -7.859274, 18.200232, 3.666163, 0.003359])
    return ((sigma8 - a[5])/(np.log(a[4]*h) * (a[2]*OMb + np.log(a[3]*OMm))) - a[1]*ns) / a[0]

#As = np.zeros(len(sigma8))
As = sigma8_to_As(sigma8, ns, h, OMb, OMm)



# colorbar stuff
norm = matplotlib.colors.Normalize(vmin=As.min(), vmax=As.max())
c_m = matplotlib.cm.viridis
mappable = matplotlib.cm.ScalarMappable(cmap=c_m, norm=norm)
mappable.set_array([])

norm2 = matplotlib.colors.Normalize(vmin=sigma8.min(), vmax=sigma8.max())
c_m2 = matplotlib.cm.cividis
mappable2 = matplotlib.cm.ScalarMappable(cmap=c_m2, norm=norm2)
mappable2.set_array([])

# quick overwrite to check others
#norm = matplotlib.colors.Normalize(vmin=OMb.min(), vmax=OMb.max())
#c_m = matplotlib.cm.viridis
#mappable = matplotlib.cm.ScalarMappable(cmap=c_m, norm=norm)
#mappable.set_array([])


# planck constraints
planck_low = 0.047 * np.ones(len(z_tau))
planck_high = 0.061 * np.ones(len(z_tau))

plt.style.use('5par.mplstyle')
fig,axs = plt.subplots(2, 1, figsize=(6,12), sharex='col')

for i in range(0, len(x)):
    axs[0].plot(z_tau, taus[i], color=mappable.to_rgba(As[i][0]))
#    axs[0].plot(z_tau, taus[i], color=mappable.to_rgba(OMb[i][0]))
#    axs[1].plot(z_tau, diff_taus[i], color=mappable.to_rgba(OMb[i][0]))
    axs[1].plot(z_tau, diff_taus[i], color=mappable2.to_rgba(sigma8[i][0]))

#planck
axs[0].axhline(y=0.061,linestyle='dotted',color='red')
axs[0].axhline(y=0.047,linestyle='dotted',color='red')
axs[0].fill_between(z_tau, planck_low, planck_high, facecolor='gray', edgecolor='red', label=r'Planck', alpha=0.2)
#tanh
axs[0].plot(z_tau, tau_planck, '--', linewidth=7, color='black', label='Tanh')
axs[1].set_xlabel(r'Redshift', fontsize=14)
axs[0].set_ylabel(r'$\tau(z)$', fontsize=14)
axs[0].set_xlim(0.,15)
axs[0].set_ylim(0,0.085)
axs[0].legend(loc='best')
axs[1].set_ylabel(r'$\Delta \tau(z)$', fontsize=14)
plt.colorbar(mappable, label=r'$A_s$', ax=axs[0], fraction=0.046, pad=0.04)
#plt.colorbar(mappable, label=r'$\Omega_b$', ax=axs[0], fraction=0.046, pad=0.04)
#plt.colorbar(mappable, label=r'$\Omega_b$', ax=axs[1], fraction=0.046, pad=0.04)
plt.colorbar(mappable2, label=r'$\sigma_8$', ax=axs[1], fraction=0.046, pad=0.04)
plt.subplots_adjust(hspace=0)
plt.savefig('app_optical_depth.pdf')
