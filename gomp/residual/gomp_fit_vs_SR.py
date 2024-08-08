import numpy as np
from numpy.polynomial import Polynomial
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.interpolate import interp1d
from scipy.integrate import quad
import sys

# core or edge?
if len(sys.argv) != 2:
        print("Usage: python gomp_fit_vs_SR.py <input_string>")
        print("Inputs: core or edge")
        sys.exit(1)

input = sys.argv[1]

if input == 'core':
    print('Doing core')
    a, x, s8, ns, h, Ob, Om, zt = np.loadtxt('../../data/xHI_core.txt', unpack=True, usecols=[0,1,2,3,4,5,6,7])
    d = [0.0, 1.0, 1.15994003e-01, 2.69992540e-02, 6.53080653e-04, -7.13122781e-05] # core
    pivot_a, tilt_a = np.loadtxt('pivottilt_6_core.txt', unpack=True)
    num_sim = 128
elif input == 'total':
    print('Doing core + edge')
    a_core, x_core, s8_core, ns_core, h_core, Ob_core, Om_core, zt_core = np.loadtxt('../../data/xHI_core.txt', unpack=True, usecols=[0,1,2,3,4,5,6,7])
    a_edge, x_edge, s8_edge, ns_edge, h_edge, Ob_edge, Om_edge, zt_edge = np.loadtxt('../../data/xHI.txt', unpack=True, usecols=[0,1,2,3,4,5,6,7])
    d = [0.0, 1.0, 1.12988593e-01, 2.59887121e-02, 5.49059964e-04, -6.51788022e-05]
    pivot_a, tilt_a = np.loadtxt('../pivottilt_6.txt', unpack=True)
    a = np.concatenate((a_edge, a_core))
    x = np.concatenate((x_edge, x_core))
    s8 = np.concatenate((s8_edge, s8_core))
    ns = np.concatenate((ns_edge, ns_core))
    h = np.concatenate((h_edge, h_core))
    Ob = np.concatenate((Ob_edge, Ob_core))
    Om = np.concatenate((Om_edge, Om_core))
    zt = np.concatenate((zt_edge, zt_core))
    num_sim = 128 + 128
else:
    print('Doing edge')
    a, x, s8, ns, h, Ob, Om, zt = np.loadtxt('../../data/xHI.txt', unpack=True, usecols=[0,1,2,3,4,5,6,7])
    d = [0.0, 1.0, 0.109881282, 0.0245923405, 0.000280982197, -7.76864358e-05] # large zeta
    pivot_a, tilt_a = np.loadtxt('../pivottilt_6.txt', unpack=True)
    num_sim = 128


    

num_a = 127
a = a.reshape(num_sim, num_a)
x = x.reshape(num_sim, num_a)
s8 = s8.reshape(num_sim, num_a)
ns = ns.reshape(num_sim, num_a)
h = h.reshape(num_sim, num_a)
Om = Om.reshape(num_sim, num_a)
Ob = Ob.reshape(num_sim, num_a)
zt = zt.reshape(num_sim, num_a)
lna = np.log(a)
xp = - np.gradient(x, lna[0], axis=1)  # - dx/dlna


P6 = Polynomial(d)
P6_deriv = Polynomial(d).deriv()

# grab the fit
pivot_fit = np.ones((num_sim,num_a))
tilt_fit = np.ones((num_sim,num_a))

for i in range(0,num_sim):
    pivot_fit[i,:] = pivot_a[i]
    tilt_fit[i,:] = tilt_a[i]


def fit_xHI(lna, pivot_fit, tilt_fit):
    lnar = (lna - pivot_fit) * tilt_fit
    return np.exp(- np.exp(P6(lnar)))
    
def fit_grad(lna, pivot_fit, tilt_fit):
    lnar = (lna - pivot_fit) * tilt_fit
    exponentials = np.exp(P6(lnar))
    gompertz = np.exp(- exponentials)
    return gompertz * exponentials * P6_deriv(lnar)


def SR_edge_core_gomp1_comp22_comp25(s8, ns, h, Ob, Om, zt):
    #pareto-hull contact #1
    pivot_SR = ((((ns - (np.log(0.11230898 * zt) * -0.35580978)) * (0.048352774 - s8)) - (Om + ns)) + ((Ob / Om)**h))
    tilt_SR = ((np.log(Ob) * (((0.005659511**Om) / 0.601493) - (np.log(zt - ((Om + (ns * h))**15.051933)) - h))) + (h / s8))
    return pivot_SR, tilt_SR

def SR_edge_core_gomp2_comp22_comp11(s8, ns, h, Ob, Om, zt):
    #pareto-hull contact #1
    pivot_SR = ((((Ob / Om)**Om) - (np.log(((zt + (Ob**-0.49822742))**s8) * h)**0.5721157)) - (ns**1.8340757))
    tilt_SR = (((zt - (Om**-1.583228)) / (Ob * h))**0.31627414)
    return pivot_SR, tilt_SR

def SR_gomp1_comp9_comp1(s8, ns, h, Ob, Om, zt):
    pivot_SR = (Ob**Om - s8 - ns) / 0.68492776
    tilt_SR = 7.62781
    return pivot_SR, tilt_SR


def SR_gomp1_comp16_comp9(s8, ns, h, Ob, Om, zt):
    pivot_SR = ((81.40768 * Ob / zt)**h - s8 - np.exp(Ob)) * (ns + Om)
    tilt_SR = ((zt - Om**(-1.5069426))/ Ob)**0.3348278
    return pivot_SR, tilt_SR
    
def SR_gomp1_comp19_comp15(s8, ns, h, Ob, Om, zt):
    # this is the one with pareto-convex hull touch
    pivot_SR = ((85.08853 * Ob / zt - Ob)**h - ns - s8/ns) * (ns + Om)
    tilt_SR = np.log(1./s8 * (3.9115524 * Om / Ob)**(np.log(zt) - h - Om))
    return pivot_SR, tilt_SR
    
def SR_gomp1_comp19_comp28(s8, ns, h, Ob, Om, zt):
    # this is the one with pareto-convex hull touch
    pivot_SR = ((85.08853 * Ob / zt - Ob)**h - ns - s8/ns) * (ns + Om)
    tilt_SR = np.log(((np.exp(1.3495045) / (Ob / Om))**(np.log(zt) - (((h + (Om * ns))**(np.log(h) + ns))**np.exp(h + Ob)))) / s8)
    return pivot_SR, tilt_SR
    
    
def SR_gomp1_comp30_comp28(s8, ns, h, Ob, Om, zt):
    # this is the one with pareto-convex hull touch
    pivot_SR = (((((Ob / (0.7067988 / (ns**-1.3519806))) / np.exp(s8))**(h * ((0.03342909 * zt)**h))) - (ns + ((Om - Ob) + 0.14749499))) * (0.797138 + s8))
    tilt_SR = np.log(((np.exp(1.3495045) / (Ob / Om))**(np.log(zt) - (((h + (Om * ns))**(np.log(h) + ns))**np.exp(h + Ob)))) / s8)
    return pivot_SR, tilt_SR
    
    
    
    
def SR_gomp2_comp22_comp10(s8, ns, h, Ob, Om, zt):
    # pareto-convex hull first touch
    pivot_SR = np.log(0.33022612/s8 * Om**(-0.50538677) * ((zt + s8**0.41064402 / Ob) * h**0.7619934)**(-0.50538677 * ns))
    tilt_SR = (np.log(zt) + Om/h)**(Ob**-0.1410175)
    return pivot_SR, tilt_SR



def SR_gomp2_comp22_comp27(s8, ns, h, Ob, Om, zt):
    # pareto-convex hull first touch
    pivot_SR = np.log(0.33022612/s8 * Om**(-0.50538677) * ((zt + s8**0.41064402 / Ob) * h**0.7619934)**(-0.50538677 * ns))
    tilt_SR = (((Ob**-0.32638636) + ((Om - ((np.log(ns + h) + Om)**zt))**s8)) * ((np.log(zt) - (h / 1.1266154)) + (-0.06340148 / Om)))
    return pivot_SR, tilt_SR


def SR_gomp2_comp36_comp27(s8, ns, h, Ob, Om, zt):
    # pareto-convex hull first touch
    pivot_SR = np.log(((((((((zt + ((s8 / Ob)**0.872185)) * h)**ns) + 0.7606463) * Om) * ns)**(-0.47219694 + (Ob**((Om + 0.66671)**((h + Om)**2.8784664))))) * 0.24528304) / s8)
    tilt_SR = (((Ob**-0.32638636) + ((Om - ((np.log(ns + h) + Om)**zt))**s8)) * ((np.log(zt) - (h / 1.1266154)) + (-0.06340148 / Om)))
    return pivot_SR, tilt_SR



def SR_gomp2_comp9_comp1(s8, ns, h, Ob, Om, zt):
    pivot_SR =  (ns + Om) * (-h - s8 / 0.8919422)
    tilt_SR = 7.6348734
    return pivot_SR, tilt_SR

def SR_xHI(lna, pivot_SR, tilt_SR):
    lnar = (lna - pivot_SR) * tilt_SR
    return np.exp(- np.exp(P6(lnar)))

def SR_grad(lna, pivot_SR, tilt_SR):
    lnar = (lna - pivot_SR) * tilt_SR
    exponentials = np.exp(P6(lnar))
    gompertz = np.exp(- exponentials)
    return gompertz * exponentials * P6_deriv(lnar)
    
plt.style.use('../..//5par.mplstyle')
fig, axes = plt.subplots(nrows=3, ncols=3, sharex=True,
                             gridspec_kw={'wspace': 0.25, 'hspace': 0.}, figsize=(9, 4))

#pivot_gomp1, tilt_gomp1 = SR_gomp1_comp19_comp15(s8, ns, h, Ob, Om, zt)
pivot_gomp1, tilt_gomp1 = SR_edge_core_gomp1_comp22_comp25(s8, ns, h, Ob, Om, zt)
lna_rescaled_SR = (lna - pivot_gomp1) * tilt_gomp1
a_rescaled_SR = np.exp(lna_rescaled_SR)
xp_rescaled_SR = xp / tilt_gomp1

#pivot_gomp2, tilt_gomp2 = SR_gomp2_comp22_comp10(s8, ns, h, Ob, Om, zt)
pivot_gomp2, tilt_gomp2 = SR_edge_core_gomp2_comp22_comp11(s8, ns, h, Ob, Om, zt)
lna_rescaled_SR2 = (lna - pivot_gomp2) * tilt_gomp2
a_rescaled_SR2 = np.exp(lna_rescaled_SR2)
xp_rescaled_SR2 = xp / tilt_gomp2

lna_rescaled_fit = (lna - pivot_fit) * tilt_fit
a_rescaled_fit = np.exp(lna_rescaled_fit)
xp_rescaled_fit = xp / tilt_fit


axes[0,0].set_title('Poly fit')
axes[0,0].plot(a_rescaled_fit.T, fit_xHI(lna, pivot_fit, tilt_fit).T - x.T, c='C6', lw=0.3, alpha=0.2)
axes[0,0].set_ylabel(r'$\Delta x_\mathrm{HI}$')

axes[0,1].set_title('SR fit gomp1')
axes[0,1].plot(a_rescaled_SR.T, SR_xHI(lna, pivot_gomp1, tilt_gomp1).T - x.T, c='C5', lw=0.3, alpha=0.2)
axes[0,2].set_title('SR fit gomp2')
axes[0,2].plot(a_rescaled_SR2.T, SR_xHI(lna, pivot_gomp2, tilt_gomp2).T - x.T, c='C4', lw=0.3, alpha=0.2)
axes[0,2].set_ylim(-0.08, 0.08)
axes[0,0].set_ylim(-0.08, 0.08)
axes[0,1].set_ylim(-0.08, 0.08)



#axes[0,1].set_ylabel(r'$\Delta x_\mathrm{HI}$')


axes[1,1].plot(a_rescaled_SR.T, SR_grad(lna, pivot_gomp1, tilt_gomp1).T - xp_rescaled_SR.T, c='C5', lw=0.3, alpha=0.2)
axes[1,2].plot(a_rescaled_SR2.T, SR_grad(lna, pivot_gomp2, tilt_gomp2).T - xp_rescaled_SR2.T, c='C4', lw=0.3, alpha=0.2)
axes[1,2].set_ylim(-0.09, 0.09)
axes[1,0].set_ylim(-0.09, 0.09)
axes[1,1].set_ylim(-0.09, 0.09)
#axes[1,1].set_ylabel(r'$\Delta - \mathrm{d}x_\mathrm{HI}/\mathrm{d}\ln\tilde{a}$')

axes[1,0].plot(a_rescaled_fit.T, fit_grad(lna, pivot_fit, tilt_fit).T - xp_rescaled_fit.T, c='C6', lw=0.3, alpha=0.2)
axes[1,0].set_ylabel(r'$\Delta - \mathrm{d}x_\mathrm{HI}/\mathrm{d}\ln\tilde{a}$')
#axes[1,0].set_yscale('log')


axes[2,0].plot(a_rescaled_fit.T, np.log(-np.log(fit_xHI(lna, pivot_fit, tilt_fit).T)) - np.log(-np.log(x.T)), c='C06', lw=0.3, alpha=0.2)
axes[2,0].set_ylabel(r'$\Delta \ln(-\ln x_\mathrm{HI})$')
#axes[2,0].set_ylim(-9, 4)
axes[2,0].set_xlabel(r'$\tilde{a}$')
axes[2,0].set_xscale('log')
axes[2,0].set_xlim(2e-3, 8)


axes[2,1].plot(a_rescaled_SR.T, np.log(-np.log(SR_xHI(lna, pivot_gomp1, tilt_gomp1).T)) - np.log(-np.log(x.T)), c='C5', lw=0.3, alpha=0.2)
axes[2,2].plot(a_rescaled_SR2.T, np.log(-np.log(SR_xHI(lna, pivot_gomp2, tilt_gomp2).T)) - np.log(-np.log(x.T)), c='C4', lw=0.3, alpha=0.2)
axes[2,2].set_ylim(-1.1, 1.1)
axes[2,0].set_ylim(-1.1, 1.1)
axes[2,1].set_ylim(-1.1, 1.1)
axes[2,1].set_xlabel(r'$\tilde{a}$')
axes[2,1].set_xscale('log')
axes[2,1].set_xlim(2e-3, 8)

if input == 'core':
    fig.savefig(f'residuals_core.pdf')
elif input == 'total':
    fig.savefig(f'residuals_total.pdf')
else:
    fig.savefig(f'residuals.pdf') # large

plt.close(fig)


# let's compute the mean squared error of tau

def z_to_a(z):
    return 1. / (1. + z)
    
MU = 1. + 0.7428 * 0.2454


# first let's compute the approx. SR taus

def dtau_SR1(z, s8, ns, h, Ob, Om, zt):
    aa = z_to_a(z)
#    pivot_SR, tilt_SR = SR_gomp1_comp19_comp15(s8, ns, h, Ob, Om, zt)
    pivot_SR, tilt_SR = SR_edge_core_gomp1_comp22_comp25(s8, ns, h, Ob, Om, zt)
    xHI = SR_xHI(np.log(aa), pivot_SR, tilt_SR)
    if xHI >= 0.990:
        xHI = 1.
    E_z = np.sqrt(Om * (1. + z)**3 + (1 - Om))
    dtau = (1. - xHI) * 0.0691 * (Ob * h**2) * (1. + z)**2 / (MU * h * E_z)
    return dtau

def tau_SR1(s8, ns, h, Ob, Om, zt):
    return quad(dtau_SR1, 0.000, 20, args=(s8, ns, h, Ob, Om, zt), limit=100, epsabs=1.49e-05, epsrel=1.49e-05)[0]


def dtau_SR2(z, s8, ns, h, Ob, Om, zt):
    aa = z_to_a(z)
#    pivot_SR, tilt_SR = SR_gomp2_comp22_comp10(s8, ns, h, Ob, Om, zt)
    pivot_SR, tilt_SR = SR_edge_core_gomp2_comp22_comp11(s8, ns, h, Ob, Om, zt)
    xHI = SR_xHI(np.log(aa), pivot_SR, tilt_SR)
    if xHI >= 0.990:
        xHI = 1.
    E_z = np.sqrt(Om * (1. + z)**3 + (1 - Om))
    dtau = (1. - xHI) * 0.0691 * (Ob * h**2) * (1. + z)**2 / (MU * h * E_z)
    return dtau

def tau_SR2(s8, ns, h, Ob, Om, zt):
    return quad(dtau_SR2, 0.000, 15, args=(s8, ns, h, Ob, Om, zt), limit=100, epsabs=1.49e-05, epsrel=1.49e-05)[0]

#  for robust gomp we want to see fit tau too, add it here
def dtau_fi(z, i_sim):
    aa = z_to_a(z)
    piv = pivot_a[i]
    til = tilt_a[i]
    xHI = fit_xHI(np.log(aa), piv, til)
    if xHI >= 0.990:
        xHI = 1.
    E_z = np.sqrt(Om[i_sim, 0] * (1. + z)**3 + (1 - Om[i_sim, 0]))
    dtau = (1. - xHI) * 0.0691 * (Ob[i_sim, 0] * h[i_sim, 0]**2) * (1. + z)**2 / (MU * h[i_sim, 0] * E_z)
    return dtau

def tau_fi(i_sim):
    return quad(dtau_fi, 0.000, 15, args=(i_sim), limit=100, epsabs=1.49e-05, epsrel=1.49e-05)[0]


# we also need the x values
def dtau_x(z, i_sim):
    x_HI = interp1d(a[i_sim,:], x[i_sim,:], kind='linear')
    aa = z_to_a(z)
    
#    print('made it here')
    
    if z < 2.:
        xHI = 0.
    else:
        xHI = x_HI(aa)
        if xHI >= 0.990:
            xHI = 1.
    E_z = np.sqrt(Om[i_sim, 0] * (1. + z)**3 + (1 - Om[i_sim, 0]))
    dtau = (1. - xHI) * 0.0691 * (Ob[i_sim, 0] * h[i_sim, 0]**2) * (1. + z)**2 / (MU * h[i_sim, 0] * E_z)
    return dtau

def tau_x(i_sim):
    return quad(dtau_x, 0.000, 15, args=(i_sim), limit=100, epsabs=1.49e-05, epsrel=1.49e-05)[0]

tau_gomp1 = np.zeros(num_sim)
tau_gomp2 = np.zeros(num_sim)
tau_xx = np.zeros(num_sim)
tau_fit = np.zeros(num_sim)
MSE_gomp1 = 0
MSE_gomp2 = 0
MSE_fit = 0
MAE_gomp1 = 0
MAE_gomp2 = 0
MAE_fit = 0
#c_gomp1 = np.zeros(num_sim)
#c_gomp2 = np.zeros(num_sim)
for i in range(0, num_sim):
    tau_gomp1[i] = tau_SR1(s8[i,0], ns[i,0], h[i,0], Ob[i,0], Om[i,0], zt[i,0])
    tau_gomp2[i] = tau_SR2(s8[i,0], ns[i,0], h[i,0], Ob[i,0], Om[i,0], zt[i,0])
    tau_xx[i] = tau_x(i)
    tau_fit[i] = tau_fi(i)
    MSE_gomp1 += (tau_gomp1[i] - tau_xx[i])**2 / tau_xx[i]**2
    MAE_gomp1 += np.abs(tau_gomp1[i] - tau_xx[i]) / tau_xx[i]
    MSE_gomp2 += (tau_gomp2[i] - tau_xx[i])**2 / tau_xx[i]**2
    MAE_gomp2 += np.abs(tau_gomp2[i] - tau_xx[i]) / tau_xx[i]
    MSE_fit += (tau_fit[i] - tau_xx[i])**2 / tau_xx[i]**2
    MAE_fit += np.abs(tau_fit[i] - tau_xx[i]) / tau_xx[i]
    # fishing for outliers
#    c_gomp1[i] = np.abs(tau_gomp1[i] - tau_xx[i]) / tau_xx[i]
#    c_gomp2[i] = np.abs(tau_gomp2[i] - tau_xx[i]) / tau_xx[i]
#    
#    if (i != 85 and i != 87 and i != 108 and i!= 118 and i!=26 and i!=44 and i!=100 and i!=104 and i!=127):
#        MSE_gomp1 += (tau_gomp1[i] - tau_xx[i])**2 / tau_xx[i]**2
#        
#        MAE_gomp1 += np.abs(tau_gomp1[i] - tau_xx[i]) / tau_xx[i]
#        
#    if (i != 85 and i != 76 and i != 119 and i != 59 and i!=79 and i!=82 and i!=95 and i!=98 and i!=51 and i!=60 and i!=88 and i!=90 and i!=95 and i!=98 and i!=104 and i!=106 and i!=113):
#        MSE_gomp2 += (tau_gomp2[i] - tau_xx[i])**2 / tau_xx[i]**2
#        MAE_gomp2 += np.abs(tau_gomp2[i] - tau_xx[i]) / tau_xx[i]


 
print('App. tau for gomp1 is ', tau_gomp1)
print('App. tau for gomp2 is ', tau_gomp2)
print('App. tau for fit is ', tau_fit)
print('App. tau for x is ', tau_xx)
print('sq. MSE rel. error for gomp1 is ', np.sqrt(MSE_gomp1 / (1. * num_sim)) )
print('sq. MSE rel. error for gomp2 is ', np.sqrt(MSE_gomp2 / (1. * num_sim)) )
print('sq. MSE rel. error for fit is ', np.sqrt(MSE_fit / (1. * num_sim)) )
print('MAE rel. error for gomp1 is ', MAE_gomp1 / (1. * num_sim) )
print('MAE rel. error for gomp2 is ', MAE_gomp2 / (1. * num_sim) )
print('MAE rel. error for fit is ', MAE_fit / (1. * num_sim) )

# fishing for outliers
#print('sq. MSE rel. error for gomp1 is ', np.sqrt(MSE_gomp1 / (1. * num_sim-9)) )
#print('sq. MSE rel. error for gomp2 is ', np.sqrt(MSE_gomp2 / (1. * num_sim-15)) )
#print('MAE rel. error for gomp1 is ', MAE_gomp1 / (1. * num_sim-9) )
#print('MAE rel. error for gomp2 is ', MAE_gomp2 / (1. * num_sim-15) )

#print('##### outliers? #####')
#i_max_gomp1 = np.where(c_gomp1 >= 0.50 * c_gomp1.max())[0]
#i_min_gomp1 = np.where(c_gomp1 <= 10 * c_gomp1.min())[0]
#print('Who has the maximum difference? Gomp 1: ', c_gomp1[i_max_gomp1], i_max_gomp1)
#print('Corresponding parameters are: ', s8[i_max_gomp1][0], ns[i_max_gomp1][0], h[i_max_gomp1][0], Ob[i_max_gomp1][0], Om[i_max_gomp1][0], zt[i_max_gomp1][0])
#print('Who has the minimum difference? Gomp 1: ', c_gomp1[i_min_gomp1], i_min_gomp1)
#i_max_gomp2 = np.where(c_gomp2 >= 0.50 * c_gomp2.max())[0]
#i_min_gomp2 = np.where(c_gomp2 <= 10 * c_gomp2.min())[0]
#print('Who has the maximum difference? Gomp 2: ', c_gomp2[i_max_gomp2], i_max_gomp2)
#print('Corresponding parameters are: ', s8[i_max_gomp2][0], ns[i_max_gomp2][0], h[i_max_gomp2][0], Ob[i_max_gomp2][0], Om[i_max_gomp2][0], zt[i_max_gomp2][0])
#print('Who has the minimum difference? Gomp 2: ', c_gomp2[i_min_gomp2], i_min_gomp2)

#print('############### Gomp1 ###############')
#print('############### s8 ###############')
#s8_out = np.array([s8[26][0], s8[44][0], s8[85][0], s8[87][0], s8[100][0], s8[104][0], s8[108][0], s8[118][0], s8[127][0]])
#print('Values: ', s8_out)
##print('Values: ', s8_out.sort())
#print('############### ns ###############')
#ns_out = np.array([ns[26][0], ns[44][0], ns[85][0], ns[87][0], ns[100][0], ns[104][0], ns[108][0], ns[118][0], ns[127][0]])
#print('Values: ', ns_out)
#
#print('############### h ###############')
#h_out = np.array([h[26][0], h[44][0], h[85][0], h[87][0], h[100][0], h[104][0], h[108][0], h[118][0], h[127][0]])
#print('Values: ', h_out)
#
#print('############### Ob ###############')
#Ob_out = np.array([Ob[26][0], Ob[44][0], Ob[85][0], Ob[87][0], Ob[100][0], Ob[104][0], Ob[108][0], Ob[118][0], Ob[127][0]])
#print('Values: ', Ob_out)
#
#print('############### Om ###############')
#Om_out = np.array([Om[26][0], Om[44][0], Om[85][0], Om[87][0], Om[100][0], Om[104][0], Om[108][0], Om[118][0], Om[127][0]])
#print('Values: ', Om_out)
#
#print('############### zt ###############')
#zt_out = np.array([zt[26][0], zt[44][0], zt[85][0], zt[87][0], zt[100][0], zt[104][0], zt[108][0], zt[118][0], zt[127][0]])
#print('Values: ', zt_out)
