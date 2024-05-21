import numpy as np
import matplotlib.pyplot as plt
from numpy.polynomial import Polynomial
from numpy import exp, log
from scipy.interpolate import interp1d

import matplotlib.gridspec as gridspec

"""
    Code is in charge of producing the "best-fit" reionization timelines obtained from the CMB analysis
"""


def xHI(z, sigma8, ns, h, Ob, Om, model):
    a = 1./ (1. + z)
    if model == '0227':
        pivot = (h**Om + sigma8) * (Ob - ns - Om)
        tilt = 8.331045
    elif model == '0226':
        pivot = (-1.0389123 - sigma8) * (Om * h - Ob + ns)
        tilt = 8.289981
    lnar = (np.log(a) - pivot) * tilt
    c = np.array([0, 1, 0.15034337, 0.04849586, 0.00526138, 0.0002182])
    poly = Polynomial(c)
    P5 = poly(lnar)
    return exp(- exp(P5))

# tanh stuuf
def useful_y(z):
    """ used in the hyperbolic tangent model"""
    return 1.0 * pow(1.0 + z, 1.5)
def tanH_model(z_re,z):
    """ hyperbolyc tangent model, without Helium contribution """
    delta_z = 0.5 # as defined by Planck
    y_z_re = useful_y(z_re)
    y_z = useful_y(z)
    delta_y = 1.5 * pow(1.0 + z_re, 0.5) * delta_z # this is definitely a derivative
    temp = y_z_re - y_z
    temp = temp / (1.0 * delta_y)
    # include helium in a consistent fashion with CLASS for this plot
    # now we will include the helium correction
#    Y_factor = 0.2454
#    He_ratio = Y_factor / (1.0 - Y_factor) / 4.0
    He_ratio = 0
    x_e = (1.0 + He_ratio) * (1.0 + np.tanh(temp)) / 2.0
    return x_e


z_vals, tau_gomp1_z = np.loadtxt('gomp1_tau_z.txt', unpack=True)
z_vals, tau_gomp2_z = np.loadtxt('gomp2_tau_z.txt', unpack=True)
z_vals, tau_tanh0_z = np.loadtxt('tanh0_tau_z.txt', unpack=True)

# planck constraints
planck_low = 0.047 * np.ones(len(z_vals))
planck_high = 0.061 * np.ones(len(z_vals))

low_0227 = (0.05115 - 0.00061) * np.ones(len(z_vals))
high_0227 = (0.05115 + 0.00061) * np.ones(len(z_vals))

low_0226 = (0.05140 - 0.00063) * np.ones(len(z_vals))
high_0226 = (0.05140 + 0.00063) * np.ones(len(z_vals))

val = 0.02233 / 0.6722**2
val2 = 0.02233 / 0.6724**2

# need two figures actually one full with everything including caption
# and the second a chibi version with only the main point
#z = np.linspace(5.8, 16, 100, endpoint=True) - 1
z = np.linspace(5., 16, 100, endpoint=True) - 1
b = 0.7
plt.style.use('../5par.mplstyle')



gs = gridspec.GridSpec(2, 1, height_ratios=[1, 1], hspace=0)

fig = plt.figure(figsize=(5.5, 6.2))

ax1 = fig.add_subplot(gs[0], sharex=ax2)
ax1.fill_between(z_vals, planck_low, planck_high, facecolor='darkblue', edgecolor='darkblue', label=r'Planck PR3', alpha=0.1)
ax1.fill_between(z_vals, low_0227, high_0227, facecolor='green', edgecolor='green', label=r'gomp 1', alpha=0.2)
ax1.fill_between(z_vals, low_0226, high_0226, facecolor='purple', edgecolor='pink', label=r'gomp 2', alpha=0.2)
#ax1.set_xlabel(r'$z$', fontsize=14)
ax1.set_ylabel(r'$\tau$', fontsize=14)
#ax1.set_xlim(2,15)
ax1.set_ylim(0.0301, 0.065)
#ax1.set_ylim(0.0301, 0.065)
ax1.legend(loc='lower right')
ax1.plot(1+z_vals, tau_tanh0_z, '-', linewidth=3, color='darkblue', label='Planck PR3')
ax1.plot(1+z_vals, tau_gomp1_z, '--', linewidth=3, color='green', label='gomp 1')
ax1.plot(1+z_vals, tau_gomp2_z, ':', linewidth=3, color='purple', label='gomp 2')

ax2 = fig.add_subplot(gs[1])
ax2.plot(1+z, xHI(z,sigma8=0.8092,ns=0.9634,h=0.6722,Ob=val,Om=0.3171,model='0227'),  '--', c='green', linewidth=3, label=r'gomp 1', zorder=7)
ax2.plot(1+z, xHI(z,sigma8=0.8092,ns=0.9634,h=0.6724,Ob=val2,Om=0.3168,model='0226'),  ':', c='purple', linewidth=3, label=r'gomp 2', zorder=8, alpha=0.8)
ax2.axvline(x=5.90, ls='dashed', c='purple')
#ax1.plot(1+z,(1.0 - tanH_model(7.67 - 0.75, z)),':',c='darkblue',zorder=5)
#ax1.plot(1+z,(1.0 - tanH_model(7.67 + 0.75, z)),':',c='darkblue',zorder=5)
ax2.plot(1+z,(1.0 - tanH_model(7.67, z)),linewidth=3,c='darkblue',label=r'Planck PR3',zorder=5)
#ax1.errorbar(6.9,0.11,0.06,uplims=True,marker='o',markersize=7, ls='none', label=r'Dark Pixel 2015',zorder=6, alpha=b)
ax2.errorbar(7.3,0.79,0.04,uplims=True,marker='o', c='gold', markersize=7, ls='none', label=r'Dark Pixel 2023',zorder=6, alpha=b)
ax2.errorbar(7.5,0.87,0.03,uplims=True,marker='o', c='gold', markersize=7, ls='none', zorder=6, alpha=b)
ax2.errorbar(7.7,0.94,0.09,uplims=True,marker='o', c='gold', markersize=7, ls='none', zorder=6, alpha=b)
ax2.errorbar(7.1,0.69,0.06,uplims=True,marker='o', c='gold', markersize=7, ls='none', zorder=6, alpha=b)
ax2.errorbar(7.9,0.4,0.1,uplims=False,lolims=True,marker='s',markersize=7, ls='none', label=r'Ly$\alpha$ Fraction',zorder=6, alpha=b)
ax2.errorbar(7.6,0.5,0.1,uplims=True,lolims=False,marker='*',markersize=7, ls='none', label=r'LAE Clustering',zorder=6, alpha=b)
ax2.errorbar(8.,0.7,yerr=np.array([[0.23,0.20]]).T,uplims=False,marker='p',markersize=7,capthick=2,capsize=4, ls='none', label=r'J0252-0503',zorder=6, alpha=b)
ax2.errorbar(8.5,0.39,yerr=np.array([[0.13,0.22]]).T,uplims=False,marker='p',markersize=7,capthick=2,capsize=4, ls='none', label=r'J1007+2115',zorder=6, alpha=b)
ax2.errorbar(8.1,0.4,yerr=np.array([[0.19,0.21]]).T,uplims=False,marker='p',markersize=7,capthick=2,capsize=4, ls='none', label=r'J1120+0641',zorder=6, alpha=b)
ax2.errorbar(8.5,0.21,yerr=np.array([[0.19,0.17]]).T,uplims=False,marker='p',markersize=7,capthick=2,capsize=4, ls='none', label=r'J1342+0928a',zorder=6, alpha=b)
# this is really a 7.5 that I moved, so let's put it back
ax2.errorbar(8.5,0.56,yerr=np.array([[0.18,0.21]]).T,uplims=False,marker='p',markersize=7,capthick=2,capsize=4, ls='none', label=r'J1342+0928b',zorder=6, alpha=b)
# lets also move this one back to 7.5
ax2.errorbar(8.5,0.60,yerr=np.array([[0.23,0.20]]).T,uplims=False,marker='p',markersize=7,capthick=2,capsize=4, c='rosybrown', ls='none', label=r'J1342+0928c',zorder=6, alpha=b)
ax2.errorbar(8.1,0.48,yerr=np.array([[0.26,0.26]]).T,uplims=False,marker='p',markersize=7,capthick=2,capsize=4, c='rosybrown', ls='none', label=r'J1120+0641',zorder=6, alpha=b)
ax2.errorbar(8.29,0.49,yerr=np.array([[0.11,0.11]]).T,uplims=False,marker='p',markersize=7,capthick=2,capsize=4, ls='none', label=r'Combined quasars',zorder=6, alpha=b)
ax2.errorbar(8.0,0.59,yerr=np.array([[0.15,0.11]]).T,xerr=0.5,uplims=False,marker='d',markersize=7,capthick=2,capsize=4, ls='none', label=r'Ly$\alpha$ EWa',zorder=6, alpha=b)
ax2.errorbar(8.6,0.88,yerr=np.array([[0.10,0.05]]).T,xerr=0.6,uplims=False,marker='d',markersize=7,capthick=2,capsize=4, ls='none', label=r'Ly$\alpha$ EWb',zorder=6, alpha=b)
ax2.errorbar(9.0,0.76,yerr=0.22,xerr=0.6,uplims=False,lolims=True,marker='d',markersize=7,capthick=2,capsize=4, ls='none', label=r'Ly$\alpha$ EWc',zorder=6, alpha=b)

ax2.set_ylabel(r'$x_\mathrm{HI}$', fontsize=14)
#ax1.set_ylim(0, 1)
#ax2.set_xlim(6.8, 13)
ax2.set_xlim(6., 13)
ax2.set_xscale('log')
ax2.set_xlabel(r'$z$',fontsize=14)
#ax2.set_xticks(range(7, 17), [str(z) for z in range(6, 16)])
ax2.set_xticks(range(6, 17), [str(z) for z in range(5, 16)])
handles, labels = ax2.get_legend_handles_labels()
line_legend = ax2.legend(handles[:3], labels[:3], loc='center right', bbox_to_anchor=(1, 0.72))
ax2.add_artist(line_legend)
ax2.legend(handles[3:], labels[3:], loc='lower right', ncols=2,
           borderpad=0, handletextpad=0.2, columnspacing=0.5)
#plt.margins(0,0)
plt.setp(ax1.get_xticklabels(), visible=False)
plt.subplots_adjust(hspace=0)
plt.tight_layout()
plt.savefig('history.pdf')






# chibi version
fig = plt.figure(figsize=(3,3))
ax1 = fig.add_subplot(111)
# the curves
ax1.plot(z, xHI(z,sigma8=0.8092,ns=0.9634,h=0.6722,Ob=val,Om=0.3171,model='0227'), '--', color='green', linewidth=3, label=r'0227', zorder=7)
ax1.axvline(x=5.90, ls='dashed', color='purple')
ax1.plot(z,(1.0 - tanH_model(7.67 - 0.73, z)),':',color='darkblue',label=r'Planck 1-$\sigma$',zorder=5)
ax1.plot(z,(1.0 - tanH_model(7.67 + 0.73, z)),':',color='darkblue',zorder=5)
ax1.plot(z,(1.0 - tanH_model(7.67, z)),linewidth=3,color='darkblue',zorder=5)
ax1.errorbar(5.9,0.11,0.06,uplims=True,marker='o',markersize=7,label=r'Dark Pixel 2015',zorder=6, alpha=b)
ax1.errorbar(6.3,0.79,0.04,uplims=True,marker='o', color='gold',markersize=7,label=r'Dark Pixel 2023',zorder=6, alpha=b)
ax1.errorbar(6.5,0.87,0.03,uplims=True,marker='o', color='gold',markersize=7,zorder=6, alpha=b)
ax1.errorbar(6.7,0.94,0.09,uplims=True,marker='o', color='gold',markersize=7,zorder=6, alpha=b)
ax1.errorbar(6.1,0.69,0.06,uplims=True,marker='o', color='gold',markersize=7,zorder=6, alpha=b)
ax1.errorbar(6.9,0.4,0.1,uplims=False,lolims=True,marker='s',markersize=7,label=r'Ly$\alpha$ Fraction',zorder=6, alpha=b)
ax1.errorbar(6.6,0.5,0.1,uplims=True,lolims=False,marker='*',markersize=7,label=r'LAE Clustering',zorder=6, alpha=b)
ax1.errorbar(7.,0.7,yerr=np.array([[0.23,0.20]]).T,uplims=False,marker='p',markersize=7,capthick=2,capsize=4,label=r'J0252-0503',zorder=6, alpha=b)
ax1.errorbar(7.5,0.39,yerr=np.array([[0.13,0.22]]).T,uplims=False,marker='p',markersize=7,capthick=2,capsize=4,label=r'J1007+2115',zorder=6, alpha=b)
ax1.errorbar(7.1,0.4,yerr=np.array([[0.19,0.21]]).T,uplims=False,marker='p',markersize=7,capthick=2,capsize=4,label=r'J1120+0641',zorder=6, alpha=b)
ax1.errorbar(7.5,0.21,yerr=np.array([[0.19,0.17]]).T,uplims=False,marker='p',markersize=7,capthick=2,capsize=4,label=r'J1342+0928a',zorder=6, alpha=b)
# this is really a 7.5 that I moved, so let's put it back
ax1.errorbar(7.5,0.56,yerr=np.array([[0.18,0.21]]).T,uplims=False,marker='p',markersize=7,capthick=2,capsize=4,label=r'J1342+0928b',zorder=6, alpha=b)
# lets also move this one back to 7.5
ax1.errorbar(7.5,0.60,yerr=np.array([[0.23,0.20]]).T,uplims=False,marker='p',markersize=7,capthick=2,capsize=4, color='rosybrown', label=r'J1342+0928c',zorder=6, alpha=b)
ax1.errorbar(7.1,0.48,yerr=np.array([[0.26,0.26]]).T,uplims=False,marker='p',markersize=7,capthick=2,capsize=4, color='rosybrown', label=r'J1120+0641',zorder=6, alpha=b)
ax1.errorbar(7.29,0.49,yerr=np.array([[0.11,0.11]]).T,uplims=False,marker='p',markersize=7,capthick=2,capsize=4,label=r'Combined quasars',zorder=6, alpha=b)
ax1.errorbar(7.0,0.59,yerr=np.array([[0.15,0.11]]).T,xerr=0.5,uplims=False,marker='d',markersize=7,capthick=2,capsize=4,label=r'Ly$\alpha$ EWa',zorder=6, alpha=b)
ax1.errorbar(7.6,0.88,yerr=np.array([[0.10,0.05]]).T,xerr=0.6,uplims=False,marker='d',markersize=7,capthick=2,capsize=4,label=r'Ly$\alpha$ EWb',zorder=6, alpha=b)
ax1.errorbar(8.0,0.76,yerr=0.22,xerr=0.6,uplims=False,lolims=True,marker='d',markersize=7,capthick=2,capsize=4,label=r'Ly$\alpha$ EWc',zorder=6, alpha=b)

ax1.set_ylabel(r'$x_\mathrm{HI}$')
#ax1.set_ylim(0, 1)
ax1.set_xlim(6, 10)
#ax1.set_yscale('log')
#ax1.set_xlabel(r'$\ln a_\mathrm{rescaled}$')
ax1.set_xlabel(r'$z$')
#ax1.legend(loc='center left',bbox_to_anchor=(1.,0.5))
#plt.savefig('history_chibi.pdf')

# let's use this to also figure out the midpoint
inter = interp1d(xHI(z,sigma8=0.8092,ns=0.9634,h=0.6722,Ob=val,Om=0.3171,model='0227'),z, kind='linear')

inter2 = interp1d(xHI(z,sigma8=0.8092,ns=0.9634,h=0.6724,Ob=val2,Om=0.3168,model='0226'),z, kind='linear')

print('#'*5+' 0227 '+'#'*5)
print('Midpoint is ', inter(0.5))
print('Duration -- 0.1 to 0.99 completion -- is ', inter(0.1) - inter(0.97))
print('Redshift 1: ', inter(0.05))
print('Redshift 2: ', inter(0.95))

print('#'*5+' 0226 '+'#'*5)
print('Midpoint is ', inter2(0.5))
print('Duration -- 0.1 to 0.99 completion -- is ', inter2(0.1) - inter2(0.97))
print('Redshift 1: ', inter2(0.05))
print('Redshift 2: ', inter2(0.95))


# the chibi one is going to be a xHI(a)
a, x = np.loadtxt('../data/xHI.txt', unpack=True)[:2]
num_sim = 128
num_a = 92
a = a[:num_a]
x = x.reshape(num_sim, num_a)

plt.style.use('../5par.mplstyle')
#plt.tick_params(
#    axis='both',
#    which='both',
#    bottom='False',
#    top='False')
plt.rcParams['xtick.bottom'] = False
plt.rcParams['xtick.labelbottom'] = False
plt.rcParams['ytick.left'] = False
plt.rcParams['ytick.labelleft'] = False
plt.rcParams['xtick.top'] = False
plt.rcParams['ytick.right'] = False
fig = plt.figure(figsize=(3,2))
ax1 = fig.add_subplot(111)
ax1.plot(a.T, x.T, c='gray', lw=0.3, alpha=0.2)
#ax1.set_xlabel(r'$a$')
#ax1.set_ylabel(r'$x_\mathrm{HI}$')
plt.savefig('21cmfast_chibi.pdf')
