#!/usr/bin/env python

import sys

import numpy as np
import py21cmfast as p21c

# not sure of how code is going to run, so easier thing to generalize is if I focus on
# generating xHI for given set of parameters from the Sobol sample

# Let's get the neutral hydrogen history
# Global parameters: can include WDM and so on

Sobol_table = np.loadtxt('sobol.txt')


def gen_history(i):
    sigma_8, n_s, h, Omega_b, Omega_m, zeta_eff = Sobol_table[i]

    with p21c.global_params.use(
            OMn=0.0, OMk =0.0, OMr=8.6e-5 , OMtot=1, Y_He=0.245, wl=-1.0,
            SMOOTH_EVOLVED_DENSITY_FIELD =1, R_smooth_density=0.2, HII_ROUND_ERR= 1e-5,
            N_POISSON=-1 , MAX_DVDR=0.2, DELTA_R_FACTOR=1.1, DELTA_R_HII_FACTOR=1.1,
            OPTIMIZE_MIN_MASS=1e11, SHETH_b=0.15, SHETH_c=0.05,
            ZPRIME_STEP_FACTOR=1.02):
        lightcone = p21c.run_lightcone(
            redshift = 5.0, # minimum redshift, choose 5 to compare with Catalina's
            max_redshift = 15.0, # 15 to compare with Catalina's but probably quite high
            lightcone_quantities=("brightness_temp", 'xH_box'), # need brightness to use spin, but only need xH_box
            global_quantities=("brightness_temp", 'xH_box'),
            user_params = {"HII_DIM": 256, "BOX_LEN":300,  "DIM":768, "N_THREADS":128}, # let's use a smaller box of 300 cMPc since does not matter
            cosmo_params = p21c.CosmoParams(
                SIGMA_8=sigma_8,
                hlittle=h,
                OMm=Omega_m,
                OMb=Omega_b,
                POWER_INDEX=n_s),
            astro_params = {'R_BUBBLE_MAX':50, 'L_X':40.5, "HII_EFF_FACTOR":zeta_eff},
            flag_options = {"INHOMO_RECO": True, "USE_TS_FLUCT":True, "USE_MASS_DEPENDENT_ZETA": False },
            random_seed=12345,
            direc = f'tmp{i:03d}', #here it is where I want the cached-boxes to be stored
            write = False,
        )

    # to make PySR easier for us let's massage the final data to have the parameters at each (z, xHI)
    ave_xH = lightcone.global_xH
    z = lightcone.node_redshifts
    ave_xH = np.array(ave_xH)
    z_array = np.array(z)
    zeta_eff_array = np.ones(len(z))
    OMb_array = np.ones(len(z))
    OMm_array = np.ones(len(z))
    ns_array = np.ones(len(z))
    sigma8_array = np.ones(len(z))
    hlittle_array = np.ones(len(z))

    zeta_eff_array = zeta_eff_array * lightcone.astro_params.HII_EFF_FACTOR
    OMb_array = OMb_array * lightcone.cosmo_params.OMb
    OMm_array = OMm_array * lightcone.cosmo_params.OMm
    ns_array = ns_array * lightcone.cosmo_params.POWER_INDEX
    sigma8_array =  sigma8_array * lightcone.cosmo_params.SIGMA_8
    hlittle_array = hlittle_array * lightcone.cosmo_params.hlittle

    np.savetxt(
        f'timelines/xHI_{i}.txt',
        np.stack([z_array, ave_xH, sigma8_array, ns_array, hlittle_array, OMb_array,
            OMm_array, zeta_eff_array], axis=0),
        delimiter=' ',
        fmt='%e')


if __name__ == "__main__":
    i_start, i_stop = int(sys.argv[1]), int(sys.argv[2])

    for i in range(i_start, i_stop):  # not including i_stop
        gen_history(i)
