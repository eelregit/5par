MCMC with Cobaya and CLASS
==========================


Using Planck PR3 TTTEEE ϕϕ, low-ell data, and optionally quasar data:

* `gomp_0 <gomp_0>`_ receives tau_reio as input and finds reionization
  timeline using bisection varying alpha while fixing beta.
  gomp_0 is used for validation
* `gomp_1 <gomp_1>`_ finds tau_reio using alpha(cosmo) and beta(cosmo)
* `gomp_1dw <gomp_1dw>`_ finds tau_reio using alpha(cosmo) and
  beta(cosmo), using additionally quasar damping wing data
* `gomp_1dw_logprior <gomp_1dw_logprior>`_ is similar but with log prior
* `gomp_1dwlf <gomp_1dwlf>`_ finds tau_reio using alpha(cosmo) and
  beta(cosmo), using additionally quasar damping wing and luminosity
  function data
* `gomp_2 <gomp_2>`_ finds tau_reio using alpha(cosmo) and beta(cosmo)
  but the SR expression was obtained using only half of the data
* `gomp_2dw <gomp_2dw>`_ finds tau_reio using alpha(cosmo) and
  beta(cosmo) but the SR expression was obtained using only half of the
  data, using additionally quasar damping wing data
* `gomp_2dwlf <gomp_2dwlf>`_ finds tau_reio using alpha(cosmo) and
  beta(cosmo) but the SR expression was obtained using only half of the
  data, using additionally quasar damping wing and luminosity function
  data

* `tanh_0 <tanh_0>`_ receives tau_reio as input and finds reionization
  timeline using bisection (varying z_re), tanh_o is used for validation
* `tanh_1 <tanh_1>`_ finds tau_reio by sampling over z_re
* `tanh_1dw <tanh_1dw>`_ finds tau_reio by sampling over z_re, using
  additionally quasar damping wing data
* `tanh_1dwlf <tanh_1dwlf>`_ finds tau_reio by sampling over z_re, using
  additionally quasar damping wing and luminosity function data


Using various combinations of datasets:

* CMB primary and lensing

  + Planck PR4 TTTEEE + PR3 low ell's, alternatives are

    - cmbP: Planck PR4 TTTEEE
    - cmbA: ACT DR4
    - cmbS: SPT 3G

  + SOTA (Planck PR4 ϕϕ + ACT DR6), alternatives are

    - lensP: Planck PR4 ϕϕ
    - lensA: ACT DR6
    - lensS: SPT 3G

  + quasar damping wing data needed to pin down zeta_UV/tau_reio

* BAO and/or SNe

  + baoD: DESI (alternative is baoS: SDSS eBOSS DR16)

    - baoDx: without the 1st LRG bin
    - baoDg: without Lya forest bin

  + snD: DES Y5
  + snP: Pantheon+
  + snU: Union3


Chaining log:

* CLASS or CAMB? Given that CLASS already has sigma8 as input, the
  choice should be clear.
* Deal with skeleton structure of "struct" dependences to reach n_s in
  thermodynamics.c
* Grabbing sigma8 issue
* add helium I reio contribution consistently
* add helium II reio contribution at low-z
* modify thermodynamics_reionization_fuction to include gomp instead of
  tanh
* adjust z_start in class:
  + z_start = z_mid in inputs hopefully not confusing
* achieve functional code
* preliminary comparison of Gomp vs Tanh (look at Cl's)
  + matched optical depths (0.05660 vs 0.05657),
  + situation would be different if matched z(xHI=0.5) since tanh reio
    does a poor job
* Allow for tau_reio input (with As): find pivot with bisection
* Choose actual pivot/tilt implementation
  + use 0226 and 0227 (simple)
* cobaya test run
  + which likelihood?
* bias
* constraint improvement
