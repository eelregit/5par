MCMC with Cobaya and CLASS
==========================


* `gomp_0 <gomp_0>`_ receives tau_reio as input and finds reionization
  timeline using bisection (variyng alpha), gomp_0 is used for
  validation
* `gomp_1 <gomp_1>`_ finds tau_reio using alpha(cosmology)
* `gomp_2 <gomp_2>`_ finds tau_reio using alpha(cosmology) but the SR
  expression was obtained using only half (64) of our data

* `tanh_0 <tanh_0>`_ receives tau_reio as input and finds reionization
  timeline using bisection (varying z_re), tanh_o is used for validation
* `tanh_1 <tanh_1>`_ finds tau_reio by sampling over z_re


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
