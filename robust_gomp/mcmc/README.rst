MCMC with Cobaya and CLASS
==========================


Reionization Models:

* gomp: Gompertzian universal shape with alpha and beta shape params
* gompα: Gompertzian universal shape with only the alpha shape param and
  fixed beta
* rgomp: Gompertzian shape with parameters symbolically regressed on
  LCDM + 3 astrophysical parameters
  + rgomp1: same as rgomp
  + rgomp2: symbolic regression using half of the training data
* tanh: logistic reionization with fixed width


Cosmological Models: LCDM, Mnu, w0wa, w0waMnu


Data:

* CMB primary: Planck PR4 TTTEEE (cut), PR3 low ell (TT + Sroll2 EE), ACT DR6
* reionization history data, reio_like
  + quasar damping wing (more precise data from Joseph Hennawi?)
  + Lyman-beta forest dark gaps
* CMB lensing ϕϕ: Planck PR4, ACT DR6
* BAO DESI DR2


Chains:

* for gomp univ shape
  + data: cmb_reio, cmb_reio_lens, cmb_reio_bao, cmb_reio_lens_bao
  + theory, cosmological models except combos with w0wa but without bao
* for robust gomp SR or tanh, cover only one or a few combinations
  + tanh doesn't fit quasar damping wings, but we need some cases to
    show that, for other cases we fit it to only the Lyman-beta forest
    dark gaps part of reio_like
* How to quantify discrepancy? Compare gomp_LCDM/cmb_reio_* vs
  gomp_LCDM/cmb_reio_bao_*, to tanh_LCDM/cmb_* vs tanh_LCDM/cmb_bao_*?
