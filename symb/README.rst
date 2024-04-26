Symbolic Regression with PySR
=============================


Training logs:

* no sigmoid in pivot and tilt
* warmup maxsize, removed now
* 01-03: implied Gompertz curve, but ignored param deps in tilt
* 01-06: failed: sigmoid(lnar^2)
* tune complexities of op, const, & var
* 01-13: failed: complexities too small, so is adaptive parsimony?
* 01-16: forcing Gom at the end; ignored param deps in tilt
* 01-17: const & var complexities from 0.5 to 0.2
* 01-17: add weight to suppress near 0 & 1; reset complexities & ops;
  decrease maxsize to 40
* 01-18: incorporate Miles suggestions on perf
* 01-22: shape(ar, tweak); make adaptive parsimony great (1000) again
* 02-26: pivot tilt gompertz-polynomial fit
* 02-26: pivot tilt symbolic regression
* 02-27: re-train with all 128, with adjusted parsimony
* 02-27: re-train with all 128, with adjusted parsimony
* 02-28: re-train tilt model without zeta effective
* 03-01: re-train tilt model without zeta effective on only first 64
* test of non-universality, depending sensitively on Om, less so on Ob,
  even more weakly on all other params other than zt. Is there a 1D
  combination of all params that captures all this?
