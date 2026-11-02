## Update submission (0.7.1 -> 0.9.0)

See NEWS.md for details.

* 0.9.0: new `weights` argument (non-negative case weights) in
  `densemlp()`, `densemlp_metrics()`, `cv_densemlp()` and
  `tune_densemlp()`. Unweighted fits are unchanged.
* 0.8.1: `ncores > 1` could hang with a threaded BLAS (MKL, OpenBLAS) in
  forked workers; each worker now runs its BLAS on one thread (new import:
  'RhpcBLASctl').
* 0.8.0 (not submitted separately): `groups`/`group_penalty` (learned
  group intercepts) and `offset` arguments, `task = "count"` (Poisson),
  fractional binary outcomes, index-based `validation`, and
  `hidden_units = integer(0)` for a linear baseline. Bug fix: the binary
  validation loss used for early stopping now uses the full Bernoulli
  cross-entropy (gradients were already correct).

## Test environments

* Local: Ubuntu 24.04, R 4.5.1

## R CMD check results

0 errors | 0 warnings | 1 note

* "Compilation used the following non-portable flag(s):
  '-mno-omit-leaf-frame-pointer'". This flag comes from the local Ubuntu R
  toolchain (`Makeconf`), not from the package's `src/Makevars`.

The installed size (dominated by the compiled `libs/` directory from
RcppArmadillo template code) is expected for a package with a native
numerical backend.

## Reverse dependencies

funcml and mimar import densemlp. The arguments of `densemlp()` that
existed in 0.7.1 are unchanged; this release only adds arguments.
