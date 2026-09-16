# CRAN comments: densemlp 0.7.1

## Resubmission

Version 0.7.0 failed the incoming pretest with an installation ERROR on
r-devel Windows: the RcppArmadillo numerical kernels were left with
undefined BLAS/LAPACK references (`dgemm_`, `dgemv_`, `ddot_`, `dsyrk_`)
at link time, so no DLL was created.

Fixed by adding `src/Makevars` and `src/Makevars.win` with

    PKG_LIBS = $(LAPACK_LIBS) $(BLAS_LIBS) $(FLIBS)

so the compiled code is linked against R's BLAS/LAPACK on every platform.
This is the standard RcppArmadillo link line; it was missing because the
local Linux toolchain resolved those symbols transitively through libR.

No other changes; only the version, NEWS.md and these files were touched.

## Notes carried over from 0.7.0

This is a major update from the CRAN-published 0.5.0.

* The model backend was rewritten from `torch` to a native
  `RcppArmadillo` implementation with hand-derived closed-form gradients.
  The package no longer depends on `torch` / `libtorch` or on `ggplot2`.
  It now has compiled code (`src/`, `LinkingTo: Rcpp, RcppArmadillo`).
* Added `task = "survival"` (Cox and IPCW integrated Brier score losses), a
  formula interface, `batch_norm` and `input_projection` arguments, and
  retained `perm_importance()` and `plot_history()` on the new backend.
* The fitted object's class changed from `"densemlp_fit"` to `"densemlp"`;
  the removed `torch`-era arguments (`activation`, `optimizer`,
  `weight_decay`) are documented in `NEWS.md`.

## R CMD check results

0 errors | 0 warnings | 1 note

* "Possibly misspelled words in DESCRIPTION: Breslow, Brier, multilayer,
  natively, perceptrons" - these are all valid: "Breslow" and "Brier" are
  the standard names of the survival-analysis estimator and score,
  "multilayer perceptrons" is the standard term for the network class, and
  "natively" is used in the ordinary sense. They are listed in
  `inst/WORDLIST`.

The INFO line about installed size (~5 Mb, dominated by the compiled
`libs/` directory from RcppArmadillo template code) is expected for a
package with a native numerical backend.

## Test environments

* Local: Ubuntu 24.04, R 4.5.1
* win-builder (devel and release)
