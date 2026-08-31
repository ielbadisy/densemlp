# CRAN comments: densemlp 0.7.0

This is a major update from the CRAN-published 0.5.0 to 0.7.0.

## Summary of changes

* The model backend was rewritten from `torch` to a native
  `RcppArmadillo` implementation with hand-derived closed-form gradients.
  The package no longer depends on `torch` / `libtorch`, and no longer
  depends on `ggplot2`. It now has compiled code (`src/`, `LinkingTo:
  Rcpp, RcppArmadillo`).
* Added `task = "survival"` (Cox and IPCW integrated Brier score losses), a
  formula interface, `batch_norm` and `input_projection` arguments, and
  retained `perm_importance()` and `plot_history()` on the new backend.
* The fitted object's class changed from `"densemlp_fit"` to `"densemlp"`;
  the removed `torch`-era arguments (`activation`, `optimizer`,
  `weight_decay`) are documented in `NEWS.md`.

## R CMD check results

0 errors | 0 warnings | 1 note

* "Compilation used the following non-portable flag(s):
  '-mno-omit-leaf-frame-pointer'" - this flag is injected by the local
  R/compiler configuration, not by the package; it does not appear on a
  default toolchain.

There is also an INFO line: installed size ~5.2Mb, dominated by the
compiled `libs/` directory (RcppArmadillo template code), which is
expected for a package with a native numerical backend.

## Test environments

* Local: Ubuntu 24.04, R 4.5.1
