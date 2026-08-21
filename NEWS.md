# densemlp 0.6.0

## New features

* New `cv_densemlp()`: k-fold cross-validation, following the same
  formula/x-y interface and task-aware defaults as `densemlp()`/
  `tune_densemlp()`. Fits `densemlp()` on each training fold and scores
  the held-out fold via `densemlp_metrics()`, returning per-fold metrics
  plus a mean/SD summary. Has its own `print()` method.

## Other

* Standardized the maintainer's family-name casing to "El Badisy" (title
  case) in `Authors@R`, matching the rest of the package suite.

# densemlp 0.5.0

* Version published on CRAN. No `NEWS.md` was kept prior to this release;
  see the git history for the full development log.
