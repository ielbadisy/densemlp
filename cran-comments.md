# CRAN comments: densemlp 0.6.0

This is an update from the CRAN-published 0.5.0 to 0.6.0.

## Submission notes

* Added `cv_densemlp()`: k-fold cross-validation, following the same
  formula/x-y interface and task-aware defaults as `densemlp()`/
  `tune_densemlp()`.
* The maintainer's family-name casing in `Authors@R` changed from
  "EL BADISY" to "El Badisy" (title case); same person, same email
  address, no change in maintainership.

## R CMD check results

0 errors | 0 warnings | 2 notes

* "unable to verify current time" - a local clock-check note unrelated to
  the package.
* "New maintainer" - the Maintainer field's family-name casing changed
  from "EL BADISY" to "El Badisy" (title case); same person, same email
  address, no change in maintainership.

## Test environments

* Local: Ubuntu 24.04, R 4.5.1
