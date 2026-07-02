# CRAN comments

This is a resubmission of `densemlp` after addressing the issues raised in the previous CRAN review.

## Package notes

- The package provides a formula interface for dense multilayer perceptrons on tabular data.
- Unit tests are included under `tests/testthat/`.
- A vignette is included under `vignettes/`.

## Resubmission notes

- Torch CUDA detection is now guarded so the package checks safely on systems without GPU support.
- The package layout has been flattened and the package name now matches the CRAN-facing root package.
- No additional comments for this submission.
