# CRAN comments

This is a resubmission of `densemlp` after addressing the issues raised in the previous CRAN review.

## Package notes

- The package provides a formula interface for dense multilayer perceptrons on tabular data.
- Unit tests are included under `tests/testthat/`.
- A vignette is included under `vignettes/`.

## Resubmission notes

- Torch backend handling now follows the same pattern as `survdnn`: startup is silent, seed and device helpers are guarded, and Torch is only touched when the backend is available and initialized.
- Torch-dependent tests are skipped when the backend is not installed, and the getting-started vignette does not evaluate model-training chunks without Torch.
- No additional comments for this submission.
