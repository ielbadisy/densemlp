# densemlp 0.7.1

* Add `src/Makevars` / `src/Makevars.win` linking the numerical kernels
  against R's BLAS/LAPACK (`$(LAPACK_LIBS) $(BLAS_LIBS) $(FLIBS)`).
  Without the explicit link line the Armadillo calls (`dgemm_`, `dgemv_`,
  `ddot_`, `dsyrk_`, ...) were left undefined at link time on the CRAN
  Windows builder. No user-visible changes.

# densemlp 0.7.0

## Backend rewrite

* The model backend no longer uses `torch`. Forward propagation,
  backpropagation and Adam optimization are now implemented natively in
  C++ via `RcppArmadillo`, with hand-derived closed-form gradients. The
  `torch` (and transitive `libtorch`) dependency is gone, so the package
  installs and trains without downloading a deep-learning runtime.
  `ggplot2` is likewise no longer a dependency; plotting uses base
  graphics.

## New capabilities

* `task = "survival"`: train against a `survival::Surv(time, event)`
  outcome with either `loss = "cox"` (batch-wise Breslow-tie Cox partial
  likelihood, the default) or `loss = "brier"` (discrete-time hazard head
  trained on the IPCW integrated Brier score). `predict(type = "survival")`
  returns the survival-probability curve for Brier-loss models;
  `densemlp_metrics()` reports Harrell's concordance index;
  `densemlp_integrated_brier_score()` evaluates a Brier-loss model's IBS on
  new data.
* Formula interface: `densemlp(formula = y ~ ., data = df)` (including
  `survival::Surv(time, status) ~ .`) as an alternative to `x`/`y`. It must
  be passed by name, since `x`/`y` keep the first two positional slots.
* `residual`, `gated`, learned cross-feature `interaction`, exponential
  moving-average weights (`ema_decay`), learning-rate schedules
  (`lr_schedule`), internal bootstrap ensembles (`ensemble`, `ncores`) and
  a `tune_densemlp()` grid search.

## Argument changes

* New `batch_norm` (default `TRUE`): set to `FALSE` for `Linear -> ReLU`
  hidden blocks with no normalization and no BN affine parameters.
* New `input_projection`: an optional bare linear layer mapping the encoded
  predictors to a chosen dimension before the first hidden block. Cannot be
  combined with `interaction`.
* `task` now accepts `"binary"` and `"multiclass"` explicitly (in addition
  to `"regression"`, `"survival"` and the default `"auto"`).
* Removed the `torch`-era arguments that no longer apply: `activation`,
  `optimizer`, `weight_decay` and `input_projection`'s old list form. The
  fitted object's class is now `"densemlp"` (was `"densemlp_fit"`), and its
  `autoplot()` method is replaced by `plot()` / `plot_history()`.

## Retained helpers

* `perm_importance()` (model-agnostic permutation importance, now also
  covering survival) and `plot_history()` are kept, on the new backend.

# densemlp 0.6.0

* New `cv_densemlp()`: k-fold cross-validation with the same formula/x-y
  interface and task-aware defaults as `densemlp()` / `tune_densemlp()`.
* Standardized the maintainer's family-name casing to "El Badisy" in
  `Authors@R`.

# densemlp 0.5.0

* First CRAN release. No `NEWS.md` was kept prior to this version; see the
  git history for the earlier development log.
